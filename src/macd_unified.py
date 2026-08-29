from __future__ import annotations
import logging
import aot_bridge
import os
import sys
import time
import asyncio
import signal
import uuid
import argparse
import psutil
import gc
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

# ── bot_config : only what macd_unified.py touches directly ──
from bot_config import (
    Constants, PIVOT_LEVELS_BUY, PIVOT_LEVELS_SELL,
    TRACE_ID, PAIR_ID, cfg, logger, logger_main,
    format_ist_time, MEMORY_CHECK_INTERVAL_PAIRS, validate_runtime_config,
    json_dumps, json_loads, JSON_BACKEND, shutdown_event, __version__,
)
# ── fetcher : only orchestrator-level I/O ──
from fetcher import (
    SessionManager, DataFetcher, PriceData, parse_candles_to_numpy,
)

# ── indicators : only helpers macd_unified.py calls directly ──
from indicators import (
    get_utc_date_key, should_reset_daily_state, warmup_if_needed,
    _normalize_samples, _prune_stale_samples, _oi_funding_gate_reason,
)

# ── state / gates / alerts : trim to direct usage ──
from state import (
    _blanket_reset_pair, _clear_all_redis_states, build_products_map_from_cfg,
    RedisKeyPrefix, RedisStateStore, RedisLock,
)

from gates import compute_confluence_score, _eval_gate

from alerts import (
    TelegramQueue, ALERT_KEYS, _eval_alerts, _apply_and_dispatch_alerts, escape_markdown_v2,
) 
_pair_eval_counter = 0

def _sync_signal_handler(sig: int, frame: Any) -> None:
    logger.warning(f"Received signal {sig}, initiating async shutdown...")
    try:
        loop = asyncio.get_running_loop()
        loop.call_soon_threadsafe(shutdown_event.set)
    except RuntimeError:
        pass

signal.signal(signal.SIGTERM, _sync_signal_handler)
signal.signal(signal.SIGINT, _sync_signal_handler)

_STARTUP_BANNER_PRINTED = False
def print_startup_banner_once() -> None:
    global _STARTUP_BANNER_PRINTED
    if _STARTUP_BANNER_PRINTED:
        return
    _STARTUP_BANNER_PRINTED = True
    logger.info(
        f"📡 Bot v{__version__} | Pairs: {len(cfg.PAIRS)} | Workers: {cfg.MAX_PARALLEL_FETCH} | "
        f"Timeout: {cfg.RUN_TIMEOUT_SECONDS}s | Redis Lock: {cfg.REDIS_LOCK_EXPIRY}s"
    )
print_startup_banner_once()

def get_trigger_timestamp() -> int:
    trigger_ts_str = os.getenv("TRIGGER_TIMESTAMP")
    if trigger_ts_str:
        try:
            trigger_ts = int(trigger_ts_str)
            now = int(time.time())
            if abs(now - trigger_ts) > 600:
                logger.warning(f"TRIGGER_TIMESTAMP ({trigger_ts}) is >10 min from now ({now}), using current time")
                return now
            logger.debug(f"Using TRIGGER_TIMESTAMP from env: {trigger_ts}")
            return trigger_ts
        except (ValueError, TypeError):
            logger.warning(f"Invalid TRIGGER_TIMESTAMP: {trigger_ts_str}, using current time")
    
    return int(datetime.now(timezone.utc).timestamp())

async def evaluate_pair_and_alert(pair_name: str, data_15m: PriceData, data_5m: PriceData,
    data_daily: Optional[Dict[str, np.ndarray]], sdb: RedisStateStore, telegram_queue: TelegramQueue, correlation_id: str,
    reference_time: int, fetcher: DataFetcher, symbol: str, alerts_sent_ref: List[int] = None, alerts_sent_lock: asyncio.Lock = None,
    max_alerts_per_run: int = cfg.MAX_ALERTS_PER_RUN,
    oi_gate_data: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[Tuple[str, Dict[str, Any]]]:

    logger_pair = logging.getLogger(f"macd_bot.{pair_name}.{correlation_id}")

    pair_oi = (oi_gate_data or {}).get(pair_name)
    gr = await _eval_gate(pair_name, data_15m, data_5m, data_daily, sdb, correlation_id, reference_time, pair_oi)
    if gr is None:
        return None
    if isinstance(gr, tuple):
        return gr  # hard reject / wick reject / gate blocked -- already final

    confluence_score: Optional[float] = None
    confluence_total: Optional[float] = None
    confluence_votes: Optional[Dict[str, bool]] = None

    reversal_eligible = (
        (cfg.ENABLE_STRONG_REVERSAL_ALERT or cfg.ENABLE_OB_GATE)
        and (gr.buy_trend_common_relaxed or gr.sell_trend_common_relaxed)
    )
    gate_passed = gr.buy_common or gr.sell_common or reversal_eligible
    buy_side = gr.buy_common or gr.buy_trend_common or gr.buy_trend_common_relaxed

    confluence_score_buy: Optional[float] = None
    confluence_total_buy: Optional[float] = None
    confluence_votes_buy: Optional[Dict[str, bool]] = None
    confluence_score_sell: Optional[float] = None
    confluence_total_sell: Optional[float] = None
    confluence_votes_sell: Optional[Dict[str, bool]] = None

    if cfg.ENABLE_CONFLUENCE_GATE and gate_passed:
        score_buy, total_buy, votes_buy = compute_confluence_score(gr, is_buy=True)
        score_sell, total_sell, votes_sell = compute_confluence_score(gr, is_buy=False)
        confluence_score_buy, confluence_total_buy, confluence_votes_buy = score_buy, total_buy, votes_buy
        confluence_score_sell, confluence_total_sell, confluence_votes_sell = score_sell, total_sell, votes_sell

        score, total = (score_buy, total_buy) if buy_side else (score_sell, total_sell)
        pct_floor = total * (cfg.CONFLUENCE_MIN_PCT / 100.0)
        required = max(pct_floor, cfg.CONFLUENCE_MIN_ABS_SCORE)
        if score < required:
            logger_pair.debug(
                f"[{pair_name}] Confluence gate blocked: {score:.1f}/{total:.1f} weighted score "
                f"(need {required:.1f}, pct-floor={pct_floor:.1f}, "
                f"abs-floor={cfg.CONFLUENCE_MIN_ABS_SCORE:.1f}) — skipping Phase-2 indicators"
            )
            await _blanket_reset_pair(sdb, pair_name, logger_pair)
            return pair_name, {
                "state": "NO_SIGNAL",
                "ts": int(time.time()),
                "summary": {
                    "alerts": 0,
                    "future_cloud": "green" if gr.cloud_up else "red" if gr.cloud_down else "neutral",
                    "hist_rma": 0.0,
                    "suppression": f"Confluence gate: {score:.1f}/{total:.1f} weighted score, need {required:.1f}"
                }
            }
        confluence_score = score
        confluence_total = total
        confluence_votes = votes

    if cfg.ENABLE_OI_FUNDING_FILTER and not cfg.ENABLE_CONFLUENCE_GATE and gate_passed:
        if pair_oi is not None:
            oi_reason = _oi_funding_gate_reason(
                pair_oi.get("oi_now"), pair_oi.get("oi_history", []),
                pair_oi.get("funding"), pair_oi.get("funding_history", []), is_buy=buy_side,
                oi_usd_now=pair_oi.get("oi_usd_now"),
                price_now=pair_oi.get("price_now"), price_history=pair_oi.get("price_history", []),
            )
            if oi_reason is not None:
                logger_pair.info(f"[{pair_name}] {oi_reason}")
                fetcher.fetch_stats["oi_funding_blocks"] += 1
                await _blanket_reset_pair(sdb, pair_name, logger_pair)
                return pair_name, {
                    "state": "NO_SIGNAL",
                    "ts": int(time.time()),
                    "summary": {
                        "alerts": 0,
                        "future_cloud": "green" if gr.cloud_up else "red" if gr.cloud_down else "neutral",
                        "hist_rma": 0.0,
                        "suppression": oi_reason
                    }
                }
    if cfg.ENABLE_OB_GATE and not cfg.ENABLE_CONFLUENCE_GATE and gate_passed:
        ob_ok = gr.ob_gate_ok_buy if buy_side else gr.ob_gate_ok_sell
        if ob_ok is False:
            ob_reason = gr.ob_gate_reason or "OB gate: zone touched, no reversal confirmed"
            logger_pair.info(f"[{pair_name}] {ob_reason}")
            await _blanket_reset_pair(sdb, pair_name, logger_pair)
            return pair_name, {
                "state": "NO_SIGNAL",
                "ts": int(time.time()),
                "summary": {
                    "alerts": 0,
                    "future_cloud": "green" if gr.cloud_up else "red" if gr.cloud_down else "neutral",
                    "hist_rma": 0.0,
                    "suppression": ob_reason
                }
            }
    try: 
        alert_result = await _eval_alerts(gr, data_5m, data_daily, reference_time, sdb, correlation_id, logger_pair)
        if alert_result is None:
            return None
        if isinstance(alert_result, tuple) and len(alert_result) == 2:
            return alert_result  # reserved: RuntimeError path inside _eval_alerts
        context, conditional_states, raw_alerts = alert_result

        return await _apply_and_dispatch_alerts(
            gr, context, conditional_states, raw_alerts, sdb, telegram_queue, fetcher, symbol,
            correlation_id, logger_pair, alerts_sent_ref, alerts_sent_lock, max_alerts_per_run,
            data_5m,
            confluence_score_buy=confluence_score_buy,
            confluence_total_buy=confluence_total_buy,
            confluence_votes_buy=confluence_votes_buy,
            confluence_score_sell=confluence_score_sell,
            confluence_total_sell=confluence_total_sell,
            confluence_votes_sell=confluence_votes_sell,
        )
    finally:
        PAIR_ID.set("")
        global _pair_eval_counter
        _pair_eval_counter += 1
        if _pair_eval_counter % MEMORY_CHECK_INTERVAL_PAIRS == 0:
            try:
                process = psutil.Process()
                current_memory_mb = process.memory_info().rss / 1024 / 1024
                memory_limit_mb = cfg.MEMORY_LIMIT_BYTES / 1024 / 1024
                if current_memory_mb > (memory_limit_mb * 0.8):
                    logger_pair.warning(f"Memory spike: {current_memory_mb:.0f}MB / {memory_limit_mb:.0f}MB")
            except Exception:
                pass

async def guarded_eval(task_data, state_db, telegram_queue, correlation_id, reference_time, fetcher,
                       alerts_sent_ref=None, alerts_sent_lock=None, max_alerts_per_run=cfg.MAX_ALERTS_PER_RUN,
                       oi_gate_data: Optional[Dict[str, Dict[str, Any]]] = None):

    p_name, symbol, candles = task_data

    try:
        pd_15m = parse_candles_to_numpy(candles.get("15"))
        pd_5m = parse_candles_to_numpy(candles.get("5"))
        pd_daily = parse_candles_to_numpy(candles.get("D")) if (cfg.ENABLE_PIVOT or cfg.ENABLE_CPR) else None

        if pd_15m is None:
            logger_main.warning(f"Skipping {p_name}: 15m parse failed")
            return None
        
        if pd_5m is None:
            logger_main.warning(f"Skipping {p_name}: 5m parse failed")
            return None

        data_15m = pd_15m
        data_5m = pd_5m
        data_daily = pd_daily.as_dict() if pd_daily is not None else None

        result = await evaluate_pair_and_alert(
            p_name, data_15m, data_5m, data_daily,
            state_db, telegram_queue, correlation_id, reference_time, fetcher, symbol,
            alerts_sent_ref, alerts_sent_lock, max_alerts_per_run,
            oi_gate_data=oi_gate_data
        )
        return result
    
    except asyncio.CancelledError:
        logger_main.warning(f"Evaluation cancelled for {p_name}")
        raise
    
    except Exception as e:
        logger_main.error(f"Error in {p_name} evaluation: {e}", exc_info=False)
        return None
    
    finally:
        pass

async def process_pairs_with_workers(fetcher: DataFetcher, products_map: Dict[str, dict],
    pairs_to_process: List[str], state_db: RedisStateStore, telegram_queue: TelegramQueue,
    correlation_id: str, lock: RedisLock, reference_time: int,
    alerts_sent_ref: List[int] = None, alerts_sent_lock: asyncio.Lock = None,
    max_alerts_per_run: int = cfg.MAX_ALERTS_PER_RUN) -> List[Tuple[str, Dict[str, Any]]]:

    ticker_task = None
    if cfg.ENABLE_OI_FUNDING_FILTER:
        ticker_task = asyncio.create_task(fetcher.fetch_tickers_batch())

    logger_main.info(f"🔡 Phase 1: Fetching candles for {len(pairs_to_process)} pairs...")
    fetch_start = time.time()
    limit_15m = 300
    limit_5m = max(
        Constants.MIN_CANDLES_FOR_INDICATORS + Constants.CANDLE_SAFETY_BUFFER,
        cfg.RMA_200_PERIOD * 3 
    )
    daily_limit = cfg.PIVOT_LOOKBACK_PERIOD if (cfg.ENABLE_PIVOT or cfg.ENABLE_CPR) else 0
    fetch_daily = cfg.ENABLE_PIVOT or cfg.ENABLE_CPR
    pair_requests = []
    valid_tasks = []     
    daily_symbols = []

    for pair_name in pairs_to_process:
        product_info = products_map.get(pair_name)
        if not product_info:
            continue

        resolutions = [("15", limit_15m), ("5", limit_5m)]
        pair_requests.append((pair_name, resolutions))
        valid_tasks.append((pair_name, pair_name))
        if fetch_daily:
            daily_symbols.append(pair_name)

    all_candles = {}
    daily_task = None
    miss_symbols = []

    if fetch_daily and daily_symbols:
        day_key = get_utc_date_key(reference_time)
        cache_keys = [f"daily_cache:{sym}:{day_key}" for sym in daily_symbols]
        cached_map = await state_db.batch_get_metadata(cache_keys)

        for sym, ck in zip(daily_symbols, cache_keys):
            raw = cached_map.get(ck)
            if raw:
                all_candles.setdefault(sym, {})["D"] = json_loads(raw)
            else:
                miss_symbols.append(sym)

        if miss_symbols:
            daily_task = asyncio.gather(*(
                fetcher.fetch_daily_cached(state_db, sym, daily_limit, reference_time)
                for sym in miss_symbols
            ), return_exceptions=True)

    live_candles = await fetcher.fetch_all_candles_truly_parallel(
        pair_requests, reference_time
    )
    for sym, res in live_candles.items():
        all_candles.setdefault(sym, {}).update(res)

    if daily_task is not None:
        daily_results = await daily_task
        for symbol, daily_data in zip(miss_symbols, daily_results):
            if isinstance(daily_data, Exception):
                logger_main.warning(f"Daily fetch failed for {symbol}: {daily_data}")
                daily_data = None
            all_candles.setdefault(symbol, {})["D"] = daily_data

    fetch_elapsed = time.time() - fetch_start
    logger_main.info(f"🌀 Phase 1 complete: {fetch_elapsed:.1f}s")

    oi_gate_data: Dict[str, Dict[str, Any]] = {}

    if cfg.ENABLE_OI_FUNDING_FILTER:
        oi_funding_map = await ticker_task if ticker_task else {}
        matched_oi = sum(
            1 for p in pairs_to_process
            if (oi_funding_map.get(products_map.get(p, {}).get("symbol", p)) or {}).get("oi") is not None
        )
        logger_main.info(
            f"📈 OI/funding: {matched_oi}/{len(pairs_to_process)} pairs have OI data this run"
        )

        now_ts = int(time.time())

        # Pass 1 — which pairs actually have OI data?
        oi_entries = []                 # (pair_name, current, meta_key)
        meta_keys_to_read = []

        for pair_name in pairs_to_process:
            product_info = products_map.get(pair_name)
            if not product_info:
                continue
            current = oi_funding_map.get(pair_name)
            if not current or current.get("oi") is None:
                continue
            meta_key = f"oi_hist:{pair_name}"
            oi_entries.append((pair_name, current, meta_key))
            meta_keys_to_read.append(meta_key)

        # Pass 2 — ONE round-trip for ALL history reads
        prev_raw_map = await state_db.batch_get_metadata(meta_keys_to_read)

        new_histories: Dict[str, str] = {}
        for (pair_name, current, meta_key) in oi_entries:
            oi_hist, funding_hist, price_hist = [], [], []
            prev_raw = prev_raw_map.get(meta_key)
            if prev_raw:
                try:
                    payload = json_loads(prev_raw)
                    oi_hist      = _normalize_samples(payload.get("oi_samples", []) or [])
                    funding_hist = _normalize_samples(payload.get("funding_samples", []) or [])
                    price_hist   = _normalize_samples(payload.get("price_samples", []) or [])
                except Exception:
                    oi_hist, funding_hist, price_hist = [], [], []

            oi_hist      = _prune_stale_samples(oi_hist, cfg.OI_FUNDING_MAX_SAMPLE_AGE_SEC, now_ts)
            funding_hist = _prune_stale_samples(funding_hist, cfg.OI_FUNDING_MAX_SAMPLE_AGE_SEC, now_ts)
            price_hist   = _prune_stale_samples(price_hist, cfg.OI_FUNDING_MAX_SAMPLE_AGE_SEC, now_ts)

            oi_gate_data[pair_name] = {
                "oi_now": current["oi"],
                "oi_usd_now": current.get("oi_value_usd"),
                "oi_history": oi_hist,
                "funding": current.get("funding"),
                "funding_history": funding_hist,
                "price_now": current.get("price"),
                "price_history": price_hist,
            }
            new_oi_hist = (oi_hist + [[now_ts, current.get("oi")]])[-cfg.OI_FUNDING_HISTORY_LEN:]
            new_funding_hist = (funding_hist + [[now_ts, current.get("funding")]])[-cfg.OI_FUNDING_HISTORY_LEN:]
            new_price_hist = (price_hist + [[now_ts, current.get("price")]])[-cfg.OI_FUNDING_HISTORY_LEN:]

            new_histories[meta_key] = json_dumps({
                "oi_samples": new_oi_hist, "funding_samples": new_funding_hist,
                "price_samples": new_price_hist, "ts": now_ts,
            })

        # ONE round-trip for ALL writes
        if new_histories:
            await state_db.batch_set_metadata(new_histories)

    if cfg.ENABLE_WIN_RATE_FILTER and not state_db.degraded and state_db._redis:
        try:
            pattern = f"{RedisKeyPrefix.OUTCOME_PENDING}*"
            keys = [k async for k in state_db._redis.scan_iter(match=pattern, count=500)]
            prefix_len = len(RedisKeyPrefix.OUTCOME_PENDING)
            pending_by_pair: Dict[str, List[str]] = {}
            for k in keys:
                pair = k[prefix_len:].split(":", 1)[0]
                pending_by_pair.setdefault(pair, []).append(k)

            state_db._pending_outcome_keys_by_pair = pending_by_pair   # ← 12 spaces (FIXED)
            total = sum(len(v) for v in pending_by_pair.values())
            if total:
                logger_main.info(f"⏳ Pre-scanned {total} pending outcome(s) across {len(pending_by_pair)} pair(s)")
        except Exception as e:
            logger_main.warning(f"Pending outcome pre-scan failed: {e}")
            state_db._pending_outcome_keys_by_pair = None
    else:
        state_db._pending_outcome_keys_by_pair = None

    if cfg.ENABLE_BRAIN and cfg.BRAIN_SHADOW_MODE and not state_db.degraded and state_db._redis:
        try:
            pattern = f"{RedisKeyPrefix.SHADOW_PENDING}*"
            keys = [k async for k in state_db._redis.scan_iter(match=pattern, count=500)]
            prefix_len = len(RedisKeyPrefix.SHADOW_PENDING)
            shadow_by_pair: Dict[str, List[str]] = {}
            for k in keys:
                pair = k[prefix_len:].split(":", 1)[0]
                shadow_by_pair.setdefault(pair, []).append(k)
            state_db._shadow_pending_outcome_keys_by_pair = shadow_by_pair
            total = sum(len(v) for v in shadow_by_pair.values())
            if total:
                logger_main.info(f"👻 Pre-scanned {total} shadow pending outcome(s) across {len(shadow_by_pair)} pair(s)")
        except Exception as e:
            logger_main.warning(f"Shadow pending outcome pre-scan failed: {e}")
            state_db._shadow_pending_outcome_keys_by_pair = None
    else:
        state_db._shadow_pending_outcome_keys_by_pair = None

    logger_main.debug("⚙️ Phase 2: Preparing evaluation tasks...")

    prepared_tasks = []
    for pair_name, symbol in valid_tasks:
        candles = all_candles.get(symbol, {})
        prepared_tasks.append((pair_name, symbol, candles))

    logger_main.debug(f"Ready to evaluate {len(prepared_tasks)} pairs")

    logger_main.debug(f"🧠 Phase 3: Evaluating {len(prepared_tasks)} pairs...")
    eval_start = time.time()
    eval_semaphore = asyncio.Semaphore(cfg.EVAL_CONCURRENCY_LIMIT)  # NEW, e.g. 5

    async def _bounded_eval(t):
        async with eval_semaphore:
            return await guarded_eval(
                t, state_db, telegram_queue, correlation_id,
                reference_time, fetcher, alerts_sent_ref, alerts_sent_lock, max_alerts_per_run,
                oi_gate_data=oi_gate_data
            )

    results = await asyncio.gather(
        *[_bounded_eval(t) for t in prepared_tasks],
        return_exceptions=True,
    )
    eval_elapsed = time.time() - eval_start
    logger_main.debug(f"Evaluation complete: {eval_elapsed:.1f}s")

    valid_results = []
    for r in results:
        if isinstance(r, Exception):
            logger_main.warning(f"Evaluation raised exception: {r}")
            continue
        if r is not None:
            valid_results.append(r)

    logger_main.debug(
        f"Results: {len(valid_results)} successful, {len(results) - len(valid_results)} failed"
    )
    del results, prepared_tasks, pair_requests, valid_tasks
    
    process = psutil.Process()

    def log_memory_usage(stage: str):
        try:
            mem_mb = process.memory_info().rss / 1024 / 1024
            limit_mb = cfg.MEMORY_LIMIT_BYTES / 1024 / 1024
            usage_pct = (mem_mb / limit_mb) * 100
            if cfg.DEBUG_MODE:
                logger_main.debug(
                    f"{stage}: {mem_mb:.0f}MB / {limit_mb:.0f}MB ({usage_pct:.0f}%)"
                )
            return mem_mb, limit_mb, usage_pct
        except Exception as e:
            logger_main.debug(f"Memory reporting failed at {stage}: {e}")
            return None, None, None

    peak_memory_mb, limit_mb, usage_pct = log_memory_usage("⚠️ Peak memory after batch")
    if peak_memory_mb and peak_memory_mb > limit_mb * 0.7:
        logger_main.warning(
            f"⚠️ High memory after batch: {peak_memory_mb:.0f}MB / {limit_mb:.0f}MB "
            f"({usage_pct:.0f}%)"
        )
    logger_main.debug("🧹 Fetch-phase data deleted, GC forced")
    current_memory_mb, limit_mb, usage_pct = log_memory_usage("💾 Memory after batch cleanup")
    if current_memory_mb and current_memory_mb > limit_mb * 0.8:
        logger_main.warning(
            f"⚠️ Memory still high after cleanup: {current_memory_mb:.0f}MB ({usage_pct:.0f}%). "
            f"Possible memory leak?"
        )

    knox_approved = len(valid_results)
    knox_rejected = len(pairs_to_process) - knox_approved
    
    logger_main.info(
        f"🎯🧠 Knox: {knox_approved} approved, {knox_rejected} rejected "
        f"({len(pairs_to_process)} total evaluated)"
    )
    return valid_results

async def run_once() -> Optional[bool]:
    MAX_ALERTS_PER_RUN = cfg.MAX_ALERTS_PER_RUN
    all_results: List[Tuple[str, Dict[str, Any]]] = []
    correlation_id = uuid.uuid4().hex[:8]
    TRACE_ID.set(correlation_id)
    logger_run = logging.getLogger(f"macd_bot.run.{correlation_id}")
    start_time = time.time()
    sdb: Optional[RedisStateStore] = None
    lock: Optional[RedisLock] = None
    fetcher: Optional[DataFetcher] = None
    telegram_queue: Optional[TelegramQueue] = None
    lock_acquired = False
    lock_extension_task: Optional[asyncio.Task] = None
    alerts_sent_lock = asyncio.Lock()

    products_map: Optional[Dict[str, dict]] = None
    pairs_to_process: List[str] = []
    
    reference_time = get_trigger_timestamp()
    logger_run.info(
        f"🎯 Run started | Correlation ID: {correlation_id} | "
        f"Reference time: {reference_time} ({format_ist_time(reference_time)})"
    )
    logger_run.debug(
        f"Momentum gate active (all alerts) | 3-of-5 vote | "
        f"body_ratio_min={cfg.CPR_MOMENTUM_BODY_RATIO_MIN}"
    )
    if cfg.ENABLE_CPR:
        logger_run.info(
            f"CPR gate active | threshold={cfg.CPR_THRESHOLD_PCT} | "
            f"wide CPR requires move_from_prev_close (see CPR_THRESHOLD_PCT/adaptive)"
        )
    else:
        logger_run.debug("CPR gate disabled")
    try:
        process = psutil.Process()
        container_memory_mb = process.memory_info().rss / 1024 / 1024
        limit_mb = cfg.MEMORY_LIMIT_BYTES / 1024 / 1024

        if container_memory_mb >= limit_mb:
            logger_run.critical(
                f"🚨 Memory limit exceeded at startup "
                f"({container_memory_mb:.1f}MB / {limit_mb:.1f}MB)"
            )
            return False

        logger_run.debug("📦 Initializing HTTP fetcher...")
        fetcher = DataFetcher(cfg.DELTA_API_BASE)
        pairs_to_process = list(cfg.PAIRS)
        products_map = build_products_map_from_cfg()

        if not pairs_to_process:
            logger_run.error("❌ No pairs configured - aborting")
            return False

        logger_run.info(f"🔄 Processing {len(pairs_to_process)} pairs from config")

        logger_run.debug("Connecting to Redis...")
        sdb = RedisStateStore(cfg.REDIS_URL)
        await sdb.connect()

        if sdb and not sdb.degraded:
            try:
                cb_state_raw = await sdb.get_metadata("circuit_breaker_state")
                if cb_state_raw:
                    await fetcher.circuit_breaker.restore(json_loads(cb_state_raw))
            except Exception as e:
                logger_run.warning(f"Could not restore circuit breaker state from Redis: {e}")

        if os.getenv("CLEAR_ALL_STATES", "false").lower() == "true": 
            if sdb and not sdb.degraded:
                logger_run.warning("🚨 CLEAR_ALL_STATES requested — purging selected Redis states...")
                
                def _env_bool(key: str, default: str = "false") -> bool:
                    return os.getenv(key, default).lower() == "true"
                
                st, dd, pend, sp, ast, sst, shc, strm = await _clear_all_redis_states(
                    sdb, pairs_to_process, logger_run,
                    clear_active_states=_env_bool("CLEAR_ACTIVE_STATES", "true"),
                    clear_dedups=_env_bool("CLEAR_DEDUPS", "true"),
                    clear_pending_outcomes=_env_bool("CLEAR_PENDING_OUTCOMES", "true"),
                    clear_shadow_pending=_env_bool("CLEAR_SHADOW_PENDING", "true"),
                    clear_alert_stats=_env_bool("CLEAR_WINRATE_STATS", "false"),
                    clear_shadow_stats=_env_bool("CLEAR_SHADOW_TRACKING", "false"),
                    clear_outcome_streams=_env_bool("CLEAR_OUTCOME_HISTORY", "false"),
                )
                
                parts = []
                if st: parts.append(f"States: {st}")
                if dd: parts.append(f"Dedups: {dd}")
                if pend: parts.append(f"Pending: {pend}")
                if sp: parts.append(f"ShadowPending: {sp}")
                if ast: parts.append(f"AlertStats: {ast}")
                if sst: parts.append(f"ShadowStats: {sst}")
                if shc: parts.append(f"ShadowHiConf: {shc}")
                if strm: parts.append(f"Streams: {strm}")
                cleared_str = " | ".join(parts) if parts else "Nothing selected to clear"
                
                if telegram_queue is None:
                    telegram_queue = TelegramQueue(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID)
                await telegram_queue.send(escape_markdown_v2(
                    f"🧹 {cfg.BOT_NAME} Redis purge complete\n"
                    f"{cleared_str}\n"
                    f"Time: {format_ist_time()}"
                ))
            else:
                logger_run.error("CLEAR_ALL_STATES=true but Redis is unavailable/degraded")

        if sdb.degraded and not sdb.degraded_alerted:
            logger_run.critical(
                "🚨 Redis is in degraded mode – alert deduplication disabled!"
            )

        if sdb and not sdb.degraded and (cfg.ENABLE_PIVOT or cfg.ENABLE_VWAP):
            logger_run.debug("Checking daily reset conditions...")
            day_tracker_key = "global:last_reset_date"
            current_date_str = get_utc_date_key(reference_time)
            
            last_reset_date_str = None
            try:
                last_reset_date_str = await sdb.get_metadata(day_tracker_key)
            except Exception as e:
                logger_run.warning(f"Failed to get last reset date: {e}")

            if should_reset_daily_state(reference_time, last_reset_date_str):
                logger_run.info(f"🔄 New day detected ({current_date_str}). Resetting daily states...")
    
                all_delete_keys = []
    
                if cfg.ENABLE_PIVOT:
                    pivot_alerts = (
                        [f"pivot_up_{level}" for level in PIVOT_LEVELS_BUY] +
                        [f"pivot_down_{level}" for level in PIVOT_LEVELS_SELL]
                    )
        
                    for pair in pairs_to_process:
                        for alert_key in pivot_alerts:
                            redis_key = ALERT_KEYS.get(alert_key)
                            if redis_key:
                                all_delete_keys.append(f"{pair}:{redis_key}")
    
                if cfg.ENABLE_VWAP:
                    vwap_alerts = ["vwap_up", "vwap_down"]
                    for pair in pairs_to_process:
                        for alert_key in vwap_alerts:
                            redis_key = ALERT_KEYS.get(alert_key)
                            if redis_key:
                                all_delete_keys.append(f"{pair}:{redis_key}")
   
                if all_delete_keys:
                    try:
                        await sdb.atomic_batch_update([], deletes=all_delete_keys)
                        logger_run.info(
                            f"✅ Cleared {len(all_delete_keys)} daily alert keys "
                            f"from {len(pairs_to_process)} pairs"
                        )
                    except Exception as e:
                        logger_run.error(f"❌ Failed to delete daily reset keys: {e}")
                        raise
    
                try:
                    await sdb.set_metadata(day_tracker_key, current_date_str)
                    logger_run.info(f"✅ Daily reset complete ({current_date_str})")
                except Exception as e:
                    logger_run.error(f"❌Failed to save reset date: {e}")
            else:
                logger_run.debug(f"No daily reset needed (last reset: {last_reset_date_str})")

        if sdb.degraded and not sdb.degraded_alerted:
            telegram_queue = TelegramQueue(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID)
            await telegram_queue.send(escape_markdown_v2(
                f"⚠️ {cfg.BOT_NAME} - REDIS DEGRADED MODE\n"
                f"Alert deduplication is disabled. You may receive duplicate alerts.\n"
                f"Time: {format_ist_time()}"
            ))
            sdb.degraded_alerted = True

        if telegram_queue is None:
            telegram_queue = TelegramQueue(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID)

        if sdb.degraded:
            logger_run.warning(
                "⚠️ Redis degraded — skipping distributed lock, proceeding without "
                "duplicate-run protection (core alerting still runs)"
            )
            lock = None
            lock_acquired = False
        else:
            lock = RedisLock(sdb._redis, "macd_bot_run")
            lock_acquired = await lock.acquire(timeout=5.0)
            if not lock_acquired:
                logger_run.warning(
                    "⚠️ Another instance is running (Redis lock held) - exiting gracefully"
                )
                return None

        async def extend_lock_periodically(lock_obj: RedisLock, telegram_queue: TelegramQueue):
            while not shutdown_event.is_set():
                try:
                    if lock_obj.should_extend():
                        success = await lock_obj.extend(timeout=3.0)
                        if success:
                            logger_run.debug("🔒 Lock extended successfully")
                        else:
                            logger_run.critical("✘ Lock extension failed...")
                            try:
                                await telegram_queue.send(escape_markdown_v2(
                                    f"⚠️ Lock extension failed for {lock_obj.lock_key}"
                                ))
                            except Exception as e:
                                logger_run.error(f"Failed to send lock failure alert: {e}")
                            shutdown_event.set()
                            return

                    time_since_extend = time.monotonic() - lock_obj.last_extend_time
                    time_until_threshold = max(0, lock_obj.get_lock_extend_interval() - time_since_extend)
                    sleep_time = max(30, min(180, int(time_until_threshold * 0.75)))

                    try:
                        await asyncio.wait_for(shutdown_event.wait(), timeout=sleep_time)
                    except asyncio.TimeoutError:
                        pass
            
                except asyncio.CancelledError:
                    break

                except Exception as e:
                    logger_run.error(f"Lock extension task error: {e}")
                    await asyncio.sleep(60)

        lock_extension_task = (
            asyncio.create_task(extend_lock_periodically(lock, telegram_queue))
            if lock is not None else None
        )
        if cfg.SEND_TEST_MESSAGE:
            await telegram_queue.send(escape_markdown_v2(
                f"🔥 {cfg.BOT_NAME} - Run Started\n"
                f"Date: {format_ist_time(datetime.now(timezone.utc))}\n"
                f"Correlation ID: {correlation_id}\n"
                f"Pairs: {len(pairs_to_process)}"
            ))

        logger_run.debug(
            f"🔔 Processing {len(pairs_to_process)} pairs using optimized parallel architecture"
        )

        logger_run.info("Starting evaluation phase...")  
        alerts_sent_ref = [0] 
        all_results = await process_pairs_with_workers(
            fetcher, products_map, pairs_to_process, sdb, telegram_queue, 
            correlation_id, lock, reference_time,
            alerts_sent_ref, alerts_sent_lock, MAX_ALERTS_PER_RUN
        ) 

        logger_run.debug("Cleanup phase with normal garbage collection...")

        fetcher_stats = fetcher.get_stats()

        total_required = fetcher_stats['candles']['success'] + fetcher_stats['candles']['failed']
        candles_str = f"{fetcher_stats['candles']['success']}/{total_required}"

        logger_run.info(
            f"Fetch Stats | "
            f"Products: config only | "
            f"Candles: {candles_str}"
        )

        if "rate_limiter" in fetcher_stats and fetcher_stats["rate_limiter"].get("total_waits", 0) > 0:
            rate_stats = fetcher_stats["rate_limiter"]
            logger_run.info(
                f"🚦 Rate limiting | "
                f"Waits: {rate_stats['total_waits']} | "
                f"Total wait: {rate_stats['total_wait_time_seconds']:.1f}s"
            )
        final_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_delta = final_memory_mb - container_memory_mb
        run_duration = time.time() - start_time
        redis_status = "OK" if (sdb and not sdb.degraded) else "DEGRADED"

        summary = (
            f"🎯🌏 RUN COMPLETE | "
            f"Duration: {run_duration:.1f}s | "
            f"Pairs: {len(all_results)}/{len(pairs_to_process)} | "
            f"Alerts: {alerts_sent_ref[0]} | "
            f"OI/Funding blocks: {fetcher_stats.get('oi_funding_blocks', 0)} | "
            f"Memory: {int(final_memory_mb)}MB (Δ{memory_delta:+.0f}MB) | "
            f"Redis: {redis_status}"
        )
        logger_run.info(summary)

        if cfg.ENABLE_BRAIN:
            try:
                from brain import BrainEngine
                await BrainEngine(sdb).maybe_generate_report(pairs_to_process, telegram_queue, logger_run)
            except Exception as e:
                logger_run.warning(f"Brain report generation failed: {e}")
        if alerts_sent_ref[0] > MAX_ALERTS_PER_RUN:
            await telegram_queue.send(escape_markdown_v2(
                f"⚠️ HIGH ALERT VOLUME\n"
                f"Alerts sent: {alerts_sent_ref[0]}\n"
                f"Pairs processed: {len(all_results)}\n"
                f"Time: {format_ist_time()}"
            ))

        return True

    except asyncio.TimeoutError:
        logger_run.error("⚠️ Run timed out - exceeded RUN_TIMEOUT_SECONDS")
        return False

    except asyncio.CancelledError:
        logger_run.warning("❌ Run cancelled (shutdown signal received)")
        return False

    except Exception as e:
        logger_run.exception(f"❌ Fatal error in run_once: {e}")

        if telegram_queue:
            try:
                await telegram_queue.send(escape_markdown_v2(
                    f"❌ {cfg.BOT_NAME} - FATAL ERROR\n"
                    f"Error: {str(e)[:200]}\n"
                    f"Correlation ID: {correlation_id}\n"
                    f"Time: {format_ist_time()}"
                ))
            except Exception:
                logger_run.error("Failed to send error notification")
     
        return False

    finally:
        
        logger_run.debug("🧹 Starting resource cleanup...")
        if lock_extension_task:
            try:
                lock_extension_task.cancel()
                await asyncio.wait_for(lock_extension_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as e:
                logger_run.error(f"Error cancelling lock extension task: {e}")

        if lock_acquired and lock and lock.acquired_by_me:
            try:
                await asyncio.wait_for(lock.release(timeout=3.0), timeout=4.0)
                logger_run.debug("🔏 Redis lock released")
            except asyncio.TimeoutError:
                logger_run.error("Timeout releasing lock")
            except Exception as e:
                logger_run.error(f"Error releasing lock: {e}", exc_info=False)

        if sdb and not sdb.degraded and fetcher:
            try:
                cb_snapshot = await fetcher.circuit_breaker.snapshot()
                await asyncio.wait_for(
                    sdb.set_metadata("circuit_breaker_state", json_dumps(cb_snapshot), ttl=3600),
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                logger_run.error("Timeout persisting circuit breaker state")
            except Exception as e:
                logger_run.error(f"Error persisting circuit breaker state: {e}", exc_info=False)

        if sdb:
            try:
                await asyncio.wait_for(sdb.close(), timeout=3.0)
                logger_run.debug("✅ Redis connection closed")
            except asyncio.TimeoutError:
                logger_run.error("Timeout closing Redis")
            except Exception as e:
                logger_run.error(f"Error closing Redis: {e}", exc_info=False)
        try:
            await asyncio.wait_for(
                RedisStateStore.shutdown_global_pool(),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logger_run.error("Timeout shutting down Redis pool")
        except Exception as e:
            logger_run.error(f"Error shutting down Redis pool: {e}")

        try:
            await asyncio.wait_for(
                SessionManager.close_session(),
                timeout=5.0
            )
            logger_run.debug("HTTP session closed")
        except asyncio.TimeoutError:
            logger_run.error("Timeout closing HTTP session")
        except Exception as e:
            logger_run.error(f"Error closing HTTP session: {e}", exc_info=False)

        try:
            TRACE_ID.set("")
            PAIR_ID.set("")
        except Exception:
            pass

        try:
            gc.collect()
            if cfg.DEBUG_MODE:
                logger_run.debug("🥃 Final garbage collection completed")
        except Exception as e:
            logger_run.debug(f"GC error: {e}")

        logger_run.debug("🧹 Resource cleanup finished")
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    try:
        import hiredis
        _hiredis_status = f"hiredis {hiredis.__version__} enabled"
    except ImportError:
        _hiredis_status = "hiredis NOT found (redis-py using pure-python parser)"
    logger.info(f"🌎 uvloop enabled | orjson enabled | {_hiredis_status}")
except ImportError:
    logger.info(f"❌ uvloop not available (using default) | {JSON_BACKEND} enabled")

if __name__ == "__main__":  
    aot_bridge.ensure_initialized()
    
    if not aot_bridge.is_using_aot():
        reason = aot_bridge.get_fallback_reason() or "Unknown"
        logger.warning("❌ AOT not available, using JIT fallback. Reason: %s", reason)
        logger.warning("⚠️ Performance may be degraded. First run may be slow.")

        if os.getenv("REQUIRE_AOT", "false").lower() == "true":
            logger.critical("❌ REQUIRE_AOT=true but AOT unavailable - exiting")
            sys.exit(1)
    else:
        logger.info("✅ Verified: AOT artifacts loaded successfully")

    if os.getenv("NUMERIC_SELFTEST_ENABLED", "true").lower() == "true":
        import numeric_selftest
        selftest_ok, selftest_failures = numeric_selftest.run_self_test()
        if selftest_ok:
            logger.info(
                "✅ Numeric self-test passed (%s backend)",
                "AOT" if aot_bridge.is_using_aot() else "JIT"
            )
        else:
            for msg in selftest_failures:
                logger.critical("❌ Numeric self-test failure: %s", msg)
            if os.getenv("NUMERIC_SELFTEST_STRICT", "true").lower() == "true":
                logger.critical("❌ NUMERIC_SELFTEST_STRICT=true and self-test failed - exiting")
                sys.exit(1)
            else:
                logger.warning("⚠️ Continuing despite self-test failure (NUMERIC_SELFTEST_STRICT=false)")

    parser = argparse.ArgumentParser(
        prog="macd_unified",
        description="Unified MACD/alerts runner with NumPy optimization"
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--validate-only", action="store_true", help="Validate config and exit")
    parser.add_argument("--skip-warmup", action="store_true", help="Skip Numba JIT warmup")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        for h in logger.handlers:
            h.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled via CLI flag")

    try:
        validate_runtime_config()
    except ValueError as e:
        logger.critical(f"Configuration validation failed: {e}")
        sys.exit(1)

    if args.validate_only:
        logger.info("Configuration validation passed - exiting (--validate-only mode)")
        sys.exit(0)

    if not args.skip_warmup:
        warmup_if_needed()
    else:
        logger.info("Skipping Numba warmup (faster startup)")

    async def main_with_cleanup():
        try:
            async with asyncio.timeout(cfg.RUN_TIMEOUT_SECONDS):
                return await run_once()
        except TimeoutError:
            logger.critical(
                "Run exceeded hard deadline: %ss",
                cfg.RUN_TIMEOUT_SECONDS
            )
            return False
    try:
        success = asyncio.run(main_with_cleanup()) 
        if success is None:
            logger.info("ℹ️ Run skipped (another instance already running) — not a failure")
            sys.exit(0)
        elif success:
            sys.exit(0)
        else:
            logger.error("❌ Bot run failed")
            sys.exit(1)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Bot stopped by timeout or user interrupt")
        sys.exit(130)
    except Exception as exc:
        logger.critical(f"Fatal error: {exc}", exc_info=True)
        sys.exit(1)
