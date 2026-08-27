from __future__ import annotations
import time
import asyncio
import logging
import uuid
from typing import Dict, Any, Optional, Tuple, List, Set, ClassVar, Callable
import numpy as np
import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError, RedisError

from bot_config import cfg, logger, json_dumps, json_loads, JSONDecodeError
from fetcher import compute_backoff

async def _blanket_reset_pair(sdb: RedisStateStore, pair_name: str, logger_pair: logging.Logger) -> int:
    from alerts import ALERT_KEYS
    all_keys = list(ALERT_KEYS.values())
    previous_states = await sdb.batch_get_all_alert_states(pair_name, all_keys)
    resets = [
        (f"{pair_name}:{rk}", "INACTIVE", None)
        for rk in all_keys
        if previous_states.get(rk, False)
    ]
    if resets:
        await sdb.atomic_batch_update(resets)
        logger_pair.debug(
            f"[{pair_name}] Blanket reset: {len(resets)} active state(s) cleared"
        )
    return len(resets)

async def _clear_all_redis_states(sdb: RedisStateStore, pairs: List[str], logger: logging.Logger) -> Tuple[int, int, int]:
    if sdb.degraded or not sdb._redis:
        logger.warning("Redis degraded — skipping mass state purge")
        return 0, 0, 0

    deleted_states = 0
    deleted_dedups = 0
    deleted_outcomes = 0

    async def _scan_keys_with_timeout(match: str, count: int = 100, timeout: float = 10.0) -> List[str]:
        """Safely consume an async scan_iter with a timeout to prevent runaway loops."""
        async def _consume():
            return [k async for k in sdb._redis.scan_iter(match=match, count=count)]
        
        try:
            return await asyncio.wait_for(_consume(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"⏱️ Redis scan for '{match}' timed out after {timeout}s. Aborting scan for this prefix to protect run deadline.")
            return []  # Fail-safe: return empty so we don't block the bot

    async def _batch_unlink(keys: List[str], batch_size: int = 100) -> int:
        """Unlink keys in batches to avoid blocking Redis with massive argument lists."""
        if not keys:
            return 0
        total_deleted = 0
        for i in range(0, len(keys), batch_size):
            batch = keys[i:i + batch_size]
            try:
                # unlink() frees memory in a background thread on the Redis server
                total_deleted += await sdb._redis.unlink(*batch)
            except Exception as e:
                logger.error(f"Batch unlink failed for {len(batch)} keys: {e}")
        return total_deleted

    try:
        # ── 1. State hashes: scan for pair_state:* ──
        state_keys = await _scan_keys_with_timeout(f"{RedisKeyPrefix.PAIR_STATE}*", count=100)
        deleted_states = await _batch_unlink(state_keys)

        # ── 2. Dedup keys: scan for recent_alert:* ──
        dedup_keys = await _scan_keys_with_timeout(f"{RedisKeyPrefix.RECENT_ALERT}*", count=500)
        deleted_dedups = await _batch_unlink(dedup_keys)

        # ── 3. Pending Outcomes (Win-rate tracking queue) ──
        outcome_keys = await _scan_keys_with_timeout(f"{RedisKeyPrefix.OUTCOME_PENDING}*", count=100)
        deleted_outcomes = await _batch_unlink(outcome_keys)

        logger.info(
            f"🧹 MASS RESET complete | "
            f"States: {deleted_states} | Dedups: {deleted_dedups} | Outcomes: {deleted_outcomes}"
        )
        return deleted_states, deleted_dedups, deleted_outcomes

    except Exception as e:
        logger.error(f"Mass reset failed: {e}")
        return 0, 0, 0

def build_products_map_from_cfg() -> Dict[str, dict]:
    products_map: Dict[str, dict] = {}
    for pair in cfg.PAIRS:
        products_map[pair] = {
            "id": pair,                 
            "symbol": pair,
            "contract_type": "perpetual_futures"
        }
    logger.info(
        f"📦 Product map built from cfg: {len(products_map)}/{len(cfg.PAIRS)} matched | "
        f"Coverage: {(len(products_map)/len(cfg.PAIRS))*100:.0f}%"
    )
    return products_map

class RedisKeyPrefix:
    """Centralized Redis key prefixes"""
    PAIR_STATE = "pair_state:"
    METADATA = "metadata:"
    ALERT = "alert:"
    RECENT_ALERT = "recent_alert:"
    LOCK = "lock:"
    OUTCOME_PENDING = "outcome_pending:"
    ALERT_STATS = "alert_stats:"
    OUTCOME_LOG_STREAM = "outcome_log_stream"
    TLR_TOUCH = "tlr_touch:"
    SHADOW_PENDING = "shadow_pending:"
    SHADOW_STATS = "shadow_stats:"
    SHADOW_LOG_STREAM = "shadow_log_stream"
    SHADOW_HICONF_STATS = "shadow_hiconf:"
    BRAIN_RUN_COUNTER = "brain_run_counter"

class RedisStateStore:
    POOL_MAX_AGE_SECONDS = 3600
    SCRIPT_RELOAD_LOCK_TIMEOUT = 2.0

    _global_pools: ClassVar[Dict[str, Optional[redis.Redis]]] = {}
    _pool_healthy: ClassVar[Dict[str, bool]] = {}
    _pool_created_at: ClassVar[Dict[str, float]] = {}
    _pool_reuse_count: ClassVar[Dict[str, int]] = {}
    _pool_lock: ClassVar[Optional[asyncio.Lock]] = None
    _script_reload_lock: ClassVar[Optional[asyncio.Lock]] = None

    @classmethod
    def _get_pool_lock(cls) -> asyncio.Lock:
        if cls._pool_lock is None:
            cls._pool_lock = asyncio.Lock()
        return cls._pool_lock

    @classmethod
    def _get_script_reload_lock(cls) -> asyncio.Lock:
        if cls._script_reload_lock is None:
            cls._script_reload_lock = asyncio.Lock()
        return cls._script_reload_lock

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._redis: Optional[redis.Redis] = None

        self.state_prefix = RedisKeyPrefix.PAIR_STATE
        self.meta_prefix = RedisKeyPrefix.METADATA
        self.alert_prefix = RedisKeyPrefix.ALERT

        self.expiry_seconds = max(cfg.STATE_EXPIRY_DAYS * 86400 if cfg.STATE_EXPIRY_DAYS > 0 else 0, 7 * 86400)
        self.alert_expiry_seconds = cfg.STATE_EXPIRY_DAYS * 86400
        self.metadata_expiry_seconds = 7 * 86400

        self.degraded = False
        self.degraded_alerted = False
        self._connection_attempts = 0

        if cfg.DEBUG_MODE and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"RedisStateStore initialized | "
                f"State TTL: {cfg.STATE_EXPIRY_DAYS}d | "
                f"Alert TTL: {cfg.STATE_EXPIRY_DAYS}d | "
                f"Metadata TTL: 7d"
            )

    async def _record_redis_failure(self, operation: str, exc: Exception) -> None:
        logger.error(f"Redis operation '{operation}' failed: {exc}")
        if self.degraded:
            return
        self.degraded = True
        logger.warning(f"Redis marked degraded after failure in '{operation}' — attempting one reconnect")
        try:
            reconnected = await self._attempt_connect(timeout=5.0)
            if reconnected:
                logger.info(f"Redis reconnected after failure in '{operation}'")
                self.degraded = False
            else:
                logger.critical(f"Redis reconnect failed after '{operation}' — staying degraded for remainder of run")
        except Exception as reconnect_exc:
            logger.critical(f"Redis reconnect attempt itself failed: {reconnect_exc} — staying degraded")

    async def _attempt_connect(self, timeout: float = 5.0) -> bool:
        try:
            self._redis = redis.from_url(
                self.redis_url,
                socket_connect_timeout=timeout,
                socket_timeout=timeout,
                retry_on_timeout=True,
                max_connections=32,
                decode_responses=True,
            )

            ok = await self._ping_with_retry(timeout)
            if not ok:
                raise RedisConnectionError("ping failed after retries")

            logger.info("Redis connected")
            self.degraded = False
            self.degraded_alerted = False
            self._connection_attempts = 0

            async with RedisStateStore._get_pool_lock():
                existing_pool = RedisStateStore._global_pools.get(self.redis_url)
                pool_is_healthy = False
                if existing_pool:
                    try:
                        pool_is_healthy = await asyncio.wait_for(existing_pool.ping(), timeout=1.0)
                    except Exception:
                        pool_is_healthy = False

                if existing_pool and pool_is_healthy:
                    if self._redis is not existing_pool:
                        await self._redis.aclose()
                    self._redis = existing_pool
                    logger.debug("Using pool created by another coroutine")
                else:
                    if existing_pool and existing_pool is not self._redis:
                        try:
                            await existing_pool.aclose()
                        except Exception:
                            pass
                    RedisStateStore._global_pools[self.redis_url] = self._redis
                    RedisStateStore._pool_healthy[self.redis_url] = True
                    RedisStateStore._pool_created_at[self.redis_url] = time.time()
                    RedisStateStore._pool_reuse_count[self.redis_url] = 0
                    if cfg.DEBUG_MODE:
                        logger.debug("Redis connection saved to per-URL pool")

                return True

        except Exception as exc:
            logger.error(f"Redis connection attempt failed: {exc}")
            if self._redis:
                try:
                    await self._redis.aclose()
                except Exception:
                    pass
                self._redis = None
            return False

    async def connect(self, timeout: float = 5.0) -> None:
        pool_reused = False

        async with RedisStateStore._get_pool_lock():
            pool = RedisStateStore._global_pools.get(self.redis_url)
            healthy = RedisStateStore._pool_healthy.get(self.redis_url, False)

            if pool and healthy:
                pool_age = time.time() - RedisStateStore._pool_created_at.get(self.redis_url, 0.0)
                if pool_age > self.POOL_MAX_AGE_SECONDS:
                    logger.info(f"Redis pool aged {pool_age:.0f}s, refreshing")
                    RedisStateStore._pool_healthy[self.redis_url] = False
                    try:
                        await pool.aclose()
                    except Exception:
                        pass
                    RedisStateStore._global_pools[self.redis_url] = None
                else:
                    try:
                        self._redis = pool
                        ok = await self._ping_with_retry(timeout)
                        if ok:
                            RedisStateStore._pool_reuse_count[self.redis_url] = \
                                RedisStateStore._pool_reuse_count.get(self.redis_url, 0) + 1
                            self.degraded = False
                            pool_reused = True
                            return
                    except Exception as e:
                        if cfg.DEBUG_MODE:
                            logger.debug(f"Pool health check failed: {e}, creating new pool")
                        RedisStateStore._pool_healthy[self.redis_url] = False
                        pool_reused = False

        if pool_reused:
            return

        for attempt in range(1, cfg.REDIS_CONNECTION_RETRIES + 1):
            if await self._attempt_connect(timeout):
                max_conn = getattr(self._redis.connection_pool, "max_connections", "?")
                logger.info(f"✅ Redis connected ({max_conn} max)")
                self.degraded = False
                self.degraded_alerted = False
                return

            if attempt < cfg.REDIS_CONNECTION_RETRIES:
                delay = compute_backoff(cfg.REDIS_RETRY_DELAY, attempt)
                logger.warning(f"Retrying Redis connection in {delay:.1f}s...")
                await asyncio.sleep(delay)

        logger.critical("❌ Redis connection failed after all retries")
        self.degraded = True
        if self._redis:
            try:
                await self._redis.aclose()
            except Exception:
                pass
        self._redis = None

        logger.warning("""
    🚨 REDIS DEGRADED MODE ACTIVE:
    - Alert deduplication:  DISABLED (may get duplicates)
    - State persistence:    DISABLED (alerts reset each run)
    - Trading alerts:       STILL ACTIVE (core functionality preserved)
    """)

        if cfg.FAIL_ON_REDIS_DOWN:
            raise RedisConnectionError("Redis unavailable after all retries – FAIL_ON_REDIS_DOWN=true")
      
    async def close(self) -> None:
        self._redis = None

    @classmethod
    async def shutdown_global_pool(cls, redis_url: Optional[str] = None) -> None:
        async with cls._get_pool_lock():
            urls = [redis_url] if redis_url else list(cls._global_pools.keys())
            for url in urls:
                pool = cls._global_pools.get(url)
                if pool:
                    try:
                        pool_age = time.time() - cls._pool_created_at.get(url, 0.0)
                        reuse_count = cls._pool_reuse_count.get(url, 0)
                        logger.debug(f"Shutting down Redis pool | url={url} | Age: {pool_age:.1f}s | Reuses: {reuse_count}")

                        await pool.aclose()
                        await asyncio.sleep(0.25)

                    except Exception as e:
                        logger.error(f"Error shutting down Redis pool {url}: {e}")

                cls._global_pools.pop(url, None)
                cls._pool_healthy.pop(url, None)
                cls._pool_created_at.pop(url, None)
                cls._pool_reuse_count.pop(url, None)
            
    async def _ping_with_retry(self, timeout: float) -> bool:
        result = await self._safe_redis_op(lambda: self._redis.ping(), timeout, "ping")
        return bool(result)

    async def _safe_redis_op(self, fn: Callable[[], Any], timeout: float, op_name: str, parser: Optional[Callable[[Any], Any]] = None):
        if not self._redis:
            return None
        try:
            coro = fn()
            result = await asyncio.wait_for(coro, timeout=timeout)
            return parser(result) if parser else result
        except (asyncio.TimeoutError, RedisConnectionError, RedisError) as e:
            logger.error(f"Redis {op_name} failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to {op_name}: {e}")
            return None

    async def get(self, key: str, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
        return await self._safe_redis_op(
            lambda: self._redis.get(f"{self.state_prefix}{key}"),
            timeout,
            f"get {key}",
            parser=lambda r: json_loads(r) if r else None,
        )

    async def set(self, key: str, state: Optional[Any], ts: Optional[int] = None, timeout: float = 2.0) -> None:
        ts = int(ts or time.time())
        redis_key = f"{self.state_prefix}{key}"
        data = json_dumps({"state": state, "ts": ts})
        await self._safe_redis_op(
            lambda: self._redis.set(
                redis_key,
                data,
                ex=self.expiry_seconds if self.expiry_seconds > 0 else None,
            ),
            timeout,
            f"set {key}",
        )

    async def get_metadata(self, key: str, timeout: float = 2.0) -> Optional[str]:
        return await self._safe_redis_op(
            lambda: self._redis.get(f"{self.meta_prefix}{key}"),
            timeout,
            f"get_metadata {key}",
            parser=lambda r: r if r else None,
        )
    async def set_metadata(self, key: str, value: str, timeout: float = 2.0,
                             ttl: Optional[int] = None) -> None:
        await self._safe_redis_op(
            lambda: self._redis.set(
                f"{self.meta_prefix}{key}",
                value,
                ex=ttl if ttl is not None else self.metadata_expiry_seconds
            ),
            timeout,
            f"set_metadata {key}",
        )
    async def batch_get_metadata(self, keys: List[str], timeout: float = 5.0) -> Dict[str, Optional[str]]:
        """Fetch many metadata keys in ONE Redis round-trip (pipeline)."""
        if not self._redis or self.degraded or not keys:
            return {k: None for k in keys}
        try:
            async with self._redis.pipeline() as pipe:
                for k in keys:
                    pipe.get(f"{self.meta_prefix}{k}")
                values = await asyncio.wait_for(pipe.execute(), timeout=timeout)
            return {k: v for k, v in zip(keys, values)}
        except Exception as e:
            logger.error(f"batch_get_metadata failed for {len(keys)} keys: {e}")
            return {k: None for k in keys}

    async def batch_set_metadata(self, items: Dict[str, str], timeout: float = 5.0) -> bool:
        """Write many metadata keys in ONE Redis round-trip (pipeline)."""
        if not self._redis or self.degraded or not items:
            return True
        try:
            async with self._redis.pipeline() as pipe:
                for k, v in items.items():
                    pipe.set(f"{self.meta_prefix}{k}", v, ex=self.metadata_expiry_seconds)
                await asyncio.wait_for(pipe.execute(), timeout=timeout)
            return True
        except Exception as e:
            logger.error(f"batch_set_metadata failed: {e}")
            return False

    async def check_recent_alert(self, pair: str, alert_key: str, ts: int, window_sec: Optional[int] = None) -> bool:
        if self.degraded:
            return True
        if not self._redis:
            logger.critical(
                f"check_recent_alert: degraded=False but _redis is None (state desync) — "
                f"failing closed for {pair}:{alert_key}, this alert will be blocked"
            )
            return False  # fail-closed, consistent with the except-branch policy below
        recent_key = f"{RedisKeyPrefix.RECENT_ALERT}{pair}:{alert_key}"
        effective_window = window_sec if window_sec is not None else cfg.ALERT_DEDUP_WINDOW_SEC
        try:
            result = await asyncio.wait_for(
                self._redis.set(recent_key, str(ts), nx=True, ex=effective_window),
                timeout=3.0
            )
            should_send = bool(result)
            if cfg.DEBUG_MODE and not should_send:
                logger.debug(f"Dedup: Skipping duplicate {pair}:{alert_key}")
            return should_send
        except Exception as e:
            logger.error(f"Dedup check FAILED for {pair}:{alert_key}: {e}")
            return False   # fail-closed, not fail-open

    async def release_recent_alert(self, pair: str, alert_key: str) -> None:
        """Undo a dedup claim if the message didn't actually get delivered."""
        if self.degraded:
            return
        recent_key = f"{RedisKeyPrefix.RECENT_ALERT}{pair}:{alert_key}"
        try:
            await asyncio.wait_for(self._redis.delete(recent_key), timeout=1.0)
        except Exception as e:
            logger.warning(f"Failed to release dedup claim for {pair}:{alert_key}: {e}")

    async def record_pending_outcome(self, pair: str, alert_key: str, direction: str,
                                       entry_ts: int, entry_price: float,
                                       confluence_score: Optional[float] = None,
                                       confluence_total: Optional[float] = None) -> None:
        if self.degraded or not cfg.ENABLE_WIN_RATE_FILTER:
            return
        key = f"{RedisKeyPrefix.OUTCOME_PENDING}{pair}:{alert_key}:{entry_ts}"
        try:
            payload = json_dumps({
                "direction": direction, "entry_ts": entry_ts, "entry_price": entry_price,
                "confluence_score": confluence_score, "confluence_total": confluence_total,
            })
        except Exception as e:
            logger.warning(f"Failed to serialize pending outcome for {pair}:{alert_key}: {e}")
            return
        ttl = (cfg.OUTCOME_LOOKAHEAD_CANDLES + 4) * 15 * 60  # lookahead + buffer, in seconds
        try:
            await asyncio.wait_for(self._redis.set(key, payload, ex=ttl), timeout=2.0)
        except Exception as e:
            logger.warning(f"Failed to record pending outcome for {pair}:{alert_key}: {e}")

    async def record_shadow_pending_outcome(self, pair: str, alert_key: str, direction: str,
                                              entry_ts: int, entry_price: float,
                                              confluence_score: Optional[float] = None,
                                              confluence_total: Optional[float] = None) -> None:
        """Twin of record_pending_outcome for alerts REJECTED by the win-rate filter.
        Lets the brain see what would have happened had the alert fired."""
        if self.degraded or not getattr(cfg, "ENABLE_BRAIN", False):
            return
        key = f"{RedisKeyPrefix.SHADOW_PENDING}{pair}:{alert_key}:{entry_ts}"
        try:
            payload = json_dumps({
                "direction": direction, "entry_ts": entry_ts, "entry_price": entry_price,
                "confluence_score": confluence_score, "confluence_total": confluence_total,
            })
        except Exception as e:
            logger.warning(f"Failed to serialize shadow pending outcome for {pair}:{alert_key}: {e}")
            return
        ttl = (cfg.OUTCOME_LOOKAHEAD_CANDLES + 4) * 15 * 60
        try:
            await asyncio.wait_for(self._redis.set(key, payload, ex=ttl), timeout=2.0)
        except Exception as e:
            logger.warning(f"Failed to record shadow pending outcome for {pair}:{alert_key}: {e}")

    async def get_tlr_touch_state(self, pair: str, is_buy: bool) -> Optional[Dict[str, Any]]:
        """Load persisted TLR touch-count state for a pair/direction. Returns
        None if degraded, unset, or on any Redis error (caller treats None
        the same as 'no prior state' — safe default, starts a fresh count)."""
        if self.degraded or not self._redis:
            return None
        direction = "buy" if is_buy else "sell"
        key = f"{RedisKeyPrefix.TLR_TOUCH}{pair}:{direction}"
        return await self._safe_redis_op(
            lambda: self._redis.get(key),
            2.0,
            f"get_tlr_touch_state {pair}:{direction}",
            parser=lambda r: json_loads(r) if r else None,
        )

    async def save_tlr_touch_state(self, pair: str, is_buy: bool, state: Dict[str, Any]) -> None:
        if self.degraded or not self._redis:
            return
        direction = "buy" if is_buy else "sell"
        key = f"{RedisKeyPrefix.TLR_TOUCH}{pair}:{direction}"
        try:
            payload = json_dumps(state)
        except Exception as e:
            logger.warning(f"Failed to serialize TLR touch state for {pair}:{direction}: {e}")
            return
        try:
            await asyncio.wait_for(
                self._redis.set(key, payload, ex=cfg.TLR_TOUCH_STATE_TTL_SEC), timeout=2.0
            )
        except Exception as e:
            logger.warning(f"Failed to save TLR touch state for {pair}:{direction}: {e}")

    async def resolve_pending_outcomes(self, pair: str, data_15m: "PriceData", i15: int,
                                         logger_pair: logging.Logger) -> None:
        if self.degraded or not cfg.ENABLE_WIN_RATE_FILTER or not self._redis:
            return

        # Use the run-level pre-scan when available (falls back to per-pair scan otherwise)
        precomputed = getattr(self, "_pending_outcome_keys_by_pair", None)
        if precomputed is not None:
            keys = precomputed.get(pair, [])
            if not keys:
                return
        else:
            try:
                pattern = f"{RedisKeyPrefix.OUTCOME_PENDING}{pair}:*"
                keys = [k async for k in self._redis.scan_iter(match=pattern, count=100)]
            except Exception as e:
                logger_pair.debug(f"Failed to scan pending outcomes for {pair}: {e}")
                return

        if not keys:
            return

        try:
            async with self._redis.pipeline() as read_pipe:
                for key in keys:
                    read_pipe.get(key)
                raw_values = await asyncio.wait_for(read_pipe.execute(), timeout=2.0)
        except Exception as e:
            logger_pair.warning(f"Failed to batch-fetch pending outcomes for {pair}: {e}")
            return

        resolved_count = 0
        not_ready_count = 0
        ts_mismatch_count = 0
        missing_score_count = 0
        bad_payload_count = 0

        stats_ttl = max(cfg.STATE_EXPIRY_DAYS * 86400, 7 * 86400)

        try:
            async with self._redis.pipeline() as write_pipe:
                pending_writes = 0

                for key, raw in zip(keys, raw_values):
                    try:
                        if raw is None:
                            # Race: expired between scan and fetch; harmless, skip silently
                            continue

                        data = json_loads(raw)
                        entry_ts = int(data["entry_ts"])
                        direction = data["direction"]
                        entry_price = float(data["entry_price"])
                        conf_score = data.get("confluence_score")
                        conf_total = data.get("confluence_total")

                        if entry_price <= 0:
                            logger_pair.debug(
                                f"Invalid entry_price {entry_price} for pending outcome {key}; skipping"
                            )
                            bad_payload_count += 1
                            continue

                        direction_norm = str(direction).lower()
                        if direction_norm in ("buy", "long"):
                            is_buy = True
                        elif direction_norm in ("sell", "short"):
                            is_buy = False
                        else:
                            logger_pair.debug(
                                f"Unknown direction '{direction}' for pending outcome {key}; skipping"
                            )
                            bad_payload_count += 1
                            continue

                        entry_idx = int(np.searchsorted(data_15m.ts, entry_ts))
                        # ── DEFENSIVE: fallback to exact match if searchsorted missed due to out-of-order timestamps ──
                        if entry_idx >= len(data_15m.ts) or data_15m.ts[entry_idx] != entry_ts:
                            exact_matches = np.flatnonzero(data_15m.ts == entry_ts)
                            if exact_matches.size == 0:
                                ts_mismatch_count += 1
                                if cfg.DEBUG_MODE:
                                    logger_pair.debug(
                                        f"[{pair}] Outcome entry_ts not found | "
                                        f"entry_ts={entry_ts} | "
                                        f"first_ts={data_15m.ts[0] if len(data_15m.ts) else None} | "
                                        f"last_ts={data_15m.ts[-1] if len(data_15m.ts) else None}"
                                    )
                                continue  # scrolled out of window; leave pending for next run
                            entry_idx = int(exact_matches[-1])
                        # ── END DEFENSIVE ──

                        target_idx = entry_idx + cfg.OUTCOME_LOOKAHEAD_CANDLES
                        if target_idx > i15:
                            not_ready_count += 1
                            continue  # not enough candles closed yet; grade on a later run

                        future_price = float(data_15m.close[target_idx])
                        pct_move = (future_price - entry_price) / entry_price * 100.0
                        win = (
                            pct_move >= cfg.OUTCOME_FAVORABLE_MOVE_PCT
                            if is_buy
                            else pct_move <= -cfg.OUTCOME_FAVORABLE_MOVE_PCT
                        )

                        alert_key = key.split(":")[-2]
                        stats_key = f"{RedisKeyPrefix.ALERT_STATS}{pair}:{alert_key}"
                        write_pipe.hincrby(stats_key, "wins" if win else "losses", 1)
                        write_pipe.expire(stats_key, stats_ttl)

                        stream_fields = None
                        if conf_score is not None and conf_total is not None:
                            stream_fields = {
                                "pair": str(pair),
                                "alert_key": str(alert_key),
                                "direction": str(direction),
                                "score": str(conf_score),
                                "total": str(conf_total),
                                "pct_move": f"{pct_move:.4f}",
                                "win": "1" if win else "0",
                                "entry_ts": str(entry_ts),
                            }
                        else:
                            missing_score_count += 1
                            logger_pair.debug(
                                f"[{pair}] Outcome for {alert_key} has no confluence score/total; "
                                f"stats updated but stream entry skipped"
                            )

                        if stream_fields is not None:
                            write_pipe.xadd(
                                RedisKeyPrefix.OUTCOME_LOG_STREAM,
                                stream_fields,
                                maxlen=50000,
                                approximate=True,
                            )

                        write_pipe.delete(key)
                        pending_writes += 1
                        resolved_count += 1

                    except Exception as e:
                        logger_pair.debug(f"Failed to resolve pending outcome {key}: {e}")
                        bad_payload_count += 1
                        continue

                if pending_writes:
                    await asyncio.wait_for(write_pipe.execute(), timeout=2.0)

        except Exception as e:
            logger_pair.debug(f"Failed to persist resolved outcomes for {pair}: {e}")
            return

        logger_pair.info(
            f"[{pair}] Outcome resolution | "
            f"pending={len(keys)} | "
            f"resolved={resolved_count} | "
            f"not_ready={not_ready_count} | "
            f"ts_mismatch={ts_mismatch_count} | "
            f"missing_score={missing_score_count} | "
            f"bad_payload={bad_payload_count}"
        )

    async def resolve_shadow_pending_outcomes(self, pair: str, data_15m: "PriceData", i15: int,
                                                logger_pair: logging.Logger) -> None:
        """Twin of resolve_pending_outcomes for shadow (rejected) alerts. Same grading logic,
        writes to SHADOW_STATS/SHADOW_LOG_STREAM instead, and additionally pools outcomes whose
        confluence was in the 'rewardable' bucket into SHADOW_HICONF_STATS for override checks."""
        if self.degraded or not getattr(cfg, "ENABLE_BRAIN", False) or not getattr(cfg, "BRAIN_SHADOW_MODE", True) or not self._redis:
            return

        precomputed = getattr(self, "_shadow_pending_outcome_keys_by_pair", None)
        if precomputed is not None:
            keys = precomputed.get(pair, [])
            if not keys:
                return
        else:
            try:
                pattern = f"{RedisKeyPrefix.SHADOW_PENDING}{pair}:*"
                keys = [k async for k in self._redis.scan_iter(match=pattern, count=100)]
            except Exception as e:
                logger_pair.debug(f"Failed to scan shadow pending outcomes for {pair}: {e}")
                return

        if not keys:
            return

        try:
            async with self._redis.pipeline() as read_pipe:
                for key in keys:
                    read_pipe.get(key)
                raw_values = await asyncio.wait_for(read_pipe.execute(), timeout=2.0)
        except Exception as e:
            logger_pair.warning(f"Failed to batch-fetch shadow pending outcomes for {pair}: {e}")
            return

        resolved_count = 0
        hiconf_pct = getattr(cfg, "BRAIN_REWARDABLE_MIN_CONFLUENCE_PCT", 80.0)
        stats_ttl = max(cfg.STATE_EXPIRY_DAYS * 86400, 7 * 86400)

        try:
            async with self._redis.pipeline() as write_pipe:
                pending_writes = 0

                for key, raw in zip(keys, raw_values):
                    try:
                        if raw is None:
                            continue

                        data = json_loads(raw)
                        entry_ts = int(data["entry_ts"])
                        direction = data["direction"]
                        entry_price = float(data["entry_price"])
                        conf_score = data.get("confluence_score")
                        conf_total = data.get("confluence_total")

                        if entry_price <= 0:
                            continue

                        direction_norm = str(direction).lower()
                        if direction_norm in ("buy", "long"):
                            is_buy = True
                        elif direction_norm in ("sell", "short"):
                            is_buy = False
                        else:
                            continue

                        entry_idx = int(np.searchsorted(data_15m.ts, entry_ts))
                        if entry_idx >= len(data_15m.ts) or data_15m.ts[entry_idx] != entry_ts:
                            exact_matches = np.flatnonzero(data_15m.ts == entry_ts)
                            if exact_matches.size == 0:
                                continue  # scrolled out of window; leave pending for next run
                            entry_idx = int(exact_matches[-1])

                        target_idx = entry_idx + cfg.OUTCOME_LOOKAHEAD_CANDLES
                        if target_idx > i15:
                            continue  # not enough candles closed yet

                        future_price = float(data_15m.close[target_idx])
                        pct_move = (future_price - entry_price) / entry_price * 100.0
                        win = (
                            pct_move >= cfg.OUTCOME_FAVORABLE_MOVE_PCT
                            if is_buy
                            else pct_move <= -cfg.OUTCOME_FAVORABLE_MOVE_PCT
                        )

                        alert_key = key.split(":")[-2]
                        stats_key = f"{RedisKeyPrefix.SHADOW_STATS}{pair}:{alert_key}"
                        write_pipe.hincrby(stats_key, "wins" if win else "losses", 1)
                        write_pipe.expire(stats_key, stats_ttl)

                        if conf_score is not None and conf_total is not None and conf_total > 0:
                            conf_pct = (conf_score / conf_total) * 100.0
                            write_pipe.xadd(
                                RedisKeyPrefix.SHADOW_LOG_STREAM,
                                {
                                    "pair": str(pair), "alert_key": str(alert_key),
                                    "direction": str(direction), "score": str(conf_score),
                                    "total": str(conf_total), "pct_move": f"{pct_move:.4f}",
                                    "win": "1" if win else "0", "entry_ts": str(entry_ts),
                                },
                                maxlen=50000, approximate=True,
                            )
                            if conf_pct >= hiconf_pct:
                                hiconf_key = f"{RedisKeyPrefix.SHADOW_HICONF_STATS}{alert_key}"
                                write_pipe.hincrby(hiconf_key, "wins" if win else "losses", 1)
                                write_pipe.expire(hiconf_key, stats_ttl)

                        write_pipe.delete(key)
                        pending_writes += 1
                        resolved_count += 1

                    except Exception as e:
                        logger_pair.debug(f"Failed to resolve shadow pending outcome {key}: {e}")
                        continue

                if pending_writes:
                    await asyncio.wait_for(write_pipe.execute(), timeout=2.0)

        except Exception as e:
            logger_pair.debug(f"Failed to persist resolved shadow outcomes for {pair}: {e}")
            return

        if resolved_count:
            logger_pair.debug(f"[{pair}] Shadow outcome resolution | resolved={resolved_count}")

    async def get_alert_win_rate(self, pair: str, alert_key: str) -> Tuple[Optional[float], int]:
        """Returns (win_rate, sample_size). win_rate is None until MIN_WIN_RATE_SAMPLE is reached."""
        if self.degraded or not cfg.ENABLE_WIN_RATE_FILTER:
            return None, 0
        stats_key = f"{RedisKeyPrefix.ALERT_STATS}{pair}:{alert_key}"
        try:
            data = await asyncio.wait_for(self._redis.hgetall(stats_key), timeout=2.0)
            wins = int(data.get("wins", 0))
            losses = int(data.get("losses", 0))
            total = wins + losses
            if total < cfg.MIN_WIN_RATE_SAMPLE:
                return None, total
            return wins / total, total
        except Exception as e:
            logger.warning(f"Failed to read win rate for {pair}:{alert_key}: {e}")
            return None, 0

    async def batch_get_alert_win_rates(self, pair: str, alert_keys: List[str], timeout: float = 3.0) -> Dict[str, Tuple[Optional[float], int]]:
        if self.degraded or not cfg.ENABLE_WIN_RATE_FILTER or not alert_keys:
            return {k: (None, 0) for k in alert_keys}
        try:
            async with self._redis.pipeline() as pipe:
                for ak in alert_keys:
                    pipe.hgetall(f"{RedisKeyPrefix.ALERT_STATS}{pair}:{ak}")
                raw_results = await asyncio.wait_for(pipe.execute(), timeout=timeout)
            out: Dict[str, Tuple[Optional[float], int]] = {}
            for ak, data in zip(alert_keys, raw_results):
                if not data:
                    out[ak] = (None, 0)
                    continue
                wins = int(data.get("wins", 0))
                losses = int(data.get("losses", 0))
                total = wins + losses
                if total < cfg.MIN_WIN_RATE_SAMPLE:
                    out[ak] = (None, total)
                else:
                    out[ak] = (wins / total, total)
            return out
        except Exception as e:
            logger.warning(f"batch_get_alert_win_rates({pair}) failed for {len(alert_keys)} keys: {e}")
            return {k: (None, 0) for k in alert_keys}

    async def batch_get_all_alert_states(self, pair: str, alert_keys: List[str], timeout: float = 3.0) -> Dict[str, bool]:
        if not self._redis or self.degraded or not alert_keys:
            return {k: False for k in alert_keys}

        try:
            hash_key = f"{self.state_prefix}{pair}"
            hash_data = await asyncio.wait_for(
                self._redis.hgetall(hash_key),
                timeout=timeout,
            )

            states: Dict[str, bool] = {}
            for key in alert_keys:
                val = hash_data.get(key)

                if val is None:
                    states[key] = False
                    continue

                try:
                    parsed_state = json_loads(val)
                    states[key] = parsed_state.get("state") == "ACTIVE"
                except (JSONDecodeError, TypeError) as e:
                    if cfg.DEBUG_MODE:
                        logger.debug(f"Failed to parse state for {pair}:{key}: {e}")
                    states[key] = False
                except Exception as e:
                    logger.error(f"Unexpected error parsing state for {pair}:{key}: {e}")
                    states[key] = False

            return states
        except asyncio.TimeoutError as e:
            await self._record_redis_failure(f"batch_get_all_alert_states({pair})", e)
            return {k: False for k in alert_keys}
        except Exception as e:
            await self._record_redis_failure(f"batch_get_all_alert_states({pair})", e)
            return {k: False for k in alert_keys}

    async def atomic_batch_update(self, updates: List[Tuple[str, Any, Optional[int]]], deletes: Optional[List[str]] = None, timeout: float = 4.0) -> bool:
        if self.degraded or not self._redis:
            return False

        if not updates and not deletes:
            return True

        try:
            async with self._redis.pipeline() as pipe:
                now = int(time.time())
                touched_hashes: Set[str] = set()

                hash_writes: Dict[str, Dict[str, str]] = {}
                for key, state, custom_ts in (updates or []):
                    pair, sep, field = key.partition(":")
                    if not sep:
                        logger.error(
                            f"Skipping malformed state key (expected 'pair:field'): {key}"
                        )
                        continue
                    ts = custom_ts if custom_ts is not None else now
                    try:
                        data = json_dumps({"state": state, "ts": ts})
                    except Exception as e:
                        logger.error(f"Failed to serialize state for {key}: {e}")
                        continue
                    hash_key = f"{self.state_prefix}{pair}"
                    hash_writes.setdefault(hash_key, {})[field] = data

                for hash_key, mapping in hash_writes.items():
                    pipe.hset(hash_key, mapping=mapping)
                    touched_hashes.add(hash_key)

                hash_deletes: Dict[str, List[str]] = {}
                for key in (deletes or []):
                    if not key:
                        continue
                    raw_key = (
                        key[len(self.state_prefix) :]
                        if key.startswith(self.state_prefix)
                        else key
                    )
                    pair, sep, field = raw_key.partition(":")
                    if not sep:
                        logger.error(
                            f"Skipping malformed delete key (expected 'pair:field'): {key}"
                        )
                        continue
                    hash_key = f"{self.state_prefix}{pair}"
                    hash_deletes.setdefault(hash_key, []).append(field)

                for hash_key, fields in hash_deletes.items():
                    pipe.hdel(hash_key, *fields)
                    touched_hashes.add(hash_key)

                if self.expiry_seconds > 0:
                    for hash_key in touched_hashes:
                        pipe.expire(hash_key, self.expiry_seconds)

                await asyncio.wait_for(pipe.execute(), timeout=timeout)
            return True
        except asyncio.TimeoutError as e:
            await self._record_redis_failure("atomic_batch_update", e)
            return False
        except Exception as e:
            await self._record_redis_failure("atomic_batch_update", e)
            return False

class RedisLock:    
    RELEASE_LUA = """
    if redis.call("GET", KEYS[1]) == ARGV[1] then
        return redis.call("DEL", KEYS[1])
    else
        return 0
    end
    """
    EXTEND_LUA = """
    if redis.call("GET", KEYS[1]) == ARGV[1] then
        return redis.call("EXPIRE", KEYS[1], ARGV[2])
    else
        return 0
    end
    """
    def __init__(self, redis_client: Optional[redis.Redis], lock_key: str, expire: int | None = None):
        self.redis = redis_client
        self.lock_key = f"{RedisKeyPrefix.LOCK}{lock_key}"
        self.expire = expire or cfg.REDIS_LOCK_EXPIRY
        self.token: Optional[str] = None
        self.lost = False
        self.acquired_by_me = False
        self.last_extend_time = time.monotonic() 

    async def acquire(self, timeout: float = 5.0) -> bool:  
        if not self.redis:
            logger.warning("Redis not available; cannot acquire lock")
            return False
        
        try:
            token = str(uuid.uuid4())
            ok = await asyncio.wait_for(
                self.redis.set(self.lock_key, token, nx=True, ex=self.expire),
                timeout=timeout,
            )
            
            if ok:
                self.token = token
                self.acquired_by_me = True
                self.lost = False
                self.last_extend_time = time.monotonic()
                
                logger.info(
                    f"🔐 Lock acquired: {self.lock_key.replace('lock:', '')} ({self.expire}s)"
                )
                return True

            logger.warning(f"Could not acquire Redis lock (held): {self.lock_key}")
            return False
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout acquiring lock {self.lock_key} after {timeout}s")
            return False
        except Exception as e:
            logger.error(f"Redis lock acquisition failed: {e}")
            return False

    async def extend(self, timeout: float = 3.0) -> bool:     
        if not self.token or not self.redis or not self.acquired_by_me:
            self.lost = True
            return False    
        try:
            result = await asyncio.wait_for(
                self.redis.eval(
                    self.EXTEND_LUA,
                    1,
                    self.lock_key,
                    self.token,
                    self.expire,
                ),
                timeout=timeout,
            )

            if result:
                self.last_extend_time = time.monotonic()
                if cfg.DEBUG_MODE:
                    logger.debug(f"Extended Redis lock: {self.lock_key} (now {self.expire}s)")
                return True
            else:
                logger.warning("Lock lost during extend (token mismatch or key missing)")
                self.lost = True
                self.acquired_by_me = False
                return False
                
        except asyncio.TimeoutError:
            logger.error(f"Timeout extending lock {self.lock_key} after {timeout}s")
            self.lost = True
            self.acquired_by_me = False
            return False
        except Exception as e:
            logger.error(f"Error extending Redis lock: {e}")
            self.lost = True
            self.acquired_by_me = False
            return False

    @classmethod
    def get_lock_extend_interval(cls) -> int:    
        extend_at = int(cfg.REDIS_LOCK_EXPIRY * 0.7)
        return max(60, min(extend_at, 540)) 

    def should_extend(self) -> bool:     
        if not self.acquired_by_me or self.lost:
            return False

        extend_threshold = self.__class__.get_lock_extend_interval()       
        elapsed = time.monotonic() - self.last_extend_time 
        should_extend = elapsed >= extend_threshold
        
        if cfg.DEBUG_MODE and should_extend:
            logger.debug(
                f"Lock extension eligible | "
                f"Elapsed: {elapsed:.0f}s | "
                f"Threshold: {extend_threshold}s"
            )
        
        return should_extend

    async def release(self, timeout: float = 3.0) -> None:     
        if not self.token or not self.redis or not self.acquired_by_me:
            return
        try:
            result = await asyncio.wait_for(
                self.redis.eval(self.RELEASE_LUA, 1, self.lock_key, self.token),
                timeout=timeout,
            )
        
            if result:
                logger.info(f"🔏 Lock released: {self.lock_key.replace('lock:', '')}")
                self.acquired_by_me = False
                self.token = None
            else:
                logger.warning(
                    f"Lock release failed (token mismatch): {self.lock_key} | "
                    f"Lock was stolen or lost"
                )
                self.lost = True
                self.acquired_by_me = False
    
        except asyncio.TimeoutError:
            logger.error(f"Timeout releasing lock {self.lock_key} after {timeout}s")
            self.lost = True
            self.acquired_by_me = False
        except Exception as e:
            logger.error(f"Error releasing Redis lock: {e}")
            self.lost = True
            self.acquired_by_me = False
    
        finally:
            self.token = None

    def __repr__(self) -> str:
        status = "HELD" if self.acquired_by_me else ("LOST" if self.lost else "RELEASED")
        token_display = self.token[:8] + "..." if self.token else "None"
        return f"RedisLock({self.lock_key}:{status}:{token_display})"

class TokenBucket:
    def __init__(self, rate: int, burst: int):
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self.lock:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.tokens = min(self.burst, self.tokens + elapsed * (self.rate / 60))
                self.last_update = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait_time = (1 - self.tokens) / (self.rate / 60)
            await asyncio.sleep(wait_time)

