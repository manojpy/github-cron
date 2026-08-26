from __future__ import annotations
import time
import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Set, Union
import numpy as np
from bot_config import cfg, Constants, PAIR_ID, normalize_timestamp, normalize_timestamp_array, format_ist_time
from fetcher import PriceData, get_last_closed_index_from_array, validate_candle_for_alerts
from state import RedisStateStore, _blanket_reset_pair
from indicators import (
    calculate_gate_indicators_numpy, get_atr_percentile, get_adx_percentile,
    get_volume_percentile, get_adaptive_rvol_threshold, get_adaptive_ppo_threshold,
    get_adaptive_rsi_thresholds, get_adaptive_cpr_threshold, _order_block_gate_reason,
    _oi_funding_gate_reason, get_adaptive_adx_threshold_smoothed, _get_smoothed_pctl,
    _choch_gate_reason, _tlr_evaluate_touch, TrendlineState,
)

@dataclass(slots=True)
class IndicatorCache:
    """Typed replacement for the merged gate_indicators/alert_indicators dict."""
    # -- gate indicators (Phase 1, cheap) --
    rma50_15: np.ndarray
    rma200_5: np.ndarray
    ichimoku_cloud_upper: np.ndarray
    ichimoku_cloud_lower: np.ndarray
    ichimoku_future_green: np.ndarray
    ichimoku_future_red: np.ndarray
    ichimoku_conversion_line: np.ndarray
    ichimoku_base_line: np.ndarray
    adx: np.ndarray
    atr_short: np.ndarray
    atr_long: np.ndarray
    volume_ema: np.ndarray
    ppo_gate: np.ndarray
    ppo_gate_signal: np.ndarray
    rsi_guard_smooth: np.ndarray
    rsi_guard_ema: np.ndarray
    rma_cloud_fast_15: np.ndarray
    cpr_ok: bool = True
    nr_cpr: float = float("nan")
    prev_day_close: float = float("nan")
    # -- alert indicators (Phase 2, expensive) --
    ppo: Optional[np.ndarray] = None
    ppo_signal: Optional[np.ndarray] = None
    smooth_rsi: Optional[np.ndarray] = None
    smooth_rsi_ema: Optional[np.ndarray] = None
    vwap: Optional[np.ndarray] = None
    hist_rma: Optional[np.ndarray] = None
    pivots: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dicts(cls, gate: Dict[str, Any], alert: Optional[Dict[str, Any]] = None) -> "IndicatorCache":
        merged = {**gate, **(alert or {})}
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in merged.items() if k in known})

    def as_dict(self) -> Dict[str, Any]:
        """Back-compat shim — mirrors the old `{**gate_indicators, **alert_indicators}` merge."""
        return {f: getattr(self, f) for f in self.__dataclass_fields__}

@dataclass(slots=True)
class GateResult:
    # -- identity / indices --
    pair_name: str
    i15: int
    i5: int
    ts_curr: int
    reference_time: int

    # -- candle info --
    candle_info: Dict[str, Any]
    o: float; h: float; l: float; c: float
    open_curr: float; high_curr: float; low_curr: float; close_curr: float
    close_prev: float
    close_5m_val: float
    is_green: bool; is_red: bool
    is_valid_for_buy: bool; is_valid_for_sell: bool
    candle_index: int
    min_wick_ratio: float
    buy_wick_ratio: float; sell_wick_ratio: float

    # -- gate/alert indicator dicts (still raw dicts — untouched by this pass) --
    gate_indicators: Dict[str, Any]
    
    # -- trend --
    base_buy_trend: bool; base_sell_trend: bool
    rma50_15_val: float; rma200_5_val: float

    # -- ichimoku cloud --
    cloud_up: Optional[bool]; cloud_down: Optional[bool]
    cloud_upper_val: float; cloud_lower_val: float
    cloud_upper_prev: float; cloud_lower_prev: float
    ichimoku_gate_ok_buy: Optional[bool]; ichimoku_gate_ok_sell: Optional[bool]
    confirmation_buy: bool; confirmation_sell: bool
    cloud_group_ok_buy: bool; cloud_group_ok_sell: bool

    # -- TK guard --
    tk_conversion_curr: float; tk_conversion_prev: float
    tk_base_curr: float; tk_base_prev: float
    tk_guard_ok_buy: Optional[bool]; tk_guard_ok_sell: Optional[bool]

    # -- oscillator group votes --
    oscillator_group_ok_buy: bool; oscillator_group_ok_sell: bool
    ppo_gate_arr: np.ndarray; ppo_gate_signal_arr: np.ndarray
    ppo_gate_curr: float; ppo_gate_prev: float
    ppo_gate_sig_curr: float; ppo_gate_sig_prev: float
    ppo_gate_ok_buy: bool; ppo_gate_ok_sell: bool
    rsi_guard_smooth_curr: float; rsi_guard_ema_curr: float
    rsi_guard_ok_buy: bool; rsi_guard_ok_sell: bool
    rma_cloud_fast_curr: float
    rma_cloud_ok_buy: bool; rma_cloud_ok_sell: bool

    # -- trend gate combination --
    trend_gate_ok_buy: bool; trend_gate_ok_sell: bool

    # -- volatility / ADX / RVOL --
    adx_val: float; adx_adaptive_threshold: float; adx_ok: bool
    rvol_bypass_ok: bool; rvol_ok: bool
    adaptive_rvol_check: bool
    momentum_count: int
    volatility_filter_ok: bool

    # -- CPR --
    cpr_ok: bool; nr_cpr: float
    effective_cpr_ok: bool
    cpr_adaptive_min_pct_move: float
    move_from_prev_close_ok: bool

    # -- adaptive thresholds carried into Phase 2 --
    ppo_adaptive_threshold: float
    rsi_adaptive_buy: float; rsi_adaptive_sell: float

    # -- final gate decision --
    buy_common: bool
    sell_common: bool
    buy_trend_common: bool
    sell_trend_common: bool
    buy_trend_common_relaxed: bool
    sell_trend_common_relaxed: bool

    # -- misc data passed through --
    data_15m: PriceData
    close_prev_invalid: bool = False

    # -- OI/funding (optional confluence vote) --
    oi_funding_ok_buy: Optional[bool] = None
    oi_funding_ok_sell: Optional[bool] = None
    oi_funding_reason: Optional[str] = None

    # -- order block / supply-demand (optional confluence vote) --
    ob_gate_ok_buy: Optional[bool] = None
    ob_gate_ok_sell: Optional[bool] = None
    ob_gate_reason: Optional[str] = None
    direction_is_buy: bool = True

    # -- CHoCH liquidity-sweep reversal (optional confluence vote) --
    choch_gate_ok_buy: Optional[bool] = None
    choch_gate_ok_sell: Optional[bool] = None
    choch_reason: Optional[str] = None
    choch_fvg_buy: bool = False
    choch_fvg_sell: bool = False
    choch_poi_tap_buy: bool = False
    choch_poi_tap_sell: bool = False

    atr_short_arr: Optional[np.ndarray] = None
    tlr_touch_gate_ok_buy: bool = False
    tlr_touch_gate_ok_sell: bool = False
    tlr_touch_reason: Optional[str] = None
    tlr_trendline_buy: Optional[TrendlineState] = None
    tlr_trendline_sell: Optional[TrendlineState] = None
    tlr_prior_touch_idx_buy: Optional[int] = None
    tlr_prior_touch_idx_sell: Optional[int] = None

    # -- percentile-rank confluence votes (optional, all default-disabled) --
    adx_pctl: Optional[float] = None
    adx_strength_ok: Optional[bool] = None
    atr_pctl: Optional[float] = None
    atr_pctl_ok: Optional[bool] = None
    volume_pctl: Optional[float] = None
    volume_pctl_ok: Optional[bool] = None

    # -- momentum-direction confluence votes (rising/falling vs prior bar) --
    ppo_gate_momentum_ok_buy: Optional[bool] = None
    ppo_gate_momentum_ok_sell: Optional[bool] = None
    rsi_guard_momentum_ok_buy: Optional[bool] = None
    rsi_guard_momentum_ok_sell: Optional[bool] = None
    rma_cloud_momentum_ok_buy: Optional[bool] = None
    rma_cloud_momentum_ok_sell: Optional[bool] = None 
    vwap_momentum_ok_buy: Optional[bool] = None
    vwap_momentum_ok_sell: Optional[bool] = None

CONFLUENCE_WEIGHTS: Dict[str, float] = {
    "base_trend": 3.0,
    "ichimoku_cloud": 2.0,
    "rma_cloud": 2.0,
    "ppo_cross": 2.0,
    "rsi_guard": 2.0,
    "tk_guard": 2.0,
    "adx": 1.0,
    "rvol": 1.5,
    "cpr": 1.0,
    "oi_funding": 2.5,
    "order_block": 2.5,
    "adx_strength": 1.5,
    "atr_percentile": 1.5,
    "volume_percentile": 1.5, 
    "ppo_gate_momentum":  1.0,
    "rsi_guard_momentum": 1.0,
    "rma_cloud_momentum": 1.0,
    "vwap_momentum": 1.0,
}

def compute_confluence_score(gr: "GateResult", is_buy: bool, exclude: Optional[Set[str]] = None) -> Tuple[float, float]:
    exclude = exclude or set()
    score = 0.0
    total = 0.0

    base_trend    = gr.base_buy_trend if is_buy else gr.base_sell_trend
    ichimoku_ok   = gr.ichimoku_gate_ok_buy if is_buy else gr.ichimoku_gate_ok_sell
    rma_cloud_ok  = gr.rma_cloud_ok_buy if is_buy else gr.rma_cloud_ok_sell
    ppo_cross_ok  = gr.ppo_gate_ok_buy if is_buy else gr.ppo_gate_ok_sell
    rsi_guard_ok  = gr.rsi_guard_ok_buy if is_buy else gr.rsi_guard_ok_sell
    tk_guard_ok   = gr.tk_guard_ok_buy if is_buy else gr.tk_guard_ok_sell
    oi_funding_ok = gr.oi_funding_ok_buy if is_buy else gr.oi_funding_ok_sell
    ob_gate_ok    = gr.ob_gate_ok_buy if is_buy else gr.ob_gate_ok_sell

    base_trend_included = "base_trend" not in exclude
    if base_trend_included:
        w = CONFLUENCE_WEIGHTS["base_trend"]
        total += w
        if base_trend: score += w

    if cfg.ICHIMOKU_CLOUD_ENABLED and ichimoku_ok is not None and "ichimoku_cloud" not in exclude:
        w = CONFLUENCE_WEIGHTS["ichimoku_cloud"]
        total += w
        if ichimoku_ok: score += w

    if cfg.RMA_CLOUD_ENABLED and rma_cloud_ok is not None and "rma_cloud" not in exclude:
        w = CONFLUENCE_WEIGHTS["rma_cloud"]
        total += w
        if rma_cloud_ok: score += w

    if cfg.ENABLE_PPO_GATE and "ppo_cross" not in exclude:
        w = CONFLUENCE_WEIGHTS["ppo_cross"]
        total += w
        if ppo_cross_ok: score += w

    if cfg.RSI_GUARD_ENABLED and "rsi_guard" not in exclude:
        w = CONFLUENCE_WEIGHTS["rsi_guard"]
        total += w
        if rsi_guard_ok: score += w

    if cfg.ICHIMOKU_TK_GUARD_ENABLED and tk_guard_ok is not None and "tk_guard" not in exclude:
        w = CONFLUENCE_WEIGHTS["tk_guard"]
        total += w
        if tk_guard_ok: score += w
    
    ppo_gate_mom_ok = gr.ppo_gate_momentum_ok_buy if is_buy else gr.ppo_gate_momentum_ok_sell
    if cfg.ENABLE_PPO_GATE_MOMENTUM_VOTE and ppo_gate_mom_ok is not None and "ppo_gate_momentum" not in exclude:
        w = CONFLUENCE_WEIGHTS["ppo_gate_momentum"]
        total += w
        if ppo_gate_mom_ok:
            score += w

    rsi_guard_mom_ok = gr.rsi_guard_momentum_ok_buy if is_buy else gr.rsi_guard_momentum_ok_sell
    if cfg.ENABLE_RSI_GUARD_MOMENTUM_VOTE and rsi_guard_mom_ok is not None and "rsi_guard_momentum" not in exclude:
        w = CONFLUENCE_WEIGHTS["rsi_guard_momentum"]
        total += w
        if rsi_guard_mom_ok:
            score += w

    rma_cloud_mom_ok = gr.rma_cloud_momentum_ok_buy if is_buy else gr.rma_cloud_momentum_ok_sell
    if cfg.ENABLE_RMA_CLOUD_MOMENTUM_VOTE and rma_cloud_mom_ok is not None and "rma_cloud_momentum" not in exclude:
        w = CONFLUENCE_WEIGHTS["rma_cloud_momentum"]
        total += w
        if rma_cloud_mom_ok:
            score += w

    vwap_mom_ok = gr.vwap_momentum_ok_buy if is_buy else gr.vwap_momentum_ok_sell
    if cfg.ENABLE_VWAP_MOMENTUM_VOTE and vwap_mom_ok is not None and "vwap_momentum" not in exclude:
        w = CONFLUENCE_WEIGHTS["vwap_momentum"]
        total += w
        if vwap_mom_ok:
            score += w

    if cfg.ENABLE_ADX_FILTER and "adx" not in exclude:
        w = CONFLUENCE_WEIGHTS["adx"]
        total += w
        if gr.adx_ok: score += w

    if (cfg.ENABLE_RVOL_ALERT or cfg.ATR_ADAPTIVE_ENABLED) and "rvol" not in exclude:
        w = CONFLUENCE_WEIGHTS["rvol"]
        total += w
        if gr.rvol_ok: score += w

    if cfg.ENABLE_CPR and "cpr" not in exclude:
        w = CONFLUENCE_WEIGHTS["cpr"]
        total += w
        if gr.effective_cpr_ok: score += w

    if cfg.ENABLE_ADX_STRENGTH_VOTE and gr.adx_strength_ok is not None and "adx_strength" not in exclude:
        w = CONFLUENCE_WEIGHTS["adx_strength"]
        total += w
        if gr.adx_strength_ok: score += w

    if cfg.ENABLE_ATR_PCTL_VOTE and gr.atr_pctl_ok is not None and "atr_percentile" not in exclude:
        w = CONFLUENCE_WEIGHTS["atr_percentile"]
        total += w
        if gr.atr_pctl_ok: score += w

    if cfg.ENABLE_VOLUME_PCTL_VOTE and gr.volume_pctl_ok is not None and "volume_percentile" not in exclude:
        w = CONFLUENCE_WEIGHTS["volume_percentile"]
        total += w
        if gr.volume_pctl_ok: score += w

    if cfg.ENABLE_OI_FUNDING_FILTER and oi_funding_ok is not None and "oi_funding" not in exclude:
        w = CONFLUENCE_WEIGHTS["oi_funding"]
        total += w
        if oi_funding_ok: score += w

    if cfg.ENABLE_OB_GATE and ob_gate_ok is not None and "order_block" not in exclude:
        w = CONFLUENCE_WEIGHTS["order_block"]
        total += w
        if ob_gate_ok:
            score += w
            base_trend_weight = CONFLUENCE_WEIGHTS["base_trend"]
            other_score = score - w - (base_trend_weight if (base_trend and base_trend_included) else 0.0)
            if other_score < cfg.OB_MIN_OTHER_SCORE:
                score -= w

    return score, total

async def _eval_gate(pair_name: str, data_15m: PriceData, data_5m: PriceData,
    data_daily: Optional[Dict[str, np.ndarray]], sdb: RedisStateStore, correlation_id: str,
    reference_time: int, pair_oi: Optional[Dict[str, Any]] = None) -> Union[GateResult, Tuple[str, Dict[str, Any]], None]:
    logger_pair = logging.getLogger(f"macd_bot.{pair_name}.{correlation_id}")
    PAIR_ID.set(pair_name)
    close_15m = None
    timestamps_15m = None
    rma50_15 = None
    rma200_5 = None

    try:
        i15 = get_last_closed_index_from_array(data_15m.ts, 15, reference_time, pair_name)
        if i15 is None or i15 < Constants.MIN_CLOSED_CANDLES_15M:
            return None

        if cfg.ENABLE_WIN_RATE_FILTER:
            await sdb.resolve_pending_outcomes(pair_name, data_15m, i15, logger_pair)

        is_valid_for_buy, is_valid_for_sell, candle_info, error_msg = validate_candle_for_alerts(
            data_15m=data_15m.as_dict(),
            candle_index=i15,
            reference_time=reference_time,
            pair_name=pair_name,
            min_wick_ratio=Constants.MIN_WICK_RATIO
        )
        if not is_valid_for_buy and not is_valid_for_sell:
            if candle_info is None:
                logger_pair.debug(
                    f"[{pair_name}] Hard-rejecting candle: {error_msg}"
                )
                await _blanket_reset_pair(sdb, pair_name, logger_pair)
                return pair_name, {
                    "state": "HARD_REJECT",
                    "ts": int(time.time()),
                    "summary": {
                        "alerts": 0,
                        "future_cloud": "neutral",
                        "hist_rma": 0.0,
                        "suppression": f"Hard reject: {error_msg}"
                    }
                }
            if not cfg.ENABLE_STRONG_REVERSAL_ALERT:
                logger_pair.debug(
                    f"[{pair_name}] Wick-rejected candle → blanket reset only. Reason: {error_msg}"
                )
                await _blanket_reset_pair(sdb, pair_name, logger_pair)
                return pair_name, {
                    "state": "NO_SIGNAL",
                    "ts": int(time.time()),
                    "summary": {
                        "alerts": 0,
                        "future_cloud": "neutral",
                        "hist_rma": 0.0,
                        "suppression": f"Wick rejected: {error_msg}"
                    }
                }
            logger_pair.debug(
                f"[{pair_name}] Shape-rejected candle kept for strong-reversal check: {error_msg}"
            )
        o = candle_info["open"]
        h = candle_info["high"]
        l = candle_info["low"]
        c = candle_info["close"]
        ts_curr = candle_info["timestamp"]
        is_green = candle_info["is_green"]
        is_red = candle_info["is_red"]
        buy_wick_ratio = candle_info["upper_wick_ratio"]
        sell_wick_ratio = candle_info["lower_wick_ratio"]
  
        if is_valid_for_buy and not is_green:
            raise RuntimeError(
                f"[{pair_name}] INVARIANT VIOLATED: is_valid_for_buy=True on non-green candle | "
                f"O={o:.2f} C={c:.2f}"
            )
        if is_valid_for_sell and not is_red:
            raise RuntimeError(
                f"[{pair_name}] INVARIANT VIOLATED: is_valid_for_sell=True on non-red candle | "
                f"O={o:.2f} C={c:.2f}"
            )

        logger_pair.debug(
            f"[{pair_name}] 🕯️ Candle | O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f} | "
            f"{'🟢 GREEN' if is_green else '🔴 RED'} | "
            f"ValidBuy={is_valid_for_buy} ValidSell={is_valid_for_sell}"
        )
        open_curr = o
        high_curr = h
        low_curr = l
        close_curr = c
        candle_range = h - l

        close_15m = data_15m.close
        timestamps_15m = data_15m.ts

        interval_5m_sec = 5 * 60
        expected_5m_open = (reference_time // interval_5m_sec) * interval_5m_sec - interval_5m_sec

        ts_5m_arr = normalize_timestamp_array(data_5m.ts)

        matches_5m = np.flatnonzero(np.abs(ts_5m_arr - expected_5m_open) <= 30)

        if matches_5m.size > 0:
            i5 = int(matches_5m[-1])
            actual_5m_ts = int(ts_5m_arr[i5])
        else:
            ts_15m_val = int(normalize_timestamp(int(data_15m.ts[i15])))
            window_mask = (ts_5m_arr >= ts_15m_val) & (ts_5m_arr < ts_15m_val + 900)
            if np.any(window_mask):
                fallback_idx = int(np.flatnonzero(window_mask)[-1])
                i5 = fallback_idx
                actual_5m_ts = int(ts_5m_arr[fallback_idx])
                if logger_pair.isEnabledFor(logging.DEBUG):
                    logger_pair.debug(
                        f"[{pair_name}] 5m fallback: using {format_ist_time(actual_5m_ts)} "
                        f"(expected {format_ist_time(expected_5m_open)} not available)"
                    )    
            else:
                logger_pair.warning(
                    f"[{pair_name}] 5m candle not found at {format_ist_time(expected_5m_open)} "
                    f"and no fallback in 15m window. Range: {format_ist_time(int(ts_5m_arr[0]))} "
                    f"to {format_ist_time(int(ts_5m_arr[-1]))}"
                )
                return None

        time_since_5m_closed = reference_time - (actual_5m_ts + interval_5m_sec)
        if time_since_5m_closed < cfg.CANDLE_MIN_AGE_BUFFER:
            logger_pair.warning(
                f"[{pair_name}] 5m candle at {format_ist_time(actual_5m_ts)} not stable yet "
                f"(closed {time_since_5m_closed}s ago, need {cfg.CANDLE_MIN_AGE_BUFFER}s). Skipping."
            )
            return None

        ts_15m_val = int(normalize_timestamp(int(data_15m.ts[i15])))
        if actual_5m_ts < ts_15m_val or actual_5m_ts >= ts_15m_val + 900:
            logger_pair.error(
                f"[{pair_name}] 5m/15m misalignment: 5m={format_ist_time(actual_5m_ts)} "
                f"outside 15m window {format_ist_time(ts_15m_val)}-{format_ist_time(ts_15m_val + 900)}"
            )
            return None

        expected_last_5m = ts_15m_val + 600
        if actual_5m_ts != expected_last_5m:
            if logger_pair.isEnabledFor(logging.DEBUG):
                logger_pair.debug(
                    f"[{pair_name}] Using non-last 5m candle: got {format_ist_time(actual_5m_ts)}, "
                    f"expected {format_ist_time(expected_last_5m)}"
                )

        if i5 < Constants.MIN_ALIGNED_5M_CANDLES:
            return None

        if logger_pair.isEnabledFor(logging.DEBUG):
            logger_pair.debug(
                f"[{pair_name}] 5m candle selected | "
                f"Open={format_ist_time(actual_5m_ts)} | i5={i5} | "
                f"Close={data_5m.close[i5]:.2f}"
            )

        # ══════════════════════════════════════════════════════
        # PHASE 1 — Gate indicators only (cheap)
        # ═══════════════════════════════════════════════════════
        gate_indicators = await asyncio.to_thread(
            calculate_gate_indicators_numpy, data_15m.as_dict(), data_5m.as_dict(), data_daily, reference_time
        )
        if gate_indicators is None:
            logger_pair.error(f"Skipping {pair_name}: gate indicators failed")
            return None

        # ── Extract gate values ──
        rma50_15 = gate_indicators["rma50_15"]
        rma200_5 = gate_indicators["rma200_5"]
        ichimoku_cloud_upper = gate_indicators["ichimoku_cloud_upper"]
        ichimoku_cloud_lower = gate_indicators["ichimoku_cloud_lower"]
        ichimoku_future_green = gate_indicators["ichimoku_future_green"]
        ichimoku_future_red = gate_indicators["ichimoku_future_red"]
        ichimoku_conversion_line = gate_indicators["ichimoku_conversion_line"]
        ichimoku_base_line = gate_indicators["ichimoku_base_line"]
        adx_arr = gate_indicators["adx"]
        atr_short_arr = gate_indicators["atr_short"]
        atr_long_arr = gate_indicators["atr_long"]
        volume_ema_arr = gate_indicators["volume_ema"]
        ppo_gate_arr = gate_indicators["ppo_gate"]
        ppo_gate_signal_arr = gate_indicators["ppo_gate_signal"]
        rsi_guard_smooth_arr = gate_indicators["rsi_guard_smooth"]
        rsi_guard_ema_arr = gate_indicators["rsi_guard_ema"]
        rma_cloud_fast_arr = gate_indicators["rma_cloud_fast_15"]
        cpr_ok = gate_indicators.get('cpr_ok', not cfg.ENABLE_CPR)
        nr_cpr = gate_indicators.get('nr_cpr', float('nan'))
        prev_day_close = gate_indicators.get('prev_day_close', float('nan'))

        future_green = ichimoku_future_green[i15]
        future_red = ichimoku_future_red[i15]

        cloud_upper_val = ichimoku_cloud_upper[i15]
        cloud_lower_val = ichimoku_cloud_lower[i15]
        cloud_upper_prev = ichimoku_cloud_upper[i15 - 1]
        cloud_lower_prev = ichimoku_cloud_lower[i15 - 1]

        ichimoku_cloud_ready = not (
            np.isnan(cloud_upper_val) or np.isnan(cloud_lower_val)
            or np.isnan(cloud_upper_prev) or np.isnan(cloud_lower_prev)
        )
        if ichimoku_cloud_ready:
            above_cloud = close_curr > cloud_upper_val
            below_cloud = close_curr < cloud_lower_val
            cloud_up = bool(future_green and above_cloud)
            cloud_down = bool(future_red and below_cloud)
        else:
            logger_pair.debug(
                f"[{pair_name}] Ichimoku cloud NaN at i15={i15} (warmup/gap). "
                f"Ichimoku cloud gate abstains (None) — not counted in cloud-group vote."
            )
            above_cloud = None
            below_cloud = None
            cloud_up = None
            cloud_down = None

        tk_conversion_curr = ichimoku_conversion_line[i15]
        tk_conversion_prev = ichimoku_conversion_line[i15 - 1]
        tk_base_curr = ichimoku_base_line[i15]
        tk_base_prev = ichimoku_base_line[i15 - 1]
        tk_guard_valid = not (np.isnan(tk_conversion_curr) or np.isnan(tk_base_curr))

        if cfg.ICHIMOKU_TK_GUARD_ENABLED:
            if tk_guard_valid:
                tk_guard_ok_buy = bool((tk_conversion_curr >= tk_base_curr) and (close_curr > tk_base_curr))
                tk_guard_ok_sell = bool((tk_conversion_curr <= tk_base_curr) and (close_curr < tk_base_curr))

            else:
                logger_pair.debug(
                    f"[{pair_name}] TK lines not ready at i15={i15}. "
                    f"TK guard abstains (None) this run — not counted in majority vote."
                )
                tk_guard_ok_buy = None
                tk_guard_ok_sell = None
        else:
            tk_guard_ok_buy = None
            tk_guard_ok_sell = None

        close_prev = close_15m[i15 - 1]

        close_prev_invalid = False
        if np.isnan(close_prev) or np.isinf(close_prev) or close_prev <= 0:
            logger_pair.warning(
                f"[{pair_name}] Previous candle close invalid ({close_prev}). "
                f"Skipping all cross-based alerts this run."
            )
            close_prev_invalid = True

        if close_prev_invalid:
            logger_pair.warning(
                f"[{pair_name}] close_prev invalid — skipping all cross alerts"
            )
            await _blanket_reset_pair(sdb, pair_name, logger_pair)
            return pair_name, {
                "state": "INVALID_PREV_CLOSE",
                "ts": int(time.time()),
                "summary": {
                    "alerts": 0,
                    "future_cloud": "neutral",
                    "hist_rma": 0.0,
                    "suppression": "close_prev was NaN/Inf/≤0"
                }
            }  
        close_5m_val = data_5m.close[i5]
        rma50_15_val = rma50_15[i15]
        rma200_5_val = rma200_5[i5]

        base_buy_trend = bool((rma50_15_val < close_curr) and (rma200_5_val < close_5m_val))
        base_sell_trend = bool((rma50_15_val > close_curr) and (rma200_5_val > close_5m_val))

        if cfg.ICHIMOKU_CLOUD_ENABLED:
            ichimoku_gate_ok_buy = cloud_up
            ichimoku_gate_ok_sell = cloud_down
        else:
            ichimoku_gate_ok_buy = None
            ichimoku_gate_ok_sell = None

        adx_val = adx_arr[i15] if not np.isnan(adx_arr[i15]) else 0.0
        adx_adaptive_threshold = get_adaptive_adx_threshold_smoothed(adx_arr, i15, cfg)
        adx_raw_check = adx_val >= adx_adaptive_threshold
        adx_ok = adx_raw_check if cfg.ENABLE_ADX_FILTER else True
        adx_bypass_ok = adx_raw_check

        atr_short_val = atr_short_arr[i15]
        atr_long_val = atr_long_arr[i15]

        atr_ratio_valid = (
            not np.isnan(atr_short_val) and not np.isnan(atr_long_val) and atr_long_val > 1e-9
        )
        atr_ratio = (atr_short_val / atr_long_val) if atr_ratio_valid else float('nan')
        shared_smoothed_pctl = _get_smoothed_pctl(atr_long_arr, i15, cfg)
        adaptive_threshold = get_adaptive_rvol_threshold(atr_long_arr, i15, cfg, pctl=shared_smoothed_pctl)
        ppo_adaptive_threshold = get_adaptive_ppo_threshold(atr_long_arr, i15, cfg, pctl=shared_smoothed_pctl)
        rsi_adaptive_buy, rsi_adaptive_sell = get_adaptive_rsi_thresholds(atr_long_arr, i15, cfg, pctl=shared_smoothed_pctl)
        cpr_adaptive_min_pct_move = get_adaptive_cpr_threshold(atr_long_arr, i15, cfg, pctl=shared_smoothed_pctl)

        volume_curr = data_15m.volume[i15]
        volume_ema_curr = volume_ema_arr[i15]
        if not np.isnan(volume_curr) and not np.isnan(volume_ema_curr) and volume_ema_curr > 1e-9:
            volume_above_ema_ok = volume_curr > volume_ema_curr
        else:
            volume_above_ema_ok = False

        rvol_bypass_ok = atr_ratio_valid and (atr_ratio >= cfg.RVOL_THRESHOLD)

        adaptive_rvol_check = (
            atr_ratio_valid
            and adaptive_threshold is not None
            and atr_ratio >= adaptive_threshold
        )
        adx_pass = adx_raw_check if cfg.ENABLE_ADX_FILTER else False
        rvol_static_pass = rvol_bypass_ok if cfg.ENABLE_RVOL_ALERT else False
        rvol_adaptive_pass = adaptive_rvol_check  # False if ATR_ADAPTIVE_ENABLED=False
        
        adx_prev = adx_arr[i15 - 1] if i15 >= 1 else adx_val
        adx_rising = (
            not np.isnan(adx_val) and not np.isnan(adx_prev)
            and adx_prev > 0 and adx_val > adx_prev
        )

        rvol_vote_ok = rvol_static_pass or rvol_adaptive_pass

        body_conviction_ok = (
            candle_range > 1e-9
            and (abs(close_curr - open_curr) / candle_range) >= cfg.CPR_MOMENTUM_BODY_RATIO_MIN
        )
        momentum_conditions = [
            adx_bypass_ok,         # 1. ADX level >= threshold
            adx_rising,            # 2. ADX rising vs prior bar
            rvol_vote_ok,          # 3. RVOL (static or adaptive, single vote — not both)
            volume_above_ema_ok,   # 4. Volume > EMA(volume)
            body_conviction_ok,    # 5. Candle body conviction (|close-open|/range)
        ]
        momentum_count = sum(momentum_conditions)

        any_vol_feature_enabled = cfg.ENABLE_ADX_FILTER or cfg.ENABLE_RVOL_ALERT or cfg.ATR_ADAPTIVE_ENABLED
        volatility_filter_ok = (not any_vol_feature_enabled) or (momentum_count >= 3)
        rvol_ok = volatility_filter_ok

        adx_pctl = None
        adx_strength_ok = None
        if cfg.ENABLE_ADX_STRENGTH_VOTE:
            adx_pctl = get_adx_percentile(adx_arr, i15, cfg)
            adx_strength_ok = (adx_pctl is not None) and (adx_pctl * 100.0 >= cfg.ADX_STRENGTH_PCTL)

        atr_pctl = None
        atr_pctl_ok = None
        if cfg.ENABLE_ATR_PCTL_VOTE:
            atr_pctl = get_atr_percentile(atr_long_arr, i15, cfg)
            atr_pctl_ok = (atr_pctl is not None) and (atr_pctl >= cfg.ATR_PCTL_VOTE_MIN)

        volume_pctl = None
        volume_pctl_ok = None
        if cfg.ENABLE_VOLUME_PCTL_VOTE:
            volume_pctl = get_volume_percentile(data_15m.volume, i15, cfg)
            volume_pctl_ok = (volume_pctl is not None) and (volume_pctl >= cfg.VOLUME_PCTL_VOTE_MIN)

        if not np.isnan(prev_day_close) and prev_day_close > 0:
            pct_move_from_prev_close = abs(close_curr - prev_day_close) / prev_day_close * 100.0
            move_from_prev_close_ok = pct_move_from_prev_close >= cpr_adaptive_min_pct_move
        else:
            pct_move_from_prev_close = float('nan')
            move_from_prev_close_ok = False

        if cfg.ENABLE_CPR:
            if cpr_ok:  # Narrow CPR: momentum now enforced globally via volatility_filter_ok
                effective_cpr_ok = True
            else:       # Wide CPR: same, plus mandatory min % move from prior close
                effective_cpr_ok = move_from_prev_close_ok
        else:
            effective_cpr_ok = True

        if cfg.DEBUG_MODE and cfg.ENABLE_CPR:
            logger_pair.debug(
                f"[{pair_name}] CPR {'narrow' if cpr_ok else 'WIDE'} | "
                f"effective={effective_cpr_ok} | momentum={momentum_count}/5 "
                f"(adx={adx_val:.1f}[{adx_bypass_ok},{adx_rising}], "
                f"rvol={rvol_vote_ok}[static={rvol_static_pass},adaptive={rvol_adaptive_pass}]"
                f"[thr={adaptive_threshold if adaptive_threshold is not None else float('nan'):.3f}], "
                f"vol_ema={volume_above_ema_ok}, body={body_conviction_ok}) | "
                f"move_from_prev_close={pct_move_from_prev_close:.2f}%[{move_from_prev_close_ok}] | "
                f"NR_CPR={nr_cpr:.4f}"
            )
        if cfg.DEBUG_MODE:
            ratio_str = f"{atr_ratio:.3f}" if atr_ratio_valid else "n/a"
            adaptive_str = f"{adaptive_threshold:.3f}" if adaptive_threshold is not None else "n/a"
            logger_pair.debug(
                f"[{pair_name}] Volatility filter | "
                f"ratio={ratio_str} | "
                f"static={cfg.RVOL_THRESHOLD:.3f}[{rvol_bypass_ok}] | "
                f"adaptive={adaptive_str}[{adaptive_rvol_check}] | "
                f"adx={adx_val:.1f}[{adx_pass}] | "
                f"market_filter={volatility_filter_ok}"
            )
        ppo_gate_curr = ppo_gate_arr[i15]
        ppo_gate_prev = ppo_gate_arr[i15 - 1] if i15 >= 1 else ppo_gate_arr[i15]
        ppo_gate_sig_curr = ppo_gate_signal_arr[i15]
        ppo_gate_sig_prev = ppo_gate_signal_arr[i15 - 1] if i15 >= 1 else ppo_gate_signal_arr[i15]
        rsi_guard_smooth_curr = rsi_guard_smooth_arr[i15]
        rsi_guard_ema_curr = rsi_guard_ema_arr[i15]
        rma_cloud_fast_curr = rma_cloud_fast_arr[i15]

        if cfg.ENABLE_PPO_GATE:
            if not np.isnan(ppo_gate_curr) and not np.isnan(ppo_gate_sig_curr):
                ppo_gate_ok_buy = bool(ppo_gate_curr > ppo_gate_sig_curr)
                ppo_gate_ok_sell = bool(ppo_gate_curr < ppo_gate_sig_curr)
            else:
                ppo_gate_ok_buy = None
                ppo_gate_ok_sell = None
        else:
            ppo_gate_ok_buy = None
            ppo_gate_ok_sell = None

        if cfg.RSI_GUARD_ENABLED:
            if not np.isnan(rsi_guard_smooth_curr) and not np.isnan(rsi_guard_ema_curr):         
                rsi_guard_ok_buy = bool(rsi_guard_smooth_curr > rsi_guard_ema_curr)
                rsi_guard_ok_sell = bool(rsi_guard_smooth_curr < rsi_guard_ema_curr)
            else:
                rsi_guard_ok_buy = None
                rsi_guard_ok_sell = None
        else:
            rsi_guard_ok_buy = None
            rsi_guard_ok_sell = None

        if cfg.RMA_CLOUD_ENABLED:
            if not np.isnan(rma_cloud_fast_curr) and not np.isnan(rma50_15_val):
                rma_cloud_ok_buy = bool(rma_cloud_fast_curr > rma50_15_val)
                rma_cloud_ok_sell = bool(rma_cloud_fast_curr < rma50_15_val)
            else:
                rma_cloud_ok_buy = None
                rma_cloud_ok_sell = None
        else:
            rma_cloud_ok_buy = None
            rma_cloud_ok_sell = None

        ppo_gate_prev_val = ppo_gate_arr[i15 - 1] if i15 >= 1 else ppo_gate_curr
        if (cfg.ENABLE_PPO_GATE_MOMENTUM_VOTE
                and not np.isnan(ppo_gate_curr) and not np.isnan(ppo_gate_prev_val)):
            ppo_gate_momentum_ok_buy = bool(ppo_gate_curr > ppo_gate_prev_val)
            ppo_gate_momentum_ok_sell = bool(ppo_gate_curr < ppo_gate_prev_val)
        else:
            ppo_gate_momentum_ok_buy = None
            ppo_gate_momentum_ok_sell = None

        rsi_guard_smooth_prev_val = rsi_guard_smooth_arr[i15 - 1] if i15 >= 1 else rsi_guard_smooth_curr
        if (cfg.ENABLE_RSI_GUARD_MOMENTUM_VOTE
                and not np.isnan(rsi_guard_smooth_curr) and not np.isnan(rsi_guard_smooth_prev_val)):
            rsi_guard_momentum_ok_buy = bool(rsi_guard_smooth_curr > rsi_guard_smooth_prev_val)
            rsi_guard_momentum_ok_sell = bool(rsi_guard_smooth_curr < rsi_guard_smooth_prev_val)
        else:
            rsi_guard_momentum_ok_buy = None
            rsi_guard_momentum_ok_sell = None

        rma_cloud_fast_prev_val = rma_cloud_fast_arr[i15 - 1] if i15 >= 1 else rma_cloud_fast_curr
        if (cfg.ENABLE_RMA_CLOUD_MOMENTUM_VOTE
                and not np.isnan(rma_cloud_fast_curr) and not np.isnan(rma_cloud_fast_prev_val)):
            rma_cloud_momentum_ok_buy = bool(rma_cloud_fast_curr > rma_cloud_fast_prev_val)
            rma_cloud_momentum_ok_sell = bool(rma_cloud_fast_curr < rma_cloud_fast_prev_val)
        else:
            rma_cloud_momentum_ok_buy = None
            rma_cloud_momentum_ok_sell = None

        vwap_gate_arr = gate_indicators["vwap_gate"]
        vwap_gate_curr = vwap_gate_arr[i15]
        vwap_gate_prev_val = vwap_gate_arr[i15 - 1] if i15 >= 1 else vwap_gate_curr
        same_utc_day = (
            i15 >= 1
            and normalize_timestamp(int(data_15m.ts[i15])) // 86400
                == normalize_timestamp(int(data_15m.ts[i15 - 1])) // 86400
        )
        if (cfg.ENABLE_VWAP_MOMENTUM_VOTE and same_utc_day
                and not np.isnan(vwap_gate_curr) and not np.isnan(vwap_gate_prev_val)):
            vwap_momentum_ok_buy = bool(vwap_gate_curr > vwap_gate_prev_val)
            vwap_momentum_ok_sell = bool(vwap_gate_curr < vwap_gate_prev_val)
        else:
            vwap_momentum_ok_buy = None
            vwap_momentum_ok_sell = None

        cloud_group_enabled = cfg.RMA_CLOUD_ENABLED or cfg.ICHIMOKU_CLOUD_ENABLED
        oscillator_group_enabled = cfg.ENABLE_PPO_GATE or cfg.RSI_GUARD_ENABLED or cfg.ICHIMOKU_TK_GUARD_ENABLED
  
        cloud_votes_buy = []
        if cfg.ICHIMOKU_CLOUD_ENABLED:
            cloud_votes_buy.append(ichimoku_gate_ok_buy)
        if cfg.RMA_CLOUD_ENABLED:
            cloud_votes_buy.append(rma_cloud_ok_buy)

        if cloud_votes_buy:
            cloud_group_ok_buy = all(v is True for v in cloud_votes_buy)
            if not cloud_group_ok_buy and cfg.DEBUG_MODE:
                logger_pair.debug(
                    f"[{pair_name}] Cloud group buy blocked: need ALL true "
                    f"(Ichimoku={ichimoku_gate_ok_buy}, RMA={rma_cloud_ok_buy})"
                )
        elif not cloud_group_enabled:
            cloud_group_ok_buy = True
        else:
            cloud_group_ok_buy = False

        cloud_votes_sell = []
        if cfg.ICHIMOKU_CLOUD_ENABLED:
            cloud_votes_sell.append(ichimoku_gate_ok_sell)
        if cfg.RMA_CLOUD_ENABLED:
            cloud_votes_sell.append(rma_cloud_ok_sell)

        if cloud_votes_sell:
            cloud_group_ok_sell = all(v is True for v in cloud_votes_sell)
            if not cloud_group_ok_sell and cfg.DEBUG_MODE:
                logger_pair.debug(
                    f"[{pair_name}] Cloud group sell blocked: need ALL true "
                    f"(Ichimoku={ichimoku_gate_ok_sell}, RMA={rma_cloud_ok_sell})"
                )
        elif not cloud_group_enabled:
            cloud_group_ok_sell = True
        else:
            cloud_group_ok_sell = False

        confirmation_buy = cloud_group_ok_buy
        confirmation_sell = cloud_group_ok_sell

        active_osc_buy = [g for g in (ppo_gate_ok_buy, rsi_guard_ok_buy, tk_guard_ok_buy) if g is not None]
        if active_osc_buy:
            oscillator_group_ok_buy = sum(active_osc_buy) >= min(Constants.OSCILLATOR_GROUP_MIN_VOTES, len(active_osc_buy))
        elif oscillator_group_enabled:
            logger_pair.debug(
                f"[{pair_name}] Oscillator group: all gates abstained (warmup/gap) — buy denied."
            )
            oscillator_group_ok_buy = False
        else:
            oscillator_group_ok_buy = True

        active_osc_sell = [g for g in (ppo_gate_ok_sell, rsi_guard_ok_sell, tk_guard_ok_sell) if g is not None]
        if active_osc_sell:
            oscillator_group_ok_sell = sum(active_osc_sell) >= min(Constants.OSCILLATOR_GROUP_MIN_VOTES, len(active_osc_sell))
        elif oscillator_group_enabled:
            logger_pair.debug(
                f"[{pair_name}] Oscillator group: all gates abstained (warmup/gap) — sell denied."
            )
            oscillator_group_ok_sell = False
        else:
            oscillator_group_ok_sell = True

        trend_gate_ok_buy = cloud_group_ok_buy and oscillator_group_ok_buy
        trend_gate_ok_sell = cloud_group_ok_sell and oscillator_group_ok_sell

        buy_trend_common = (
            base_buy_trend
            and volatility_filter_ok and effective_cpr_ok
            and trend_gate_ok_buy
        )
        sell_trend_common = (
            base_sell_trend
            and volatility_filter_ok and effective_cpr_ok
            and trend_gate_ok_sell
        )
        buy_common  = buy_trend_common and is_valid_for_buy
        sell_common = sell_trend_common and is_valid_for_sell
        buy_trend_common_relaxed = (
            base_buy_trend
            and effective_cpr_ok
            and trend_gate_ok_buy
        )
        sell_trend_common_relaxed = (
            base_sell_trend
            and effective_cpr_ok
            and trend_gate_ok_sell
        )
        reversal_candidate = (
            (cfg.ENABLE_STRONG_REVERSAL_ALERT or cfg.ENABLE_OB_GATE or cfg.ENABLE_CHOCH_ALERT or cfg.ENABLE_TLR_ALERT)
            and (buy_trend_common_relaxed or sell_trend_common_relaxed)
        )
        if not buy_common and not sell_common and not reversal_candidate:
            await _blanket_reset_pair(sdb, pair_name, logger_pair)
            reasons = []
            if not base_buy_trend and not base_sell_trend:
                reasons.append("base_trend=False")
            if not confirmation_buy and not confirmation_sell:
                reasons.append("cloud_align=False")
            if not volatility_filter_ok:
                reasons.append(
                    f"market_filter=False (adx={adx_val:.1f}, "
                    f"rvol_static={rvol_bypass_ok}, rvol_adaptive={adaptive_rvol_check})"
                )
            if not effective_cpr_ok:
                reasons.append("cpr=False")
            if not trend_gate_ok_buy and not trend_gate_ok_sell:
                reasons.append("trend_gate=False")
            logger_pair.debug(
                f"😒 {pair_name} | Gate blocked | "
                f"Suppression: {', '.join(reasons)}"
            )
            return pair_name, {
                "state": "NO_SIGNAL",
                "ts": int(time.time()),
                "summary": {
                    "alerts": 0,
                    "future_cloud": "green" if cloud_up else "red" if cloud_down else "neutral",
                    "hist_rma": 0.0,
                    "suppression": f"Gate blocked: {', '.join(reasons)}"
                }
            }
        oi_funding_ok_buy = oi_funding_ok_sell = None
        oi_funding_reason = None
        if cfg.ENABLE_OI_FUNDING_FILTER and cfg.ENABLE_CONFLUENCE_GATE and pair_oi is not None:
            buy_reason = _oi_funding_gate_reason(
                pair_oi.get("oi_now"), pair_oi.get("oi_history", []),
                pair_oi.get("funding"), pair_oi.get("funding_history", []), is_buy=True,
                oi_usd_now=pair_oi.get("oi_usd_now"),
                price_now=pair_oi.get("price_now"), price_history=pair_oi.get("price_history", []),
            )
            sell_reason = _oi_funding_gate_reason(
                pair_oi.get("oi_now"), pair_oi.get("oi_history", []),
                pair_oi.get("funding"), pair_oi.get("funding_history", []), is_buy=False,
                oi_usd_now=pair_oi.get("oi_usd_now"),
                price_now=pair_oi.get("price_now"), price_history=pair_oi.get("price_history", []),
            )
            oi_funding_ok_buy = buy_reason is None
            oi_funding_ok_sell = sell_reason is None
            oi_funding_reason = buy_reason or sell_reason

        ob_gate_ok_buy = ob_gate_ok_sell = None
        ob_gate_reason = None
        if cfg.ENABLE_OB_GATE:
            ob_gate_ok_buy, ob_gate_ok_sell, ob_gate_reason = await asyncio.to_thread(
                _order_block_gate_reason,
                data_15m.open, data_15m.high, data_15m.low, data_15m.close,
                atr_short_arr, i15, cfg,
            )

            if ob_gate_reason:
                logger_pair.debug(f"[{pair_name}] OB gate: {ob_gate_reason}")

        choch_gate_ok_buy = choch_gate_ok_sell = None
        choch_reason = None
        choch_fvg_buy = choch_fvg_sell = False
        choch_poi_tap_buy = choch_poi_tap_sell = False
        if cfg.ENABLE_CHOCH_ALERT:
            (choch_gate_ok_buy, choch_gate_ok_sell, choch_reason, choch_fvg_buy, choch_fvg_sell,
             choch_poi_tap_buy, choch_poi_tap_sell) = await asyncio.to_thread(
                _choch_gate_reason,
                data_15m.open, data_15m.high, data_15m.low, data_15m.close, data_15m.ts,
                atr_short_arr, i15, cfg,
            )
            if choch_reason:
                logger_pair.debug(f"[{pair_name}] CHoCH gate: {choch_reason}")

        tlr_touch_gate_ok_buy = tlr_touch_gate_ok_sell = False
        tlr_touch_reason = None
        tlr_trendline_buy = tlr_trendline_sell = None
        tlr_prior_touch_idx_buy = tlr_prior_touch_idx_sell = None
        if cfg.ENABLE_TLR_ALERT:
            prior_state_buy = await sdb.get_tlr_touch_state(pair_name, True)
            prior_state_sell = await sdb.get_tlr_touch_state(pair_name, False)
            tlr_prior_touch_idx_buy = (prior_state_buy or {}).get("last_touch_idx")
            tlr_prior_touch_idx_sell = (prior_state_sell or {}).get("last_touch_idx")

            touch_result_buy = await asyncio.to_thread(
                _tlr_evaluate_touch,
                data_15m.open, data_15m.high, data_15m.low, data_15m.close,
                atr_short_arr, i15, cfg, True, prior_state_buy,
            )
            touch_result_sell = await asyncio.to_thread(
                _tlr_evaluate_touch,
                data_15m.open, data_15m.high, data_15m.low, data_15m.close,
                atr_short_arr, i15, cfg, False, prior_state_sell,
            )
            await sdb.save_tlr_touch_state(pair_name, True, touch_result_buy["state"])
            await sdb.save_tlr_touch_state(pair_name, False, touch_result_sell["state"])

            tlr_touch_gate_ok_buy = touch_result_buy["gate_ok"]
            tlr_touch_gate_ok_sell = touch_result_sell["gate_ok"]
            tlr_trendline_buy = touch_result_buy["tl"]
            tlr_trendline_sell = touch_result_sell["tl"]
            tlr_touch_reason = (
                touch_result_buy["reason"] if tlr_touch_gate_ok_buy
                else touch_result_sell["reason"] if tlr_touch_gate_ok_sell
                else None
            )
            if tlr_touch_gate_ok_buy or tlr_touch_gate_ok_sell:
                logger_pair.debug(
                    f"[{pair_name}] TLR touch gate: buy={tlr_touch_gate_ok_buy} "
                    f"sell={tlr_touch_gate_ok_sell} | {tlr_touch_reason}"
                )

        return GateResult(
            pair_name=pair_name, i15=i15, i5=i5, ts_curr=ts_curr, reference_time=reference_time,
            candle_info=candle_info, o=o, h=h, l=l, c=c,
            open_curr=open_curr, high_curr=high_curr, low_curr=low_curr, close_curr=close_curr,
            close_prev=close_prev, close_5m_val=close_5m_val,
            is_green=is_green, is_red=is_red,
            is_valid_for_buy=is_valid_for_buy, is_valid_for_sell=is_valid_for_sell,
            candle_index=i15, min_wick_ratio=Constants.MIN_WICK_RATIO,
            buy_wick_ratio=buy_wick_ratio, sell_wick_ratio=sell_wick_ratio,
            gate_indicators=gate_indicators,
            base_buy_trend=base_buy_trend, base_sell_trend=base_sell_trend,
            rma50_15_val=rma50_15_val, rma200_5_val=rma200_5_val,
            cloud_up=cloud_up, cloud_down=cloud_down,
            cloud_upper_val=cloud_upper_val, cloud_lower_val=cloud_lower_val,
            cloud_upper_prev=cloud_upper_prev, cloud_lower_prev=cloud_lower_prev,
            ichimoku_gate_ok_buy=ichimoku_gate_ok_buy, ichimoku_gate_ok_sell=ichimoku_gate_ok_sell,
            confirmation_buy=confirmation_buy, confirmation_sell=confirmation_sell,
            cloud_group_ok_buy=cloud_group_ok_buy, cloud_group_ok_sell=cloud_group_ok_sell,
            tk_conversion_curr=tk_conversion_curr, tk_conversion_prev=tk_conversion_prev,
            tk_base_curr=tk_base_curr, tk_base_prev=tk_base_prev,
            tk_guard_ok_buy=tk_guard_ok_buy, tk_guard_ok_sell=tk_guard_ok_sell,
            oscillator_group_ok_buy=oscillator_group_ok_buy, oscillator_group_ok_sell=oscillator_group_ok_sell,
            ppo_gate_arr=ppo_gate_arr, ppo_gate_signal_arr=ppo_gate_signal_arr,
            ppo_gate_curr=ppo_gate_curr, ppo_gate_prev=ppo_gate_prev,
            ppo_gate_sig_curr=ppo_gate_sig_curr, ppo_gate_sig_prev=ppo_gate_sig_prev,
            ppo_gate_ok_buy=ppo_gate_ok_buy, ppo_gate_ok_sell=ppo_gate_ok_sell,
            rsi_guard_smooth_curr=rsi_guard_smooth_curr, rsi_guard_ema_curr=rsi_guard_ema_curr,
            rsi_guard_ok_buy=rsi_guard_ok_buy, rsi_guard_ok_sell=rsi_guard_ok_sell,
            rma_cloud_fast_curr=rma_cloud_fast_curr,
            rma_cloud_ok_buy=rma_cloud_ok_buy, rma_cloud_ok_sell=rma_cloud_ok_sell,
            trend_gate_ok_buy=trend_gate_ok_buy, trend_gate_ok_sell=trend_gate_ok_sell,
            adx_val=adx_val, adx_adaptive_threshold=adx_adaptive_threshold, adx_ok=adx_ok,
            rvol_bypass_ok=rvol_bypass_ok, rvol_ok=rvol_ok, adaptive_rvol_check=adaptive_rvol_check,
            momentum_count=momentum_count, volatility_filter_ok=volatility_filter_ok,
            adx_pctl=adx_pctl, adx_strength_ok=adx_strength_ok,
            atr_pctl=atr_pctl, atr_pctl_ok=atr_pctl_ok,
            volume_pctl=volume_pctl, volume_pctl_ok=volume_pctl_ok,
            cpr_ok=cpr_ok, nr_cpr=nr_cpr, effective_cpr_ok=effective_cpr_ok,
            cpr_adaptive_min_pct_move=cpr_adaptive_min_pct_move, move_from_prev_close_ok=move_from_prev_close_ok,
            ppo_adaptive_threshold=ppo_adaptive_threshold,
            rsi_adaptive_buy=rsi_adaptive_buy, rsi_adaptive_sell=rsi_adaptive_sell,
            buy_common=buy_common, sell_common=sell_common,
            buy_trend_common=buy_trend_common, sell_trend_common=sell_trend_common,
            buy_trend_common_relaxed=buy_trend_common_relaxed, sell_trend_common_relaxed=sell_trend_common_relaxed,
            data_15m=data_15m, close_prev_invalid=close_prev_invalid,
            oi_funding_ok_buy=oi_funding_ok_buy, oi_funding_ok_sell=oi_funding_ok_sell,
            oi_funding_reason=oi_funding_reason,
            ob_gate_ok_buy=ob_gate_ok_buy, ob_gate_ok_sell=ob_gate_ok_sell,
            ob_gate_reason=ob_gate_reason,
            choch_gate_ok_buy=choch_gate_ok_buy, choch_gate_ok_sell=choch_gate_ok_sell,
            choch_reason=choch_reason,
            choch_fvg_buy=choch_fvg_buy, choch_fvg_sell=choch_fvg_sell,
            choch_poi_tap_buy=choch_poi_tap_buy, choch_poi_tap_sell=choch_poi_tap_sell,
            atr_short_arr=atr_short_arr,
            tlr_touch_gate_ok_buy=tlr_touch_gate_ok_buy, tlr_touch_gate_ok_sell=tlr_touch_gate_ok_sell,
            tlr_touch_reason=tlr_touch_reason,
            tlr_trendline_buy=tlr_trendline_buy, tlr_trendline_sell=tlr_trendline_sell,
            tlr_prior_touch_idx_buy=tlr_prior_touch_idx_buy, tlr_prior_touch_idx_sell=tlr_prior_touch_idx_sell,
            direction_is_buy=bool(buy_common or buy_trend_common),
            ppo_gate_momentum_ok_buy=ppo_gate_momentum_ok_buy,
            ppo_gate_momentum_ok_sell=ppo_gate_momentum_ok_sell,
            rsi_guard_momentum_ok_buy=rsi_guard_momentum_ok_buy,
            rsi_guard_momentum_ok_sell=rsi_guard_momentum_ok_sell,
            rma_cloud_momentum_ok_buy=rma_cloud_momentum_ok_buy,
            rma_cloud_momentum_ok_sell=rma_cloud_momentum_ok_sell,
            vwap_momentum_ok_buy=vwap_momentum_ok_buy,
            vwap_momentum_ok_sell=vwap_momentum_ok_sell,
        )
    except asyncio.CancelledError:
        logger_pair.warning(f"Evaluation cancelled for {pair_name}")
        raise
    except RuntimeError as e:
        logger_pair.critical(f"🚨 INVARIANT VIOLATION in {pair_name}: {e}")
        return pair_name, {
            "state": "INVARIANT_VIOLATION",
            "ts": int(time.time()),
            "summary": {
                "alerts": 0,
                "future_cloud": "neutral",
                "hist_rma": 0.0,
                "error": str(e)
            }
        }
    except Exception as e:
        logger_pair.exception(
            f"❌ Error in _eval_gate for {pair_name}: {e} | Correlation: {correlation_id}"
        )
        return None







