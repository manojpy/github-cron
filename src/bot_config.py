"""bot_config — the BotConfig settings model, load_config()/cfg singleton, logger setup and runtime validation. Static building blocks live in config_base and are re-exported here.

Split out of the original module without logic changes; the original
module remains as a facade re-exporting every name."""
from __future__ import annotations
import os
import sys
import re
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Literal
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict, PrivateAttr
from config_base import (
    CONFLUENCE_WEIGHTS,
    Constants,
    SafeFormatter,
    TraceContextFilter,
    _IST_TZ,
    json_loads,
)

class BotConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    _validation_warnings: List[str] = PrivateAttr(default_factory=list)
    TELEGRAM_BOT_TOKEN: str = Field(..., min_length=1)
    TELEGRAM_CHAT_ID: str = Field(..., min_length=1)
    REDIS_URL: str = Field(..., min_length=1)
    DELTA_API_BASE: str = Field(..., min_length=1)
    DEBUG_MODE: bool = Field(default=False)
    SEND_TEST_MESSAGE: bool = Field(default=False, description="Send test message on startup")
    BOT_NAME: str = "Unified Alert Bot"
    PAIRS: List[str] = Field(default=["ETHUSD", "AVAXUSD", "XRPUSD", "BNBUSD", "LTCUSD", "DOTUSD", "ADAUSD", "SUIUSD", "AAVEUSD", "SOLUSD", "PAXGUSD", "PIPPINUSD", "RIVERUSD", "BLESSUSD", "BASEDUSD","SKYAIUSD","HUSD","EDENUSD","XAUTUSD", "ZECUSD", "LABUSD", "BTCUSD", "LINKUSD", "ARBUSD", "KITEUSD", "VVVUSD", "BEATUSD", "BILLUSD", "BCHUSD", "WLDUSD" ], min_length=1) 
    PPO_FAST: int = Field(default=7, ge=1, le=50, description="PPO fast period")
    PPO_SLOW: int = Field(default=16, ge=2, le=100, description="PPO slow period")
    PPO_SIGNAL: int = Field(default=5, ge=1, le=25, description="PPO signal period")
    RMA_50_PERIOD: int = Field(default=50, ge=10, le=200, description="RMA 50 period")
    RMA_200_PERIOD: int = Field(default=200, ge=50, le=500, description="RMA 200 period")
    VOLUME_EMA_LENGTH: int = Field(default=20, ge=2, le=100, description="EMA period for 15m volume, used as wide-CPR confirmation (candle volume > EMA)")
    CPR_ADAPTIVE_CALM: float = Field(default=1.0, ge=0.1, le=20.0, description="Min % move from prev close for wide-CPR bypass, calm regime")
    CPR_ADAPTIVE_VOLATILE: float = Field(default=3.5, ge=0.1, le=20.0, description="Min % move from prev close for wide-CPR bypass, volatile regime") 
    ENABLE_PPO_ALERTS: bool = Field(default=True, description="Master switch for PPO cross alerts: signal-line cross, zero-line cross, adaptive-threshold cross (6 alert types)")
    ENABLE_PPOHIST_ALERT: bool = Field(default=True, description="Enable PPO Histogram Reversal alerts (ppohist_buy/sell). Independent of ENABLE_PPO_GATE, which drives the unrelated PPO trend-gate/confluence logic")   
    ENABLE_RSI_ALERTS: bool = Field(default=True, description="Master switch for RSI cross alerts: EMA5 cross, adaptive-threshold cross (4 alert types). Independent of RSI_GUARD_ENABLED, which is an unrelated trend gate")
    ENABLE_HIST_RMA: bool = Field(default=True, description="Enable RMA 10/30 histogram reversal alerts")
    HIST_RMA_FAST: int = Field(default=10, ge=2, le=100, description="Histogram RMA fast period")
    HIST_RMA_SLOW: int = Field(default=30, ge=5, le=200, description="Histogram RMA slow period")
    ENABLE_PPO_GATE: bool = Field(default=True, description="Enable PPO(32,84,20) as trend gate")
    PPO_GATE_FAST: int = Field(default=32, ge=1, le=100, description="Gate PPO fast period")
    PPO_GATE_SLOW: int = Field(default=84, ge=2, le=200, description="Gate PPO slow period")
    PPO_GATE_SIGNAL: int = Field(default=20, ge=1, le=50, description="Gate PPO signal period")
    PPOHIST_WARMUP_BUFFER_BARS: int = Field(default=56, ge=0, le=200)
    RSI_GUARD_ENABLED: bool = Field(default=False, description="Enable RSI(89) Kalman-smoothed vs EMA(50) as an alternate trend gate, OR'd with PPO gate")
    RSI_GUARD_RSI_LEN: int = Field(default=89, ge=2, le=200, description="RSI Guard RSI length")
    RSI_GUARD_KALMAN_LEN: int = Field(default=9, ge=1, le=50, description="RSI Guard Kalman smoothing length")
    RSI_GUARD_EMA_LEN: int = Field(default=50, ge=1, le=200, description="RSI Guard EMA length applied to Kalman-smoothed RSI")
    MOMENTUM_OSCILLATOR_ENABLED: bool = Field(default=True, description="Enable the AlgoAlpha Momentum Oscillator as the third oscillator-group vote (buy: vf > 0, sell: vf < 0). Replaces the removed Tenkan/Kijun guard")
    MOMENTUM_OSCILLATOR_LENGTH: int = Field(default=10, ge=2, le=200, description="Underlying Momentum Oscillator length (Pine: Underlying Momentum Oscillator Length)")
    SRSI_RSI_LEN: int = 14
    SRSI_KALMAN_LEN: int = 9
    SRSI_EMA_LEN: int = 5
    ATR_SHORT: int = Field(default=5, ge=1, le=50)
    ATR_LONG: int = Field(default=14, ge=2, le=200)
    MAX_PARALLEL_FETCH: int = Field(15, ge=1, le=20)
    HTTP_TIMEOUT: int = 15
    CANDLE_FETCH_RETRIES: int = 3
    CANDLE_FETCH_BACKOFF: float = 1.5
    RUN_TIMEOUT_SECONDS: int = 600
    FETCH_PHASE_TIMEOUT_SEC: int = 90
    TCP_CONN_LIMIT: int = 16
    TCP_CONN_LIMIT_PER_HOST: int = 16
    TELEGRAM_RETRIES: int = 3
    MEMORY_LIMIT_BYTES: int = 400_000_000
    STATE_EXPIRY_DAYS: int = 11
    LOG_LEVEL: str = "INFO"
    ENABLE_BATCHED_ALERTS: bool = Field(default=True, description="Combine all pair alerts into one run-level message instead of sending per-pair")
    ENABLE_ADX_FILTER: bool = Field(default=True)
    ENABLE_RVOL_ALERT: bool = Field(default=True)
    ENABLE_VWAP: bool = Field(default=True)
    ENABLE_PIVOT: bool = Field(default=True)
    ENABLE_CPR: bool = Field(default=False)
    CPR_THRESHOLD_PCT: float = Field(default=0.010, ge=0.001, le=0.10)
    CPR_MOMENTUM_BODY_RATIO_MIN: float = Field(default=0.50, ge=0.0, le=1.0, description="Min |close-open|/range for candle-body-conviction momentum vote")
    PIVOT_LOOKBACK_PERIOD: int = 15
    FAIL_ON_REDIS_DOWN: bool = False
    FAIL_ON_TELEGRAM_DOWN: bool = False
    TELEGRAM_RATE_LIMIT_PER_MINUTE: int = 20
    TELEGRAM_BURST_SIZE: int = 5
    REDIS_CONNECTION_RETRIES: int = 3
    REDIS_RETRY_DELAY: float = 2.0
    REDIS_RECOVERY_COOLDOWN_SEC: float = Field(default=30.0, ge=1.0, le=300.0, description="Minimum seconds between mid-run Redis recovery probes while degraded")
    REDIS_LOCK_EXPIRY: int = Field(default=600, ge=300, description="Redis lock TTL in seconds")
    ALERT_DEDUP_WINDOW_SEC: int = Field(default=120, ge=0, description="Dedup window for repeat alerts")
    ENABLE_ALERT_COALESCING: bool = Field(default=True) 
    ENABLE_SINGLE_ACTIVE_SHADOW: bool = Field(default=True, description="One open shadow trade per pair, like Recorded trades (no cooldown). While a shadow trade for a pair has not hit its stop/target or reached its horizon, later signals on that pair are not registered as shadow trades, so one market move is counted once. Within one candle the alert key with the highest confluence ratio represents the pair.")
    ENABLE_SINGLE_ACTIVE_TRADE: bool = Field(default=True, description="One recorded trade per pair at a time. While a recorded trade for a pair is still open (its stop or target not yet hit, and its outcome horizon not yet reached), further alerts for that pair are still sent but are labelled 'Ignored' and NOT recorded for the Brain. A same-candle batch of several alert keys records only one trade (the strongest edge), with the other keys kept in its context.")
    ENABLE_TRADE_CLOSE_NOTICE: bool = Field(default=True, description="When a Recorded or Shadowed trade touches its target or stop, send one 'Trade updates' Telegram message after the run's alerts, and start the pair's re-entry cooldown.")
    ENABLE_HORIZON_CLOSE_NOTICE: bool = Field(default=True, description="Also list in the 'Trade updates' message every Recorded or Shadowed trade that reached the end of its OUTCOME_LOOKAHEAD_CANDLES window without touching target or stop, with its directional % move at the close. Needs ENABLE_TRADE_CLOSE_NOTICE")
    TRADE_CLOSE_COOLDOWN_CANDLES: int = Field(default=3, ge=0, le=96, description="After a target is hit, the pair is not recorded for the hit candle plus this many following 15m candles. Shadow trades are still registered in that window and flagged cooldown_active. 0 disables the cooldown.")
    TRADE_CLOSE_COOLDOWN_ON_STOP: bool = Field(default=True, description="If True (default), a stop-loss or same-candle both-touched close also starts the re-entry cooldown, like a target hit.")
    SHADOW_TRACK_CONFLUENCE_BLOCKS: bool = Field(default=True, description="When the early confluence gate blocks a pair, still evaluate which alerts would have fired and shadow-track them (rejection_reason=confluence_gate). Needs ENABLE_BRAIN and BRAIN_SHADOW_MODE. Costs one extra alert evaluation for each blocked pair; set False to skip it.")
    ENABLE_FETCH_COALESCING: bool = Field(default=True, description="If True, identical candle requests (same symbol, resolution, limit and reference time) inside one run share a single API call. Confirmation re-fetches are never coalesced.")
    ENABLE_ADAPTIVE_DEDUP_WINDOWS: bool = Field(default=False, description="If True, the live dedup window per alert key comes from the Brain's historical inter-arrival analysis (metadata:adaptive_dedup_windows) instead of the fixed ALERT_DEDUP_WINDOW_SEC; the coalesce window may only be LENGTHENED above COALESCE_DEDUP_WINDOW_SEC. The Brain always computes and persists the windows so they can be inspected before enabling.")
    ADAPTIVE_DEDUP_MIN_GAPS: int = Field(default=30, ge=10, le=1000, description="Minimum same-pair inter-arrival gaps for an alert key before an adaptive window is derived; below this the fixed window is used.")
    ADAPTIVE_DEDUP_PERCENTILE: float = Field(default=10.0, ge=1.0, le=50.0, description="Percentile of the inter-arrival gap distribution used as the window (low = only suppress the fast repeats).")
    ADAPTIVE_DEDUP_MIN_SEC: int = Field(default=120, ge=0, le=3600, description="Hard lower bound for an adaptive window.")
    ADAPTIVE_DEDUP_MAX_SEC: int = Field(default=1800, ge=300, le=7200, description="Hard upper bound for an adaptive window.")
    COALESCE_DEDUP_WINDOW_SEC: int = Field(default=1800, ge=0)
    ENABLE_UNCONFIRMED_SIGNAL_NOTICE: bool = Field(default=True, description="If True, a signal whose candle/mark-price confirmation stays inconclusive after in-run retries is reported as a clearly labelled UNCONFIRMED notice (never as an actionable alert, no trade recorded) instead of being dropped silently.")
    ENABLE_TELEGRAM_DLQ: bool = Field(default=True, description="If True, an alert whose Telegram send failed is parked in a Redis dead-letter queue and re-sent on the next run instead of being lost.")
    ENABLE_RUN_DIAGNOSTIC_TELEGRAM: bool = Field(default=False, description="If True, send one compact operator diagnostic (dedup claims, DLQ, stale candles) to Telegram at the end of a run that had any dedup/DLQ/stale activity.")
    TELEGRAM_DLQ_MAX_AGE_SEC: int = Field(default=1800, ge=300, le=86400, description="A parked alert whose candle is older than this is dropped (stale) and its dedup claim released.")
    TELEGRAM_DLQ_MAX_ATTEMPTS: int = Field(default=3, ge=1, le=10, description="Replay attempts before a parked alert is abandoned and its dedup claim released.")
    TELEGRAM_DLQ_MAX_ITEMS_PER_RUN: int = Field(default=10, ge=1, le=100, description="Cap on parked alerts replayed per run.")
    ENABLE_CONFLUENCE_GATE: bool = Field(default=False) 
    CONFLUENCE_MIN_PCT: float = Field(default=60.0, ge=1.0, le=100.0, description="Min percentage of the achievable confluence total required to pass. Denominator = sum of weights of enabled, non-abstaining votes this cycle, so the threshold auto-scales when votes are enabled/disabled — no manual retuning needed")
    CONFLUENCE_MIN_ABS_SCORE: float = Field(default=18.0, ge=0.0, le=50.0, description="Absolute weighted-score floor required to pass the confluence gate, applied alongside CONFLUENCE_MIN_PCT. The stricter of the two (percentage-of-total vs this fixed floor) wins, so a low-vote-count cycle can't clear the gate on percentage alone")
    ENABLE_PAIR_THRESHOLDS: bool = Field(default=False, description="If true, a per-pair abs-score floor learned by the brain and stored in Redis (metadata:pair_confluence_thresholds) is used in place of CONFLUENCE_MIN_ABS_SCORE for that pair's confluence gate. Falls back to CONFLUENCE_MIN_ABS_SCORE when no pair-specific value is stored yet, or when this is off")
    BRAIN_PAIR_THRESHOLD_MIN_SAMPLE: int = Field(default=30, ge=1, description="Minimum resolved-outcome sample size a pair needs before the brain will compute/apply a per-pair confluence threshold for it")
    BRAIN_AUTO_APPLY_DYNAMIC_WEIGHTS: bool = Field(default=False, description="If True, the Brain persists optimized CONFLUENCE_WEIGHTS to Redis (metadata:dynamic_weights) after each report. The bot loads them at startup and uses them instead of the static config. Should only be enabled after shadow-validating the weights for at least one full analysis window.")
    BRAIN_USE_FILE_STORAGE: bool = Field(default=True)
    OUTCOME_DATA_DIR: str = Field(default="outcome-data", min_length=1)
    OB_MIN_OTHER_SCORE: float = Field(default=3.0, ge=0.0, le=50.0, description="Min weighted score from votes OTHER than base_trend and order_block required before the OB vote is allowed to count toward the confluence score and total. When the guard trips, the OB weight is removed from both score and total, so conf_pct is unaffected. base_trend is excluded because it's a precondition for evaluation, not independent confluence. Default 3.0 is set above oi_funding's 2.0 weight so oi_funding alone can't pair with OB to clear the gate")
    ENABLE_MACRO_CONTEXT_GATE: bool = Field(default=False, description="Computes a BTC-trend-alignment confluence multiplier (correlation + relative-strength based) for every alt-pair alert and records it alongside the outcome for later brain analysis. SHADOW MODE ONLY — the multiplier is never applied to the live confluence 'required' floor yet; see MACRO_CONTEXT_LIVE")
    MACRO_CONTEXT_LIVE: bool = Field(default=False, description="If True, the macro multiplier is APPLIED to the confluence 'required' floor. If False (default), it's computed and logged in shadow mode only.")
    MACRO_REFERENCE_PAIR: str = Field(default="BTCUSD", description="Which pair's GateResult is treated as macro/BTC trend context. Must be present in cfg.PAIRS and included in the current run's pairs_to_process for the gate to compute anything")
    ENABLE_CLUSTER_GATE: bool = Field(default=False, description="If true, before dispatch a same-run gate-only pre-pass counts how many pairs are directionally aligned (buy vs sell) across the whole universe. If more than CLUSTER_PCT_THRESHOLD of pairs lean the same way as an alert's direction, that alert's confluence score is reduced by CLUSTER_PENALTY_PCT before the confluence gate check — a systemic 'beta trap' veto (BTC pumps, 8 correlated alts fire as one trade). LIVE — roughly doubles gate-eval cost per run (a lightweight pre-pass over every pair)")
    CLUSTER_PCT_THRESHOLD: float = Field(default=0.25, ge=0.0, le=1.0, description="Fraction of the pair universe leaning the same direction (gate-level confirmation_buy/sell + adx_ok) before the cluster penalty kicks in")
    CLUSTER_PENALTY_PCT: float = Field(default=0.15, ge=0.0, le=0.9, description="Fractional haircut applied to an alert's confluence score when its direction matches a detected cluster above CLUSTER_PCT_THRESHOLD")
    ENABLE_RECENCY_WEIGHTING: bool = Field(default=False, description="If true, the Brain's per-alert auto-disable/recovery win-rate check (generate_recommendations) uses an exponentially recency-weighted win rate instead of a flat count — a trade from today counts ~e (2.72x) more than one exactly RECENCY_DECAY_DAYS old, and one 3x that age is nearly discarded. Reacts to regime shifts faster than the flat BRAIN_ANALYSIS_WINDOW_DAYS window. Does NOT yet apply to threshold-optimization (recommend_threshold's bucket/knee/EV machinery) — that stays unweighted for now")
    RECENCY_DECAY_DAYS: float = Field(default=7.0, gt=0, description="Exponential time-constant for recency weighting: weight = exp(-age_days / this). Not a strict half-life (that would be this * ln(2), ~4.85 days at the default) — matches the simple exp(-age/decay) formulation")
    OB_MIN_PENETRATION_ATR_MULT: float = Field(default=0.05, ge=0.0, le=2.0, description="Minimum close penetration beyond the zone edge (top for demand, bottom for supply), scaled by ATR_SHORT, required to count as a confirmed reversal. 0 disables the check. Prevents a close a fraction of a tick beyond the zone from counting as 'reversed'")
    OB_CONFIRM_LOOKAHEAD_CANDLES: int = Field(default=5, ge=0, le=10, description="Candles of grace after a zone is first touched during which a close beyond the opposite edge (+ OB_MIN_PENETRATION_ATR_MULT) still counts as a confirmed reversal. 0 restores the old same-candle-only behavior. A close that fully breaks the zone in the invalidating direction during the grace window kills it immediately")
    OB_PERSISTENCE_CANDLES: int = Field(default=2, ge=0, le=10, description="How many additional closed 15m candles after an OB confirmation to keep the gate valid. 0 = exact-candle-only (legacy).")
    ENABLE_WIN_RATE_FILTER: bool = Field(default=False)
    ENABLE_OI_FUNDING_FILTER: bool = Field(default=False, description="Block BUY/SELL when OI isn't rising (vs pair's own history) AND funding is crowded (vs pair's own history) in the alert direction")
    OI_FUNDING_HISTORY_LEN: int = Field(default=30, ge=5, le=200, description="Rolling window of past OI/funding samples kept per pair (in run cycles, e.g. 30 runs @ 15m cadence ≈ 7.5h)")
    MIN_OI_FUNDING_SAMPLES: int = Field(default=8, ge=3, description="Min warm-up samples before the adaptive gate activates for a pair; fail-open until then")
    OI_RISING_PERCENTILE: float = Field(default=0.50, ge=0.0, le=1.0, description="OI delta must exceed this percentile of the pair's own recent |delta| history to count as 'rising with conviction'")
    OI_DELTA_REF_SAMPLES: int = Field(default=3, ge=1, le=20, description="Number of most-recent OI history samples averaged to form the reference point for oi_delta, instead of comparing oi_now against only the single last sample. Smooths out a single anomalous tick (exchange glitch, brief liquidation cascade spike) from distorting the delta. 1 reproduces the old single-sample behavior")
    FUNDING_CROWDED_PERCENTILE: float = Field(default=0.80, ge=0.5, le=1.0, description="Current funding must be at/above this percentile (BUY) or at/below its complement (SELL) of the pair's own recent funding history to count as 'crowded'")
    FUNDING_ABS_FLOOR: float = Field(default=0.0005, ge=0.0, description="Min |funding| required before percentile-crowding applies at all, so a flat near-zero history can't self-trigger 'crowded'")
    MIN_OI_USD: float = Field(default=75000, ge=0.0, description="Ignore the OI/funding gate entirely for pairs whose current OI is below this floor (quote currency). 0 disables the floor")
    OI_FUNDING_MAX_SAMPLE_AGE_SEC: int = Field(default=10800, ge=300, description="Prune OI/funding samples older than this (default 180min ≈ 12 cycles @15m, matching OI_DIVERGENCE_LOOKBACK_SAMPLES). Prevents comparing a stale pre-outage sample as if only one cycle passed")
    ENABLE_OI_PRICE_DIVERGENCE: bool = Field(default=False, description="Block BUY when price is rising but OI is falling (short-covering, not new demand); block SELL when price is falling but OI is falling (long liquidation, not new supply). Requires ticker mark price to be available.")
    OI_DIVERGENCE_LOOKBACK_SAMPLES: int = Field(default=12, ge=2, le=200, description="How many OI/price history samples back to compare against for divergence (default 12 runs @15m ≈ 3h)")
    OI_DIVERGENCE_MIN_PRICE_ROC_PCT: float = Field(default=0.3, ge=0.0, le=50.0, description="Min absolute price move (%) over the lookback window before divergence logic applies at all")
    OI_DIVERGENCE_MIN_OI_FALL_PCT: float = Field(default=2.0, ge=0.0, le=100.0, description="Min OI decline (%) over the lookback window to count as 'falling with conviction' (closing/covering, not new positioning)")
    ENABLE_OB_GATE: bool = Field(default=False, description="Add institutional order-block (supply/demand) reversal on 15m as a confluence vote. Abstains (None) unless a fresh, first-touch reversal off an unmitigated zone confirms this cycle")
    OB_FILTER_CONFLUENCE: bool = Field(default=False) 
    OB_LOOKBACK_CANDLES: int = Field(default=50, ge=20, le=500, description="How many closed 15m candles back to scan for order-block zones (default 96 ≈ 24h)")
    OB_IMPULSE_LOOKAHEAD: int = Field(default=3, ge=1, le=10, description="Candles after a candidate base candle checked for the impulsive displacement that confirms it as an order block")
    ENABLE_OB_PREMIUM_DISCOUNT_FILTER: bool = Field(default=False, description="Only accept demand-zone OB reversals below the 50% equilibrium of the OB_LOOKBACK_CANDLES dealing range (discount), and supply-zone reversals above it (premium). Zones on the wrong side are skipped entirely.")
    OUTCOME_LOOKAHEAD_CANDLES: int = Field(default=12, ge=1, le=96) 
    OUTCOME_FAVORABLE_MOVE_PCT: float = Field(default=0.5, ge=0.01, le=10.0) 
    OUTCOME_FILL_DELAY_CANDLES: int = Field(default=1, ge=0, le=4, description="15m candles of latency assumed between signal and simulated fill, used to compute fill_price/entry_slip_pct for realistic net P&L. 0 = no live-order bot exists, so this can never be a real broker fill.")
    MIN_WIN_RATE_SAMPLE: int = Field(default=20, ge=1)    
    MIN_WIN_RATE: float = Field(default=0.55, ge=0.0, le=1.0)    
    OUTCOME_MAE_LOSS_PCT: float = Field(default=0.5, ge=0.01, le=10.0)  
    OUTCOME_PRIMARY_METRIC: str = Field(default="mfe") 
    OUTCOME_RR_TARGET: float = Field(default=2.0, ge=1.0, le=5.0) 
    OUTCOME_BONUS_RR: float = Field(default=3.0, ge=2.0, le=10.0) 
    OUTCOME_BONUS_WEIGHT: float = Field(default=1.5, ge=1.0, le=3.0) 
    OUTCOME_TIMEOUT_WIN_WEIGHT: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight of a timed-out win (neither target nor stop hit; closed on the right side of the fill) in weighted win rate and model training. A real target win weighs 1.0 or more.") 
    ENABLE_SESSION_FILTER: bool = Field(default=False, description="If true, alerts are also checked against a pair:alert_key:session win-rate (session = asian/london/ny/dead per IST trading hours). Blocks alongside the existing pair:alert_key MIN_WIN_RATE check — whichever of the two is lower decides. Requires ENABLE_WIN_RATE_FILTER")
    MIN_WIN_RATE_SESSION_SAMPLE: int = Field(default=15, ge=1, description="Minimum resolved-outcome sample size for a pair:alert_key:session combo before its win rate is trusted enough to block dispatch")
    ENABLE_BRAIN: bool = Field(default=False, description="Master switch for the Brain analysis/shadow-mode/reporting layer. Requires ENABLE_WIN_RATE_FILTER to be meaningful")
    BRAIN_SHADOW_MODE: bool = Field(default=True, description="When an alert is rejected by the win-rate filter, keep tracking what would have happened instead of discarding it")
    BRAIN_REPORT_INTERVAL_RUNS: int = Field(default=1, ge=1, le=2000) 
    BRAIN_REWARDABLE_MIN_CONFLUENCE_PCT: float = Field(default=80.0, ge=50.0, le=100.0, description="Min confluence % required for a win-rate-rejected alert to be eligible for a rewardable override")
    BRAIN_REWARDABLE_MIN_SHADOW_SAMPLE: int = Field(default=10, ge=3, description="Min resolved shadow samples in the high-confluence bucket for this alert_key before an override is trusted")
    BRAIN_REWARDABLE_MIN_SHADOW_WR: float = Field(default=0.60, ge=0.5, le=1.0, description="Shadow win rate required in the high-confluence bucket to allow rewardable overrides through")
    BRAIN_REPORT_STREAM_SAMPLE: int = Field(default=5000, ge=100, le=50000, description="Max recent OUTCOME_LOG_STREAM/SHADOW_LOG_STREAM entries the brain reads per report")
    BRAIN_ALERT_DISABLE_THRESHOLD_WR: float = Field(default=0.40, ge=0.0, le=1.0, description="Pooled win rate (across all pairs) below which the brain recommends disabling an alert_key entirely in its report")
    BRAIN_OVERRIDE_COOLDOWN_SECONDS: int = Field(default=14400, ge=600, le=86400, description="Min seconds between rewardable overrides for the same alert_key")
    BRAIN_STAR_ALERT_WR: float = Field(default=0.70, ge=0.5, le=1.0, description="Win rate above which an alert is flagged as a star performer")
    BRAIN_ANALYSIS_WINDOW_DAYS: int = Field(default=30, ge=7, le=365, description="Only analyze outcomes from the last N days")
    BRAIN_MEDIUM_WINDOW_DAYS: int = Field(default=90, ge=7, le=365, description="Medium-history window for layered_window_analysis — compared against BRAIN_ANALYSIS_WINDOW_DAYS (recent) and BRAIN_LONG_WINDOW_DAYS (long) to tell 'historically good but currently weak' apart from 'always weak'")
    BRAIN_LONG_WINDOW_DAYS: int = Field(default=180, ge=7, le=730, description="Long-history window for layered_window_analysis — see BRAIN_MEDIUM_WINDOW_DAYS")
    ENABLE_LAYERED_WINDOW_ANALYSIS: bool = Field(default=True, description="Adds a recent/medium/long-history (30d/90d/180d by default) per-alert net-EV comparison to the brain report. Purely diagnostic — never changes live gating on its own")
    BRAIN_LONG_WINDOW_STREAM_SAMPLE: int = Field(default=15000, ge=100, le=100000, description="Max OUTCOME_LOG_STREAM entries fetched for the long-history layer of layered_window_analysis — needs to be large enough to actually reach BRAIN_LONG_WINDOW_DAYS back, unlike BRAIN_REPORT_STREAM_SAMPLE which only needs to cover BRAIN_ANALYSIS_WINDOW_DAYS")
    ENABLE_MARKET_STATE_MODEL: bool = Field(default=True, description="Train a logistic market-state model (votes + RSI/PPO/ADX/wick/session/direction context) each report cycle and persist it if it clears OOS validation. Report-only: see ENABLE_MARKET_STATE_LIVE_SCORE for whether dispatch actually uses it.")
    MARKET_STATE_MODEL_MIN_SAMPLE: int = Field(default=150, ge=50, description="Minimum pooled rows before attempting to train/refresh the market-state model. Below this, the previous persisted model (if any) is left in place rather than overwritten by a noisy fit.")
    MARKET_STATE_MODEL_MIN_OOS_P: float = Field(default=0.70, ge=0.5, le=0.99, description="Minimum OOS P(net EV > 0) on the holdout split required before a freshly trained market-state model is accepted/persisted. A model that fails this is discarded, not persisted — the previous one keeps serving live predictions.")
    ENABLE_MARKET_STATE_LIVE_SCORE: bool = Field(default=False, description="If True, get_trade_quality() blends the persisted market-state model's live P(win) into the dispatch-time quality verdict. Keep False (report-only, logged but not shown/used) until the model has been observed passing OOS validation for at least one full BRAIN_ANALYSIS_WINDOW_DAYS window.")
    ENABLE_PNL_WEIGHTED_TRAINING: bool = Field(default=True, description="Weight each training row in train_market_state_model()/oos_permutation_importance() by |net_pnl_pct| instead of counting every row equally, so the fit is pulled toward correctly classifying economically significant trades rather than just maximizing hit-rate. Set False to fall back to uniform weighting for comparison.")
    BRAIN_AUTO_DISABLE_ENABLED: bool = Field(default=True, description="If True, the brain writes disable_alert/reinstate verdicts directly to the live config_override in Redis instead of only reporting them")
    BRAIN_AUTO_DISABLE_MIN_SAMPLE: int = Field(default=200, ge=50, description="Min samples per individual alert_key (not pooled) before the brain will auto-disable or auto-reinstate its shared config path")
    BRAIN_MAX_DISABLED_KEYS: int = Field(default=6, ge=1, le=46, description="Hard cap on how many alert keys can be disabled at the same time. Once this many are disabled, further Brain auto-disables are skipped (and reported as NOT APPLIED) until a key is re-enabled. Re-enables are never blocked by this cap.")
    BRAIN_REENABLE_COOLDOWN_HOURS: int = Field(default=72, ge=0, le=720, description="Minimum hours an alert key stays brain-disabled before it can be auto re-enabled")
    BRAIN_REENABLE_MIN_NEW_SAMPLES: int = Field(default=30, ge=10, le=500, description="Min post-disable shadow outcomes (de-clustered to one per pair per outcome horizon) before a disabled alert key can be auto re-enabled")
    BRAIN_REENABLE_MIN_WR_LO: float = Field(default=0.40, ge=0.30, le=0.90, description="Wilson lower bound of the post-disable shadow win rate required to re-enable. Disable fires when the upper bound is below BRAIN_ALERT_DISABLE_THRESHOLD_WR; re-enable needs the lower bound above this, so there is a dead band between the two")
    BRAIN_REENABLE_PROBATION_DAYS: int = Field(default=30, ge=0, le=180, description="After a re-enable, the disable verdict only counts outcomes recorded after the re-enable for this many days (0 = off). The re-enable was justified by fresher evidence, so the pre-disable losses are superseded; 30 matches how long they stay in the analysis window. Auto-disable still needs BRAIN_AUTO_DISABLE_MIN_SAMPLE post-re-enable outcomes")
    BRAIN_MC_SIMULATIONS: int = Field(default=50, ge=0, le=500, description="Block-bootstrap Monte Carlo simulations for the robustness check in the periodic brain report. 0 disables it (offline-only, never affects live gating).")
    BRAIN_COUNTERFACTUAL_PROMOTE_MIN_SHADOW: int = Field(default=15, ge=5, le=200, description="Min shadow (rejected-path) sample a counterfactual scenario needs, in addition to shadow_validated=True, before the SIMULATIONS report section marks it PROMOTION-ELIGIBLE instead of NOT YET. Matches the min_n=15 that _shadow_weight_check already uses for the same class of decision (shadow-validating a proposed change before trusting it)")
    BRAIN_WEIGHT_OPTIMIZER_MAX_DELTA: float = Field(default=2.0, ge=0.5, le=5.0)
    BRAIN_WEIGHT_OPTIMIZER_WALK_FORWARD: bool = Field(default=True) 
    BRAIN_WEIGHT_OPTIMIZER_MIN_CONFIDENCE: float = Field(default=0.4, ge=0.0, le=1.0)      
    BRAIN_PERMUTATION_IMPORTANCE: bool = Field(default=True) 
    DRY_RUN_MODE: bool = Field(default=False)
    SKIP_WARMUP: bool = Field(default=False)
    REJECT_HIGH_DEVIATION: bool = Field( default=False)
    SANITIZE_BAD_CANDLES: bool = Field(default=False, description="If True, drop individual invalid candles instead of rejecting the whole fetch")
    ICHIMOKU_CLOUD_ENABLED: bool = Field(default=True, description="Compute the Ichimoku cloud (BIAS_ICHIMOKU_* = 23/65/130/65) used by the cloud-cross alerts. It no longer votes in the cloud group (Gaussian Trend does)")
    GAUSSIAN_TREND_ENABLED: bool = Field(default=True, description="Gaussian Filter Trend (QuantAlgo Pine port) as the trend-cloud vote of the cloud group; replaces the old Ichimoku 9/26/52/26 cloud vote")
    GAUSSIAN_PRESET: Literal['Default', 'Fast Response', 'Smooth Trend'] = Field(default='Default', description="Pine preset. Anything other than Default overrides the GAUSSIAN_* length/pole/ATR/efficiency inputs below, exactly like the Pine script")
    GAUSSIAN_SOURCE: Literal['close', 'hl2', 'hlc3'] = Field(default='close', description="Price source passed through the Gaussian filter (Pine: Price Source)")
    GAUSSIAN_LENGTH: int = Field(default=14, ge=2, le=200, description="Gaussian filter sampling length in bars (Pine: Length)")
    GAUSSIAN_POLES: int = Field(default=4, ge=1, le=4, description="Number of cascaded filter poles (Pine: Poles)")
    GAUSSIAN_ATR_LENGTH: int = Field(default=14, ge=1, le=200, description="ATR length for the trend dead-band (Pine: ATR Length)")
    GAUSSIAN_FIXED_MULTIPLIER: float = Field(default=1.5, ge=0.1, le=20.0, description="Fixed dead-band width in ATR units; only used when GAUSSIAN_ADAPTIVE_WIDTH is False")
    GAUSSIAN_ADAPTIVE_WIDTH: bool = Field(default=True, description="Scale the dead-band by the smoothed Efficiency Ratio (Pine: Adaptive Width)")
    GAUSSIAN_EFFICIENCY_LENGTH: int = Field(default=10, ge=2, le=200, description="Efficiency Ratio lookback (Pine: Efficiency Length)")
    GAUSSIAN_TREND_MULTIPLIER: float = Field(default=0.8, ge=0.1, le=20.0, description="ATR width when efficiency is high (Pine: Adaptive Trend Multiplier)")
    GAUSSIAN_CHOP_MULTIPLIER: float = Field(default=2.5, ge=0.1, le=20.0, description="ATR width when efficiency is low (Pine: Adaptive Chop Multiplier)")
    GAUSSIAN_EFFICIENCY_SMOOTH: int = Field(default=5, ge=1, le=200, description="EMA length applied to the Efficiency Ratio (Pine: Efficiency Smoothing)")
    ICHIMOKU_TK_CONVERSION_PERIODS: int = Field(default=23, ge=1, le=300, description="Tenkan (conversion) length used for TK guard + cross alerts, independent of cloud conversion length")
    ICHIMOKU_TK_BASE_PERIODS: int = Field(default=65, ge=1, le=400, description="Kijun (base) length used for TK guard + cross alerts, independent of cloud base length")
    ENABLE_BIAS_HEADER: bool = Field(default=True, description="Prepend a pair-universe directional-bias header (Cirrus Cloud, or Ichimoku when BIAS_USE_CIRRUS=False; cosmetic only, does not gate alerts) to every outgoing Telegram message")
    ALERT_TAKE_MIN_CONVICTION: float = Field(default=70.0, ge=0.0, le=100.0, description="Telegram alert verdict: conviction % at or above which an alert reads TAKE")
    ALERT_WATCH_MIN_CONVICTION: float = Field(default=50.0, ge=0.0, le=100.0, description="Telegram alert verdict: conviction % at or above which an alert reads WATCH (below = AVOID)")
    ALERT_SHADOW_CONVICTION_CAP: float = Field(default=55.0, ge=0.0, le=100.0, description="Conviction ceiling while the only evidence is SHADOW/INSUFFICIENT, so nothing reads TAKE before it has real results")
    ALERT_BIAS_MIN_EDGE: float = Field(default=0.10, ge=0.0, le=1.0, description="Fraction of pairs by which the dominant bias bucket must lead the opposite one before an alert counts as with/against the market; otherwise the alert is treated as neutral")
    ENABLE_TAKE_SKIP_BUTTONS: bool = Field(default=False, description="Attach Took/Skip buttons to alerts and read the taps at the start of each run (one getUpdates call). Needs no webhook on the bot; off by default.")
    TAKE_SKIP_MAX_ROWS: int = Field(default=8, ge=1, le=20, description="Maximum pairs (button rows) per alert message")
    PLAYBOOK_MODE: str = Field(default="shadow", description="Learner playbook in alerts: off | shadow (record only) | live (restrict-only overlay)")
    PLAYBOOK_ALLOW_UPGRADE: bool = Field(default=False, description="Live mode may lift WATCH to TAKE for a VALIDATED champion plan (off = restrict-only)")
    PLAYBOOK_MAX_AGE_HOURS: float = Field(default=36.0, gt=0, description="Ignore a playbook older than this (learner outage = no change)")
    PLAYBOOK_WINDOW_DAYS: int = Field(default=90, ge=7, description="Learner looks back this many days of outcome archives")
    PLAYBOOK_MIN_N: int = Field(default=40, ge=20, description="Path rows a group needs before the learner tests it")
    PLAYBOOK_MIN_HOLDOUT: int = Field(default=15, ge=5, description="Minimum out-of-sample rows")
    PLAYBOOK_PROMOTE_CONSECUTIVE: int = Field(default=3, ge=1, description="Consecutive re-validations of a frozen plan before promotion")
    PLAYBOOK_MIN_FORWARD: int = Field(default=15, ge=5, description="Forward-only rows (after the plan was frozen) required to promote")
    PLAYBOOK_REQUIRE_CONTROL: bool = Field(default=True, description="Promotion needs positive alpha against the no-alert control baseline")
    PLAYBOOK_CONTROL_DAYS: int = Field(default=14, ge=3, le=30, description="Days of 15m candles used for the no-alert control")
    PLAYBOOK_FAMILY_ALPHA: float = Field(default=0.2, gt=0, lt=1, description="Family-wise false-positive budget across all groups tested")
    PLAYBOOK_RECONFIRM_DAYS: float = Field(default=7.0, gt=0, description="Champion flagged RECONFIRM_DUE after this many days without fresh confirmation")
    PLAYBOOK_EXPIRE_DAYS: float = Field(default=14.0, gt=0, description="Champion expires after this many days without confirmation")
    PLAYBOOK_MAX_DD_R: float = Field(default=15.0, gt=0, description="A plan's out-of-sample drawdown may not exceed this many stops (promotion criterion and scoreboard T4)")
    PLAYBOOK_STAB_FLOOR_R: float = Field(default=0.25, ge=0, description="Neither half of the holdout may average worse than minus this many stops")
    PLAYBOOK_AVOID_CONSECUTIVE: int = Field(default=2, ge=1, description="Consecutive learner cycles of proof before a group is restricted to AVOID")
    PLAYBOOK_MAX_RULES: int = Field(default=120, ge=10, le=500, description="Cap on 'do not take when' rule candidates scored per group per cycle")
    PLAYBOOK_SIZE_FULL_LCB: float = Field(default=0.30, gt=0, description="EV lower bound (pct) that earns a full advisory size multiple")
    PLAYBOOK_SB_TAKE_MIN_BLOCKS: int = Field(default=60, ge=1, description="Scoreboard T1: independent 3h blocks of forward trades needed")
    PLAYBOOK_SB_AVOID_MIN_BLOCKS: int = Field(default=30, ge=1, description="Scoreboard A1: independent 3h blocks of forward trades needed")
    PLAYBOOK_SB_CALIB_TOL: float = Field(default=0.10, gt=0, lt=1, description="Scoreboard T3: allowed gap between realized and predicted win share")
    PLAYBOOK_RECON_MAX_CALLS: int = Field(default=120, ge=0, le=600, description="Candle-history requests per learner run used to rebuild missing paths of old real alerts (0 = off)")
    PLAYBOOK_NOTIFY: str = Field(default="changes", description="Learner Telegram summary: changes | always | never")
    ENABLE_TRADE_AUTOPSY: bool = Field(default=True, description="Learner also builds the per-trade autopsy, edge map, shadow rejection audit and Trust Meter (report only, never changes a verdict) and stores it under the brain_autopsy metadata key")
    AUTOPSY_NOTIFY_HOURS: float = Field(default=24.0, ge=0.0, description="Send the autopsy digest to Telegram at most this often (0 = never send, still stored)")
    AUTOPSY_MIN_N: int = Field(default=12, ge=5, description="Earlier trades a condition needs before it can count as a strength or weakness of a trade")
    AUTOPSY_EDGE_MIN_N: int = Field(default=20, ge=10, description="Trades on each side a condition needs to enter the edge map")
    AUTOPSY_REJECT_MIN_N: int = Field(default=20, ge=5, description="Independent shadow trades a rejection reason needs before it is judged")
    AUTOPSY_SHRINK_K: float = Field(default=10.0, gt=0, description="Shrinkage strength: a condition with n trades is pulled toward its baseline with weight k/(n+k)")
    TRUST_TARGET_PRECISION: float = Field(default=0.80, gt=0, lt=1, description="TAKE precision target. Reported as met only when the 95% Wilson lower bound reaches it")
    TRUST_MIN_TAKE_N: int = Field(default=100, ge=30, description="Resolved TAKE trades needed before the Trust Meter can reach level L2 or L3")
    TRUST_WINDOWS: int = Field(default=4, ge=2, le=8, description="Chronological periods TAKE trades are split into to test stability")
    ENABLE_ALERT_UNPROVEN_TAKE: bool = Field(default=True, description="Let a very strong technical setup with no brain history read TAKE (small size) instead of always WATCH")
    ALERT_UNPROVEN_MIN_CONFLUENCE_PCT: float = Field(default=90.0, ge=0.0, le=100.0, description="Confluence % needed for an unproven (no brain history) alert to read TAKE (small size)")
    ALERT_UNPROVEN_SIZE_MULT: float = Field(default=0.25, ge=0.0, le=1.0, description="Advisory size multiple shown for unproven TAKE alerts")
    ALERT_BASKET_NOTE_MIN: int = Field(default=3, ge=2, le=50, description="Add a 'size as one basket' note when this many same-direction TAKE alerts land in one batch")
    ALERT_CORRELATED_PAIRS: List[List[str]] = Field(default_factory=lambda: [["PAXGUSD", "XAUTUSD"]], description="Groups of pairs on the same underlying; later same-direction members of a group get a 'count as one trade' note")
    BIAS_ICHIMOKU_CONVERSION_PERIODS: int = Field(default=23, ge=1, le=300, description="Conversion (Tenkan) length for the standalone bias-header Ichimoku cloud, independent of the alert-gate cloud")
    BIAS_ICHIMOKU_BASE_PERIODS: int = Field(default=65, ge=1, le=400, description="Base (Kijun) length for the bias-header cloud")
    BIAS_ICHIMOKU_SPANB_PERIODS: int = Field(default=130, ge=1, le=500, description="Leading Span B length for the bias-header cloud")
    BIAS_ICHIMOKU_DISPLACEMENT: int = Field(default=65, ge=1, le=400, description="Forward displacement for the bias-header cloud")
    RMA_CLOUD_ENABLED: bool = Field(default=True, description="Enable the 5m RMA(fast)/RMA(slow) cloud as one of the cloud-group votes; green (buy) when RMA_fast>RMA_slow, red (sell) when RMA_fast<RMA_slow, both computed on 5m closes (fast = RMA_CLOUD_FAST_PERIOD, slow = RMA_50_PERIOD). Independent of the 15m RMA50 used for base trend.")
    RMA_CLOUD_FAST_PERIOD: int = Field(default=20, ge=2, le=200, description="RMA Cloud fast period (5m). Slow leg reuses RMA_50_PERIOD (also on 5m).")
    CIRRUS_CLOUD_ENABLED: bool = Field(default=True, description="Cirrus Cloud (Pine port), calculated on 5m closes, as one of the cloud-group votes: buy needs Trend Up, sell needs Trend Down. The cloud group passes when at least CLOUD_GROUP_MIN_VOTES of the enabled clouds (RMA 5m, Cirrus 5m, Dynamic Flow 15m, Gaussian 15m) agree")
    CIRRUS_X1: int = Field(default=22, ge=1, le=200, description="Cirrus Cloud slow range EMA length (Pine x1)")
    CIRRUS_X2: float = Field(default=9.0, gt=0.0, le=100.0, description="Cirrus Cloud slow range multiplier (Pine x2)")
    CIRRUS_X3: int = Field(default=15, ge=1, le=200, description="Cirrus Cloud fast range EMA length (Pine x3)")
    CIRRUS_X4: float = Field(default=5.0, gt=0.0, le=100.0, description="Cirrus Cloud fast range multiplier (Pine x4)")
    CIRRUS_MIN_GAP_PCT: float = Field(default=0.0, ge=0.0, le=5.0, description="Neutral band for Cirrus Cloud: when the two filter lines are closer than this percent of price, the trend is neither Up nor Down (the Cirrus vote abstains). 0 = exact Pine replica. Near-parallel lines flip on tiny price differences between the bot's data and TradingView's, so a small band (e.g. 0.10) removes those edge-case votes")
    ENABLE_CLOUD_VOTE_LINE: bool = Field(default=True, description="Add one line to every alert showing how each of the four clouds (Cirrus 5m, Dynamic Flow 15m, Gaussian 15m, RMA 5m) voted at decision time")
    BIAS_USE_CIRRUS: bool = Field(default=True, description="Base the pair-universe bias header on Cirrus Cloud (Trend Up / Trend Down / neutral) instead of the standalone Ichimoku cloud (BIAS_ICHIMOKU_*). False restores the Ichimoku bias")
    DYNAMIC_FLOW_RIBBON_ENABLED: bool = Field(default=True, description="Enable the 15m Dynamic Flow Ribbon (BigBeluga) as a third cloud-group trend gate alongside Ichimoku Cloud and RMA Cloud; green (buy) when the band-flip direction is bullish, red (sell) when bearish")
    DYNAMIC_FLOW_FACTOR: float = Field(default=3.0, ge=0.1, le=20.0, description="Dynamic Flow Ribbon band-width multiplier (Pine 'Length' input) — bands sit at basis \u00b1 factor*dist")
    DYNAMIC_FLOW_BASIS_LENGTH: int = Field(default=15, ge=2, le=200, description="Dynamic Flow Ribbon basis EMA period (15m, applied to hlc3)")
    DYNAMIC_FLOW_DIST_LENGTH: int = Field(default=200, ge=10, le=500, description="Dynamic Flow Ribbon distance SMA period (15m, applied to high-low) used to size the bands")
    ENABLE_DYNAMIC_FLOW_CROSS_ALERT: bool = Field(default=False, description="Add Dynamic Flow Ribbon crossover/cross-under alert: fires on the candle where price crosses the ribbon line while the ribbon direction is aligned, gated by buy_trend_common_relaxed/sell_trend_common_relaxed plus a wick-ratio-or-reversal-pattern condition — same gating style as CHoCH. Requires DYNAMIC_FLOW_RIBBON_ENABLED. Timeframe controlled by CROSS_ALERT_TF.")
    ENABLE_TK_CONVERSION_CROSS: bool = Field(default=True, description="Enable alert when close crosses above/below the Ichimoku conversion (Tenkan) line, subject to all other buy/sell common conditions. Timeframe controlled by CROSS_ALERT_TF.")
    ENABLE_CLOUD_CROSS_ALERT: bool = Field(default=True, description="Enable alert when close crosses above/below the Ichimoku cloud, subject to all other buy/sell common conditions. Timeframe controlled by CROSS_ALERT_TF.") 
    ENABLE_KIJUN_CROSS: bool = Field(default=True, description="Enable alert when close crosses above/below the Ichimoku base (Kijun) line, subject to all other buy/sell common conditions. Timeframe controlled by CROSS_ALERT_TF.")
    ENABLE_MOMENTUM_CROSS_ALERT: bool = Field(default=False, description="Add Momentum Oscillator zero-line crossover/cross-under alert: fires on the candle where the Momentum Oscillator (vf) crosses above/below 0, gated exactly like the Dynamic Flow cross (buy_trend_common_relaxed/sell_trend_common_relaxed plus a wick-ratio-or-reversal-pattern condition). When MOMENTUM_CROSS_TF is '5m' the 15m oscillator must also be on the same side of 0. Requires MOMENTUM_OSCILLATOR_ENABLED.")
    MOMENTUM_CROSS_TF: str = Field(default="5m", pattern="^(5m|15m)$", description="Timeframe for the Momentum cross alert: '5m' (cross detected on 5m data and additionally requires the 15m oscillator to be on the favourable side of 0) or '15m' (cross detected on 15m data).")
    ENABLE_GAUSSIAN_CROSS_ALERT: bool = Field(default=False, description="Add Gaussian Filter Trend line crossover/cross-under alert: fires on the candle where price crosses the Gaussian trend line while the Gaussian direction is aligned, gated exactly like the Dynamic Flow cross (buy_trend_common_relaxed/sell_trend_common_relaxed plus a wick-ratio-or-reversal-pattern condition). Requires GAUSSIAN_TREND_ENABLED. Timeframe controlled by GAUSSIAN_CROSS_TF.")
    GAUSSIAN_CROSS_TF: str = Field(default="5m", pattern="^(5m|15m)$", description="Timeframe for the Gaussian cross alert: '5m' (cross detected on a 5m Gaussian line and additionally requires the 15m Gaussian trend to be in the favourable direction) or '15m' (cross detected on the 15m Gaussian line).")
    CROSS_ALERT_TF: str = Field(default="5m", description="Timeframe for cloud_cross / tk_conversion / kijun_cross / dynamic_flow_cross detection: '5m' (default) or '15m'. When '5m', the cross is detected on 5m data and additionally requires the corresponding 15m indicator to be in the favorable direction (e.g. 15m Dynamic Flow green for a buy cross). All other common conditions remain on their existing timeframes (mostly 15m + RMA200 on 5m).")
    ENABLE_STRONG_REVERSAL_ALERT: bool = Field(default=True, description="Enable candlestick reversal-pattern alert (Engulfing/Piercing/Star/Soldiers-Crows/Tweezer/Harami/Marubozu/Pinbar) on top of full buy_common/sell_common confluence")
    ENABLE_EQUILIBRIUM_CROSS: bool = Field(default=False, description="Add 15m SMC Equilibrium cross alert: fires when close crosses the 50% midpoint (equilibrium) of the OB_LOOKBACK_CANDLES dealing range — same premium/discount equilibrium used by ENABLE_OB_PREMIUM_DISCOUNT_FILTER — gated by buy_trend_common_relaxed/sell_trend_common_relaxed plus a wick-ratio-or-reversal-pattern condition and the PPO/RSI signal-cross guard, same relaxed gating style as CHoCH")
    ENABLE_CHOCH_ALERT: bool = Field(default=False, description="Add 15m Change-of-Character (CHoCH) alert: fires on the displacement candle that recovers back through a swept short-term low/high, before any structural pivot is broken, gated by buy_trend_common/sell_trend_common plus a wick or reversal-pattern condition")
    CHOCH_SWING_LEN: int = Field(default=3, ge=2, le=20, description="Bars on each side used to confirm a short-term (minor) swing pivot for CHoCH structure — the lower highs / higher lows the break is measured against")
    CHOCH_LOOKBACK_CANDLES: int = Field(default=40, ge=10, le=200, description="How many closed 15m candles back from the current candle to scan for a qualifying CHoCH structure (swing pivots + sweep)")
    CHOCH_CONFIRM_WINDOW_CANDLES: int = Field(default=6, ge=1, le=20, description="Max candles allowed between the liquidity-sweep candle and the displacement candle. A displacement found further back than this is rejected as stale")
    CHOCH_ALLOW_SAME_CANDLE_SWEEP: bool = Field(default=False, description="If True, the sweep and the displacement candle may be the same 15m candle. If False (default), the displacement must be strictly after the sweep candle, removing same-candle sweep/displacement ambiguity")
    CHOCH_MIN_SWEEP_DISTANCE_ATR: float = Field(default=0.05, ge=0.0, le=2.0, description="Minimum distance (in ATR_SHORT multiples) the sweep wick must pierce beyond the prior short-term low/high to count as a real liquidity sweep rather than noise")
    CHOCH_MIN_DISPLACEMENT_BODY_RATIO: float = Field(default=0.45, ge=0.0, le=1.0, description="Minimum body-to-range ratio required on the displacement/entry candle so the CHoCH is backed by a real move rather than a thin/indecisive close")
    CHOCH_REQUIRE_FVG: bool = Field(default=False, description="If True, an unfilled direction-specific Fair Value Gap must also exist within the sweep-to-displacement window for the CHoCH to qualify. If False, FVG presence is still detected and reported in the alert reason as a bonus, not a requirement")
    CHOCH_CHECK_POI_TAP: bool = Field(default=False, description="Bonus confluence only, not a hard requirement: also check whether the sweep-to-displacement window touched an existing demand/supply order-block zone (POI). That window is now typically 1 candle (often the same candle), so POI taps will register less often than under the old break-based logic. Reuses the OB gate's zone detection (same OB_LOOKBACK_CANDLES/OB_FILTER_CONFLUENCE settings) and appends 'POI tap' to the CHoCH alert reason when true")
    CHOCH_PERSISTENCE_CANDLES: int = Field(default=1, ge=0, le=9, description="How many additional closed 15m candles after a displacement candle to keep the gate valid. Invalidation now checks price against the swept level, not a structural pivot — see below. 0 = exact-candle-only (fresh displacement required every cycle)")
    ENABLE_FIB_REVERSAL_ALERT: bool = Field(default=False, description="Enable Fibonacci Pivot Reversal alerts: price retraces into the 50-78.6% zone of the last major swing leg, with a confluence vote across the zone touch, oscillator divergence, wick/pattern rejection, and volume exhaustion")
    FIB_REVERSAL_CONFLUENCE_REQUIRED: int = Field(default=3, ge=1, le=4, description="Minimum number of the 4 confluence checks (wick/pattern rejection, Fibonacci zone, oscillator divergence, volume exhaustion) that must pass for a Fib Reversal alert to fire")
    FIB_REVERSAL_SWING_LENGTH: int = Field(default=5, ge=2, le=200, description="Bars on each side used to confirm a major swing pivot for the Fibonacci leg — matches the OB detection swing_len so the zone is anchored to the same structural swings")
    FIB_REVERSAL_SWING_LOOKBACK_CANDLES: int = Field(default=150, ge=20, le=2000, description="How many candles back to search for the swing pivots that anchor the Fibonacci leg and the divergence comparison")
    FIB_REVERSAL_ZONE_LOW: float = Field(default=0.5, ge=0.0, le=1.0, description="Lower bound of the Fibonacci retracement zone (as a fraction of the leg from the anchor swing to the extreme reached since) that counts as a zone touch")
    FIB_REVERSAL_ZONE_HIGH: float = Field(default=0.786, ge=0.0, le=1.0, description="Upper bound of the Fibonacci retracement zone — default 0.5-0.786 is the conventional 'golden zone'")
    FIB_REVERSAL_VOL_DRYUP_LOOKBACK: int = Field(default=6, ge=2, le=50, description="Number of candles compared for the volume dry-up check: mean volume over the N candles before the touch candle vs mean volume over the N candles before that")
    FIB_REVERSAL_VOL_SPIKE_MULT: float = Field(default=1.3, ge=1.0, le=5.0, description="Touch candle's volume must exceed its volume EMA by this multiple to count as an exhaustion/reversal spike") 
    FIB_REVERSAL_MAX_DIVERGENCE_AGE_BARS: int = Field(default=50, ge=5, le=500, description="Max bars between the anchor swing and the prior swing for divergence comparison")
    FIB_REVERSAL_MAJOR_SWING_LENGTH: int = Field(default=50, ge=2, le=200, description="Fallback only when no minor pivot exists...")
    ENABLE_TRENDLINE_ALERTS: bool = Field(default=False, description="Master toggle for Auto Trendline alerts (tl_break / tl_bounce / tl_retest). Trendlines join the two most recent confirmed 15m fractal highs (resistance) and lows (support), exactly as the Pine 'Auto Trendline' indicator does. Off by default.")
    TRENDLINE_SHADOW_MODE: bool = Field(default=True, description="While True, trendline alerts are evaluated and recorded as counterfactual (shadow) trades but NEVER sent to Telegram. Set False only after reviewing the shadow results.")
    ENABLE_TL_BREAK_ALERT: bool = Field(default=True, description="Trendline break alert: a closed candle crosses the line after the previous close was on the other side. Only active when ENABLE_TRENDLINE_ALERTS is True.")
    ENABLE_TL_BOUNCE_ALERT: bool = Field(default=True, description="Trendline bounce alert: price wicks into the support (BUY) / resistance (SELL) line and closes back on the right side. Only active when ENABLE_TRENDLINE_ALERTS is True.")
    ENABLE_TL_RETEST_ALERT: bool = Field(default=True, description="Trendline retest alert: after a confirmed break, price returns to the broken line for its first retest and holds. Only active when ENABLE_TRENDLINE_ALERTS is True.")
    TRENDLINE_FRACTAL_PERIOD: int = Field(default=20, ge=2, le=100, description="Bars on each side of a fractal pivot (Pine ta.pivothigh(n, n)). A pivot is usable only n+1 closed bars after it forms, matching the Pine indicator's [1] shift, so lines never repaint.")
    TRENDLINE_LOOKBACK_CANDLES: int = Field(default=300, ge=60, le=2000, description="How many closed 15m candles back to search for pivots. Raised automatically to at least 6x the fractal period.")
    TRENDLINE_MIN_SLOPE_PCT: float = Field(default=0.0001, ge=0.0, le=0.05, description="Ignore lines flatter than this, as a fraction of price per bar (0.0001 = 0.01% per bar). 0 disables the filter and allows horizontal lines.")
    TRENDLINE_BREAK_MIN_ATR: float = Field(default=0.0, ge=0.0, le=3.0, description="Minimum distance (ATR_SHORT multiples) the breakout close must be beyond the line. 0 = any close beyond the line.")
    TRENDLINE_BREAK_MAX_ATR: float = Field(default=1.5, ge=0.1, le=10.0, description="Anti-chase limit: a break is only an EARLY signal if the close is no further than this many ATR_SHORT beyond the line. A candle that already ran further is skipped.")
    TRENDLINE_BREAK_MIN_BODY_RATIO: float = Field(default=0.4, ge=0.0, le=1.0, description="False-breakout guard: the break candle's body must be at least this fraction of its range, so a wick-only poke or a doji through the line does not count.")
    TRENDLINE_REJECT_MIN_CLOSE_POS: float = Field(default=0.5, ge=0.0, le=1.0, description="False-bounce guard for bounce and retest: the candle must close in the favourable part of its range (BUY: at/above this fraction from the low; SELL: from the high). 0.5 = closing in the better half.")
    TRENDLINE_BOUNCE_ATR_MULT: float = Field(default=0.25, ge=0.0, le=3.0, description="Touch tolerance (ATR_SHORT multiples) for bounce and retest: the wick must come within this distance of the line.")
    TRENDLINE_BOUNCE_MAX_PIERCE_ATR: float = Field(default=0.5, ge=0.0, le=3.0, description="Maximum distance (ATR_SHORT multiples) the wick may pierce THROUGH the line while still counting as a bounce/retest. Deeper wicks are treated as a failed line.")
    TRENDLINE_RETEST_ATR_MULT: float = Field(default=0.15, ge=0.0, le=3.0, description="Touch tolerance (ATR_SHORT multiples) for the retest of a broken line.")
    TRENDLINE_RETEST_MAX_BARS: int = Field(default=12, ge=2, le=96, description="A retest is only accepted within this many closed candles after the break candle.")
    TRENDLINE_MIN_TOUCHES: int = Field(default=2, ge=2, le=10, description="Minimum touch points a line needs. The 2 anchor pivots count as 2, so 2 = anchors only; 3 requires one more separate touch (wick within TRENDLINE_TOUCH_ATR_MULT*ATR of the line without closing through it) between the anchors and the signal candle.")
    TRENDLINE_TOUCH_ATR_MULT: float = Field(default=0.25, ge=0.0, le=3.0, description="How close (ATR_SHORT multiples) a wick must come to the line to count as an extra touch.")
    TRENDLINE_MAX_AGE_BARS: int = Field(default=200, ge=20, le=2000, description="Reject a line whose newest anchor is older than this many closed candles.")
    TRENDLINE_REJECT_VIOLATED: bool = Field(default=True, description="Reject a line if any candle CLOSED through it between its first anchor and the signal. A line price has already closed through is not a valid support/resistance.")
    ENABLE_PULLBACK_CONTINUATION: bool = Field(default=False, description="Master toggle for the trend-aligned pullback-continuation layer (Trend -> Pullback Exhaustion -> Confirmed Continuation). When True, reversal-family and RMA/PPO histogram-reversal alerts are tagged when they fire as a confirmed pullback continuation, and an informational WATCH message can be sent. Existing alert conditions are never loosened. False = layer fully off")
    PULLBACK_REQUIRE_CONFIRMED: bool = Field(default=False, description="Restrict-only gate mode: if True, the host alerts (strong_reversal, choch, fib_reversal, ob_reversal, dynamic_flow_cross, hist_rma, ppohist) fire only when the pullback setup is CONFIRMED. If False (default) they fire as today and are merely tagged")
    PULLBACK_SEND_WATCH: bool = Field(default=True, description="Send a separate, non-actionable WATCH Telegram message when a pullback shows exhaustion but the trigger candle has not closed yet")
    PULLBACK_WATCH_MAX_PER_RUN: int = Field(default=6, ge=1, le=30, description="Max pullback WATCH entries per run (all go out in ONE message after the real alerts). Extra setups are released and retried next run")
    PULLBACK_WATCH_DEDUP_SEC: int = Field(default=10800, ge=900, le=86400, description="Dedup window for one WATCH message per pair/direction/pullback-low candle")
    PULLBACK_LOOKBACK_CANDLES: int = Field(default=24, ge=8, le=96, description="Closed 15m candles scanned back from the current candle for the impulse top and the pullback low")
    PULLBACK_IMPULSE_LOOKBACK_CANDLES: int = Field(default=24, ge=8, le=96, description="Candles before the impulse top searched for the impulse start (swing low for BUY, swing high for SELL); that level is also the invalidation level")
    PULLBACK_MIN_IMPULSE_ATR: float = Field(default=2.0, ge=0.5, le=20.0, description="Minimum size of the trend leg before the pullback, in ATR_SHORT multiples")
    PULLBACK_MIN_DEPTH_ATR: float = Field(default=0.8, ge=0.2, le=10.0, description="Minimum pullback depth in ATR_SHORT multiples, so tiny wiggles are not treated as pullbacks")
    PULLBACK_MIN_RETRACE: float = Field(default=0.2, ge=0.05, le=0.9, description="Minimum pullback depth as a fraction of the preceding trend leg")
    PULLBACK_MAX_RETRACE: float = Field(default=0.786, ge=0.2, le=0.99, description="Maximum pullback depth as a fraction of the preceding trend leg. Deeper pullbacks are treated as a weak trend and not tagged")
    PULLBACK_MIN_BARS: int = Field(default=2, ge=1, le=20, description="Minimum candles between the impulse top and the pullback low")
    PULLBACK_MAX_AGE_BARS: int = Field(default=8, ge=1, le=48, description="A setup expires when the pullback low is more than this many candles old")
    PULLBACK_EXHAUSTION_MIN_VOTES: int = Field(default=2, ge=1, le=3, description="Exhaustion votes needed out of 3: rejection wick/pattern, PPO histogram turning up off its pullback trough, RSI turning up")
    PULLBACK_REJECTION_WICK_RATIO: float = Field(default=0.40, ge=0.1, le=0.9, description="Lower-wick (BUY) / upper-wick (SELL) share of the candle range that counts as a rejection candle")
    PULLBACK_TRIGGER_BUFFER_ATR: float = Field(default=0.0, ge=0.0, le=2.0, description="Extra distance (ATR_SHORT multiples) the trigger candle must close beyond the pullback trigger level")
    PULLBACK_MIN_BARS_AFTER_LOW: int = Field(default=1, ge=0, le=6, description="Exhaustion (WATCH) needs at least this many closed candles after the pullback extreme, so a one-candle bounce top/bottom cannot qualify and the trigger level always exists. 0 = old behaviour (exhaustion allowed on the extreme candle itself)")
    PULLBACK_CONFIRM_WINDOW_BARS: int = Field(default=1, ge=1, le=4, description="A trigger close on any of the last N candles confirms the setup. 1 = the current candle only (strict). 2 lets a host alert that fires one candle after the trigger still be tagged")
    EXHAUSTION_MODE: str = Field(default="shadow", pattern="^(off|shadow|live)$", description="Late-trend exhaustion filter for the host alerts (same list as the pullback layer). off = not computed; shadow = score computed, stored with every Recorded/Shadowed trade and shown on the alert, nothing blocked; live = a host alert whose exhaustion score reaches EXHAUSTION_VETO_SCORE (in at least EXHAUSTION_MIN_CATEGORIES categories) is not sent and is shadow-tracked as 'exhaustion_veto'. Existing alert conditions are never changed. Keep shadow until the Brain's exhaustion study shows the veto catches early stops without losing winners")
    EXHAUSTION_SCOPE: str = Field(default="all", pattern="^(all|hosts)$", description="Which alerts the exhaustion filter covers. all = every alert (pivot, cross, reversal, trendline...), measured in the alert's own direction; hosts = only the pullback-layer host alerts")
    EXHAUSTION_EXCLUDE_KEYS: List[str] = Field(default_factory=list, description="Alert keys the exhaustion filter never touches, e.g. [\"tl_break_buy\"]. Empty = none excluded")
    EXHAUSTION_VETO_SCORE: int = Field(default=4, ge=1, le=6, description="Exhaustion score (0-6) at which a host alert is vetoed in live mode")
    EXHAUSTION_MIN_CATEGORIES: int = Field(default=2, ge=1, le=3, description="Independent categories (extension, momentum, failed continuation) that must each score at least 1 before a veto, so one noisy indicator can never block a trade alone")
    EXHAUSTION_SHOW_MIN_SCORE: int = Field(default=3, ge=0, le=6, description="Show a warning line on the Telegram alert when the exhaustion score is at least this. 0 = never show")
    EXHAUSTION_LOOKBACK_CANDLES: int = Field(default=96, ge=40, le=300, description="Closed 15m candles scanned for swing highs/lows, leg count and divergence")
    EXHAUSTION_RUN_BARS: int = Field(default=24, ge=8, le=96, description="Candles scanned back for the latest run extreme used by the extension and rejection tests")
    EXHAUSTION_BASELINE_EMA: int = Field(default=50, ge=10, le=200, description="EMA period of the trend baseline that extension is measured from")
    EXHAUSTION_EXT_ATR: float = Field(default=3.5, ge=1.0, le=15.0, description="Extension vote: run extreme is at least this many ATR_SHORT from the baseline EMA")
    EXHAUSTION_SWING_BARS: int = Field(default=2, ge=1, le=5, description="Bars on each side that confirm a swing high/low (a swing is only used after it is confirmed, so nothing repaints)")
    EXHAUSTION_MIN_LEGS: int = Field(default=4, ge=2, le=12, description="Maturity vote: this many consecutive higher swing highs (lower swing lows for SELL) ending at the latest confirmed swing")
    EXHAUSTION_DIV_MAX_AGE_BARS: int = Field(default=30, ge=6, le=96, description="Divergence only counts when the latest swing is at most this many candles old")
    EXHAUSTION_RSI_DIV_MIN: float = Field(default=1.0, ge=0.0, le=20.0, description="RSI must be lower at the latest swing than at the previous one by at least this many points to count as divergence")
    EXHAUSTION_MIN_ROOM_R: float = Field(default=0.75, ge=0.1, le=5.0, description="No-room vote: distance from entry close to the run extreme is less than this many stop distances (OUTCOME_MAE_LOSS_PCT)")
    EXHAUSTION_TOP_WICK_RATIO: float = Field(default=0.5, ge=0.2, le=0.95, description="Rejection vote: the run-extreme candle has an upper (BUY) / lower (SELL) wick of at least this share of its range, or broke the previous swing and closed back inside it")
    EXHAUSTION_VERDICT_MODE: str = Field(default="shadow", pattern="^(off|shadow|live)$", description="Feed the exhaustion score into the TAKE/WATCH/AVOID verdict. off = ignored; shadow = what it WOULD change is stored with the trade (advisor_seen.exhaustion_adj) but the verdict is untouched; live = restrict-only: lowers conviction and can turn TAKE into WATCH or AVOID, never raises. Needs EXHAUSTION_MODE shadow or live (the score must be computed)")
    EXHAUSTION_VERDICT_WATCH_SCORE: int = Field(default=3, ge=1, le=6, description="With at least EXHAUSTION_MIN_CATEGORIES categories scoring, this score turns a TAKE into WATCH (live verdict mode)")
    EXHAUSTION_VERDICT_AVOID_SCORE: int = Field(default=5, ge=1, le=6, description="With at least EXHAUSTION_MIN_CATEGORIES categories scoring, this score turns TAKE/WATCH into AVOID (live verdict mode). Must be >= the WATCH score")
    EXHAUSTION_VERDICT_PENALTY_PER_POINT: float = Field(default=5.0, ge=0.0, le=20.0, description="Conviction points removed for every exhaustion score point above 2 (live verdict mode). 0 = no conviction penalty")
    EXHAUSTION_EARLY_STOP_BARS: int = Field(default=2, ge=1, le=6, description="A trade stopped within this many candles after fill counts as an early stop in the Brain's exhaustion study")
    EVAL_CONCURRENCY_LIMIT: int = Field(default=2, ge=1, le=30, description="Max pairs evaluated concurrently")
    MIN_RUN_TIMEOUT: int = Field(default=480, ge=300, le=1800)  # Min/max run timeout in seconds (5-30 min)
    MAX_ALERTS_PER_PAIR: int = Field(default=8, ge=5, le=15)  # Max alerts per pair per run    
    MAX_ALERTS_PER_RUN: int = Field(default=50, ge=10, le=200)  
    PIVOT_MAX_DISTANCE_PCT: float = Field(default=1.0)  # Max distance from pivot to trigger alert (1.5%)
    RVOL_THRESHOLD: float = Field(default=1.0, ge=0.5, le=2.0)  # Volatility expansion threshold (1.0=baseline, 1.5=50% expansion required) 
    ATR_ADAPTIVE_ENABLED: bool = Field(default=True)
    ATR_PCTL_LOOKBACK: int = Field(default=96, ge=20, le=500)
    ATR_PCTL_MIN_HISTORY: int = Field(default=50, ge=10, le=400)
    ADAPTIVE_MULT_CALM: float = Field(default=0.85, ge=0.1, le=2.0)
    ADAPTIVE_MULT_VOLATILE: float = Field(default=1.4, ge=0.5, le=3.0)
    ADX_DI_LENGTH: int = Field(default=14, ge=5, le=30)
    ADX_SMOOTHING_LENGTH: int = Field(default=14, ge=5, le=30)
    ADX_ADAPTIVE_TARGET_PCTL: float = Field(default=60.0, ge=1.0, le=99.0, description="ADX threshold = this percentile of the pair's own trailing ADX history")
    ENABLE_ADX_STRENGTH_VOTE: bool = Field(default=False, description="Confluence vote: ADX in top ADX_STRENGTH_PCTL of its own history  a stricter secondary bar on top of the existing adx_ok gate, not a duplicate of it")
    ADX_STRENGTH_PCTL: float = Field(default=80.0, ge=1.0, le=99.0, description="Percentile threshold for the adx_strength confluence vote. Should be set meaningfully above ADX_ADAPTIVE_TARGET_PCTL so this vote and the base 'adx' vote aren't answering the same question")
    ENABLE_ATR_PCTL_VOTE: bool = Field(default=False, description="Confluence vote: current volatility (ATR) in top ATR_PCTL_VOTE_MIN of its own history — a volatility-regime check, distinct from the existing rvol vote which checks short/long ATR expansion trend")
    ATR_PCTL_VOTE_MIN: float = Field(default=0.60, ge=0.0, le=1.0, description="Min ATR percentile rank (0-1) required for the atr_percentile confluence vote to pass")
    ENABLE_VOLUME_PCTL_VOTE: bool = Field(default=False, description="Confluence vote: current volume in top VOLUME_PCTL_VOTE_MIN of its own trailing history — more robust than the existing EMA-based volume_above_ema_ok check, which a single spike can drag upward for several bars")
    VOLUME_PCTL_VOTE_MIN: float = Field(default=0.70, ge=0.0, le=1.0, description="Min volume percentile rank (0-1) required for the volume_percentile confluence vote to pass")
    VOLUME_PCTL_LOOKBACK: int = Field(default=96, ge=20, le=500, description="Rolling window (in 15m candles) for volume percentile ranking")
    VOLUME_PCTL_MIN_HISTORY: int = Field(default=50, ge=10, le=400, description="Min warm-up samples before volume percentile activates; fails open (vote excluded) until then")
    ADX_ADAPTIVE_BAND_WIDTH: float = Field(default=0.0, ge=0.0, le=40.0)
    ADX_ADAPTIVE_FALLBACK: float = Field(default=18.0, ge=5.0, le=50.0, description="ADX threshold used during warm-up or when ATR_ADAPTIVE_ENABLED=False")
    PPO_ADAPTIVE_CALM: float = Field(default=0.08, ge=0.01, le=1.0, description="PPO cross threshold in calm regime")
    PPO_ADAPTIVE_VOLATILE: float = Field(default=0.20, ge=0.01, le=1.0, description="PPO cross threshold in volatile regime")
    RSI_ADAPTIVE_BUY_CALM: float = Field(default=55.0, ge=45.0, le=90.0, description="RSI buy level in calm regime")
    RSI_ADAPTIVE_BUY_VOLATILE: float = Field(default=70.0, ge=45.0, le=90.0, description="RSI buy level in volatile regime")
    RSI_ADAPTIVE_SELL_CALM: float = Field(default=45.0, ge=10.0, le=55.0, description="RSI sell level in calm regime")
    RSI_ADAPTIVE_SELL_VOLATILE: float = Field(default=30.0, ge=10.0, le=55.0, description="RSI sell level in volatile regime")
    MAX_CANDLE_STALENESS_SEC: int = Field(default=1200, ge=600, le=3600)  # Max candle age in seconds (10-60 min)
    LAST_CANDLE_STALE_AFTER_SEC: int = Field(default=2700, ge=900, le=86400, description="Run summary flags a pair as STALE when its last successfully evaluated 15m candle opened more than this many seconds ago (default 2700 = 3 candles).")
    RATE_LIMIT_PER_MINUTE: int = Field(default=400, ge=90, le=600)
    CONFIRM_RATE_LIMIT_PER_MINUTE: int = Field(default=20, ge=5, le=60)
    CB_FAILURE_THRESHOLD: int = Field(default=3, ge=1, le=10)  # Failures before circuit breaker opens
    CB_RECOVERY_TIMEOUT: int = Field(default=60, ge=10, le=600)  # Circuit breaker recovery wait time (seconds)
    DAILY_RESET_BUFFER_SEC: int = Field(default=300, ge=0, le=3600)  # Buffer after midnight before allowing daily resets (VWAP/pivots)
    MIN_CANDLES_PER_DAY: int = Field(default=94, ge=50, le=100)  # Minimum candles for complete day (94=23h for 15m candles)
    CANDLE_MIN_AGE_BUFFER: int = Field(default=60, ge=0, le=600)  # Seconds to wait after candle interval before using (ensures finalized data)
    ENABLE_PPO_GATE_MOMENTUM_VOTE: bool = Field(default=False) 
    ENABLE_RSI_GUARD_MOMENTUM_VOTE: bool = Field(default=False) 
    ENABLE_RMA_CLOUD_MOMENTUM_VOTE: bool = Field(default=False) 
    ENABLE_VWAP_MOMENTUM_VOTE: bool = Field(default=False)
    BRAIN_STABILITY_MIN_HISTORY: int = Field(default=3, ge=1, le=20, description="StabilityGate: min threshold history entries before gating kicks in")
    BRAIN_STABILITY_MAX_JUMP: float = Field(default=2.0, ge=0.1, le=10.0, description="StabilityGate: max allowed deviation (in score points) from median history")
    BRAIN_CUSUM_DRIFT_DELTA: float = Field(default=0.10, ge=0.01, le=0.50, description="CUSUM: sensitivity to WR shift (delta parameter)")
    BRAIN_CUSUM_THRESHOLD: float = Field(default=2.0, ge=0.5, le=10.0, description="CUSUM: alarm threshold (h parameter)")
    BRAIN_CUSUM_MIN_SAMPLE: int = 30
    BRAIN_FEE_PCT: float = Field(default=0.0006, ge=0.0, le=0.01, description="Taker fee per side (0.06%) used in EV/Kelly calculations")
    BRAIN_SLIPPAGE_PCT: float = Field(default=0.0003, ge=0.0, le=0.01, description="Estimated slippage per side used in EV/Kelly calculations")
    BRAIN_OOD_ENABLED: bool = Field(default=True, description="Vote-count OOD gate on/off")
    BRAIN_REPORT_ON_DEMAND: bool = Field(default=False, description="If true, force a brain report to be generated and sent on this run regardless of the normal BRAIN_REPORT_INTERVAL_RUNS cadence")
    BRAIN_ARCHIVE_SHALLOW: bool = Field(default=False, description="Set by the workflow on alert-only runs that check out just a few days of the outcome archive. The Brain must never analyse a truncated archive, so report generation is skipped when this is true")
    BRAIN_MAX_PLAN_ENTRIES: int = Field(default=3, ge=0, le=50) 
    BRAIN_REPAIR_SHOP_MAX: int = Field(default=3, ge=1, le=10) 
    BRAIN_EV_GATE_P_THRESHOLD: float = Field(default=0.85, ge=0.5, le=1.0, description="Min P(EV>0) required for a threshold to clear the EV gate")
    BRAIN_EV_GATE_P5_FLOOR: float = Field(default=-0.10, ge=-1.0, le=0.0, description="5th-percentile EV must be above this for the EV gate to pass")
    BRAIN_ACTION_GATE_ENABLED: bool = Field(default=True, description="If True, config patches are suppressed unless the multi-layer action gate passes")
    BRAIN_AUDIT_MIN_ROWS_FOR_RECOMMENDATION: int = Field(default=100, ge=20, le=1000, description="Minimum resolved rows before any recommendation can reach ACTIONABLE tier")
    OOD_MIN_HISTORY: int = Field(default=10, ge=5, le=100, description="Min historical samples before OOD gate activates")
    OOD_MARGIN: int = Field(default=2, ge=0, le=10, description="Extra votes allowed beyond 5th-95th percentile before flagging as OOD")
    OOD_RELAXED_MODE: bool = Field(default=True, description="If True, use margin-based OOD check (tolerant of small deviations); if False, use strict percentile check")
    OOD_P5: int = Field(default=5, ge=1, le=50, description="Lower percentile for OOD range (5th)")
    OOD_P95: int = Field(default=95, ge=50, le=99, description="Upper percentile for OOD range (95th)")
    ENABLE_CALIBRATION_GATE: bool = Field(default=False) 
    CALIBRATION_BUCKET_PCT: float = Field(default=5.0, ge=1.0, le=20.0)
    CALIBRATION_MIN_SAMPLE: int = Field(default=15, ge=5, le=200)
    CALIBRATION_INCREMENTAL_MAX_ROWS: int = Field(default=80, ge=10, le=500)     
    CALIBRATION_SLACK: float = Field(default=0.05, ge=0.0, le=0.20) 
    ENABLE_ONLINE_CALIBRATION: bool = Field(default=True, description="Between full rebuilds, fold newly resolved outcomes into the existing calibration buckets.")
    ENABLE_OUTCOME_STREAM_TAIL: bool = Field(default=True, description="When BRAIN_USE_FILE_STORAGE is on, still write each resolved outcome to the capped (2000) Redis outcome/shadow streams. The online calibration fold reads only these streams, so without this it folds nothing. Set False to save Redis memory (online calibration then does nothing; the 2-hourly rebuild still works).")
    CALIBRATION_REFRESH_MAX_AGE_HOURS: float = Field(default=2.0, ge=0.5, le=48.0) 
    ENABLE_PORTFOLIO_HEAT_GATE: bool = Field(default=False) 
    MAX_CONCURRENT_POSITIONS: int = Field(default=6, ge=1, le=50)
    MAX_NET_DIRECTIONAL_POSITIONS: int = Field(default=4, ge=1, le=50) 
    PORTFOLIO_MAX_SAME_DIRECTION_PCT: float = Field(default=1.0, ge=0.10, le=1.0) 
    PORTFOLIO_POSITION_MAX_AGE_MIN: int = Field(default=180, ge=15, le=1440, description="A trade you tapped 'Took' counts as open for the heat gate until you reply with an exit price or this many minutes pass (default 180 = the 12-candle outcome horizon). Needs ENABLE_TAKE_SKIP_BUTTONS.")
    KILL_SWITCH_MODE: str = Field(default="off", pattern="^(off|shadow|live)$", description="off = kill switch not evaluated (unless the legacy ENABLE_KILL_SWITCH is true, which means live); shadow = evaluate and record would-be trips but never block dispatch; live = block dispatch while tripped")
    ENABLE_KILL_SWITCH: bool = Field(default=False)
    KILL_SWITCH_MAX_CONSECUTIVE_LOSSES: int = Field(default=6, ge=2, le=20)
    KILL_SWITCH_MAX_DRAWDOWN_PCT: float = Field(default=3.0, ge=0.5, le=20.0) 
    KILL_SWITCH_LOOKBACK_HOURS: int = Field(default=24, ge=1, le=168)
    KILL_SWITCH_COOLDOWN_HOURS: int = Field(default=12, ge=1, le=168) 
    ENABLE_FILL_RECONCILIATION: bool = Field(default=False) 
    ENABLE_HIERARCHICAL_COMBINATION_ANALYSIS: bool = Field(default=True, description="Adds a pair+direction+alert_key+regime breakdown to the brain report, using empirical-Bayes shrinkage toward each combo's alert+direction+regime parent so small leaf combinations don't overfit. Purely diagnostic — never changes live gating on its own")
    FAMILY_POOLING_MIN_SAMPLE: int = Field(default=25, ge=10, le=500, description="Minimum de-clustered outcomes a family+direction(+regime) pool needs before it can stand in for an alert key that has no EV model of its own")
    HIERARCHICAL_MIN_LEAF_SAMPLE: int = Field(default=15, ge=1, le=1000, description="Minimum raw sample size for a pair+direction+alert+regime leaf to be reported at all in hierarchical_combination_analysis — shrinkage still pulls it toward its parent above this floor, this just filters out leaves too thin to report on")
    HIERARCHICAL_SHRINKAGE_K: float = Field(default=20.0, ge=1.0, le=500.0, description="Equivalent-sample-size prior strength for hierarchical_combination_analysis's empirical-Bayes shrinkage — higher pulls leaf estimates harder toward their parent bucket regardless of the leaf's own sample size")
    ENABLE_ALERT_FAMILY_ANALYSIS: bool = Field(default=True, description="Treat alert families as separate learning entities with hierarchical family→pair→regime stats. Purely diagnostic.")
    ALERT_FAMILY_MIN_SAMPLE: int = Field(default=20, ge=5, le=1000, description="Minimum sample size for an alert-family bucket to be reported.")
    ENABLE_REGIME_TRANSITION_ANALYSIS: bool = Field(default=True, description="Detect regime transitions (TREND↔RANGE) and compare post-transition vs stable-regime alert outcomes. Purely diagnostic.")
    REGIME_TRANSITION_LOOKBACK_BARS: int = Field(default=4, ge=1, le=48, description="Consecutive same-regime outcomes required before a flip counts as a transition.")
    REGIME_TRANSITION_POST_WINDOW_HOURS: int = Field(default=6, ge=1, le=72, description="Hours after a transition during which an alert is tagged post-transition.")
    ENABLE_STRATEGY_VS_REGIME_ATTRIBUTION: bool = Field(default=True, description="When recent performance drops, attribute to strategy degradation vs under-represented current regime. Purely diagnostic.")
    STRATEGY_VS_REGIME_MIN_SAMPLE: int = Field(default=30, ge=10, le=1000, description="Minimum sample for strategy-vs-regime attribution buckets.")
    ENABLE_ENSEMBLE_DECISION: bool = Field(default=True, description="Blend hierarchical Bayesian + ML P(profit) + EV model + recent performance into one calibrated ensemble layer. Advisory by default.")
    ENSEMBLE_WEIGHT_BAYESIAN: float = Field(default=0.30, ge=0.0, le=1.0)
    ENSEMBLE_WEIGHT_ML: float = Field(default=0.30, ge=0.0, le=1.0)
    ENSEMBLE_WEIGHT_EV: float = Field(default=0.25, ge=0.0, le=1.0)
    ENSEMBLE_WEIGHT_RECENT: float = Field(default=0.15, ge=0.0, le=1.0)
    ENSEMBLE_GATE_MODE: str = Field(default="shadow", pattern="^(off|shadow|live)$", description="Ensemble quality gate. off = ignored; shadow = annotate the trade-quality result with what the gate WOULD do but change nothing; live = a low ensemble probability lowers the verdict one step (HIGH->MEDIUM, MEDIUM->LOW). Restrict-only: never raises a verdict and never overrides a BLOCKED/LOW verdict. Needs OOS-validated evidence and ENSEMBLE_GATE_MIN_N resolved trades.")
    ENSEMBLE_GATE_MIN_N: int = Field(default=100, ge=30, le=5000, description="Minimum out-of-sample resolved trades before the ensemble gate may act.")
    ENSEMBLE_GATE_MAX_P: float = Field(default=0.40, ge=0.05, le=0.60, description="Ensemble probability at or below this lowers the verdict one step (when the gate is eligible and not off).")
    EPISTEMIC_GATE_MODE: str = Field(default="shadow", pattern="^(off|shadow|live)$", description="Novelty / missing-feature check on the market-state model input. off = ignored; shadow = annotate what the check WOULD do but change nothing; live = a novel input lowers the verdict one step (HIGH->MEDIUM, MEDIUM->LOW). Restrict-only: never raises a verdict.")
    UNCERTAINTY_GATE_MODE: str = Field(default="shadow", pattern="^(off|shadow|live)$", description="Future Edge Score gate. The score is net EV minus its uncertainty (sampling error of the EV estimate plus disagreement between the probability sources). off = not computed; shadow = annotate what the gate WOULD do; live = a verdict whose edge does not survive the uncertainty is lowered one step (HIGH->MEDIUM when the score is not positive; MEDIUM->LOW when net EV itself is not positive). Restrict-only: never raises a verdict.")
    UNCERTAINTY_GATE_MIN_N: int = Field(default=100, ge=20, le=5000, description="Minimum out-of-sample trades behind an alert key before the Future Edge Score gate may act (thin evidence is already handled by the evidence cap).")
    HYPOTHESIS_GEN_MODE: str = Field(default="shadow", pattern="^(off|shadow)$", description="Self-generated 'skip when' hypotheses. off = nothing is generated; shadow = each report proposes up to HYPOTHESIS_BUDGET new candidate rules, tests them on older-then-newer trades with a confidence level tightened for every rule ever tried, and follows survivors forward in time. Reported only: a confirmed rule is a candidate for human review and is never applied.")
    HYPOTHESIS_BUDGET: int = Field(default=300, ge=20, le=3000, description="Maximum new candidate rules proposed per report. Every one counts toward the experiment ledger and tightens the evidence needed for all later rules.")
    HYPOTHESIS_SHADOW_MAX: int = Field(default=10, ge=1, le=50, description="Maximum rules followed forward at once.")
    HYPOTHESIS_FORWARD_MIN_N: int = Field(default=30, ge=10, le=500, description="Forward trades a followed rule needs before it can be confirmed or rejected.")
    HYPOTHESIS_EXPIRE_DAYS: int = Field(default=30, ge=7, le=180, description="A followed rule that has not gathered enough forward trades within this many days is dropped.")
    LAYER_BOUNDARY_MODE: str = Field(default="report", pattern="^(off|report|block)$", description="Research / Validation / Policy boundary check (brain_layers.py). off = not run; report = the Brain report lists any research or validation module that changes live behaviour outside the documented exceptions; block = additionally, a stored plan is NOT applied while such a violation exists. Never loosens anything.")
    EPISTEMIC_Z_LIMIT: float = Field(default=4.0, ge=2.0, le=10.0, description="A numeric feature further than this many training standard deviations from its training mean counts as extreme.")
    EPISTEMIC_MAX_EXTREME: int = Field(default=2, ge=1, le=20, description="This many extreme features (or more) marks the input as novel.")
    EPISTEMIC_MAX_MISSING_FRAC: float = Field(default=0.34, ge=0.05, le=1.0, description="A larger share of missing numeric features than this marks the input as novel.")
    MARKET_STATE_REQUIRE_BASELINE_SKILL: bool = Field(default=False, description="When True, a retrained market-state model is only accepted if its holdout Brier score beats the plain base-rate baseline. The skill figure is always recorded on the model, whatever this is set to.")
    MARKET_STATE_MIN_HORIZONS_POSITIVE: int = Field(default=0, ge=0, le=4, description="A retrained market-state model is only accepted when its edge is positive at this many of the four label horizons (4/6/8/12 candles) on newer trades. 0 = the check is recorded and reported but never rejects a model.")
    FAMILY_ACCOUNTABILITY_MODE: str = Field(default="shadow", pattern="^(off|shadow|live)$", description="Recommendation-family accountability. off = ignored; shadow = report which families WOULD be suspended and tag plan items, change nothing; live = items from a suspended family are dropped from the stored plan. Restrict-only: never adds a change.")
    ACCOUNTABILITY_CONTROL_MODE: str = Field(default="shadow", pattern="^(off|shadow|live)$", description="Same-period comparison group for repair verdicts. off = not computed; shadow = each measured repair also gets a difference-in-differences check against trades the repair did not touch (stored and reported, verdicts unchanged); live = family accountability treats a verdict as neutral when the comparison group moved the same way. Never changes auto-rollback.")
    FAMILY_ACCOUNTABILITY_WINDOW_DAYS: int = Field(default=60, ge=7, le=365, description="Only repair verdicts from the last N days count toward a family's record.")
    FAMILY_HURT_STREAK: int = Field(default=3, ge=2, le=10, description="This many most-recent verdicts in a row all 'hurt' suspends the family.")
    FAMILY_SUSPEND_DAYS: int = Field(default=14, ge=1, le=180, description="A suspended family returns to probation this many days after its last 'hurt' verdict.")
    PREDICTION_AUDIT_MIN_N: int = Field(default=30, ge=10, le=1000, description="Minimum resolved alerts carrying a stored prediction before the prediction track record states a skill verdict.")
    NET_EV_AUDIT_MIN_TEST: int = Field(default=40, ge=20, le=2000, description="Minimum newest (test) trades before the market-state model is compared with the chronological baselines in net P&L terms. Report-only; changes no decision.")
    NET_EV_AUDIT_HORIZON: int = Field(default=12, ge=4, le=12, description="Candles after the fill at which the net-EV audit scores each trade (the stored path is 12 candles long).")
    ENABLE_BRAIN_PLAN_IDS: bool = Field(default=True, description="Stamp every Brain pending plan with an explicit PLAN-NNN id, model version, data window, and parameter diffs.")
    ENABLE_ALERT_WHY_REJECTED: bool = Field(default=True, description="When Brain/quality/ML-EV blocks an alert, emit structured 'why rejected' reason (symmetric to ENABLE_ALERT_WHY_SURVIVED).")
    SAMPLE_INSUFFICIENT_N: int = Field(default=15, ge=1, le=500, description="n < this → INSUFFICIENT evidence state (no adjustment).")
    SAMPLE_SHADOW_N: int = Field(default=50, ge=5, le=2000, description="minimum ≤ n < this → SHADOW (monitor / simulate only).")
    SAMPLE_ELIGIBLE_N: int = Field(default=100, ge=20, le=5000, description="robust sample floor; with OOS validation becomes ACTIONABLE.")
    ENABLE_MAE_MFE_TRADE_PLAN: bool = Field(default=True, description="Attach a historical-MAE/MFE-derived SL/TP1/TP2 suggestion ... Advisory only — never changes the bracket used to grade an alert's own win/loss.")
    MAE_MFE_MIN_SAMPLE: int = Field(default=15, ge=1, le=1000, description="Minimum sample per bucket before falling back to the next-broader one, down to global.")
    MAE_MFE_SL_PERCENTILE: float = Field(default=70.0, ge=50.0, le=95.0, description="...")
    MAE_MFE_TP1_PERCENTILE: float = Field(default=60.0, ge=40.0, le=90.0, description="...")
    MAE_MFE_TP2_PERCENTILE: float = Field(default=85.0, ge=60.0, le=99.0, description="...")
    MAE_MFE_SL_MIN_PCT: float = Field(default=0.15, ge=0.01, le=5.0, description="Safety floor ...")
    MAE_MFE_SL_MAX_PCT: float = Field(default=3.0, ge=0.1, le=20.0, description="Safety ceiling ...")
    ENABLE_ML_EV_SHADOW: bool = Field(default=False, description="If True, compute per-trade calibrated EV + qualification at dispatch and log it; never blocks. Requires a persisted ML calibration curve with acceptable ECE.")
    ML_EV_MIN_THRESHOLD: float = Field(default=0.0, ge=-1.0, le=2.0, description="Shadow-only EV floor used for the qualification verdict label (not a hard gate yet).")
    ENABLE_ML_EV_GATE: bool = Field(default=False, description="HARD gate: block dispatch when per-trade calibrated EV < ML_EV_MIN_THRESHOLD. Keep False until shadow mode has been observed for a full analysis window.")
    BRAIN_AUTO_ROLLBACK_HURT: bool = Field(default=True) 
    BRAIN_ROLLBACK_MIN_HOURS: float = Field(default=24.0, ge=1.0, le=720.0, description="Do not judge an applied plan until this many hours have passed.")
    BRAIN_ROLLBACK_MONITOR_DAYS: float = Field(default=14.0, ge=1.0, le=90.0, description="How long an applied plan stays under watch before it is marked cleared.")
    BRAIN_ROLLBACK_MIN_N: int = Field(default=30, ge=10, le=500, description="Minimum outcomes required BOTH before and after the apply to judge harm.")
    BRAIN_ROLLBACK_MIN_WR_DROP: float = Field(default=0.08, ge=0.02, le=0.5, description="Minimum win-rate drop (absolute) after apply to count as harm.")
    BRAIN_ROLLBACK_ALPHA: float = Field(default=0.05, ge=0.001, le=0.2, description="One-sided significance level for the post-apply win-rate drop.")
    BRAIN_ROLLBACK_MAX_PER_DAY: int = Field(default=1, ge=1, le=10, description="Hard cap on automatic rollbacks per rolling 24h.")
    ENABLE_ALERT_WHY_SURVIVED: bool = Field(default=True, description="Learner digest adds a 'gate margin' section: do alerts that clear the confluence floor by more actually do better? Evidence for why an alert survives, not a Telegram line.")
    BRAIN_FILTER_TELEGRAM_COOLDOWN_SEC: int = Field(default=3600, ge=60, le=86400, description="Minimum seconds between BRAIN FILTER messages for the same pair + alert + gate.")
    ENABLE_BRAIN_FILTER_TELEGRAM: bool = Field(default=False, description="Send a Telegram 'BRAIN FILTER' message when the quality, calibration, ML-EV or OOD gate blocks an alert that passed the signal gates.")
    BRAIN_FILTER_TELEGRAM_MAX_PER_RUN: int = Field(default=3, ge=1, le=20, description="Cap on BRAIN FILTER Telegram messages per bot run.")
    ENABLE_EVIDENCE_VERDICT_CAP: bool = Field(default=True, description="If True, a trade-quality verdict is capped by sample evidence: INSUFFICIENT evidence → LOW, SHADOW evidence → at most MEDIUM. Advisory label only — never changes signal gates.") 
    MEMORY_SOFT_STOP_RATIO: float = Field(default=0.80, ge=0.5, le=0.95) 
    ENABLE_CHAMPION_CHALLENGER: bool = Field(default=False, description="If True, weight-optimizer / root-cause weight changes are stored as challenger weights in Redis instead of going live. Periodic promotion is checked at the end of each Brain report via maybe_promote_challenger().")
    CHALLENGER_MIN_OOS_SAMPLE: int = Field(default=80, ge=20, le=5000, description="Minimum OOS sample size required before challenger can promote to champion.")
    CHALLENGER_MIN_EV_LIFT: float = Field(default=0.02, ge=0.0, le=1.0, description="Minimum net-EV lift (challenger - champion) required for auto-promotion.")
    CHALLENGER_SHADOW_ONLY: bool = Field(default=True, description="If True, maybe_promote_challenger() never auto-applies (returns shadow_only_requires_force). Set False to allow gated auto-promotion when ENABLE_CHAMPION_CHALLENGER is on.")
    ENFORCE_SINGLE_WEIGHT_PATH: bool = Field(default=True, description="If True, live CONFLUENCE_WEIGHTS (Redis dynamic_weights) can ONLY change through champion/challenger promotion or an auto-rollback restore. Brain plan-apply never writes live weights directly; it stores a challenger instead. Hard signal rules stay fixed.")
    ENABLE_BRAIN_SIZE_HINT: bool = Field(default=False)
    BRAIN_SIZE_HINT_HIGH: float = Field(default=1.0, ge=0.0, le=1.0)
    BRAIN_SIZE_HINT_MEDIUM: float = Field(default=0.5, ge=0.0, le=1.0)
    BRAIN_SIZE_HINT_LOW: float = Field(default=0.25, ge=0.0, le=1.0)
    BRAIN_SIZE_HINT_BLOCKED: float = Field(default=0.0, ge=0.0, le=1.0)
    ZONE_MODE: str = Field(default="live", pattern="^(off|advisory|live)$", description="Validated TP/SL zones. off = not computed; advisory = computed, persisted and reported but NOT attached to alerts; live = a PROMOTED zone replaces the advisory MAE/MFE plan in the alert and is labelled validated. Zones never change the fixed outcome bracket (OUTCOME_MAE_LOSS_PCT / OUTCOME_RR_TARGET) used for labels, and never place orders.")
    ZONE_MIN_N: int = Field(default=60, ge=30, le=5000, description="Minimum resolved trades with MAE/MFE in a bucket before a zone candidate is built.")
    ZONE_MIN_HOLDOUT: int = Field(default=20, ge=10, le=500, description="Minimum chronological holdout trades (after purge/embargo) used to validate a candidate out-of-sample.")
    ZONE_MIN_DELTA_PCT: float = Field(default=0.05, ge=0.0, le=2.0, description="Holdout mean net P&L per trade under the zone must beat the fixed bracket by at least this many percentage points.")
    ZONE_MIN_P_BETTER: float = Field(default=0.80, ge=0.5, le=0.99, description="Paired-difference probability that the zone truly beats the fixed bracket on the holdout.")
    ZONE_STABILITY_TOL: float = Field(default=0.35, ge=0.05, le=1.0, description="Max relative disagreement between the SL/TP1 learned on the training split and on the holdout split.")
    ZONE_MAX_DEVIATION: float = Field(default=2.0, ge=1.0, le=5.0, description="Safety rail: zone SL and TP1 must stay within [1/x, x] times the fixed bracket levels.")
    ZONE_MIN_RR: float = Field(default=1.0, ge=0.5, le=5.0, description="Safety rail: zone TP1 / SL must be at least this.")
    ZONE_PROMOTE_CONSECUTIVE: int = Field(default=2, ge=1, le=10, description="A candidate must pass validation on this many consecutive Brain reports before it is promoted; any failure resets the streak (demotes).")
    REGIME_GATE_MODE: str = Field(default="live", pattern="^(off|shadow|live)$", description="Regime-aware quality gate. off = ignored; shadow = annotate the trade-quality result with what the gate WOULD do but change nothing; live = the gate can lower the quality verdict (never raise it, never touch the hard signal gates). Inert until a segment has enough rows AND out-of-sample confirmation.")
    REGIME_GATE_MIN_N_DOWNGRADE: int = Field(default=50, ge=15, le=2000, description="Minimum resolved trades in an alert+direction+regime segment before it can DOWNGRADE a verdict to LOW.")
    REGIME_GATE_MIN_N_BLOCK: int = Field(default=100, ge=30, le=5000, description="Minimum resolved trades in the segment before it can BLOCK (also needs out-of-sample confirmation).")
    REGIME_GATE_MIN_HOLDOUT: int = Field(default=20, ge=10, le=500, description="Minimum chronological holdout rows (after purge/embargo) needed to confirm a BLOCK out-of-sample.")
    REGIME_GATE_DOWNGRADE_P: float = Field(default=0.35, ge=0.01, le=0.5, description="Segment P(net EV > 0) must be at or below this to DOWNGRADE.")
    REGIME_GATE_BLOCK_P: float = Field(default=0.20, ge=0.01, le=0.5, description="Segment AND holdout P(net EV > 0) must be at or below this to BLOCK.")
    REGIME_GATE_MIN_GAP_PCT: float = Field(default=0.10, ge=0.0, le=5.0, description="Segment net EV must be at least this many percentage points WORSE than the same alert+direction across all regimes. Keeps the gate regime-specific: an alert that is simply bad everywhere is the alert-level gate's job.")
    ENABLE_QUALITY_HARD_BLOCK: bool = Field(default=False, description="HARD gate: when trade-quality verdict is BLOCKED, suppress Telegram dispatch and record a counterfactual instead of only annotating the message. Keep False until quality labels have been observed in shadow for a full analysis window.")
    CHALLENGER_MIN_CONSECUTIVE_PASSES: int = Field(default=3, ge=1, le=20, description="Consecutive, separately-spaced Brain evaluations the SAME challenger must pass before auto-promotion. A failed evaluation or a new challenger resets the streak.")
    CHALLENGER_STREAK_MIN_GAP_SEC: int = Field(default=3600, ge=60, le=86400, description="Minimum seconds between two evaluations that count toward the streak (maybe_promote_challenger runs more than once per Brain run).")
    ONLINE_MODEL_MODE: str = Field(default="shadow", pattern="^(off|shadow|live)$", description="Online-updated copy of the market-state model, compared with the batch model that was being served. off = not run; shadow = learn and compare each Brain report, record what WOULD be promoted, serve nothing; live = a copy that passes the gates is served instead of the batch model (only matters while ENABLE_MARKET_STATE_LIVE_SCORE is on). Updates happen in the Brain report job only, never on the 15-minute alert path.")
    ONLINE_MODEL_MIN_EVAL: int = Field(default=60, ge=20, le=2000, description="Minimum newly resolved trades scored by both models before the online copy can be judged.")
    ONLINE_MODEL_MAX_AGE_DAYS: int = Field(default=14, ge=1, le=90, description="The online copy restarts from the current batch model after this many days (and is demoted if it was promoted).")
    ONLINE_MODEL_MAX_DRIFT: float = Field(default=1.0, ge=0.1, le=10.0, description="Largest allowed change of any coefficient of the online copy from its starting batch model; beyond this it cannot be promoted and is demoted.")
    ENABLE_FEATURE_DATASET_CACHE: bool = Field(default=True, description="Keep the materialised point-in-time dataset in Redis (keyed by feature-set version and feature hash) and only build rows for newly resolved trades. A sample of reused rows is recomputed every run; any mismatch discards the cache.")
    FEATURE_DATASET_MAX_BYTES: int = Field(default=3000000, ge=100000, le=20000000, description="Largest packed dataset the cache will store. Bigger datasets are rebuilt each run instead.")
    ABLATION_NOISE_THRESHOLD: float = Field(default=0.01, ge=0.0, le=0.20, description="Permutation importance |imp| below this → condition treated as noise " "(candidate for weight reduction). Advisory until plan gate passes.") 
    ABLATION_EDGE_THRESHOLD: float = Field(default=0.03, ge=0.0, le=0.50, description="Permutation importance above this → condition treated as carrying real edge.") 
    PORTFOLIO_POSITION_SOURCE: str = Field(default="taps", pattern="^(taps|alerts|both)$", description="Where the heat gate learns which positions are open: 'taps' = trades you tapped Took on (needs ENABLE_TAKE_SKIP_BUTTONS); 'alerts' = alerts the bot actually delivered with a verdict in PORTFOLIO_AUTO_VERDICTS, assumed open for PORTFOLIO_POSITION_MAX_AGE_MIN; 'both' = union of the two.")
    PORTFOLIO_AUTO_VERDICTS: List[str] = Field(default_factory=lambda: ["TAKE"], description="Verdicts that count as an open position when PORTFOLIO_POSITION_SOURCE is 'alerts' or 'both'. AVOID/WATCH alerts are not trades you were told to take, so they do not fill the book by default.")

    @field_validator('TELEGRAM_BOT_TOKEN')
    def validate_token(cls, v: str) -> str:
        if not re.match(r'^\d+:[A-Za-z0-9_-]+$', v):
            raise ValueError('Invalid Telegram bot token format')
        return v

    @field_validator('TELEGRAM_CHAT_ID')
    def validate_chat_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Chat ID cannot be empty')
        return v.strip()

    @field_validator('PIVOT_LOOKBACK_PERIOD')
    def validate_pivot_lookback(cls, v: int) -> int:
        if v < 5:
            raise ValueError(
                f'PIVOT_LOOKBACK_PERIOD must be >= 5 (need minimum historical data), got {v}'
            )
        if v > 365:
            raise ValueError(
                f'PIVOT_LOOKBACK_PERIOD > 365 days is excessive, got {v}'
            )
        return v

    @field_validator('DELTA_API_BASE')
    def validate_api_base(cls, v: str) -> str:
        if not re.match(r'^(https?://)[A-Za-z0-9\.\-:_/]+$', v.strip()):
            raise ValueError('DELTA_API_BASE must be a valid http(s) URL')
        return v.strip().rstrip('/')

    @field_validator('PPO_FAST', 'PPO_SLOW', 'PPO_SIGNAL')
    @classmethod
    def validate_ppo_params(cls, v):
        if not (1 <= v <= 100):
            raise ValueError(f'PPO parameter must be 1-100, got {v}')
        return v

    @model_validator(mode='after')
    def validate_adaptive_rvol(self) -> 'BotConfig':
        if self.ATR_SHORT >= self.ATR_LONG:
            raise ValueError(
                f'ATR_SHORT ({self.ATR_SHORT}) must be < ATR_LONG ({self.ATR_LONG}) '
                f'— the RVOL ratio assumes short-period ATR is compared against a longer baseline'
            )
        if self.ATR_ADAPTIVE_ENABLED:
            if self.ATR_PCTL_MIN_HISTORY >= self.ATR_PCTL_LOOKBACK:
                raise ValueError(
                    f'ATR_PCTL_MIN_HISTORY ({self.ATR_PCTL_MIN_HISTORY}) must be < '
                    f'ATR_PCTL_LOOKBACK ({self.ATR_PCTL_LOOKBACK})'
                )
        if self.ENABLE_VOLUME_PCTL_VOTE and self.VOLUME_PCTL_MIN_HISTORY >= self.VOLUME_PCTL_LOOKBACK:
            raise ValueError(
                f'VOLUME_PCTL_MIN_HISTORY ({self.VOLUME_PCTL_MIN_HISTORY}) must be < '
                f'VOLUME_PCTL_LOOKBACK ({self.VOLUME_PCTL_LOOKBACK})'
            )
        if self.ENABLE_ADX_STRENGTH_VOTE and self.ADX_STRENGTH_PCTL <= self.ADX_ADAPTIVE_TARGET_PCTL:
            raise ValueError(
                f'ADX_STRENGTH_PCTL ({self.ADX_STRENGTH_PCTL}) should be > '
                f'ADX_ADAPTIVE_TARGET_PCTL ({self.ADX_ADAPTIVE_TARGET_PCTL}), otherwise the '
                f'adx_strength vote duplicates the existing adx gate instead of adding a stricter bar'
            )
        if self.ADAPTIVE_MULT_CALM >= self.ADAPTIVE_MULT_VOLATILE:
            raise ValueError(
                f'ADAPTIVE_MULT_CALM ({self.ADAPTIVE_MULT_CALM}) must be < '
                f'ADAPTIVE_MULT_VOLATILE ({self.ADAPTIVE_MULT_VOLATILE})'
            )
        if self.PPO_ADAPTIVE_CALM >= self.PPO_ADAPTIVE_VOLATILE:
            raise ValueError(
                f'PPO_ADAPTIVE_CALM ({self.PPO_ADAPTIVE_CALM}) must be < '
                f'PPO_ADAPTIVE_VOLATILE ({self.PPO_ADAPTIVE_VOLATILE})'
            )
        if self.RSI_ADAPTIVE_BUY_CALM >= self.RSI_ADAPTIVE_BUY_VOLATILE:
            raise ValueError(
                f'RSI_ADAPTIVE_BUY_CALM ({self.RSI_ADAPTIVE_BUY_CALM}) must be < '
                f'RSI_ADAPTIVE_BUY_VOLATILE ({self.RSI_ADAPTIVE_BUY_VOLATILE})'
            )
        if self.RSI_ADAPTIVE_SELL_CALM <= self.RSI_ADAPTIVE_SELL_VOLATILE:
            raise ValueError(
                f'RSI_ADAPTIVE_SELL_CALM ({self.RSI_ADAPTIVE_SELL_CALM}) must be > '
                f'RSI_ADAPTIVE_SELL_VOLATILE ({self.RSI_ADAPTIVE_SELL_VOLATILE}) '
                f'— sell threshold drops as volatility rises'
            )
        if self.CPR_ADAPTIVE_CALM >= self.CPR_ADAPTIVE_VOLATILE:
            raise ValueError(
                f'CPR_ADAPTIVE_CALM ({self.CPR_ADAPTIVE_CALM}) must be < '
                f'CPR_ADAPTIVE_VOLATILE ({self.CPR_ADAPTIVE_VOLATILE})'
            )
        if self.ADX_ADAPTIVE_BAND_WIDTH > 0:
            lo = self.ADX_ADAPTIVE_TARGET_PCTL - self.ADX_ADAPTIVE_BAND_WIDTH / 2.0
            hi = self.ADX_ADAPTIVE_TARGET_PCTL + self.ADX_ADAPTIVE_BAND_WIDTH / 2.0
            if lo < 1.0 or hi > 99.0:
                raise ValueError(
                    f'ADX_ADAPTIVE_TARGET_PCTL ({self.ADX_ADAPTIVE_TARGET_PCTL}) ± '
                    f'band/2 ({self.ADX_ADAPTIVE_BAND_WIDTH / 2.0}) produces range '
                    f'[{lo:.1f}, {hi:.1f}] which exceeds [1, 99]'
                )
        return self

    @model_validator(mode='after')
    def validate_oi_divergence_window(self) -> 'BotConfig':
        if self.ENABLE_OI_PRICE_DIVERGENCE:
            required_age_sec = self.OI_DIVERGENCE_LOOKBACK_SAMPLES * 900  # 900s = 1 cycle @15m
            if self.OI_FUNDING_MAX_SAMPLE_AGE_SEC < required_age_sec:
                raise ValueError(
                    f'OI_FUNDING_MAX_SAMPLE_AGE_SEC ({self.OI_FUNDING_MAX_SAMPLE_AGE_SEC}s) is less than '
                    f'OI_DIVERGENCE_LOOKBACK_SAMPLES * 900 ({required_age_sec}s) — history will be pruned '
                    f'before the divergence lookback can be satisfied, so ENABLE_OI_PRICE_DIVERGENCE will '
                    f'silently never fire. Raise OI_FUNDING_MAX_SAMPLE_AGE_SEC or lower OI_DIVERGENCE_LOOKBACK_SAMPLES.'
                )
        return self

    @model_validator(mode='after')
    def validate_reenable_deadband(self) -> 'BotConfig':
        if self.BRAIN_REENABLE_MIN_WR_LO < self.BRAIN_ALERT_DISABLE_THRESHOLD_WR:
            raise ValueError(
                f'BRAIN_REENABLE_MIN_WR_LO ({self.BRAIN_REENABLE_MIN_WR_LO}) must be >= '
                f'BRAIN_ALERT_DISABLE_THRESHOLD_WR ({self.BRAIN_ALERT_DISABLE_THRESHOLD_WR}); '
                f'otherwise an alert key can be disabled and re-enabled on the same evidence.'
            )
        return self

    @model_validator(mode='after')
    def validate_ppo_ordering(self) -> 'BotConfig':
        if self.PPO_FAST >= self.PPO_SLOW:
            raise ValueError(
                f'PPO_FAST ({self.PPO_FAST}) must be strictly less than '
                f'PPO_SLOW ({self.PPO_SLOW})'
            )

        if self.PPO_GATE_FAST >= self.PPO_GATE_SLOW:
            raise ValueError(
                f'PPO_GATE_FAST ({self.PPO_GATE_FAST}) must be strictly less than '
                f'PPO_GATE_SLOW ({self.PPO_GATE_SLOW})'
            )
        if self.ENABLE_HIST_RMA and self.HIST_RMA_FAST >= self.HIST_RMA_SLOW:
            raise ValueError(
                f'HIST_RMA_FAST ({self.HIST_RMA_FAST}) must be strictly less than '
                f'HIST_RMA_SLOW ({self.HIST_RMA_SLOW})'
            )

        if self.PULLBACK_MIN_RETRACE >= self.PULLBACK_MAX_RETRACE:
            raise ValueError(
                f'PULLBACK_MIN_RETRACE ({self.PULLBACK_MIN_RETRACE}) must be strictly less than '
                f'PULLBACK_MAX_RETRACE ({self.PULLBACK_MAX_RETRACE})'
            )
        if self.EXHAUSTION_VERDICT_WATCH_SCORE > self.EXHAUSTION_VERDICT_AVOID_SCORE:
            raise ValueError(
                f'EXHAUSTION_VERDICT_WATCH_SCORE ({self.EXHAUSTION_VERDICT_WATCH_SCORE}) must not exceed '
                f'EXHAUSTION_VERDICT_AVOID_SCORE ({self.EXHAUSTION_VERDICT_AVOID_SCORE})'
            )
        if self.RMA_CLOUD_ENABLED and self.RMA_CLOUD_FAST_PERIOD >= self.RMA_50_PERIOD:
            raise ValueError(
                f'RMA_CLOUD_FAST_PERIOD ({self.RMA_CLOUD_FAST_PERIOD}) must be strictly less than '
                f'RMA_50_PERIOD ({self.RMA_50_PERIOD}), since the cloud slow leg reuses RMA_50_PERIOD'
            )
        return self

    @model_validator(mode='after')
    def validate_confluence_floor(self) -> 'BotConfig':
        if self.ENABLE_CONFLUENCE_GATE:
            max_achievable = sum(CONFLUENCE_WEIGHTS.values())
            if self.CONFLUENCE_MIN_ABS_SCORE > max_achievable:
                raise ValueError(
                    f'CONFLUENCE_MIN_ABS_SCORE ({self.CONFLUENCE_MIN_ABS_SCORE}) exceeds the max '
                    f'achievable weighted total ({max_achievable}) — every alert would be blocked forever'
                )
        return self

    @model_validator(mode='after')
    def validate_rr_consistency(self) -> 'BotConfig':
       
        if self.OUTCOME_MAE_LOSS_PCT <= 0:
            raise ValueError(
                f'OUTCOME_MAE_LOSS_PCT ({self.OUTCOME_MAE_LOSS_PCT}) '
                f'must be > 0 — it defines the stop-loss distance'
            )
        if self.OUTCOME_RR_TARGET < 1.0:
            raise ValueError(
                f'OUTCOME_RR_TARGET ({self.OUTCOME_RR_TARGET}) '
                f'must be >= 1.0'
            )
        if self.OUTCOME_BONUS_RR <= self.OUTCOME_RR_TARGET:
            raise ValueError(
                f'OUTCOME_BONUS_RR ({self.OUTCOME_BONUS_RR}) must be > '
                f'OUTCOME_RR_TARGET ({self.OUTCOME_RR_TARGET})'
            )
        return self

    @model_validator(mode='after')
    def validate_logic(self) -> 'BotConfig':   
        errors = []
        warnings = []

        if self.RUN_TIMEOUT_SECONDS < self.MIN_RUN_TIMEOUT:
            errors.append(
                f'RUN_TIMEOUT_SECONDS ({self.RUN_TIMEOUT_SECONDS}s) must be >= '
                f'MIN_RUN_TIMEOUT ({self.MIN_RUN_TIMEOUT}s)'
            )

        if self.RUN_TIMEOUT_SECONDS >= self.REDIS_LOCK_EXPIRY:
            errors.append(
                f'REDIS_LOCK_EXPIRY ({self.REDIS_LOCK_EXPIRY}s) must be > '
                f'RUN_TIMEOUT_SECONDS ({self.RUN_TIMEOUT_SECONDS}s)'
            )

        if self.TELEGRAM_RATE_LIMIT_PER_MINUTE < 10 or self.TELEGRAM_RATE_LIMIT_PER_MINUTE > 30:
            errors.append('TELEGRAM_RATE_LIMIT_PER_MINUTE must be 10-30')

        if self.ENABLE_PIVOT and self.PIVOT_MAX_DISTANCE_PCT < 1.0:
            errors.append('PIVOT_MAX_DISTANCE_PCT should be >= 1.0 for meaningful alerts')

        ranges = {
            'RMA_50_PERIOD': (self.RMA_50_PERIOD, 20, 100),
            'RMA_200_PERIOD': (self.RMA_200_PERIOD, 100, 300),
            'SRSI_RSI_LEN': (self.SRSI_RSI_LEN, 5, 50),
            'SRSI_KALMAN_LEN': (self.SRSI_KALMAN_LEN, 2, 20),
        }
        
        for name, (val, min_v, max_v) in ranges.items():
            if not (min_v <= val <= max_v):
                errors.append(f'{name} must be {min_v}-{max_v}, got {val}')

        if self.ENABLE_ALERT_COALESCING and self.COALESCE_DEDUP_WINDOW_SEC > 900:
            warnings.append(
                f'COALESCE_DEDUP_WINDOW_SEC={self.COALESCE_DEDUP_WINDOW_SEC} is longer than one 15m candle: '
                f'a same-direction alert on the NEXT candle is suppressed. Use <= 900 to dedup within a candle only.'
            )

        if self.MAX_ALERTS_PER_PAIR > 15:
            warnings.append(
                f'MAX_ALERTS_PER_PAIR={self.MAX_ALERTS_PER_PAIR} is very high, may cause spam'
            )

        if self.MAX_PARALLEL_FETCH < 1 or self.MAX_PARALLEL_FETCH > 20:
            warnings.append(
                f'MAX_PARALLEL_FETCH={self.MAX_PARALLEL_FETCH} is outside recommended range (1-20)'
            )

        if self.HTTP_TIMEOUT < 5 or self.HTTP_TIMEOUT > 60:
            warnings.append(
                f'HTTP_TIMEOUT={self.HTTP_TIMEOUT}s is outside recommended range (5-60s)'
            )

        min_batches = -(-len(self.PAIRS) // self.MAX_PARALLEL_FETCH)  # ceil division
        estimated_runtime = min_batches * Constants.INTER_BATCH_DELAY * 100  # heuristic, recalibrate with observed data
        safe_fraction = 0.8

        if min_batches > 3 or estimated_runtime > self.RUN_TIMEOUT_SECONDS * safe_fraction:
            warnings.append(
                f"PAIRS={len(self.PAIRS)} with MAX_PARALLEL_FETCH={self.MAX_PARALLEL_FETCH} "
                f"requires {min_batches} sequential fetch batches. "
                f"Estimated runtime ~{int(estimated_runtime)}s may exceed safe window "
                f"({int(self.RUN_TIMEOUT_SECONDS * safe_fraction)}s of RUN_TIMEOUT_SECONDS={self.RUN_TIMEOUT_SECONDS}s). "
                f"Verify actual runtime before adding more pairs."  
            )

        if self.MEMORY_LIMIT_BYTES < 200_000_000:
            warnings.append(
                f'MEMORY_LIMIT_BYTES={self.MEMORY_LIMIT_BYTES} is very low '
                f'(minimum recommended: 200MB)'
            )

        if self.RVOL_THRESHOLD < 0.5 or self.RVOL_THRESHOLD > 2.0:
            errors.append(f'RVOL_THRESHOLD {self.RVOL_THRESHOLD} outside range [0.5, 2.0]')

        if self.MAX_CANDLE_STALENESS_SEC < 300:
            warnings.append(f'MAX_CANDLE_STALENESS_SEC very low ({self.MAX_CANDLE_STALENESS_SEC}s)')

        # Every 15m indicator must produce a valid value at the last closed candle
        # (index FETCH_LIMIT_15M - 2) inside the fetched window. Otherwise it is
        # silently NaN and its gate evaluates False for every pair.
        last_idx = Constants.FETCH_LIMIT_15M - 2
        warmups = {
            'ICHIMOKU cloud 23/65/130/65 (SPANB+DISPLACEMENT-2)':
                self.BIAS_ICHIMOKU_SPANB_PERIODS + self.BIAS_ICHIMOKU_DISPLACEMENT - 2
                if self.ICHIMOKU_CLOUD_ENABLED else 0,
            'GAUSSIAN trend (5 x LENGTH + EFF_LEN + 3 x EFF_SMOOTH + ATR_LEN)':
                5 * self.GAUSSIAN_LENGTH + self.GAUSSIAN_EFFICIENCY_LENGTH
                + 3 * self.GAUSSIAN_EFFICIENCY_SMOOTH + self.GAUSSIAN_ATR_LENGTH
                if self.GAUSSIAN_TREND_ENABLED else 0,
            'ICHIMOKU TK lines (TK_BASE-1)':
                self.ICHIMOKU_TK_BASE_PERIODS - 1
                if (self.ENABLE_TK_CONVERSION_CROSS or self.ENABLE_KIJUN_CROSS) else 0,
            'MOMENTUM_OSCILLATOR (2 x LENGTH + sqrt(LENGTH))':
                2 * self.MOMENTUM_OSCILLATOR_LENGTH + int(round(self.MOMENTUM_OSCILLATOR_LENGTH ** 0.5))
                if self.MOMENTUM_OSCILLATOR_ENABLED else 0,
            'BIAS_ICHIMOKU cloud (SPANB+DISPLACEMENT-2)':
                self.BIAS_ICHIMOKU_SPANB_PERIODS + self.BIAS_ICHIMOKU_DISPLACEMENT - 2
                if (self.ENABLE_BIAS_HEADER and not self.BIAS_USE_CIRRUS) else 0,
            'CIRRUS cloud (3 x slowest EMA span to settle)':
                3 * (2 * max(self.CIRRUS_X1, self.CIRRUS_X3) - 1)
                if (self.CIRRUS_CLOUD_ENABLED or (self.ENABLE_BIAS_HEADER and self.BIAS_USE_CIRRUS)) else 0,
            'RSI_GUARD (RSI+KALMAN+EMA lengths)':
                self.RSI_GUARD_RSI_LEN + self.RSI_GUARD_KALMAN_LEN + self.RSI_GUARD_EMA_LEN
                if self.RSI_GUARD_ENABLED else 0,
            'DYNAMIC_FLOW_DIST_LENGTH':
                self.DYNAMIC_FLOW_DIST_LENGTH if self.DYNAMIC_FLOW_RIBBON_ENABLED else 0,
        }
        for label, needed in warmups.items():
            if needed > last_idx:
                errors.append(
                    f'{label} needs {needed} candles but only {Constants.FETCH_LIMIT_15M} 15m candles '
                    f'are fetched (Constants.FETCH_LIMIT_15M); the gate would be permanently false'
                )
            elif needed > last_idx - 50:
                warnings.append(
                    f'{label} needs {needed} of {Constants.FETCH_LIMIT_15M} fetched 15m candles '
                    f'(< 50 spare); raise Constants.FETCH_LIMIT_15M before lengthening it further'
                )

        if errors:
            error_msg = 'Configuration validation failed:\n  ' + '\n  '.join(errors)
            raise ValueError(error_msg)

        self._validation_warnings = warnings

        return self

def load_config() -> BotConfig:
    config_file = os.getenv("CONFIG_FILE", "config_macd.json")
    data: Dict[str, Any] = {}
    if Path(config_file).exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json_loads(f.read())

        except Exception as exc:
            error_msg = f"❌ ERROR: Config file {config_file} is not valid JSON: {exc}"
            print(error_msg, file=sys.stderr)
            sys.exit(1)    
    else:
        print(f"⚠️ WARNING: Config file {config_file} not found, using environment variables only", file=sys.stderr)

    for field_name, field_info in BotConfig.model_fields.items():
        env_value = os.getenv(field_name)
        if env_value is None:
            continue
        if field_info.annotation is str:
            data[field_name] = env_value  # never JSON-decode str fields (e.g. all-digit chat IDs)
        else:
            try:
                data[field_name] = json_loads(env_value)
            except Exception:
                data[field_name] = env_value

    for key in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "REDIS_URL", "DELTA_API_BASE"):
        val = data.get(key, "")
        if not val or val.startswith("__SET_IN_"):
            print(f"❌ ERROR: Missing required config: {key}", file=sys.stderr)
            print("❌ Set this in your CI/CD secrets (GitHub Actions → Secrets, GitLab → Variables)", file=sys.stderr)
            sys.exit(1)
    try:
        return BotConfig(**data)
    except Exception as exc:
        print("❌ ERROR: Pydantic validation failed", file=sys.stderr)
        print(f"❌ Details: {exc}", file=sys.stderr)
        sys.exit(1)

cfg = load_config()

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("macd_bot")
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    level = logging.DEBUG if cfg.DEBUG_MODE else getattr(logging, cfg.LOG_LEVEL, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)    
    console.setFormatter(SafeFormatter(
        fmt='%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | [%(trace_id)s] | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    # REMOVED: console.addFilter(SecretFilter())  -- SafeFormatter already redacts
    console.addFilter(TraceContextFilter())  
    logger.addHandler(console)
    logger.debug(
        f"Logging configured | Level: {logging.getLevelName(level)} | "
        f"Format: structured with trace_id | Output: stdout"
    )
    return logger

logger = setup_logging()

logger_main = logger

def format_ist_time(dt_or_ts: Any = None, fmt: str = "%Y-%m-%d %H:%M:%S IST") -> str:
    try:
        if dt_or_ts is None:
            dt = datetime.now(timezone.utc)

        elif isinstance(dt_or_ts, datetime):
            dt = dt_or_ts
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        else:
            try:
                ts = float(dt_or_ts)
                if ts > 1_000_000_000_000:
                    ts /= 1000
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            except (ValueError, TypeError):
                dt = datetime.fromisoformat(str(dt_or_ts))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(_IST_TZ).strftime(fmt)
    except Exception as e:
        if cfg.DEBUG_MODE:
            logger.debug(f"format_ist_time parsing failed for '{dt_or_ts}': {e}")
        return str(dt_or_ts)

def _get_session_from_ts(ts: Any) -> str:
    """Maps a unix timestamp (or datetime) to its IST trading session:
    asian / london / ny / dead. The four ranges below overlap by design
    (liquidity handoffs between sessions aren't clean cuts) — resolved by
    checking Asian -> London -> NY in that order, so e.g. 14:00 IST (inside
    both Asian 09:30-15:30 and London 13:30-22:00) tags as 'asian'. Anything
    not covered by the first three (03:00-09:30 IST) is the Dead Zone."""
    try:
        if isinstance(ts, datetime):
            dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        else:
            t = float(ts)
            if t > 1_000_000_000_000:
                t /= 1000
            dt = datetime.fromtimestamp(t, tz=timezone.utc)
        ist = dt.astimezone(_IST_TZ)
        minutes = ist.hour * 60 + ist.minute

        def _in_range(start_h: int, start_m: int, end_h: int, end_m: int) -> bool:
            start, end = start_h * 60 + start_m, end_h * 60 + end_m
            if start <= end:
                return start <= minutes < end
            return minutes >= start or minutes < end  # wraps past midnight (NY session)

        if _in_range(9, 30, 15, 30):
            return "asian"
        if _in_range(13, 30, 22, 0):
            return "london"
        if _in_range(18, 0, 3, 0):
            return "ny"
        return "dead"
    except Exception as e:
        if cfg.DEBUG_MODE:
            logger.debug(f"_get_session_from_ts parsing failed for '{ts}': {e}")
        return "dead"

shutdown_event = asyncio.Event()

_pair_eval_counter = 0

_VALIDATION_DONE = False

def validate_runtime_config() -> None:
    global _VALIDATION_DONE
    if _VALIDATION_DONE:       
        return   
    errors = []
    warnings = []
    if hasattr(cfg, '_validation_warnings'):
        warnings.extend(cfg._validation_warnings)
    
    try:
        from urllib.parse import urlparse
        parsed = urlparse(cfg.REDIS_URL)
        if parsed.scheme not in ('redis', 'rediss'):
            errors.append(f"Invalid REDIS_URL scheme: {parsed.scheme} (must be redis:// or rediss://)")
        if not parsed.hostname:
            errors.append("REDIS_URL missing hostname")
    except Exception as e:
        errors.append(f"Failed to parse REDIS_URL: {e}")
    
    if errors:
        logger.critical("Configuration validation FAILED:")
        for error in errors:
            logger.critical(f"  ERROR: {error}")
        raise ValueError(f"Configuration validation failed with {len(errors)} error(s)")
    
    if warnings:
        logger.warning("Configuration warnings:")
        for warning in warnings:
            logger.warning(f"  WARNING: {warning}")
    
    logger.info(
        f"Configuration validated successfully | "
        f"Pairs: {len(cfg.PAIRS)} | Workers: {cfg.MAX_PARALLEL_FETCH} | "
        f"Timeout: {cfg.RUN_TIMEOUT_SECONDS}s"
    )    
    _VALIDATION_DONE = True

# Names moved to other modules, re-exported for backward compatibility.
from config_base import (
    JSONDecodeError,
    JSON_BACKEND,
    json_dumps,
    normalize_timestamp,
    normalize_timestamp_array,
    CprNotReadyError,
    __version__,
    CONFIG_OVERRIDE_ALLOWED_FIELDS,
    BRAIN_DISABLED_KEYS_METADATA_KEY,
    CONFIG_OVERRIDE_METADATA_KEY,
    PAIR_THRESHOLDS_METADATA_KEY,
    PIVOT_LEVELS_BUY,
    PIVOT_LEVELS_SELL,
    BtcMacroContext,
    ClusterContext,
    BiasContext,
    CompiledPatterns,
    TRACE_ID,
    PAIR_ID,
    MEMORY_CHECK_INTERVAL_PAIRS,
)

__all__ = [
    "JSONDecodeError",
    "JSON_BACKEND",
    "json_dumps",
    "json_loads",
    "normalize_timestamp",
    "normalize_timestamp_array",
    "CprNotReadyError",
    "__version__",
    "CONFLUENCE_WEIGHTS",
    "CONFIG_OVERRIDE_ALLOWED_FIELDS",
    "BRAIN_DISABLED_KEYS_METADATA_KEY",
    "CONFIG_OVERRIDE_METADATA_KEY",
    "PAIR_THRESHOLDS_METADATA_KEY",
    "Constants",
    "PIVOT_LEVELS_BUY",
    "PIVOT_LEVELS_SELL",
    "BtcMacroContext",
    "ClusterContext",
    "BiasContext",
    "CompiledPatterns",
    "TRACE_ID",
    "PAIR_ID",
    "BotConfig",
    "load_config",
    "cfg",
    "TraceContextFilter",
    "SafeFormatter",
    "setup_logging",
    "logger",
    "logger_main",
    "_IST_TZ",
    "format_ist_time",
    "_get_session_from_ts",
    "shutdown_event",
    "_pair_eval_counter",
    "MEMORY_CHECK_INTERVAL_PAIRS",
    "_VALIDATION_DONE",
    "validate_runtime_config",
]
