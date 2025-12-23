import logging
import numpy as np

logger = logging.getLogger(__name__)

HAVE_AOT = False

try:
    import aot_indicators as AOT
    HAVE_AOT = True
    logger.info("✅ Loaded aot_indicators.so successfully")
except ImportError:
    logger.info("ℹ️ aot_indicators.so not found, falling back to njit functions")

# Import njit fallbacks from macd_unified
from macd_unified import (
    _sanitize_array_numba,
    _ema_loop,
    _ema_loop_alpha,
    _sma_loop,
    _rolling_mean_numba,
    _rolling_std_welford,
    _rolling_min_numba,
    _rolling_max_numba,
    _kalman_loop,
    _vwap_daily_loop,
    _rng_filter_loop,
    _smooth_range,
    _calc_mmh_worm_loop,
    _calc_mmh_value_loop,
    _calc_mmh_momentum_loop,
    _calculate_rsi_core,
    _calculate_ppo_core,
    _vectorized_wick_check_buy,
    _vectorized_wick_check_sell,
)

# --- Aliases ---
def AOT_SANITIZE(arr: np.ndarray, default: float) -> np.ndarray:
    return AOT._sanitize_array_numba(arr, default) if HAVE_AOT else _sanitize_array_numba(arr, default)

def AOT_EMA_PERIOD(data: np.ndarray, period: float) -> np.ndarray:
    return AOT._ema_loop(data, period) if HAVE_AOT else _ema_loop(data, period)

def AOT_EMA_ALPHA(data: np.ndarray, alpha: float) -> np.ndarray:
    return AOT._ema_loop_alpha(data, alpha) if HAVE_AOT else _ema_loop_alpha(data, alpha)

def AOT_SMA(data: np.ndarray, period: int) -> np.ndarray:
    return AOT._sma_loop(data, period) if HAVE_AOT else _sma_loop(data, period)

def AOT_ROLLING_MEAN_NUMBA(close: np.ndarray, period: int) -> np.ndarray:
    return AOT._rolling_mean_numba(close, period) if HAVE_AOT else _rolling_mean_numba(close, period)

def AOT_ROLLING_STD(close: np.ndarray, period: int, responsiveness: float) -> np.ndarray:
    return AOT._rolling_std_welford(close, period, responsiveness) if HAVE_AOT else _rolling_std_welford(close, period, responsiveness)

def AOT_ROLLING_MIN(arr, period):
    return AOT._rolling_min_numba(arr, period) if HAVE_AOT else _rolling_min_numba(arr, period)

def AOT_ROLLING_MAX(arr, period):
    return AOT._rolling_max_numba(arr, period) if HAVE_AOT else _rolling_max_numba(arr, period)

def AOT_KALMAN(src: np.ndarray, length: int, R: float, Q: float) -> np.ndarray:
    return AOT._kalman_loop(src, length, R, Q) if HAVE_AOT else _kalman_loop(src, length, R, Q)

def AOT_VWAP_DAILY(high, low, close, volume, timestamps):
    return AOT._vwap_daily_loop(high, low, close, volume, timestamps) if HAVE_AOT else _vwap_daily_loop(high, low, close, volume, timestamps)

def AOT_RNG_FILTER(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    return AOT._rng_filter_loop(x, r) if HAVE_AOT else _rng_filter_loop(x, r)

def AOT_SMOOTH_RANGE(close: np.ndarray, t: int, m: int) -> np.ndarray:
    return AOT._smooth_range(close, t, m) if HAVE_AOT else _smooth_range(close, t, m)

def AOT_MMH_WORM(close_arr: np.ndarray, sd_arr: np.ndarray, rows: int) -> np.ndarray:
    return AOT._calc_mmh_worm_loop(close_arr, sd_arr, rows) if HAVE_AOT else _calc_mmh_worm_loop(close_arr, sd_arr, rows)

def AOT_MMH_VALUE(temp_arr: np.ndarray, rows: int) -> np.ndarray:
    return AOT._calc_mmh_value_loop(temp_arr, rows) if HAVE_AOT else _calc_mmh_value_loop(temp_arr, rows)

def AOT_MMH_MOMENTUM_ACCUM(momentum_arr: np.ndarray, rows: int) -> np.ndarray:
    return AOT._calc_mmh_momentum_loop(momentum_arr, rows) if HAVE_AOT else _calc_mmh_momentum_loop(momentum_arr, rows)

def AOT_RSI_CORE(close: np.ndarray, rsi_len: int) -> np.ndarray:
    return AOT._calculate_rsi_core(close, rsi_len) if HAVE_AOT else _calculate_rsi_core(close, rsi_len)

def AOT_PPO_CORE(close: np.ndarray, fast: int, slow: int, signal: int):
    return AOT._calculate_ppo_core(close, fast, slow, signal) if HAVE_AOT else _calculate_ppo_core(close, fast, slow, signal)

def AOT_WICK_CHECK_BUY(open_arr, high_arr, low_arr, close_arr, min_wick_ratio: float):
    return AOT._vectorized_wick_check_buy(open_arr, high_arr, low_arr, close_arr, min_wick_ratio) if HAVE_AOT else _vectorized_wick_check_buy(open_arr, high_arr, low_arr, close_arr, min_wick_ratio)

def AOT_WICK_CHECK_SELL(open_arr, high_arr, low_arr, close_arr, min_wick_ratio: float):
    return AOT._vectorized_wick_check_sell(open_arr, high_arr, low_arr, close_arr, min_wick_ratio) if HAVE_AOT else _vectorized_wick_check_sell(open_arr, high_arr, low_arr, close_arr, min_wick_ratio)

# --- Helper ---
def use_aot() -> bool:
    return HAVE_AOT
