from numba.pycc import CC
import numpy as np

cc = CC('aot_indicators')

# --- Sanitize array ---
@cc.export('_sanitize_array_numba', 'float64[:](float64[:], float64)')
def _sanitize_array_numba(arr, default):
    out = np.empty_like(arr)
    for i in range(len(arr)):
        val = arr[i]
        out[i] = default if (np.isnan(val) or np.isinf(val)) else val
    return out

# --- EMA loops ---
@cc.export('_ema_loop', 'float64[:](float64[:], float64)')
def _ema_loop(data, alpha_or_period):
    n = len(data)
    if alpha_or_period > 1.0:
        alpha = 2.0 / (alpha_or_period + 1.0)
    else:
        alpha = alpha_or_period
    out = np.empty(n, dtype=np.float64)
    out[0] = data[0] if not np.isnan(data[0]) else 0.0
    for i in range(1, n):
        curr = data[i]
        out[i] = alpha * curr + (1 - alpha) * out[i-1] if not np.isnan(curr) else out[i-1]
    return out

@cc.export('_ema_loop_alpha', 'float64[:](float64[:], float64)')
def _ema_loop_alpha(data, alpha):
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    out[0] = data[0] if not np.isnan(data[0]) else 0.0
    for i in range(1, n):
        curr = data[i]
        out[i] = alpha * curr + (1 - alpha) * out[i-1] if not np.isnan(curr) else out[i-1]
    return out

# --- SMA loop ---
@cc.export('_sma_loop', 'float64[:](float64[:], int32)')
def _sma_loop(data, period):
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    window_sum = 0.0
    count = 0
    for i in range(n):
        val = data[i]
        if not np.isnan(val):
            window_sum += val
            count += 1
        if i >= period:
            old_val = data[i - period]
            if not np.isnan(old_val):
                window_sum -= old_val
                count -= 1
        out[i] = window_sum / count if count > 0 else np.nan
    return out

# --- Rolling mean ---
@cc.export('_rolling_mean_numba', 'float64[:](float64[:], int32)')
def _rolling_mean_numba(close, period):
    rows = len(close)
    ma = np.empty(rows, dtype=np.float64)
    for i in range(rows):
        start = max(0, i - period + 1)
        s = 0.0
        c = 0
        for j in range(start, i + 1):
            v = close[j]
            if not np.isnan(v):
                s += v
                c += 1
        ma[i] = s / c if c > 0 else np.nan
    return ma

# --- Rolling std (Welford) ---
@cc.export('_rolling_std_welford', 'float64[:](float64[:], int32, float64)')
def _rolling_std_welford(close, period, responsiveness):
    n = len(close)
    sd = np.empty(n, dtype=np.float64)
    resp = max(0.0001, min(1.0, responsiveness))
    for i in range(n):
        mean = 0.0
        m2 = 0.0
        count = 0
        start = max(0, i - period + 1)
        for j in range(start, i + 1):
            val = close[j]
            if not np.isnan(val):
                count += 1
                delta = val - mean
                mean += delta / count
                delta2 = val - mean
                m2 += delta * delta2
        sd[i] = (np.sqrt(max(0.0, m2 / count)) * resp) if count > 1 else 0.0
    return sd

# --- Rolling min ---
@cc.export('_rolling_min_numba', 'float64[:](float64[:], int32)')
def _rolling_min_numba(arr, period):
    rows = len(arr)
    out = np.empty(rows, dtype=np.float64)
    for i in range(rows):
        start = max(0, i - period + 1)
        out[i] = np.nanmin(arr[start:i + 1])
    return out

# --- Rolling max ---
@cc.export('_rolling_max_numba', 'float64[:](float64[:], int32)')
def _rolling_max_numba(arr, period):
    rows = len(arr)
    out = np.empty(rows, dtype=np.float64)
    for i in range(rows):
        start = max(0, i - period + 1)
        out[i] = np.nanmax(arr[start:i + 1])
    return out

# --- Kalman filter ---
@cc.export('_kalman_loop', 'float64[:](float64[:], int32, float64, float64)')
def _kalman_loop(src, length, R, Q):
    n = len(src)
    result = np.empty(n, dtype=np.float64)
    estimate = src[0] if not np.isnan(src[0]) else 0.0
    error_est = 1.0
    error_meas = R * max(1.0, float(length))
    Q_div_length = Q / max(1.0, float(length))
    for i in range(n):
        current = src[i]
        if np.isnan(current):
            result[i] = estimate
            continue
        if np.isnan(estimate):
            estimate = current
        prediction = estimate
        kalman_gain = error_est / (error_est + error_meas)
        estimate = prediction + kalman_gain * (current - prediction)
        error_est = (1.0 - kalman_gain) * error_est + Q_div_length
        result[i] = estimate
    return result

# --- VWAP daily ---
@cc.export('_vwap_daily_loop', 'float64[:](float64[:], float64[:], float64[:], float64[:], int64[:])')
def _vwap_daily_loop(high, low, close, volume, timestamps):
    n = len(close)
    vwap = np.empty(n, dtype=np.float64)
    cum_vol = 0.0
    cum_pv = 0.0
    current_session_day = -1
    for i in range(n):
        ts = timestamps[i]
        day = ts // 86400
        h, l, c, v = high[i], low[i], close[i], volume[i]
        if day != current_session_day:
            current_session_day = day
            cum_vol = 0.0
            cum_pv = 0.0
        if np.isnan(h) or np.isnan(l) or np.isnan(c) or np.isnan(v) or v <= 0:
            vwap[i] = vwap[i - 1] if i > 0 else c
            continue
        typical = (h + l + c) / 3.0
        cum_vol += v
        cum_pv += typical * v
        vwap[i] = cum_pv / cum_vol if cum_vol > 0 else typical
    return vwap

# --- Range filter ---
@cc.export('_rng_filter_loop', 'float64[:](float64[:], float64[:])')
def _rng_filter_loop(x, r):
    n = len(x)
    filt = np.empty(n, dtype=np.float64)
    filt[0] = x[0] if not np.isnan(x[0]) else 0.0
    for i in range(1, n):
        prev_filt = filt[i - 1]
        curr_x = x[i]
        curr_r = r[i]
        if np.isnan(curr_r) or np.isnan(curr_x):
            filt[i] = prev_filt
            continue
        if curr_x > prev_filt:
            target = curr_x - curr_r
            filt[i] = max(prev_filt, target)
        else:
            target = curr_x + curr_r
            filt[i] = min(prev_filt, target)
    return filt

# --- Smooth range (EMA twice, scaled) ---
@cc.export('_smooth_range', 'float64[:](float64[:], int32, int32)')
def _smooth_range(close, t, m):
    n = len(close)
    diff = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        diff[i] = abs(close[i] - close[i - 1])
    avrng = _ema_loop(diff, float(t))
    wper = t * 2 - 1
    smoothrng = _ema_loop(avrng, float(wper))
    return smoothrng * float(m)

# --- MMH worm/value/momentum ---
@cc.export('_calc_mmh_worm_loop', 'float64[:](float64[:], float64[:], int32)')
def _calc_mmh_worm_loop(close_arr, sd_arr, rows):
    worm_arr = np.empty(rows, dtype=np.float64)
    first_val = close_arr[0]
    worm_arr[0] = 0.0 if np.isnan(first_val) else first_val
    for i in range(1, rows):
        src = close_arr[i] if not np.isnan(close_arr[i]) else worm_arr[i - 1]
        prev_worm = worm_arr[i - 1]
        diff = src - prev_worm
        sd_i = sd_arr[i]
        if np.isnan(sd_i):
            delta = diff
        else:
            delta = (np.sign(diff) * sd_i) if (abs(diff) > sd_i) else diff
        worm_arr[i] = prev_worm + delta
    return worm_arr

@cc.export('_calc_mmh_value_loop', 'float64[:](float64[:], int32)')
def _calc_mmh_value_loop(temp_arr, rows):
    value_arr = np.zeros(rows, dtype=np.float64)
    value_arr[0] = 0.0
    for i in range(1, rows):
        prev_v = value_arr[i - 1] if not np.isnan(value_arr[i - 1]) else 0.0
        t = temp_arr[i] if not np.isnan(temp_arr[i]) else 0.5
        v = t - 0.5 + 0.5 * prev_v
        value_arr[i] = max(-0.9999, min(0.9999, v))
    return value_arr

@cc.export('_calc_mmh_momentum_loop', 'float64[:](float64[:], int32)')
def _calc_mmh_momentum_loop(momentum_arr, rows):
    out = momentum_arr.copy()
    for i in range(1, rows):
        prev = out[i - 1] if not np.isnan(out[i - 1]) else 0.0
        out[i] = out[i] + 0.5 * prev
    return out

# --- RSI core ---
@cc.export('_calculate_rsi_core', 'float64[:](float64[:], int32)')
def _calculate_rsi_core(close, rsi_len):
    n = len(close)
    delta = np.zeros(n, dtype=np.float64)
    gain = np.zeros(n, dtype=np.float64)
    loss = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        delta[i] = close[i] - close[i - 1]
        if delta[i] > 0:
            gain[i] = delta[i]
        elif delta[i] < 0:
            loss[i] = -delta[i]
    alpha = 1.0 / float(rsi_len)
    avg_gain = np.empty(n, dtype=np.float64)
    avg_loss = np.empty(n, dtype=np.float64)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, n):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i - 1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i - 1]
    rsi = np.empty(n, dtype=np.float64)
    for i in range(n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi

# --- PPO core ---
@cc.export('_calculate_ppo_core', '(float64[:], float64[:])(float64[:], int32, int32, int32)')
def _calculate_ppo_core(close, fast, slow, signal):
    # fast/slow EMA
    fast_ma = _ema_loop(close, float(fast))
    slow_ma = _ema_loop(close, float(slow))
    n = len(close)
    ppo = np.empty(n, dtype=np.float64)
    for i in range(n):
        if np.isnan(slow_ma[i]) or abs(slow_ma[i]) < 1e-12:
            ppo[i] = 0.0
        else:
            ppo[i] = ((fast_ma[i] - slow_ma[i]) / slow_ma[i]) * 100.0
    ppo_sig = _ema_loop(ppo, float(signal))
    return ppo, ppo_sig

# --- Wick quality checks ---
@cc.export('_vectorized_wick_check_buy', 'bool_[:](float64[:], float64[:], float64[:], float64[:], float64)')
def _vectorized_wick_check_buy(open_arr, high_arr, low_arr, close_arr, min_wick_ratio):
    n = len(close_arr)
    result = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        o, h, l, c = open_arr[i], high_arr[i], low_arr[i], close_arr[i]
        if c <= o:
            result[i] = False
            continue
        candle_range = h - l
        if candle_range < 1e-8:
            result[i] = False
            continue
        upper_wick = h - c
        wick_ratio = upper_wick / candle_range
        result[i] = wick_ratio < min_wick_ratio
    return result

@cc.export('_vectorized_wick_check_sell', 'bool_[:](float64[:], float64[:], float64[:], float64[:], float64)')
def _vectorized_wick_check_sell(open_arr, high_arr, low_arr, close_arr, min_wick_ratio):
    n = len(close_arr)
    result = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        o, h, l, c = open_arr[i], high_arr[i], low_arr[i], close_arr[i]
        if c >= o:
            result[i] = False
            continue
        candle_range = h - l
        if candle_range < 1e-8:
            result[i] = False
            continue
        lower_wick = c - l
        wick_ratio = lower_wick / candle_range
        result[i] = wick_ratio < min_wick_ratio
    return result

if __name__ == "__main__":
    cc.compile()
