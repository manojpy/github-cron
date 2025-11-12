import requests
import pandas as pd
import numpy as np
import time
import os
import json
import pytz
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============ CONFIGURATION ============
# Telegram settings - reads from environment variables (GitHub Secrets)
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '8462496498:AAHYZ4xDIHvrVRjmCmZyoPhupCjRaRgiITc')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '203813932')

# Enable debug mode - set to True to see detailed logs
DEBUG_MODE = os.environ.get('DEBUG_MODE', 'True').lower() == 'true'

# Send test message on startup
SEND_TEST_MESSAGE = os.environ.get('SEND_TEST_MESSAGE', 'True').lower() == 'true'

# Delta Exchange API
DELTA_API_BASE = "https://api.india.delta.exchange"

# Trading pairs to monitor
PAIRS = {
    "BTCUSD": None,
    "ETHUSD": None,
    "SOLUSD": None,
    "AVAXUSD": None,
    "BCHUSD": None,
    "XRPUSD": None,
    "BNBUSD": None,
    "LTCUSD": None,
    "DOTUSD": None,
    "ADAUSD": None,
    "SUIUSD": None,
    "AAVEUSD": None
}

# Special data requirements for pairs with limited history
SPECIAL_PAIRS = {
    "SOLUSD": {"limit_15m": 210, "min_required": 160, "limit_5m": 300, "min_required_5m": 200}
}

# Indicator settings
# PPO settings
PPO_FAST = 7
PPO_SLOW = 16
PPO_SIGNAL = 5
PPO_USE_SMA = False  # False = use EMA

# RMA settings
RMA_50_PERIOD = 50   # RMA50 on 15min
RMA_200_PERIOD = 200 # RMA200 on 5min

# Cirrus Cloud settings
CIRRUS_CLOUD_ENABLED = True
X1 = 22
X2 = 9
X3 = 15
X4 = 5

# Smoothed RSI (SRSI) settings
SRSI_RSI_LEN = 21
SRSI_KALMAN_LEN = 5

# State file paths (read from environment, fallback to defaults)
STATE_FILE = os.environ.get("STATE_FILE_PATH", "macd_state.json")
STATE_FILE_BAK = STATE_FILE + ".bak"

COOLDOWN_SECONDS = 600  # 10 minutes per signal

# Thread lock for state updates
state_lock = Lock()

# ============ UTILITY FUNCTIONS ============

def create_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

SESSION = create_session()

def debug_log(message):
    """Print debug messages if DEBUG_MODE is enabled"""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")

def load_state():
    """Load previous alert state from file with backup recovery and schema validation"""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            if not isinstance(state, dict):
                raise ValueError("State file is not a dict")
            # normalize schema: {pair: {"state": str, "ts": int}}
            normalized = {}
            for k, v in state.items():
                if isinstance(v, dict):
                    s = v.get("state")
                    ts = v.get("ts", 0)
                    if isinstance(s, str):
                        normalized[k] = {"state": s, "ts": int(ts)}
                elif isinstance(v, str):
                    normalized[k] = {"state": v, "ts": 0}
            debug_log(f"Loaded state: {normalized}")
            return normalized
    except Exception as e:
        print(f"Error loading state: {e}")
        # Try backup
        try:
            if os.path.exists(STATE_FILE_BAK):
                with open(STATE_FILE_BAK, 'r') as f:
                    state = json.load(f)
                if isinstance(state, dict):
                    print("Recovered state from backup.")
                    normalized = {}
                    for k, v in state.items():
                        if isinstance(v, dict):
                            s = v.get("state")
                            ts = v.get("ts", 0)
                            if isinstance(s, str):
                                normalized[k] = {"state": s, "ts": int(ts)}
                        elif isinstance(v, str):
                            normalized[k] = {"state": v, "ts": 0}
                    return normalized
        except Exception as e2:
            print(f"Backup state load failed: {e2}")
    debug_log("No previous state found or recovery failed, starting fresh")
    return {}

def save_state(state):
    """Save alert state to file atomically with backup and fsync"""
    try:
        temp_file = STATE_FILE + '.tmp'
        with open(temp_file, 'w') as f:
            json.dump(state, f)
            f.flush()
            os.fsync(f.fileno())
        # Write backup from existing file
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    existing = f.read()
                with open(STATE_FILE_BAK, 'w') as fb:
                    fb.write(existing)
                    fb.flush()
                    os.fsync(fb.fileno())
            except Exception as e:
                print(f"Warning: could not write backup: {e}")
        os.replace(temp_file, STATE_FILE)  # Atomic on POSIX
        debug_log(f"Saved state: {state}")
    except Exception as e:
        print(f"Error saving state: {e}")

def send_telegram_alert(message):
    """Send alert message via Telegram with validation and safe JSON parsing"""
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == 'xxxx' or not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID == 'xxxx':
        print("✗ Telegram not configured: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        return False
    try:
        debug_log(f"Attempting to send message: {message[:100]}...")
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": None  # No HTML formatting
        }
        response = SESSION.post(url, data=data, timeout=10)
        try:
            response_data = response.json()
        except Exception:
            response_data = {"ok": False, "status_code": response.status_code, "text": response.text[:200]}
        if response_data.get('ok'):
            print(f"✓ Alert sent successfully")
            return True
        else:
            print(f"✗ Telegram error: {response_data}")
            return False
    except Exception as e:
        print(f"✗ Error sending Telegram message: {e}")
        if DEBUG_MODE:
            traceback.print_exc()
        return False

def send_test_message():
    """Send a test message to verify Telegram connectivity"""
    ist = pytz.timezone('Asia/Kolkata')
    current_dt = datetime.now(ist)
    formatted_time = current_dt.strftime('%d-%m-%Y @ %H:%M IST')
    test_msg = f"🔔 Bot Started\nTest message from PPO Bot\nTime: {formatted_time}\nDebug Mode: {'ON' if DEBUG_MODE else 'OFF'}"
    print("\n" + "="*50)
    print("SENDING TEST MESSAGE")
    print("="*50)
    success = send_telegram_alert(test_msg)
    if success:
        print("✓ Test message sent successfully!")
    else:
        print("✗ Test message failed - check your bot token and chat ID")
    print("="*50 + "\n")
    return success

def get_product_ids():
    """Fetch all product IDs from Delta Exchange"""
    try:
        debug_log("Fetching product IDs from Delta Exchange...")
        response = SESSION.get(f"{DELTA_API_BASE}/v2/products", timeout=10)
        try:
            data = response.json()
        except Exception:
            print(f"API Error: Non-JSON response: {response.status_code} {response.text[:200]}")
            return False

        if data.get('success'):
            products = data['result']
            debug_log(f"Received {len(products)} products from API")
            for product in products:
                symbol = product['symbol'].replace('_USDT', 'USD').replace('USDT', 'USD')
                if product.get('contract_type') == 'perpetual_futures':
                    for pair_name in PAIRS.keys():
                        if symbol == pair_name or symbol.replace('_', '') == pair_name:
                            PAIRS[pair_name] = {
                                'id': product['id'],
                                'symbol': product['symbol'],
                                'contract_type': product['contract_type']
                            }
                            debug_log(f"Matched {pair_name} -> {product['symbol']} (ID: {product['id']})")
            return True
        else:
            print(f"API Error: {data}")
            return False
    except Exception as e:
        print(f"Error fetching products: {e}")
        if DEBUG_MODE:
            traceback.print_exc()
        return False

def get_candles(product_id, resolution="15", limit=150):
    """Fetch OHLCV candles from Delta Exchange with softened stale handling"""
    try:
        to_time = int(time.time())
        from_time = to_time - (limit * int(resolution) * 60)
        url = f"{DELTA_API_BASE}/v2/chart/history"
        params = {
            'resolution': resolution,
            'symbol': product_id,
            'from': from_time,
            'to': to_time
        }
        debug_log(f"Fetching {resolution}m candles for {product_id}, limit={limit}")
        response = SESSION.get(url, params=params, timeout=15)
        try:
            data = response.json()
        except Exception:
            print(f"Error fetching candles for {product_id}: Non-JSON response {response.status_code} {response.text[:200]}")
            return None

        if data.get('success'):
            result = data['result']
            # Verify lengths to avoid misaligned DataFrame
            arrays = [result.get('t', []), result.get('o', []), result.get('h', []), result.get('l', []), result.get('c', []), result.get('v', [])]
            min_len = min(map(len, arrays)) if arrays else 0
            if min_len == 0:
                debug_log(f"No candles returned for {product_id}")
                return None
            df = pd.DataFrame({
                'timestamp': result.get('t', [])[:min_len],
                'open': result.get('o', [])[:min_len],
                'high': result.get('h', [])[:min_len],
                'low': result.get('l', [])[:min_len],
                'close': result.get('c', [])[:min_len],
                'volume': result.get('v', [])[:min_len]
            })

            # Sort by timestamp just in case and drop duplicates
            df = df.sort_values('timestamp').drop_duplicates(subset='timestamp').reset_index(drop=True)

            # Validate price data
            if df['close'].iloc[-1] <= 0:
                debug_log(f"Invalid price (≤0) for {product_id}, skipping")
                return None

            debug_log(f"Received {len(df)} candles for {product_id} ({resolution}m)")

            # Check data freshness
            last_candle_time = df['timestamp'].iloc[-1]
            time_diff = time.time() - last_candle_time
            max_age = int(resolution) * 60 * 3  # 3 candles max age (strict)
            max_age_soft = int(resolution) * 60 * 6  # soft threshold
            if time_diff > max_age_soft:
                print(f"⚠️ Stale data for {product_id} ({resolution}m) - {time_diff/60:.1f} min old. Skipping.")
                return None
            elif time_diff > max_age:
                debug_log(f"⚠️ Mild staleness for {product_id} ({resolution}m): {time_diff/60:.1f} min. Proceeding cautiously.")

            return df
        else:
            print(f"Error fetching candles for {product_id}: {data.get('message', 'No message')}")
            return None
    except Exception as e:
        print(f"Exception fetching candles for {product_id}: {e}")
        if DEBUG_MODE:
            traceback.print_exc()
        return None

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def calculate_sma(data, period):
    return data.rolling(window=period, min_periods=max(2, period//3)).mean()

def calculate_rma(data, period):
    r = data.ewm(alpha=1/period, adjust=False).mean()
    return r.bfill().ffill()

def calculate_ppo(df, fast=7, slow=16, signal=5, use_sma=False):
    close = df['close'].astype(float)
    if use_sma:
        fast_ma = calculate_sma(close, fast)
        slow_ma = calculate_sma(close, slow)
    else:
        fast_ma = calculate_ema(close, fast)
        slow_ma = calculate_ema(close, slow)
    slow_ma = slow_ma.replace(0, np.nan).bfill().ffill()
    ppo = ((fast_ma - slow_ma) / slow_ma) * 100
    ppo = ppo.replace([np.inf, -np.inf], np.nan).bfill().ffill()
    ppo_signal = calculate_sma(ppo, signal) if use_sma else calculate_ema(ppo, signal)
    ppo_signal = ppo_signal.replace([np.inf, -np.inf], np.nan).bfill().ffill()
    return ppo, ppo_signal

def smoothrng(x, t, m):
    wper = t * 2 - 1
    avrng = calculate_ema(np.abs(x.diff().fillna(0)), t)
    smoothrng = calculate_ema(avrng, max(1, wper)) * m
    # Avoid zeros to prevent flat filter behavior
    return smoothrng.clip(lower=1e-8).bfill().ffill()

def rngfilt(x, r):
    result_list = [x.iloc[0]]
    for i in range(1, len(x)):
        prev_f = result_list[-1]
        curr_x = x.iloc[i]
        curr_r = max(float(r.iloc[i]), 1e-8)
        if curr_x > prev_f:
            f = prev_f if (curr_x - curr_r) < prev_f else (curr_x - curr_r)
        else:
            f = prev_f if (curr_x + curr_r) > prev_f else (curr_x + curr_r)
        result_list.append(f)
    return pd.Series(result_list, index=x.index)

def calculate_cirrus_cloud(df):
    close = df['close'].copy()
    smrngx1x = smoothrng(close, X1, X2)
    smrngx1x2 = smoothrng(close, X3, X4)
    filtx1 = rngfilt(close, smrngx1x)
    filtx12 = rngfilt(close, smrngx1x2)
    upw = filtx1 < filtx12
    dnw = filtx1 > filtx12
    return upw, dnw, filtx1, filtx12

def kalman_filter(src, length, R=0.01, Q=0.1):
    result_list = []
    estimate = np.nan
    error_est = 1.0
    error_meas = R * max(1, length)
    Q_div_length = Q / max(1, length)
    for i in range(len(src)):
        current_src = src.iloc[i]
        if np.isnan(estimate):
            if i > 0:
                estimate = src.iloc[i-1]
            else:
                result_list.append(np.nan)
                continue
        prediction = estimate
        kalman_gain = error_est / (error_est + error_meas)
        estimate = prediction + kalman_gain * (current_src - prediction)
        error_est = (1 - kalman_gain) * error_est + Q_div_length
        result_list.append(estimate)
    return pd.Series(result_list, index=src.index)

def calculate_smooth_rsi(df, rsi_len=SRSI_RSI_LEN, kalman_len=SRSI_KALMAN_LEN):
    close = df['close'].astype(float)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = calculate_rma(gain, rsi_len)
    avg_loss = calculate_rma(loss, rsi_len).replace(0, np.nan).bfill().ffill().clip(lower=1e-8)
    rs = avg_gain.divide(avg_loss)
    rsi_value = 100 - (100 / (1 + rs))
    rsi_value = rsi_value.replace([np.inf, -np.inf], np.nan).bfill().ffill()
    smooth_rsi = kalman_filter(rsi_value, kalman_len).bfill().ffill()
    return smooth_rsi

def calculate_magical_momentum_hist(df, period=144, responsiveness=0.9):
    """
    NaN-safe, zero-division safe Magical Momentum Histogram.
    Returns a numeric series of the same length, avoiding NaNs/inf at tail.
    """
    n = len(df)
    if n == 0:
        return pd.Series([], dtype=float)
    if n < period + 50:
        # Return zeros instead of NaNs to avoid blocking signals
        return pd.Series(np.zeros(n), index=df.index, dtype=float)

    close = df['close'].astype(float).copy()
    responsiveness = max(1e-5, float(responsiveness))

    sd = close.rolling(window=50, min_periods=10).std() * responsiveness
    sd = sd.bfill().ffill().fillna(0.001).clip(lower=1e-6)

    worm = close.copy()
    for i in range(1, n):
        diff = close.iloc[i] - worm.iloc[i - 1]
        delta = np.sign(diff) * sd.iloc[i] if abs(diff) > sd.iloc[i] else diff
        worm.iloc[i] = worm.iloc[i - 1] + delta

    ma = close.rolling(window=period, min_periods=max(5, period//3)).mean().bfill().ffill()
    denom = worm.replace(0, np.nan).bfill().ffill().clip(lower=1e-8)

    raw_momentum = ((worm - ma).fillna(0)) / denom
    raw_momentum = raw_momentum.replace([np.inf, -np.inf], 0).fillna(0)

    min_med = raw_momentum.rolling(window=period, min_periods=max(5, period//3)).min().bfill().ffill()
    max_med = raw_momentum.rolling(window=period, min_periods=max(5, period//3)).max().bfill().ffill()
    rng = (max_med - min_med).replace(0, np.nan).fillna(1e-8)

    temp = pd.Series(0.0, index=df.index)
    valid = rng > 1e-10
    temp.loc[valid] = (raw_momentum.loc[valid] - min_med.loc[valid]) / rng.loc[valid]
    temp = temp.clip(-1, 1).fillna(0)

    value = pd.Series(0.0, index=df.index)
    for i in range(1, n):
        v_prev = value.iloc[i - 1]
        v_new = (temp.iloc[i] - 0.5 + 0.5 * v_prev)
        value.iloc[i] = max(-0.9999, min(0.9999, v_new))

    temp2 = (1 + value) / (1 - value)
    temp2 = temp2.replace([np.inf, -np.inf], np.nan).clip(lower=1e-6).fillna(1e-6)

    momentum = 0.25 * np.log(temp2)
    momentum = pd.Series(momentum, index=df.index).replace([np.inf, -np.inf], 0).fillna(0)

    hist = pd.Series(0.0, index=df.index)
    for i in range(1, n):
        hist.iloc[i] = momentum.iloc[i] + 0.5 * hist.iloc[i - 1]

    return hist.replace([np.inf, -np.inf], 0).fillna(0)

def check_pair(pair_name, pair_info, last_state_for_pair):
    """Check PPO and RMA/Cirrus/SRSI conditions for a pair using last closed candle"""
    try:
        if pair_info is None:
            return None
        debug_log(f"\n{'='*60}")
        debug_log(f"Checking {pair_name}")
        debug_log(f"{'='*60}")

        if pair_name in SPECIAL_PAIRS:
            limit_15m = SPECIAL_PAIRS[pair_name]["limit_15m"]
            min_required = SPECIAL_PAIRS[pair_name]["min_required"]
            limit_5m = SPECIAL_PAIRS[pair_name].get("limit_5m", 300)
            min_required_5m = SPECIAL_PAIRS[pair_name].get("min_required_5m", 250)
        else:
            limit_15m = 210
            min_required = 200
            limit_5m = 300
            min_required_5m = 250

        df_15m = get_candles(pair_info['symbol'], "15", limit=limit_15m)
        df_5m = get_candles(pair_info['symbol'], "5", limit=limit_5m)

        if df_15m is None or len(df_15m) < (min_required + 2):
            print(f"⚠️ Insufficient 15m data for {pair_name}: {len(df_15m) if df_15m is not None else 0}/{min_required + 2} candles (needs +2 for closed indexing)")
            return None
        if df_5m is None or len(df_5m) < (min_required_5m + 2):
            print(f"⚠️ Insufficient 5m data for {pair_name}: {len(df_5m) if df_5m is not None else 0}/{min_required_5m + 2} candles (needs +2 for closed indexing)")
            return None

        # Use most recent fully closed candle indices
        last_i = -1
        prev_i = -2

        magical_hist = calculate_magical_momentum_hist(df_15m, period=144, responsiveness=0.9)
        magical_hist_curr = float(magical_hist.iloc[last_i])
        if pd.isna(magical_hist_curr):
            debug_log(f"⚠️ NaN in Magical Momentum Hist for {pair_name}, skipping")
            return None
        debug_log(f"Magical Momentum Hist (15m): {magical_hist_curr:.6f}")

        ppo, ppo_signal = calculate_ppo(df_15m, PPO_FAST, PPO_SLOW, PPO_SIGNAL, PPO_USE_SMA)
        rma_50 = calculate_rma(df_15m['close'], RMA_50_PERIOD)
        rma_200 = calculate_rma(df_5m['close'], RMA_200_PERIOD)
        upw, dnw, filtx1, filtx12 = calculate_cirrus_cloud(df_15m)
        smooth_rsi = calculate_smooth_rsi(df_15m)

        # Extract last closed values
        ppo_curr = float(ppo.iloc[last_i])
        ppo_prev = float(ppo.iloc[prev_i])
        ppo_signal_curr = float(ppo_signal.iloc[last_i])
        ppo_signal_prev = float(ppo_signal.iloc[prev_i])

        if any(pd.isna(x) for x in [ppo_curr, ppo_prev, ppo_signal_curr, ppo_signal_prev]):
            debug_log(f"⚠️ NaN values in PPO for {pair_name}, skipping")
            return None

        smooth_rsi_curr = float(smooth_rsi.iloc[last_i])
        smooth_rsi_prev = float(smooth_rsi.iloc[prev_i])
        if any(pd.isna(x) for x in [smooth_rsi_curr, smooth_rsi_prev]):
            debug_log(f"⚠️ NaN values in Smooth RSI for {pair_name}, skipping")
            return None

        close_curr = float(df_15m['close'].iloc[last_i])
        open_curr = float(df_15m['open'].iloc[last_i])
        high_curr = float(df_15m['high'].iloc[last_i])
        low_curr = float(df_15m['low'].iloc[last_i])

        rma50_curr = float(rma_50.iloc[last_i])
        if pd.isna(rma50_curr):
            debug_log(f"⚠️ NaN values in RMA50 for {pair_name}, skipping")
            return None

        # Use most recent closed 5m RMA200
        rma200_curr = float(rma_200.iloc[-1])
        if pd.isna(rma200_curr):
            debug_log(f"⚠️ NaN values in RMA200 for {pair_name}, skipping")
            return None

        # Candle metrics (candle-aware wick logic)
        total_range = high_curr - low_curr
        if total_range <= 0:
            upper_wick = 0.0
            lower_wick = 0.0
            strong_bullish_close = False
            strong_bearish_close = False
            bullish_candle = False
            bearish_candle = False
            debug_log("  Candle range is zero or negative, skipping wick checks.")
        else:
            bullish_candle = close_curr > open_curr
            bearish_candle = close_curr < open_curr

            if bullish_candle:
                # Green: upper = high - close, lower = open - low
                upper_wick = max(0.0, high_curr - close_curr)
                lower_wick = max(0.0, open_curr - low_curr)
                strong_bullish_close = (upper_wick / total_range) < 0.20
                strong_bearish_close = False
            elif bearish_candle:
                # Red: upper = high - open, lower = close - low
                upper_wick = max(0.0, high_curr - open_curr)
                lower_wick = max(0.0, close_curr - low_curr)
                strong_bearish_close = (lower_wick / total_range) < 0.20
                strong_bullish_close = False
            else:
                # Doji/neutral fallback
                upper_wick = max(0.0, high_curr - max(open_curr, close_curr))
                lower_wick = max(0.0, min(open_curr, close_curr) - low_curr)
                strong_bullish_close = (upper_wick / total_range) < 0.20
                strong_bearish_close = (lower_wick / total_range) < 0.20

        debug_log(f"\nCandle Metrics (15m):")
        debug_log(f"  O:{open_curr:.4f} H:{high_curr:.4f} L:{low_curr:.4f} C:{close_curr:.4f}")
        if total_range > 0:
            debug_log(f"  bullish={bullish_candle}, bearish={bearish_candle}, range={total_range:.6f}")
            debug_log(f"  upper_wick={upper_wick:.6f} ({(upper_wick/total_range):.2%}), lower_wick={lower_wick:.6f} ({(lower_wick/total_range):.2%})")
            debug_log(f"  Strong Bullish Close (20% Rule): {strong_bullish_close}")
            debug_log(f"  Strong Bearish Close (20% Rule): {strong_bearish_close}")

        debug_log(f"\nIndicator Values:")
        debug_log(f"Price (15m): ${close_curr:,.2f}")
        debug_log(f"PPO: {ppo_curr:.4f} (prev: {ppo_prev:.4f})")
        debug_log(f"PPO Signal: {ppo_signal_curr:.4f} (prev: {ppo_signal_prev:.4f})")
        debug_log(f"RMA50 (15m): {rma50_curr:.4f}")
        debug_log(f"RMA200 (5m): {rma200_curr:.4f}")
        debug_log(f"Smoothed RSI (15m): {smooth_rsi_curr:.2f} (prev: {smooth_rsi_prev:.2f})")
        debug_log(f"Cirrus Filter 1: {filtx1.iloc[last_i]:.6f}, Filter 2: {filtx12.iloc[last_i]:.6f}")
        debug_log(f"Cirrus Cloud - Upw: {bool(upw.iloc[last_i])}, Dnw: {bool(dnw.iloc[last_i])}")

        # Crossovers and bands
        ppo_cross_up = (ppo_prev <= ppo_signal_prev) and (ppo_curr > ppo_signal_curr)
        ppo_cross_down = (ppo_prev >= ppo_signal_prev) and (ppo_curr < ppo_signal_curr)
        ppo_cross_above_zero = (ppo_prev <= 0) and (ppo_curr > 0)
        ppo_cross_below_zero = (ppo_prev >= 0) and (ppo_curr < 0)
        ppo_cross_above_011 = (ppo_prev <= 0.11) and (ppo_curr > 0.11)
        ppo_cross_below_minus011 = (ppo_prev >= -0.11) and (ppo_curr < -0.11)

        ppo_below_020 = ppo_curr < 0.20
        ppo_above_minus020 = ppo_curr > -0.20
        ppo_above_signal = ppo_curr > ppo_signal_curr
        ppo_below_signal = ppo_curr < ppo_signal_curr

        ppo_below_030 = ppo_curr < 0.30
        ppo_above_minus030 = ppo_curr > -0.30

        close_above_rma50 = close_curr > rma50_curr
        close_below_rma50 = close_curr < rma50_curr
        close_above_rma200 = close_curr > rma200_curr
        close_below_rma200 = close_curr < rma200_curr

        srsi_cross_up_50 = (smooth_rsi_prev <= 50) and (smooth_rsi_curr > 50)
        srsi_cross_down_50 = (smooth_rsi_prev >= 50) and (smooth_rsi_curr < 50)

        debug_log(f"\nCrossover Checks:")
        debug_log(f"  PPO 15m cross up: {ppo_cross_up}")
        debug_log(f"  PPO 15m cross down: {ppo_cross_down}")
        debug_log(f"  SRSI cross up 50: {srsi_cross_up_50}")
        debug_log(f"  SRSI cross down 50: {srsi_cross_down_50}")

        # Helper to print condition sets
        def log_cond_set(label, conds):
            debug_log(label + " " + ", ".join([f"{k}={v}" for k, v in conds.items()]))

        # Current time & formatting
        ist = pytz.timezone('Asia/Kolkata')
        current_dt = datetime.now(ist)
        formatted_time = current_dt.strftime('%d-%m-%Y @ %H:%M IST')
        price = close_curr

        # Prepare last state info
        now_ts = int(time.time())
        last_state_value = None
        last_ts = 0
        if isinstance(last_state_for_pair, dict):
            last_state_value = last_state_for_pair.get("state")
            last_ts = int(last_state_for_pair.get("ts", 0))
        elif isinstance(last_state_for_pair, str):
            last_state_value = last_state_for_pair

        current_state = None
        send_message = None

        # Cirrus Cloud state (exclusive)
        cloud_state = "neutral"
        if CIRRUS_CLOUD_ENABLED:
            cloud_state = ("green" if (bool(upw.iloc[last_i]) and not bool(dnw.iloc[last_i]))
                           else "red" if (bool(dnw.iloc[last_i]) and not bool(upw.iloc[last_i]))
                           else "neutral")
        debug_log(f"Cirrus Cloud State: {cloud_state}")

        # --- ALERT LOGIC (8 SIGNALS) ---
        # BUY
        buy_conds = {
            "ppo_cross_up": ppo_cross_up,
            "ppo_below_020": ppo_below_020,
            "close_above_rma50": close_above_rma50,
            "close_above_rma200": close_above_rma200,
            "cloud_green": (cloud_state == "green"),
            "strong_bullish_close": strong_bullish_close,
            "magical_hist_curr>0": magical_hist_curr > 0,
        }
        log_cond_set("BUY condition inputs:", buy_conds)
        if all(buy_conds.values()):
            current_state = "buy"
            debug_log(f"\n🟢 BUY SIGNAL DETECTED for {pair_name}!")
            send_message = f"🟢 {pair_name} - BUY\nPPO - SIGNAL Crossover (PPO: {ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"

        # SELL
        sell_conds = {
            "ppo_cross_down": ppo_cross_down,
            "ppo_above_minus020": ppo_above_minus020,
            "close_below_rma50": close_below_rma50,
            "close_below_rma200": close_below_rma200,
            "cloud_red": (cloud_state == "red"),
            "strong_bearish_close": strong_bearish_close,
            "magical_hist_curr<0": magical_hist_curr < 0,
        }
        log_cond_set("SELL condition inputs:", sell_conds)
        if current_state is None and all(sell_conds.values()):
            current_state = "sell"
            debug_log(f"\n🔴 SELL SIGNAL DETECTED for {pair_name}!")
            send_message = f"🔴 {pair_name} - SELL\nPPO - SIGNAL Crossunder (PPO: {ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"

        # BUY SRSI 50
        buy_srsi_conds = {
            "srsi_cross_up_50": srsi_cross_up_50,
            "ppo_above_signal": ppo_above_signal,
            "ppo_below_030": ppo_below_030,
            "close_above_rma50": close_above_rma50,
            "close_above_rma200": close_above_rma200,
            "cloud_green": (cloud_state == "green"),
            "strong_bullish_close": strong_bullish_close,
            "magical_hist_curr>0": magical_hist_curr > 0,
        }
        log_cond_set("BUY (SRSI 50) condition inputs:", buy_srsi_conds)
        if current_state is None and all(buy_srsi_conds.values()):
            current_state = "buy_srsi50"
            debug_log(f"\n⬆️ BUY (SRSI 50) SIGNAL DETECTED for {pair_name}!")
            send_message = f"⬆️ {pair_name} - BUY (SRSI 50)\nSRSI 15m Cross Up 50 ({smooth_rsi_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"

        # SELL SRSI 50
        sell_srsi_conds = {
            "srsi_cross_down_50": srsi_cross_down_50,
            "ppo_below_signal": ppo_below_signal,
            "ppo_above_minus030": ppo_above_minus030,
            "close_below_rma50": close_below_rma50,
            "close_below_rma200": close_below_rma200,
            "cloud_red": (cloud_state == "red"),
            "strong_bearish_close": strong_bearish_close,
            "magical_hist_curr<0": magical_hist_curr < 0,
        }
        log_cond_set("SELL (SRSI 50) condition inputs:", sell_srsi_conds)
        if current_state is None and all(sell_srsi_conds.values()):
            current_state = "sell_srsi50"
            debug_log(f"\n⬇️ SELL (SRSI 50) SIGNAL DETECTED for {pair_name}!")
            send_message = f"⬇️ {pair_name} - SELL (SRSI 50)\nSRSI 15m Cross Down 50 ({smooth_rsi_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"

        # LONG (0)
        long0_conds = {
            "ppo_cross_above_zero": ppo_cross_above_zero,
            "ppo_above_signal": ppo_above_signal,
            "close_above_rma50": close_above_rma50,
            "close_above_rma200": close_above_rma200,
            "cloud_green": (cloud_state == "green"),
            "strong_bullish_close": strong_bullish_close,
            "magical_hist_curr>0": magical_hist_curr > 0,
        }
        log_cond_set("LONG(0) condition inputs:", long0_conds)
        if current_state is None and all(long0_conds.values()):
            current_state = "long_zero"
            debug_log(f"\n🟢 LONG (0) SIGNAL DETECTED for {pair_name}!")
            send_message = f"🟢 {pair_name} - LONG\nPPO crossing above 0 ({ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"

        # LONG (0.11)
        long011_conds = {
            "ppo_cross_above_011": ppo_cross_above_011,
            "ppo_above_signal": ppo_above_signal,
            "close_above_rma50": close_above_rma50,
            "close_above_rma200": close_above_rma200,
            "cloud_green": (cloud_state == "green"),
            "strong_bullish_close": strong_bullish_close,
            "magical_hist_curr>0": magical_hist_curr > 0,
        }
        log_cond_set("LONG(0.11) condition inputs:", long011_conds)
        if current_state is None and all(long011_conds.values()):
            current_state = "long_011"
            debug_log(f"\n🟢 LONG (0.11) SIGNAL DETECTED for {pair_name}!")
            send_message = f"🟢 {pair_name} - LONG\nPPO crossing above 0.11 ({ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"

        # SHORT (0)
        short0_conds = {
            "ppo_cross_below_zero": ppo_cross_below_zero,
            "ppo_below_signal": ppo_below_signal,
            "close_below_rma50": close_below_rma50,
            "close_below_rma200": close_below_rma200,
            "cloud_red": (cloud_state == "red"),
            "strong_bearish_close": strong_bearish_close,
            "magical_hist_curr<0": magical_hist_curr < 0,
        }
        log_cond_set("SHORT(0) condition inputs:", short0_conds)
        if current_state is None and all(short0_conds.values()):
            current_state = "short_zero"
            debug_log(f"\n🔴 SHORT (0) SIGNAL DETECTED for {pair_name}!")
            send_message = f"🔴 {pair_name} - SHORT\nPPO crossing below 0 ({ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"

        # SHORT (-0.11)
        short011_conds = {
            "ppo_cross_below_minus011": ppo_cross_below_minus011,
            "ppo_below_signal": ppo_below_signal,
            "close_below_rma50": close_below_rma50,
            "close_below_rma200": close_below_rma200,
            "cloud_red": (cloud_state == "red"),
            "strong_bearish_close": strong_bearish_close,
            "magical_hist_curr<0": magical_hist_curr < 0,
        }
        log_cond_set("SHORT(-0.11) condition inputs:", short011_conds)
        if current_state is None and all(short011_conds.values()):
            current_state = "short_011"
            debug_log(f"\n🔴 SHORT (-0.11) SIGNAL DETECTED for {pair_name}!")
            send_message = f"🔴 {pair_name} - SHORT\nPPO crossing below -0.11 ({ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"

        # Summary and final decision
        debug_log(f"{pair_name} summary: C={close_curr:.4f}, RMA50={rma50_curr:.4f}, RMA200(5m)={rma200_curr:.4f}, PPO={ppo_curr:.4f}/{ppo_signal_curr:.4f}, SRSI={smooth_rsi_curr:.2f}, MMH={magical_hist_curr:.6f}")

        if current_state is None:
            debug_log(f"No signal conditions met for {pair_name}")
            return None

        # Cooldown and duplicate suppression
        should_send = (last_state_value != current_state) or (now_ts - last_ts >= COOLDOWN_SECONDS)
        if should_send and send_message:
            send_telegram_alert(send_message)
        else:
            debug_log(f"{pair_name}: Suppressing alert; cooldown active or state unchanged (last={last_state_value}, now={current_state})")

        # Return structured state
        return {"state": current_state, "ts": now_ts}

    except Exception as e:
        print(f"Error checking {pair_name}: {e}")
        if DEBUG_MODE:
            traceback.print_exc()
        return None

def run_with_jitter(fn, *args, **kwargs):
    time.sleep(np.random.uniform(0.1, 0.7))
    return fn(*args, **kwargs)

def main():
    print("=" * 50)
    ist = pytz.timezone('Asia/Kolkata')
    start_time = datetime.now(ist)
    print(f"PPO/Cirrus Cloud Alert Bot - {start_time.strftime('%d-%m-%Y @ %H:%M IST')}")
    print(f"Debug Mode: {'ON' if DEBUG_MODE else 'OFF'}")
    print("=" * 50)

    if SEND_TEST_MESSAGE and TELEGRAM_BOT_TOKEN != 'xxxx' and TELEGRAM_CHAT_ID != 'xxxx':
        send_test_message()
    elif SEND_TEST_MESSAGE:
        print("⚠️ Skipping test message: Telegram not configured.")

    last_alerts = load_state()

    if not get_product_ids():
        print("Failed to fetch products. Exiting.")
        return

    found_count = sum(1 for v in PAIRS.values() if v is not None)
    print(f"✓ Monitoring {found_count} pairs")
    if found_count == 0:
        print("No valid pairs found. Exiting.")
        return

    results = {}
    max_workers = min(4, found_count)  # reduced threadpool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pair = {}
        for pair_name, pair_info in PAIRS.items():
            if pair_info is not None:
                last_state_for_pair = last_alerts.get(pair_name)
                future = executor.submit(run_with_jitter, check_pair, pair_name, pair_info, last_state_for_pair)
                future_to_pair[future] = pair_name

        for future in as_completed(future_to_pair):
            pair_name = future_to_pair[future]
            try:
                new_state = future.result()
                if new_state is not None:
                    results[pair_name] = new_state
            except Exception as e:
                print(f"Error processing {pair_name} in thread: {e}")
                if DEBUG_MODE:
                    traceback.print_exc()
                continue

    with state_lock:
        for k, v in results.items():
            last_alerts[k] = v  # structured state
        save_state(last_alerts)

    end_time = datetime.now(ist)
    elapsed = (end_time - start_time).total_seconds()
    print(f"✓ Check complete. {len(results)} state updates processed. ({elapsed:.1f}s)")
    print("=" * 50)

if __name__ == "__main__":
    main()
