import requests
import pandas as pd
import numpy as np
import time
import os
import json
import pytz
import traceback
from datetime import datetime
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============ CONFIGURATION ============
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '8462496498:AAHYZ4xDIHvrVRjmCmZyoPhupCjRaRgiITc')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '203813932')

DEBUG_MODE = os.environ.get('DEBUG_MODE', 'True').lower() == 'true'
SEND_TEST_MESSAGE = os.environ.get('SEND_TEST_MESSAGE', 'True').lower() == 'true'
DELTA_API_BASE = "https://api.delta.exchange"

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

SPECIAL_PAIRS = {
    "SOLUSD": {"limit_15m": 150, "min_required": 74, "limit_5m": 300, "min_required_5m": 250}
}

# Indicator settings
PPO_FAST = 7
PPO_SLOW = 16
PPO_SIGNAL = 5
PPO_USE_SMA = False
RMA_50_PERIOD = 50
RMA_200_PERIOD = 200
X1 = 22
X2 = 9
X3 = 15
X4 = 5
SRSI_RSI_LEN = 21
SRSI_KALMAN_LEN = 5
STATE_FILE = 'macd_state.json'
state_lock = Lock()

# ============ UTILITY FUNCTIONS ============
def create_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

SESSION = create_session()

def debug_log(message):
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")

def load_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                raw = json.load(f)
                # Validate structure
                if isinstance(raw, dict):
                    clean = {k: v for k, v in raw.items() if k in PAIRS and isinstance(v, str)}
                    debug_log(f"Loaded state: {clean}")
                    return clean
                else:
                    debug_log("State file invalid format")
        debug_log("No previous state found, starting fresh")
    except Exception as e:
        print(f"Error loading state: {e}")
    return {}

def save_state(state):
    try:
        temp_file = STATE_FILE + '.tmp'
        with open(temp_file, 'w') as f:
            json.dump(state, f)
        os.replace(temp_file, STATE_FILE)
        debug_log(f"Saved state: {state}")
    except Exception as e:
        print(f"Error saving state: {e}")

def send_telegram_alert(message):
    try:
        debug_log(f"Attempting to send message: {message[:100]}...")
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        response = SESSION.post(url, data=data, timeout=10)
        response_data = response.json()
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
        print("✗ Test message failed")
    print("="*50 + "\n")
    return success

def get_product_ids():
    try:
        debug_log("Fetching product IDs from Delta Exchange...")
        response = SESSION.get(f"{DELTA_API_BASE}/v2/products", timeout=10)
        data = response.json()
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
        data = response.json()
        if data.get('success'):
            result = data['result']
            df = pd.DataFrame({
                'timestamp': result['t'],
                'open': result['o'],
                'high': result['h'],
                'low': result['l'],
                'close': result['c'],
                'volume': result['v']
            })
            if len(df) == 0:
                debug_log(f"No candles returned for {product_id}")
                return None
            df = df.sort_values('timestamp').reset_index(drop=True)
            if df['close'].iloc[-1] <= 0:
                debug_log(f"Invalid price (≤0) for {product_id}, skipping")
                return None
            debug_log(f"Received {len(df)} candles for {product_id} ({resolution}m)")
            last_candle_time = df['timestamp'].iloc[-1]
            time_diff = time.time() - last_candle_time
            max_age = int(resolution) * 60 * 3
            if time_diff > max_age:
                print(f"⚠️ Warning: Stale data for {product_id} ({resolution}m) - {time_diff/60:.1f} min old")
                return None
            return df
        else:
            print(f"Error fetching candles for {product_id}: {data.get('message', 'No message')}")
            return None
    except Exception as e:
        print(f"Exception fetching candles for {product_id}: {e}")
        if DEBUG_MODE:
            traceback.print_exc()
        return None

# --- Indicator Functions ---
def calculate_ema(data, period): return data.ewm(span=period, adjust=False).mean()
def calculate_sma(data, period): return data.rolling(window=period).mean()
def calculate_rma(data, period): return data.ewm(alpha=1/period, adjust=False).mean()

def calculate_ppo(df, fast=7, slow=16, signal=5, use_sma=False):
    close = df['close']
    fast_ma = calculate_sma(close, fast) if use_sma else calculate_ema(close, fast)
    slow_ma = calculate_sma(close, slow) if use_sma else calculate_ema(close, slow)
    ppo = (fast_ma - slow_ma) / slow_ma * 100
    ppo_signal = calculate_sma(ppo, signal) if use_sma else calculate_ema(ppo, signal)
    return ppo, ppo_signal

def smoothrng(x, t, m):
    wper = t * 2 - 1
    avrng = calculate_ema(np.abs(x.diff()), t)
    return calculate_ema(avrng, wper) * m

def rngfilt(x, r):
    result_list = [x.iloc[0]]
    for i in range(1, len(x)):
        prev_f = result_list[-1]
        curr_x, curr_r = x.iloc[i], r.iloc[i]
        if curr_x > prev_f:
            f = prev_f if (curr_x - curr_r) < prev_f else (curr_x - curr_r)
        else:
            f = prev_f if (curr_x + curr_r) > prev_f else (curr_x + curr_r)
        result_list.append(f)
    return pd.Series(result_list, index=x.index)

def calculate_cirrus_cloud(df):
    close = df['close'].copy()
    filtx1 = rngfilt(close, smoothrng(close, X1, X2))
    filtx12 = rngfilt(close, smoothrng(close, X3, X4))
    upw = filtx1 < filtx12
    dnw = filtx1 > filtx12
    return upw, dnw, filtx1, filtx12

def kalman_filter(src, length, R=0.01, Q=0.1):
    result_list = []
    estimate = np.nan
    error_est = 1.0
    error_meas = R * length
    Q_div_length = Q / length
    for i in range(len(src)):
        curr = src.iloc[i]
        if np.isnan(estimate):
            estimate = src.iloc[i-1] if i > 0 else np.nan
            if np.isnan(estimate):
                result_list.append(np.nan)
                continue
        prediction = estimate
        kalman_gain = error_est / (error_est + error_meas)
        estimate = prediction + kalman_gain * (curr - prediction)
        error_est = (1 - kalman_gain) * error_est + Q_div_length
        result_list.append(estimate)
    return pd.Series(result_list, index=src.index)

def calculate_smooth_rsi(df, rsi_len=SRSI_RSI_LEN, kalman_len=SRSI_KALMAN_LEN):
    close = df['close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = calculate_rma(gain, rsi_len)
    avg_loss = calculate_rma(loss, rsi_len)
    rs = avg_gain.divide(avg_loss.replace(0, np.nan))
    rsi_value = 100 - (100 / (1 + rs))
    return kalman_filter(rsi_value, kalman_len)

def calculate_magical_momentum_hist(df, period=144, responsiveness=0.9):
    if len(df) < period + 50:
        return pd.Series([np.nan] * len(df), index=df.index)
    close = df['close'].astype(float)
    responsiveness = max(0.00001, responsiveness)
    sd = close.rolling(window=50).std() * responsiveness
    sd = sd.bfill().fillna(0.001)
    worm = close.copy()
    for i in range(1, len(close)):
        diff = close.iloc[i] - worm.iloc[i - 1]
        abs_diff = abs(diff)
        delta = np.sign(diff) * sd.iloc[i] if abs_diff > sd.iloc[i] else diff
        worm.iloc[i] = worm.iloc[i - 1] + delta
    ma = close.rolling(window=period).mean()
    raw_momentum = (worm - ma) / worm.replace(0, np.nan)
    raw_momentum = raw_momentum.fillna(0)
    min_med = raw_momentum.rolling(window=period).min().fillna(0)
    max_med = raw_momentum.rolling(window=period).max().fillna(0)
    rng = np.maximum(1e-10, max_med - min_med)
    temp = (raw_momentum - min_med) / rng
    value = pd.Series(0.0, index=df.index)
    value.iloc[0] = 0.0
    for i in range(1, len(temp)):
        v_prev = value.iloc[i - 1]
        v_new = 1.0 * (temp.iloc[i] - 0.5 + 0.5 * v_prev)
        value.iloc[i] = max(-0.9999, min(0.9999, v_new))
    temp2 = (1 + value) / (1 - value)
    temp2 = temp2.replace([np.inf, -np.inf], np.nan)
    log_temp2 = np.log(temp2)
    momentum = 0.25 * log_temp2
    momentum = momentum.fillna(0)
    hist = pd.Series(0.0, index=df.index)
    hist.iloc[0] = momentum.iloc[0]
    for i in range(1, len(momentum)):
        hist.iloc[i] = momentum.iloc[i] + 0.5 * hist.iloc[i - 1]
    return hist.fillna(0)

def diagnose_failure(pair_name, **kwargs):
    """Log why a near-miss didn't trigger"""
    missing = []
    if not kwargs.get('strong_bearish_close') and kwargs.get('bearish_intent'):
        missing.append("strong_bearish_close")
    if not kwargs.get('strong_bullish_close') and kwargs.get('bullish_intent'):
        missing.append("strong_bullish_close")
    if not kwargs.get('ppo_cross_down') and kwargs.get('short_intent'):
        missing.append("ppo_cross_down")
    if not kwargs.get('ppo_cross_up') and kwargs.get('long_intent'):
        missing.append("ppo_cross_up")
    if not kwargs.get('magical_hist_positive') and kwargs.get('long_intent'):
        missing.append("magical_hist > 0")
    if not kwargs.get('magical_hist_negative') and kwargs.get('short_intent'):
        missing.append("magical_hist < 0")
    if missing:
        debug_log(f"⚠️ NEAR-MISS {pair_name}: missing {', '.join(missing)}")

def check_pair(pair_name, pair_info, last_state_for_pair):
    try:
        if pair_info is None:
            return None
        debug_log(f"\n{'='*60}")
        debug_log(f"Checking {pair_name}")
        debug_log(f"{'='*60}")

        cfg = SPECIAL_PAIRS.get(pair_name,
            {"limit_15m": 210, "min_required": 200, "limit_5m": 300, "min_required_5m": 250})
        df_15m = get_candles(pair_info['symbol'], "15", limit=cfg["limit_15m"])
        df_5m = get_candles(pair_info['symbol'], "5", limit=cfg["limit_5m"])
        if df_15m is None or len(df_15m) < cfg["min_required"]:
            print(f"⚠️ Insufficient 15m data for {pair_name}: {len(df_15m) if df_15m is not None else 0}/{cfg['min_required']}")
            return None
        if df_5m is None or len(df_5m) < cfg["min_required_5m"]:
            print(f"⚠️ Insufficient 5m data for {pair_name}: {len(df_5m) if df_5m is not None else 0}/{cfg['min_required_5m']}")
            return None

        magical_hist = calculate_magical_momentum_hist(df_15m)
        ppo, ppo_signal = calculate_ppo(df_15m, PPO_FAST, PPO_SLOW, PPO_SIGNAL, PPO_USE_SMA)
        rma_50 = calculate_rma(df_15m['close'], RMA_50_PERIOD)
        rma_200 = calculate_rma(df_5m['close'], RMA_200_PERIOD)
        upw, dnw, filtx1, filtx12 = calculate_cirrus_cloud(df_15m)
        smooth_rsi = calculate_smooth_rsi(df_15m)

        # Get latest values with safety
        def safe_get(series, idx=-1):
            val = series.iloc[idx]
            return float(val) if not pd.isna(val) else np.nan

        ppo_curr, ppo_prev = safe_get(ppo, -1), safe_get(ppo, -2)
        sig_curr, sig_prev = safe_get(ppo_signal, -1), safe_get(ppo_signal, -2)
        smooth_curr, smooth_prev = safe_get(smooth_rsi, -1), safe_get(smooth_rsi, -2)
        rma50_curr = safe_get(rma_50)
        rma200_curr = safe_get(rma_200)
        magical_hist_curr = safe_get(magical_hist)

        for v in [ppo_curr, sig_curr, smooth_curr, rma50_curr, rma200_curr, magical_hist_curr]:
            if pd.isna(v):
                debug_log(f"⚠️ NaN in indicators for {pair_name}, skipping")
                return None

        open_curr = df_15m['open'].iloc[-1]
        high_curr = df_15m['high'].iloc[-1]
        low_curr = df_15m['low'].iloc[-1]
        close_curr = df_15m['close'].iloc[-1]
        total_range = high_curr - low_curr

        bullish = close_curr > open_curr
        bearish = close_curr < open_curr
        strong_bullish_close = False
        strong_bearish_close = False

        if total_range > 1e-10:
            upper_wick = high_curr - max(open_curr, close_curr)
            lower_wick = min(open_curr, close_curr) - low_curr
            strong_bullish_close = bullish and (upper_wick / total_range) < 0.20
            strong_bearish_close = bearish and (lower_wick / total_range) < 0.20
        else:
            strong_bullish_close = bullish
            strong_bearish_close = bearish

        # Conditions
        ppo_cross_up = (ppo_prev <= sig_prev) and (ppo_curr > sig_curr)
        ppo_cross_down = (ppo_prev >= sig_prev) and (ppo_curr < sig_curr)
        close_above_rma50 = close_curr > rma50_curr
        close_below_rma50 = close_curr < rma50_curr
        close_above_rma200 = close_curr > rma200_curr
        close_below_rma200 = close_curr < rma200_curr
        srsi_cross_up_50 = (smooth_prev <= 50) and (smooth_curr > 50)
        srsi_cross_down_50 = (smooth_prev >= 50) and (smooth_curr < 50)
        magical_hist_positive = magical_hist_curr > 0
        magical_hist_negative = magical_hist_curr < 0

        debug_log(f"\nCandle Metrics (15m): O:{open_curr:.2f} H:{high_curr:.2f} L:{low_curr:.2f} C:{close_curr:.2f}")
        debug_log(f"Strong Bullish Close: {strong_bullish_close}, Strong Bearish Close: {strong_bearish_close}")
        debug_log(f"Magical Hist: {magical_hist_curr:.4f}, PPO: {ppo_curr:.4f}, Signal: {sig_curr:.4f}")
        debug_log(f"RMA50: {rma50_curr:.2f}, RMA200: {rma200_curr:.2f}, Cloud: Upw={upw.iloc[-1]}, Dnw={dnw.iloc[-1]}")

        current_state = None
        ist = pytz.timezone('Asia/Kolkata')
        formatted_time = datetime.now(ist).strftime('%d-%m-%Y @ %H:%M IST')
        price = close_curr

        # === SIGNAL CHECKS ===
        short_intent = (close_below_rma50 and close_below_rma200 and dnw.iloc[-1] and magical_hist_negative)
        long_intent = (close_above_rma50 and close_above_rma200 and upw.iloc[-1] and magical_hist_positive)

        if (ppo_cross_up and ppo_curr < 0.20 and long_intent and strong_bullish_close):
            current_state = "buy"
            if last_state_for_pair != "buy":
                send_telegram_alert(f"🟢 {pair_name} - BUY\nPPO: {ppo_curr:.2f}\nPrice: ${price:,.2f}\n{formatted_time}")
        elif (ppo_cross_down and ppo_curr > -0.20 and short_intent and strong_bearish_close):
            current_state = "sell"
            if last_state_for_pair != "sell":
                send_telegram_alert(f"🔴 {pair_name} - SELL\nPPO: {ppo_curr:.2f}\nPrice: ${price:,.2f}\n{formatted_time}")
        elif (srsi_cross_up_50 and ppo_curr > sig_curr and ppo_curr < 0.30 and long_intent and strong_bullish_close):
            current_state = "buy_srsi50"
            if last_state_for_pair != "buy_srsi50":
                send_telegram_alert(f"⬆️ {pair_name} - BUY (SRSI 50)\nSRSI: {smooth_curr:.2f}\nPrice: ${price:,.2f}\n{formatted_time}")
        elif (srsi_cross_down_50 and ppo_curr < sig_curr and ppo_curr > -0.30 and short_intent and strong_bearish_close):
            current_state = "sell_srsi50"
            if last_state_for_pair != "sell_srsi50":
                send_telegram_alert(f"⬇️ {pair_name} - SELL (SRSI 50)\nSRSI: {smooth_curr:.2f}\nPrice: ${price:,.2f}\n{formatted_time}")
        elif ( (ppo_prev <= 0 and ppo_curr > 0) and ppo_curr > sig_curr and long_intent and strong_bullish_close):
            current_state = "long_zero"
            if last_state_for_pair != "long_zero":
                send_telegram_alert(f"🟢 {pair_name} - LONG (0)\nPPO: {ppo_curr:.2f}\nPrice: ${price:,.2f}\n{formatted_time}")
        elif ( (ppo_prev <= 0.11 and ppo_curr > 0.11) and ppo_curr > sig_curr and long_intent and strong_bullish_close):
            current_state = "long_011"
            if last_state_for_pair != "long_011":
                send_telegram_alert(f"🟢 {pair_name} - LONG (0.11)\nPPO: {ppo_curr:.2f}\nPrice: ${price:,.2f}\n{formatted_time}")
        elif ( (ppo_prev >= 0 and ppo_curr < 0) and ppo_curr < sig_curr and short_intent and strong_bearish_close):
            current_state = "short_zero"
            if last_state_for_pair != "short_zero":
                send_telegram_alert(f"🔴 {pair_name} - SHORT (0)\nPPO: {ppo_curr:.2f}\nPrice: ${price:,.2f}\n{formatted_time}")
        elif ( (ppo_prev >= -0.11 and ppo_curr < -0.11) and ppo_curr < sig_curr and short_intent and strong_bearish_close):
            current_state = "short_011"
            if last_state_for_pair != "short_011":
                send_telegram_alert(f"🔴 {pair_name} - SHORT (-0.11)\nPPO: {ppo_curr:.2f}\nPrice: ${price:,.2f}\n{formatted_time}")
        else:
            # === DIAGNOSTIC: Log near misses ===
            if short_intent and ppo_curr < sig_curr and not ppo_cross_down:
                diagnose_failure(pair_name, short_intent=True, ppo_cross_down=False,
                                strong_bearish_close=strong_bearish_close, magical_hist_negative=magical_hist_negative)
            if long_intent and ppo_curr > sig_curr and not ppo_cross_up:
                diagnose_failure(pair_name, long_intent=True, ppo_cross_up=False,
                                strong_bullish_close=strong_bullish_close, magical_hist_positive=magical_hist_positive)
            debug_log(f"No signal conditions met for {pair_name}")

        return current_state

    except Exception as e:
        print(f"Error checking {pair_name}: {e}")
        if DEBUG_MODE:
            traceback.print_exc()
        return None

def main():
    print("=" * 50)
    ist = pytz.timezone('Asia/Kolkata')
    start_time = datetime.now(ist)
    print(f"PPO/Cirrus Cloud Alert Bot - {start_time.strftime('%d-%m-%Y @ %H:%M IST')}")
    print(f"Debug Mode: {'ON' if DEBUG_MODE else 'OFF'}")
    print("=" * 50)

    if SEND_TEST_MESSAGE:
        send_test_message()

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
    # Reduce workers to avoid Delta API throttling
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_pair = {
            executor.submit(check_pair, name, info, last_alerts.get(name)): name
            for name, info in PAIRS.items() if info is not None
        }
        for future in as_completed(future_to_pair):
            name = future_to_pair[future]
            try:
                results[name] = future.result()
            except Exception as e:
                print(f"Thread error for {name}: {e}")

    with state_lock:
        # Only update non-None results
        last_alerts.update({k: v for k, v in results.items() if v is not None})
        save_state(last_alerts)

    elapsed = (datetime.now(ist) - start_time).total_seconds()
    print(f"✓ Check complete. {len({k:v for k,v in results.items() if v is not None})} state updates. ({elapsed:.1f}s)")
    print("=" * 50)

if __name__ == "__main__":
    main()
