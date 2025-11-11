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

# --- Pair Definitions ---
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

# --- Indicator Settings ---
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
MOMENTUM_HIST_PERIOD = 144
MOMENTUM_HIST_RESPONSIVENESS = 0.9

# --- Signal Thresholds (NEW: Replaces 'magic numbers' for tunability) ---
PPO_ENTRY_MAX_BUY = 0.20
PPO_ENTRY_MIN_SELL = -0.20
PPO_SRSI_MAX_BUY = 0.30
PPO_SRSI_MIN_SELL = -0.30
PPO_ZERO_LINE = 0.0
PPO_UP_LEVEL = 0.11
PPO_DOWN_LEVEL = -0.11

# --- State Management ---
# IMPROVEMENT: Use an absolute path derived from the script's directory for Cron/GitHub stability
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
STATE_FILE = os.path.join(SCRIPT_DIR, 'macd_state.json') 
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
        [span_0](start_span)print(f"[DEBUG] {message}")[span_0](end_span)

def load_state():
    # [span_1](start_span)IMPROVEMENT: Safe state parsing already implemented[span_1](end_span)
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                raw = json.load(f)
                if isinstance(raw, dict):
                    # Filter out stale pairs that might be in the JSON but not in PAIRS
                    clean = {k: v for k, v in raw.items() if 
                             [span_2](start_span)k in PAIRS and isinstance(v, str)}[span_2](end_span)
                    debug_log(f"Loaded state: {clean}")
                    return clean
    except Exception as e:
        [span_3](start_span)print(f"Error loading state (starting fresh): {e}")[span_3](end_span)
    debug_log("No previous state found, starting fresh")
    return {}

def save_state(state):
    # [span_4](start_span)IMPROVEMENT: Uses atomic file swap for safety[span_4](end_span)
    try:
        temp_file = STATE_FILE + '.tmp'
        [span_5](start_span)with open(temp_file, 'w') as f:[span_5](end_span)
            json.dump(state, f)
        os.replace(temp_file, STATE_FILE)
        debug_log(f"Saved state: {state}")
    except Exception as e:
        [span_6](start_span)print(f"Error saving state: {e}")[span_6](end_span)

def send_telegram_alert(message):
    try:
        [span_7](start_span)debug_log(f"Attempting to send message: {message[:100]}...")[span_7](end_span)
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        [span_8](start_span)response = SESSION.post(url, data=data, timeout=10)[span_8](end_span)
        response_data = response.json()
        if response_data.get('ok'):
            print(f"✓ Alert sent successfully")
            return True
        else:
            print(f"✗ Telegram error: {response_data}")
            return False
    [span_9](start_span)except Exception as e:[span_9](end_span)
        print(f"✗ Error sending Telegram message: {e}")
        if DEBUG_MODE:
            traceback.print_exc()
        return False

def send_test_message():
    [span_10](start_span)ist = pytz.timezone('Asia/Kolkata')[span_10](end_span)
    current_dt = datetime.now(ist)
    formatted_time = current_dt.strftime('%d-%m-%Y @ %H:%M IST')
    test_msg = f"🔔 Bot Started\nTest message from PPO Bot\nTime: {formatted_time}\nDebug Mode: {'ON' if DEBUG_MODE else 'OFF'}"
    [span_11](start_span)print("\n" + "="*50)[span_11](end_span)
    print("SENDING TEST MESSAGE")
    print("="*50)
    [span_12](start_span)success = send_telegram_alert(test_msg)[span_12](end_span)
    if success:
        print("✓ Test message sent successfully!")
    else:
        print("✗ Test message failed")
    print("="*50 + "\n")
    return success

# ... (get_product_ids remains unchanged)
def get_product_ids():
    try:
        [span_13](start_span)debug_log("Fetching product IDs from Delta Exchange...")[span_13](end_span)
        response = SESSION.get(f"{DELTA_API_BASE}/v2/products", timeout=10)
        data = response.json()
        if data.get('success'):
            [span_14](start_span)products = data['result'][span_14](end_span)
            debug_log(f"Received {len(products)} products from API")
            product_map = {}
            for product in products:
                [span_15](start_span)if product.get('contract_type') == 'perpetual_futures':[span_15](end_span)
                    product_map[product['symbol']] = product

            [span_16](start_span)for pair_name in list(PAIRS.keys()):[span_16](end_span)
                delta_symbol = pair_name.replace('USD', 'USDT')
                if delta_symbol in product_map:
                    product = product_map[delta_symbol]
                    PAIRS[pair_name] = {
                        [span_17](start_span)'id': product['id'],[span_17](end_span)
                        'symbol': product['symbol'],
                        'contract_type': product['contract_type']
                    }
                    [span_18](start_span)debug_log(f"Matched {pair_name} -> {product['symbol']} (ID: {product['id']})")[span_18](end_span)
                else:
                    debug_log(f"No match found for {pair_name} (tried {delta_symbol})")
            return True
        else:
            [span_19](start_span)print(f"API Error: {data}")[span_19](end_span)
            return False
    [span_20](start_span)except Exception as e:[span_20](end_span)
        print(f"Error fetching products: {e}")
        if DEBUG_MODE:
            traceback.print_exc()
        return False

# ... (get_candles remains unchanged, includes stale data check)
def get_candles(product_id, resolution="15", limit=150):
    try:
        [span_21](start_span)to_time = int(time.time())[span_21](end_span)
        from_time = to_time - (limit * int(resolution) * 60)
        url = f"{DELTA_API_BASE}/v2/chart/history"
        [span_22](start_span)params = {[span_22](end_span)
            'resolution': resolution,
            'symbol': product_id,
            'from': from_time,
            'to': to_time
        }
        [span_23](start_span)debug_log(f"Fetching {resolution}m candles for {product_id}, limit={limit}")[span_23](end_span)
        response = SESSION.get(url, params=params, timeout=15)
        [span_24](start_span)data = response.json()[span_24](end_span)
        if data.get('success'):
            result = data['result']
            df = pd.DataFrame({
                'timestamp': result['t'],
                'open': result['o'],
                'high': result['h'],
                [span_25](start_span)'low': result['l'],[span_25](end_span)
                'close': result['c'],
                'volume': result['v']
            })
            if len(df) == 0:
                debug_log(f"No candles returned for {product_id}")
                [span_26](start_span)return None[span_26](end_span)
            df = df.sort_values('timestamp').reset_index(drop=True)
            if df['close'].iloc[-1] <= 0:
                debug_log(f"Invalid price (≤0) for {product_id}, skipping")
                return None
            debug_log(f"Received {len(df)} candles for {product_id} ({resolution}m)")
            
            # Stale data check
            [span_27](start_span)last_candle_time = df['timestamp'].iloc[-1][span_27](end_span)
            time_diff = time.time() - last_candle_time
            max_age = int(resolution) * 60 * 3
            if time_diff > max_age:
                print(f"⚠️ Warning: Stale data for {product_id} ({resolution}m) - {time_diff/60:.1f} min old")
                return None
      
            [span_28](start_span)return df[span_28](end_span)
        else:
            print(f"Error fetching candles for {product_id}: {data.get('message', 'No message')}")
            return None
    except Exception as e:
        [span_29](start_span)print(f"Exception fetching candles for {product_id}: {e}")[span_29](end_span)
        if DEBUG_MODE:
            traceback.print_exc()
        return None

# ... (Indicator functions remain unchanged)
[span_30](start_span)def calculate_ema(data, period): return data.ewm(span=period, adjust=False).mean()[span_30](end_span)
[span_31](start_span)def calculate_sma(data, period): return data.rolling(window=period).mean()[span_31](end_span)
[span_32](start_span)def calculate_rma(data, period): return data.ewm(alpha=1/period, adjust=False).mean()[span_32](end_span)

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
    [span_33](start_span)return calculate_ema(avrng, wper) * m[span_33](end_span)

def rngfilt(x, r):
    result_list = [x.iloc[0]]
    for i in range(1, len(x)):
        prev_f = result_list[-1]
        curr_x, curr_r = x.iloc[i], r.iloc[i]
        if curr_x > prev_f:
            f = prev_f if (curr_x - curr_r) < prev_f else (curr_x - curr_r)
        else:
            [span_34](start_span)f = prev_f if (curr_x + curr_r) > prev_f else (curr_x + curr_r)[span_34](end_span)
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
    [span_35](start_span)error_meas = R * length[span_35](end_span)
    Q_div_length = Q / length
    for i in range(len(src)):
        curr = src.iloc[i]
        if np.isnan(estimate):
            estimate = src.iloc[i-1] if i > 0 else np.nan
            if np.isnan(estimate):
                result_list.append(np.nan)
                [span_36](start_span)continue[span_36](end_span)
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
    [span_37](start_span)loss = -delta.where(delta < 0, 0)[span_37](end_span)
    avg_gain = calculate_rma(gain, rsi_len)
    avg_loss = calculate_rma(loss, rsi_len)
    rs = avg_gain.divide(avg_loss.replace(0, np.nan))
    rsi_value = 100 - (100 / (1 + rs))
    return kalman_filter(rsi_value, kalman_len)

def calculate_magical_momentum_hist(df, period=MOMENTUM_HIST_PERIOD, responsiveness=MOMENTUM_HIST_RESPONSIVENESS):
    # IMPROVEMENT: Checks for minimal required data length for this complex indicator
    if len(df) < period + 50:
        return pd.Series([np.nan] * len(df), index=df.index)
        
    close = df['close'].astype(float)
    responsiveness = max(0.00001, responsiveness)
    sd = close.rolling(window=50).std() * responsiveness
    sd = sd.bfill().fillna(0.001)
    [span_38](start_span)worm = close.copy()[span_38](end_span)
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
 
    [span_39](start_span)rng = np.maximum(1e-10, max_med - min_med)[span_39](end_span)
    temp = (raw_momentum - min_med) / rng
    value = pd.Series(0.0, index=df.index)
    value.iloc[0] = 0.0
    for i in range(1, len(temp)):
        v_prev = value.iloc[i - 1]
        v_new = 1.0 * (temp.iloc[i] - 0.5 + 0.5 * v_prev)
        value.iloc[i] = max(-0.9999, min(0.9999, v_new))
    temp2 = (1 + value) / (1 - value)
    [span_40](start_span)temp2 = temp2.replace([np.inf, -np.inf], np.nan)[span_40](end_span)
    log_temp2 = np.log(temp2)
    momentum = 0.25 * log_temp2
    momentum = momentum.fillna(0)
    hist = pd.Series(0.0, index=df.index)
    hist.iloc[0] = momentum.iloc[0]
    for i in range(1, len(momentum)):
        hist.iloc[i] = momentum.iloc[i] + 0.5 * hist.iloc[i - 1]
    return hist.fillna(0)

def diagnose_failure(pair_name, **kwargs):
    missing = []
    [span_41](start_span)if not kwargs.get('strong_bearish_close') and kwargs.get('short_intent'):[span_41](end_span)
        missing.append("strong_bearish_close")
    if not kwargs.get('strong_bullish_close') and kwargs.get('long_intent'):
        [span_42](start_span)missing.append("strong_bullish_close")[span_42](end_span)
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
        [span_43](start_span)if pair_info is None:[span_43](end_span)
            return None
        debug_log(f"\n{'='*60}")
        debug_log(f"Checking {pair_name}")
        debug_log(f"{'='*60}")

        cfg = SPECIAL_PAIRS.get(pair_name,
            {"limit_15m": 210, "min_required": 200, "limit_5m": 300, "min_required_5m": 250})
        df_15m = get_candles(pair_info['symbol'], "15", limit=cfg["limit_15m"])
        df_5m = get_candles(pair_info['symbol'], "5", limit=cfg["limit_5m"])
  
        [span_44](start_span)if df_15m is None or len(df_15m) < cfg["min_required"]:[span_44](end_span)
            print(f"⚠️ Insufficient 15m data for {pair_name}: {len(df_15m) if df_15m is not None else 0}/{cfg['min_required']}")
            return None
        if df_5m is None or len(df_5m) < cfg["min_required_5m"]:
            print(f"⚠️ Insufficient 5m data for {pair_name}: {len(df_5m) if df_5m is not None else 0}/{cfg['min_required_5m']}")
            [span_45](start_span)return None[span_45](end_span)

        magical_hist = calculate_magical_momentum_hist(df_15m)
        ppo, ppo_signal = calculate_ppo(df_15m, PPO_FAST, PPO_SLOW, PPO_SIGNAL, PPO_USE_SMA)
        rma_50 = calculate_rma(df_15m['close'], RMA_50_PERIOD)
        rma_200 = calculate_rma(df_5m['close'], RMA_200_PERIOD)
        upw, dnw, filtx1, filtx12 = calculate_cirrus_cloud(df_15m)
        smooth_rsi = calculate_smooth_rsi(df_15m)

        def safe_get(series, idx=-1):
            val = series.iloc[idx]
            [span_46](start_span)return float(val) if not pd.isna(val) else np.nan[span_46](end_span)

        ppo_curr, ppo_prev = safe_get(ppo, -1), safe_get(ppo, -2)
        sig_curr, sig_prev = safe_get(ppo_signal, -1), safe_get(ppo_signal, -2)
        smooth_curr, smooth_prev = safe_get(smooth_rsi, -1), safe_get(smooth_rsi, -2)
        rma50_curr = safe_get(rma_50)
        rma200_curr = safe_get(rma_200)
        magical_hist_curr = safe_get(magical_hist)

        # Robust NaN check
        [span_47](start_span)for v in [ppo_curr, sig_curr, smooth_curr, rma50_curr, rma200_curr, magical_hist_curr]:[span_47](end_span)
            if pd.isna(v):
                debug_log(f"⚠️ NaN in indicators for {pair_name}, skipping")
                return None

        open_curr = df_15m['open'].iloc[-1]
        high_curr = df_15m['high'].iloc[-1]
        low_curr = df_15m['low'].iloc[-1]
        close_curr = df_15m['close'].iloc[-1]
    
        [span_48](start_span)total_range = high_curr - low_curr[span_48](end_span)

        bullish = close_curr > open_curr
        bearish = close_curr < open_curr
        strong_bullish_close = False
        strong_bearish_close = False

        # Zero Range Candle Fix (Code is already robust here, simplifying presentation)
        [span_49](start_span)if total_range > 1e-10:[span_49](end_span)
            upper_wick = high_curr - max(open_curr, close_curr)
            lower_wick = min(open_curr, close_curr) - low_curr
            [span_50](start_span)strong_bullish_close = bullish and (upper_wick / total_range) < 0.20[span_50](end_span)
            strong_bearish_close = bearish and (lower_wick / total_range) < 0.20
        else:
            # ORIGINAL FIX: Handle zero-range candles by relying only on direction (or lack thereof)
            strong_bullish_close = bullish
            [span_51](start_span)strong_bearish_close = bearish[span_51](end_span)

        [span_52](start_span)debug_log(f"\nCandle Metrics (15m):")[span_52](end_span)
        debug_log(f"  O:{open_curr:.2f} H:{high_curr:.2f} L:{low_curr:.2f} C:{close_curr:.2f}")
        if total_range > 1e-10:
            [span_53](start_span)debug_log(f"  Strong Bullish Close (20% Rule): {strong_bullish_close}")[span_53](end_span)
            debug_log(f"  Strong Bearish Close (20% Rule): {strong_bearish_close}")
        else:
            [span_54](start_span)debug_log("  Candle range is zero; using directional close only.")[span_54](end_span)

        [span_55](start_span)debug_log(f"\nIndicator Values:")[span_55](end_span)
        debug_log(f"Price (15m): ${close_curr:,.2f}")
        debug_log(f"PPO: {ppo_curr:.4f} (prev: {ppo_prev:.4f})")
        debug_log(f"PPO Signal: {sig_curr:.4f} (prev: {sig_prev:.4f})")
        debug_log(f"RMA50 (15m): {rma50_curr:.2f}")
        debug_log(f"RMA200 (5m): {rma200_curr:.2f}")
        debug_log(f"Smoothed RSI (15m): {smooth_curr:.2f} (prev: {smooth_prev:.2f})")
        [span_56](start_span)debug_log(f"Cirrus Filter 1: {filtx1.iloc[-1]:.4f}, Filter 2: {filtx12.iloc[-1]:.4f}")[span_56](end_span)
        [span_57](start_span)debug_log(f"Cirrus Cloud - Upw: {upw.iloc[-1]}, Dnw: {dnw.iloc[-1]}")[span_57](end_span)

        [span_58](start_span)ppo_cross_up = (ppo_prev <= sig_prev) and (ppo_curr > sig_curr)[span_58](end_span)
        ppo_cross_down = (ppo_prev >= sig_prev) and (ppo_curr < sig_curr)
        close_above_rma50 = close_curr > rma50_curr
        close_below_rma50 = close_curr < rma50_curr
        close_above_rma200 = close_curr > rma200_curr
        close_below_rma200 = close_curr < rma200_curr
        [span_59](start_span)srsi_cross_up_50 = (smooth_prev <= 50) and (smooth_curr > 50)[span_59](end_span)
        srsi_cross_down_50 = (smooth_prev >= 50) and (smooth_curr < 50)
        magical_hist_positive = magical_hist_curr > 0
        magical_hist_negative = magical_hist_curr < 0

        [span_60](start_span)debug_log(f"\nCrossover Checks:")[span_60](end_span)
        debug_log(f"  PPO 15m cross up: {ppo_cross_up}")
        debug_log(f"  PPO 15m cross down: {ppo_cross_down}")
        debug_log(f"  SRSI cross up 50: {srsi_cross_up_50}")
        [span_61](start_span)debug_log(f"  SRSI cross down 50: {srsi_cross_down_50}")[span_61](end_span)

        current_state = None
        ist = pytz.timezone('Asia/Kolkata')
        formatted_time = datetime.now(ist).strftime('%d-%m-%Y @ %H:%M IST')
        price = close_curr

        # Intent logic (gating condition)
        [span_62](start_span)short_intent = (close_below_rma50 and close_below_rma200 and dnw.iloc[-1] and magical_hist_negative)[span_62](end_span)
        long_intent = (close_above_rma50 and close_above_rma200 and upw.iloc[-1] and magical_hist_positive)

        # --- SIGNAL CONDITIONS (Using new constants) ---

        # 1. PPO Cross Entry
        [span_63](start_span)if (ppo_cross_up and ppo_curr < PPO_ENTRY_MAX_BUY and long_intent and strong_bullish_close): # 0.20[span_63](end_span)
            current_state = "buy"
            if last_state_for_pair != "buy":
                send_telegram_alert(f"🟢 {pair_name} - BUY\nPPO: {ppo_curr:.2f}\nPrice: ${price:,.2f}\n{formatted_time}")
        [span_64](start_span)elif (ppo_cross_down and ppo_curr > PPO_ENTRY_MIN_SELL and short_intent and strong_bearish_close): # -0.20[span_64](end_span)
            current_state = "sell"
            if last_state_for_pair != "sell":
                [span_65](start_span)send_telegram_alert(f"🔴 {pair_name} - SELL\nPPO: {ppo_curr:.2f}\nPrice: ${price:,.2f}\n{formatted_time}")[span_65](end_span)

        # 2. SRSI 50 Cross Entry
        [span_66](start_span)elif (srsi_cross_up_50 and ppo_curr > sig_curr and ppo_curr < PPO_SRSI_MAX_BUY and long_intent and strong_bullish_close): # 0.30[span_66](end_span)
            current_state = "buy_srsi50"
            if last_state_for_pair != "buy_srsi50":
                send_telegram_alert(f"⬆️ {pair_name} - BUY (SRSI 50)\nSRSI: {smooth_curr:.2f}\nPrice: ${price:,.2f}\n{formatted_time}")
        [span_67](start_span)elif (srsi_cross_down_50 and ppo_curr < sig_curr and ppo_curr > PPO_SRSI_MIN_SELL and short_intent and strong_bearish_close): # -0.30[span_67](end_span)
            current_state = "sell_srsi50"
            if last_state_for_pair != "sell_srsi50":
                [span_68](start_span)send_telegram_alert(f"⬇️ {pair_name} - SELL (SRSI 50)\nSRSI: {smooth_curr:.2f}\nPrice: ${price:,.2f}\n{formatted_time}")[span_68](end_span)

        # 3. PPO Zero Line Cross
        [span_69](start_span)elif ( (ppo_prev <= PPO_ZERO_LINE and ppo_curr > PPO_ZERO_LINE) and ppo_curr > sig_curr and long_intent and strong_bullish_close): # 0[span_69](end_span)
            current_state = "long_zero"
            if last_state_for_pair != "long_zero":
                [span_70](start_span)send_telegram_alert(f"🟢 {pair_name} - LONG (0)\nPPO: {ppo_curr:.2f}\nPrice: ${price:,.2f}\n{formatted_time}")[span_70](end_span)
        [span_71](start_span)elif ( (ppo_prev >= PPO_ZERO_LINE and ppo_curr < PPO_ZERO_LINE) and ppo_curr < sig_curr and short_intent and strong_bearish_close): # 0[span_71](end_span)
            current_state = "short_zero"
            if last_state_for_pair != "short_zero":
                [span_72](start_span)send_telegram_alert(f"🔴 {pair_name} - SHORT (0)\nPPO: {ppo_curr:.2f}\nPrice: ${price:,.2f}\n{formatted_time}")[span_72](end_span)
        
        # 4. PPO 0.11/-0.11 Cross (Trend Confirmation)
        [span_73](start_span)elif ( (ppo_prev <= PPO_UP_LEVEL and ppo_curr > PPO_UP_LEVEL) and ppo_curr > sig_curr and long_intent and strong_bullish_close): # 0.11[span_73](end_span)
            current_state = "long_011"
            if last_state_for_pair != "long_011":
                [span_74](start_span)send_telegram_alert(f"🟢 {pair_name} - LONG (0.11)\nPPO: {ppo_curr:.2f}\nPrice: ${price:,.2f}\n{formatted_time}")[span_74](end_span)
        [span_75](start_span)elif ( (ppo_prev >= PPO_DOWN_LEVEL and ppo_curr < PPO_DOWN_LEVEL) and ppo_curr < sig_curr and short_intent and strong_bearish_close): # -0.11[span_75](end_span)
            current_state = "short_011"
            if last_state_for_pair != "short_011":
                [span_76](start_span)send_telegram_alert(f"🔴 {pair_name} - SHORT (-0.11)\nPPO: {ppo_curr:.2f}\nPrice: ${price:,.2f}\n{formatted_time}")[span_76](end_span)
        
        else:
            [span_77](start_span)if short_intent and ppo_curr < sig_curr and not ppo_cross_down:[span_77](end_span)
                diagnose_failure(pair_name, short_intent=True, ppo_cross_down=False,
                                strong_bearish_close=strong_bearish_close, magical_hist_negative=magical_hist_negative)
            [span_78](start_span)if long_intent and ppo_curr > sig_curr and not ppo_cross_up:[span_78](end_span)
                diagnose_failure(pair_name, long_intent=True, ppo_cross_up=False,
                             [span_79](start_span)strong_bullish_close=strong_bullish_close, magical_hist_positive=magical_hist_positive)[span_79](end_span)
            [span_80](start_span)debug_log(f"No signal conditions met for {pair_name}")[span_80](end_span)

        return current_state

    except Exception as e:
        [span_81](start_span)print(f"Error checking {pair_name}: {e}")[span_81](end_span)
        if DEBUG_MODE:
            traceback.print_exc()
        return None

# ... (main function remains unchanged, uses ThreadPoolExecutor)
def main():
    [span_82](start_span)print("=" * 50)[span_82](end_span)
    [span_83](start_span)ist = pytz.timezone('Asia/Kolkata')[span_83](end_span)
    start_time = datetime.now(ist)
    print(f"PPO/Cirrus Cloud Alert Bot - {start_time.strftime('%d-%m-%Y @ %H:%M IST')}")
    print(f"Debug Mode: {'ON' if DEBUG_MODE else 'OFF'}")
    print("=" * 50)

    [span_84](start_span)if SEND_TEST_MESSAGE:[span_84](end_span)
        send_test_message()

    last_alerts = load_state()
    if not get_product_ids():
        [span_85](start_span)print("Failed to fetch products. Exiting.")[span_85](end_span)
        return

    [span_86](start_span)found_count = sum(1 for v in PAIRS.values() if v is not None)[span_86](end_span)
    print(f"✓ Monitoring {found_count} pairs")
    if found_count == 0:
        print("No valid pairs found. Exiting.")
        return

    results = {}
    with ThreadPoolExecutor(max_workers=6) as executor: # max_workers=6 is confirmed
        future_to_pair = {
            executor.submit(check_pair, name, info, last_alerts.get(name)): name
            [span_87](start_span)for name, info in PAIRS.items() if info is not None[span_87](end_span)
        }
        for future in as_completed(future_to_pair):
            name = future_to_pair[future]
            try:
                [span_88](start_span)res = future.result()[span_88](end_span)
                if res is not None:
                    results[name] = res
            except Exception as e:
                print(f"Thread error for {name}: {e}")

    with state_lock:
        last_alerts.update(results)
        save_state(last_alerts)

    [span_89](start_span)elapsed = (datetime.now(ist) - start_time).total_seconds()[span_89](end_span)
    print(f"✓ Check complete. {len(results)} state updates. ({elapsed:.1f}s)")
    print("=" * 50)

if __name__ == "__main__":
    main()
