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

# ============ CONFIGURATION ============
# Telegram settings - reads from environment variables (GitHub Secrets)
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '8462496498:AAHYZ4xDIHvrVRjmCmZyoPhupCjRaRgiITc')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '203813932')

# Enable debug mode - set to True to see detailed logs
DEBUG_MODE = os.environ.get('DEBUG_MODE', 'True').lower() == 'true'
# Send test message on startup
SEND_TEST_MESSAGE = os.environ.get('SEND_TEST_MESSAGE', 'True').lower() == 'true'
# Delta Exchange API
DELTA_API_BASE = "https://api.delta.exchange"
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
    "SOLUSD": {"limit_15m": 150, "min_required": 74, "limit_5m": 300, "min_required_5m": 250}
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
SRSI_EMA_LEN = 5 
# File to store last alert state
STATE_FILE = 'alert_state.json'
# Thread lock for state updates
state_lock = Lock()

# ============ UTILITY FUNCTIONS ============

def debug_log(message):
    """Print debug messages if DEBUG_MODE is enabled"""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")

def load_state():
    """Load previous alert state from file"""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                debug_log(f"Loaded state: {state}")
                return state
    except Exception as e:
        print(f"Error loading state: {e}")
    debug_log("No previous state found, starting fresh")
    return {}

def save_state(state):
    """Save alert state to file"""
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
        debug_log(f"Saved state: {state}")
    except Exception as e:
        print(f"Error saving state: {e}")

def send_telegram_alert(message):
    """Send alert message via Telegram"""
    try:
        debug_log(f"Attempting to send message: {message[:100]}...")
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": None  # No HTML formatting
        }
        response = requests.post(url, data=data, timeout=10)
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
        response = requests.get(f"{DELTA_API_BASE}/v2/products", timeout=10)
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
    """Fetch OHLCV candles from Delta Exchange"""
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
        response = requests.get(url, params=params, timeout=15)
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
            debug_log(f"Received {len(df)} candles for {product_id} ({resolution}m)")
            if len(df) > 0:
                last_candle_time = df['timestamp'].iloc[-1]
                time_diff = time.time() - last_candle_time
                max_age = int(resolution) * 60 * 3
                if time_diff > max_age:
                    print(f"⚠️ Warning: Stale data for {product_id} ({resolution}m) - {time_diff/60:.1f} min old")
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
    return data.rolling(window=period).mean()

def calculate_rma(data, period):
    return data.ewm(alpha=1/period, adjust=False).mean()

def calculate_ppo(df, fast=7, slow=16, signal=5, use_sma=False):
    close = df['close']
    if use_sma:
        fast_ma = calculate_sma(close, fast)
        slow_ma = calculate_sma(close, slow)
    else:
        fast_ma = calculate_ema(close, fast)
        slow_ma = calculate_ema(close, slow)
    ppo = (fast_ma - slow_ma) / slow_ma * 100
    if use_sma:
        ppo_signal = calculate_sma(ppo, signal)
    else:
        ppo_signal = calculate_ema(ppo, signal)
    return ppo, ppo_signal

def smoothrng(x, t, m):
    wper = t * 2 - 1
    avrng = calculate_ema(np.abs(x.diff()), t)
    smoothrng = calculate_ema(avrng, wper) * m
    return smoothrng

def rngfilt(x, r):
    result_list = [x.iloc[0]] 
    for i in range(1, len(x)):
        prev_f = result_list[-1]
        curr_x = x.iloc[i] 
        curr_r = r.iloc[i] 
        f = 0.0 
        if curr_x > prev_f:
            if curr_x - curr_r < prev_f:
                f = prev_f
            else:
                f = curr_x - curr_r
        else:
            if curr_x + curr_r > prev_f:
                f = prev_f
            else:
                f = curr_x + curr_r
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
    error_meas = R * length
    Q_div_length = Q / length
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
    close = df['close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = calculate_rma(gain, rsi_len)
    avg_loss = calculate_rma(loss, rsi_len)
    rs = avg_gain.divide(avg_loss.replace(0, np.nan)) 
    rsi_value = 100 - (100 / (1 + rs))
    smooth_rsi = kalman_filter(rsi_value, kalman_len)
    return smooth_rsi

# ============ MAGICAL MOMENTUM HISTOGRAM ============
def calculate_magical_momentum_hist(df, period=144, responsiveness=0.9):
    source = df['close'].copy()
    responsiveness = max(0.00001, responsiveness)
    sd = source.rolling(window=50).std() * responsiveness

    worm = source.copy()
    for i in range(1, len(source)):
        diff = source.iloc[i] - worm.iloc[i - 1]
        abs_diff = abs(diff)
        if abs_diff > sd.iloc[i]:
            delta = np.sign(diff) * sd.iloc[i]
        else:
            delta = diff
        worm.iloc[i] = worm.iloc[i - 1] + delta

    ma = source.rolling(window=period).mean()
    raw_momentum = (worm - ma) / worm

    min_med = raw_momentum.rolling(window=period).min()
    max_med = raw_momentum.rolling(window=period).max()
    denominator = max_med - min_med
    temp = np.where(denominator != 0, (raw_momentum - min_med) / denominator, 0.0)

    value = pd.Series(0.0, index=df.index)
    if len(value) > 0:
        value.iloc[0] = 0.0

    for i in range(1, len(temp)):
        val_prev = value.iloc[i - 1]
        new_val = 1.0 * (temp[i] - 0.5 + 0.5 * val_prev)
        new_val = max(-0.9999, min(0.9999, new_val))
        value.iloc[i] = new_val

    temp2 = (1 + value) / (1 - value)
    temp2 = temp2.replace([np.inf, -np.inf], np.nan)
    momentum = 0.25 * np.log(temp2)
    momentum = momentum.fillna(0)

    hist = pd.Series(0.0, index=df.index)
    if len(hist) > 0:
        hist.iloc[0] = momentum.iloc[0]
        for i in range(1, len(momentum)):
            hist.iloc[i] = momentum.iloc[i] + 0.5 * hist.iloc[i - 1]

    return hist

def check_pair(pair_name, pair_info, last_alerts):
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

        if df_15m is None or len(df_15m) < min_required:
            print(f"⚠️ Insufficient 15m data for {pair_name}: {len(df_15m) if df_15m is not None else 0}/{min_required} candles")
            return None
        if df_5m is None or len(df_5m) < min_required_5m:
            print(f"⚠️ Insufficient 5m data for {pair_name}: {len(df_5m) if df_5m is not None else 0}/{min_required_5m} candles")
            return None

        # Calculate Magical Momentum Histogram (15m)
        magical_hist = calculate_magical_momentum_hist(df_15m, period=144, responsiveness=0.9)
        magical_hist_curr = magical_hist.iloc[-1]
        if pd.isna(magical_hist_curr):
            debug_log(f"⚠️ NaN in Magical Momentum Hist for {pair_name}, skipping")
            return None
        debug_log(f"Magical Momentum Hist (15m): {magical_hist_curr:.6f}")

        # Core indicators
        ppo, ppo_signal = calculate_ppo(df_15m, PPO_FAST, PPO_SLOW, PPO_SIGNAL, PPO_USE_SMA)
        rma_50 = calculate_rma(df_15m['close'], RMA_50_PERIOD)
        rma_200 = calculate_rma(df_5m['close'], RMA_200_PERIOD)
        upw, dnw, filtx1, filtx12 = calculate_cirrus_cloud(df_15m)
        smooth_rsi = calculate_smooth_rsi(df_15m)

        # Latest values
        ppo_curr = ppo.iloc[-1]
        ppo_prev = ppo.iloc[-2]
        ppo_signal_curr = ppo_signal.iloc[-1]
        ppo_signal_prev = ppo_signal.iloc[-2]

        if pd.isna(ppo_curr) or pd.isna(ppo_signal_curr) or pd.isna(ppo_prev) or pd.isna(ppo_signal_prev):
            debug_log(f"⚠️ NaN values in PPO for {pair_name}, skipping")
            return None

        smooth_rsi_curr = smooth_rsi.iloc[-1]
        smooth_rsi_prev = smooth_rsi.iloc[-2]
        if pd.isna(smooth_rsi_curr) or pd.isna(smooth_rsi_prev):
            debug_log(f"⚠️ NaN values in Smooth RSI for {pair_name}, skipping")
            return None

        close_curr = df_15m['close'].iloc[-1]
        rma50_curr = rma_50.iloc[-1]
        if pd.isna(rma50_curr):
            debug_log(f"⚠️ NaN values in RMA50 for {pair_name}, skipping")
            return None

        upw_curr = upw.iloc[-1]
        dnw_curr = dnw.iloc[-1]
        rma200_curr = rma_200.iloc[-1]
        if pd.isna(rma200_curr):
            debug_log(f"⚠️ NaN values in RMA200 for {pair_name}, skipping")
            return None

        # Candle structure
        open_curr = df_15m['open'].iloc[-1]
        high_curr = df_15m['high'].iloc[-1]
        low_curr = df_15m['low'].iloc[-1]
        total_range = high_curr - low_curr
        upper_wick = high_curr - max(open_curr, close_curr)
        lower_wick = min(open_curr, close_curr) - low_curr
        bullish_candle = close_curr > open_curr
        bearish_candle = close_curr < open_curr
        wick_check_valid = total_range > 0
        strong_bullish_close = False
        strong_bearish_close = False
        if wick_check_valid:
            strong_bullish_close = bullish_candle and (upper_wick / total_range) < 0.20
            strong_bearish_close = bearish_candle and (lower_wick / total_range) < 0.20

        debug_log(f"\nCandle Metrics (15m):")
        debug_log(f"  O:{open_curr:.2f} H:{high_curr:.2f} L:{low_curr:.2f} C:{close_curr:.2f}")
        debug_log(f"  Range: {total_range:.2f}, UW: {upper_wick:.2f}, LW: {lower_wick:.2f}")
        if wick_check_valid:
            debug_log(f"  Strong Bullish Close (20% Rule): {strong_bullish_close}")
            debug_log(f"  Strong Bearish Close (20% Rule): {strong_bearish_close}")
        else:
            debug_log("  Candle range is zero, skipping wick checks.")

        debug_log(f"\nIndicator Values:")
        debug_log(f"Price (15m): ${close_curr:,.2f}")
        debug_log(f"PPO: {ppo_curr:.4f} (prev: {ppo_prev:.4f})")
        debug_log(f"PPO Signal: {ppo_signal_curr:.4f} (prev: {ppo_signal_prev:.4f})")
        debug_log(f"RMA50 (15m): {rma50_curr:.2f}, Close: {close_curr:.2f}")
        debug_log(f"RMA200 (5m): {rma200_curr:.2f}, Close (15m for comparison): {close_curr:.2f}")
        debug_log(f"Smoothed RSI (15m): {smooth_rsi_curr:.2f} (prev: {smooth_rsi_prev:.2f})") 
        debug_log(f"Cirrus Filter 1 (filtx1): {filtx1.iloc[-1]:.4f}") 
        debug_log(f"Cirrus Filter 2 (filtx12): {filtx12.iloc[-1]:.4f}") 
        debug_log(f"Cirrus Cloud - Upw: {upw_curr}, Dnw: {dnw_curr}")

        # Cross detections
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

        debug_log(f"\nCondition Checks:")
        debug_log(f"  Close above RMA50: {close_above_rma50}")
        debug_log(f"  Close below RMA50: {close_below_rma50}")
        debug_log(f"  Close above RMA200: {close_above_rma200}")
        debug_log(f"  Close below RMA200: {close_below_rma200}")

        current_state = None
        ist = pytz.timezone('Asia/Kolkata')
        current_dt = datetime.now(ist)
        formatted_time = current_dt.strftime('%d-%m-%Y @ %H:%M IST')
        price = df_15m['close'].iloc[-1]

        # --- ALERT LOGIC WITH MAGICAL MOMENTUM FILTER ---
        # 1. ORIGINAL BUY
        if (ppo_cross_up and 
            ppo_below_020 and 
            close_above_rma50 and 
            close_above_rma200 and 
            upw_curr and (not dnw_curr) and 
            strong_bullish_close and
            magical_hist_curr > 0):
            current_state = "buy"
            debug_log(f"\n🟢 BUY SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "buy":
                message = f"🟢 {pair_name} - BUY\nPPO - SIGNAL Crossover (PPO: {ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}" 
                send_telegram_alert(message)
            else:
                debug_log(f"BUY already alerted for {pair_name}, skipping duplicate")

        # 2. ORIGINAL SELL
        elif (ppo_cross_down and 
              ppo_above_minus020 and 
              close_below_rma50 and 
              close_below_rma200 and 
              dnw_curr and (not upw_curr) and 
              strong_bearish_close and
              magical_hist_curr < 0):
            current_state = "sell"
            debug_log(f"\n🔴 SELL SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "sell":
                message = f"🔴 {pair_name} - SELL\nPPO - SIGNAL Crossunder (PPO: {ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}" 
                send_telegram_alert(message)
            else:
                debug_log(f"SELL already alerted for {pair_name}, skipping duplicate")

        # 3. SRSI BUY ALERT
        elif (srsi_cross_up_50 and 
              ppo_above_signal and 
              ppo_below_030 and 
              close_above_rma50 and 
              close_above_rma200 and 
              upw_curr and (not dnw_curr) and 
              strong_bullish_close and
              magical_hist_curr > 0):
            current_state = "buy_srsi50" 
            debug_log(f"\n⬆️ BUY (SRSI 50) SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "buy_srsi50": 
                message = f"⬆️ {pair_name} - BUY (SRSI 50)\nSRSI 15m Cross Up 50 ({smooth_rsi_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}" 
                send_telegram_alert(message)
            else:
                debug_log(f"BUY (SRSI 50) already alerted for {pair_name}, skipping duplicate") 

        # 4. SRSI SELL ALERT
        elif (srsi_cross_down_50 and 
              ppo_below_signal and 
              ppo_above_minus030 and 
              close_below_rma50 and 
              close_below_rma200 and 
              dnw_curr and (not upw_curr) and 
              strong_bearish_close and
              magical_hist_curr < 0):
            current_state = "sell_srsi50" 
            debug_log(f"\n⬇️ SELL (SRSI 50) SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "sell_srsi50": 
                message = f"⬇️ {pair_name} - SELL (SRSI 50)\nSRSI 15m Cross Down 50 ({smooth_rsi_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}" 
                send_telegram_alert(message)
            else:
                debug_log(f"SELL (SRSI 50) already alerted for {pair_name}, skipping duplicate") 

        # 5. LONG (0)
        elif (ppo_cross_above_zero and 
              ppo_above_signal and 
              close_above_rma50 and 
              close_above_rma200 and 
              upw_curr and (not dnw_curr) and 
              strong_bullish_close and
              magical_hist_curr > 0):
            current_state = "long_zero"
            debug_log(f"\n🟢 LONG (0) SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "long_zero":
                message = f"🟢 {pair_name} - LONG\nPPO crossing above 0 ({ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}" 
                send_telegram_alert(message)
            else:
                debug_log(f"LONG (0) already alerted for {pair_name}, skipping duplicate")

        # 6. LONG (0.11)
        elif (ppo_cross_above_011 and 
              ppo_above_signal and 
              close_above_rma50 and 
              close_above_rma200 and 
              upw_curr and (not dnw_curr) and 
              strong_bullish_close and
              magical_hist_curr > 0):
            current_state = "long_011"
            debug_log(f"\n🟢 LONG (0.11) SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "long_011":
                message = f"🟢 {pair_name} - LONG\nPPO crossing above 0.11 ({ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}" 
                send_telegram_alert(message)
            else:
                debug_log(f"LONG (0.11) already alerted for {pair_name}, skipping duplicate")

        # 7. SHORT (0)
        elif (ppo_cross_below_zero and 
              ppo_below_signal and 
              close_below_rma50 and 
              close_below_rma200 and 
              dnw_curr and (not upw_curr) and 
              strong_bearish_close and
              magical_hist_curr < 0):
            current_state = "short_zero"
            debug_log(f"\n🔴 SHORT (0) SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "short_zero":
                message = f"🔴 {pair_name} - SHORT\nPPO crossing below 0 ({ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}" 
                send_telegram_alert(message)
            else:
                debug_log(f"SHORT (0) already alerted for {pair_name}, skipping duplicate")

        # 8. SHORT (-0.11)
        elif (ppo_cross_below_minus011 and 
              ppo_below_signal and 
              close_below_rma50 and 
              close_below_rma200 and 
              dnw_curr and (not upw_curr) and 
              strong_bearish_close and
              magical_hist_curr < 0):
            current_state = "short_011"
            debug_log(f"\n🔴 SHORT (-0.11) SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "short_011":
                message = f"🔴 {pair_name} - SHORT\nPPO crossing below -0.11 ({ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}" 
                send_telegram_alert(message)
            else:
                debug_log(f"SHORT (-0.11) already alerted for {pair_name}, skipping duplicate")
        else:
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

    alerts_sent = 0
    results = {}

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_pair = {}
        for pair_name, pair_info in PAIRS.items():
            if pair_info is not None:
                future = executor.submit(check_pair, pair_name, pair_info, last_alerts.copy())
                future_to_pair[future] = pair_name
        for future in as_completed(future_to_pair):
            pair_name = future_to_pair[future]
            try:
                new_state = future.result() 
                if new_state: 
                    results[pair_name] = new_state
                    alerts_sent += 1 
            except Exception as e:
                print(f"Error processing {pair_name} in thread: {e}")
                if DEBUG_MODE:
                    traceback.print_exc()
                continue

    with state_lock:
        last_alerts.update(results)
        save_state(last_alerts)

    end_time = datetime.now(ist)
    elapsed = (end_time - start_time).total_seconds()
    print(f"✓ Check complete. {alerts_sent} state updates processed. ({elapsed:.1f}s)")
    print("=" * 50)

if __name__ == "__main__":
    main()
