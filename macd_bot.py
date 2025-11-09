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

# ============ CONFIGURATION ============
# Telegram settings - reads from environment variables (GitHub Secrets)
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
    "SOLUSD": {"limit_15m": 150, "min_required": 74, "limit_5m": 250, "min_required_5m": 183}
}

PPO_FAST = 7
PPO_SLOW = 16
PPO_SIGNAL = 5
PPO_USE_SMA = False

RMA_50_PERIOD = 50
RMA_200_PERIOD = 200

CIRRUS_CLOUD_ENABLED = True
X1 = 22
X2 = 9
X3 = 15
X4 = 5

SRSI_RSI_LEN = 21
SRSI_KALMAN_LEN = 5
SRSI_EMA_LEN = 5

STATE_FILE = 'alert_state.json'

def debug_log(message):
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")

def load_state():
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
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
        debug_log(f"Saved state: {state}")
    except Exception as e:
        print(f"Error saving state: {e}")

def send_telegram_alert(message):
    try:
        debug_log(f"Attempting to send message: {message[:100]}...")
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": None
        }
        response = requests.post(url, data=data, timeout=10)
        response_data = response.json()
        if response_data.get('ok'):
            print("✓ Alert sent successfully")
            return True
        else:
            print(f"❌ Telegram error: {response_data}")
            return False
    except Exception as e:
        print(f"❌ Error sending Telegram message: {e}")
        if DEBUG_MODE:
            pass
        return False

def send_test_message():
    ist = pytz.timezone('Asia/Kolkata')
    current_dt = datetime.now(ist)
    formatted_time = current_dt.strftime('%d-%m-%Y @ %H:%M IST')
    test_msg = (
        f"🔔 Bot Started
"
        f"Test message from PPO Bot
"
        f"Time: {formatted_time}
"
        f"Debug Mode: {'ON' if DEBUG_MODE else 'OFF'}"
    )
    print("
" + "="*50)
    print("SENDING TEST MESSAGE")
    print("="*50)
    success = send_telegram_alert(test_msg)
    if success:
        print("✓ Test message sent successfully!")
    else:
        print("❌ Test message failed - check your bot token and chat ID")
    print("="*50 + "
")
    return success

def get_product_ids():
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
            pass
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
            return df
        else:
            print(f"Error fetching candles for {product_id}: {data.get('message', 'No message')}")
            return None
    except Exception as e:
        print(f"Exception fetching candles for {product_id}: {e}")
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

def kalman_filter(src, length, R = 0.01, Q = 0.1):
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

def calculate_magical_momentum_hist(close, responsiveness=0.9, period=144):
    responsiveness = max(0.00001, responsiveness)
    sd = close.rolling(50).std() * responsiveness

    worm = close.iloc[0]
    worms = []
    for price, std in zip(close, sd):
        diff = price - worm
        delta = np.sign(diff) * std if np.abs(diff) > std else diff
        worm = worm + delta
        worms.append(worm)
    worms = pd.Series(worms, index=close.index)
    ma = close.rolling(period).mean()
    raw_momentum = (worms - ma) / worms
    current_med = raw_momentum
    min_med = current_med.rolling(period).min()
    max_med = current_med.rolling(period).max()
    temp = (current_med - min_med) / (max_med - min_med)
    value = pd.Series(0.5 * 2, index=close.index)
    for i in range(1, len(close)):
        value.iloc[i] = value.iloc[i-1] * (temp.iloc[i] - 0.5 + 0.5 * value.iloc[i-1])
    value = value.clip(-0.9999, 0.9999)
    temp2 = (1 + value) / (1 - value)
    momentum = 0.25 * np.log(temp2)
    for i in range(1, len(momentum)):
        momentum.iloc[i] = momentum.iloc[i] + 0.5 * momentum.iloc[i-1]
    hist = momentum
    return hist

def check_pair(pair_name, pair_info, last_alerts):
    try:
        if pair_info is None:
            return None
        debug_log(f"
{'='*60}")
        debug_log(f"Checking {pair_name}")
        debug_log(f"{'='*60}")

        if pair_name in SPECIAL_PAIRS:
            limit_15m = SPECIAL_PAIRS[pair_name]["limit_15m"]
            min_required = SPECIAL_PAIRS[pair_name]["min_required"]
            limit_5m = SPECIAL_PAIRS[pair_name].get("limit_5m", 210)
            min_required_5m = SPECIAL_PAIRS[pair_name].get("min_required_5m", 200)
        else:
            limit_15m = 210
            min_required = 200
            limit_5m = 210
            min_required_5m = 200

        df_15m = get_candles(pair_info['symbol'], "15", limit=limit_15m)
        df_5m = get_candles(pair_info['symbol'], "5", limit=limit_5m)
        if df_15m is None or len(df_15m) < min_required:
            print(f"Not enough 15m data for {pair_name} ({len(df_15m) if df_15m is not None else 0}/{min_required})")
            return None
        if df_5m is None or len(df_5m) < min_required_5m:
            print(f"Not enough 5m data for {pair_name} ({len(df_5m) if df_5m is not None else 0}/{min_required_5m})")
            return None

        ppo, ppo_signal = calculate_ppo(df_15m, PPO_FAST, PPO_SLOW, PPO_SIGNAL, PPO_USE_SMA)
        rma_50 = calculate_rma(df_15m['close'], RMA_50_PERIOD)
        rma_200 = calculate_rma(df_5m['close'], RMA_200_PERIOD)
        upw, dnw, filtx1, filtx12 = calculate_cirrus_cloud(df_15m)
        smooth_rsi = calculate_smooth_rsi(df_15m)

        magical_momentum_hist = calculate_magical_momentum_hist(df_15m["close"])
        magical_momentum_hist_curr = magical_momentum_hist.iloc[-1]

        ppo_curr = ppo.iloc[-1]
        ppo_prev = ppo.iloc[-2]
        ppo_signal_curr = ppo_signal.iloc[-1]
        ppo_signal_prev = ppo_signal.iloc[-2]
        smooth_rsi_curr = smooth_rsi.iloc[-1]
        smooth_rsi_prev = smooth_rsi.iloc[-2]
        close_curr = df_15m['close'].iloc[-1]
        rma50_curr = rma_50.iloc[-1]
        upw_curr = upw.iloc[-1]
        dnw_curr = dnw.iloc[-1]
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

        close_5m_curr = df_5m['close'].iloc[-1]
        rma200_curr = rma_200.iloc[-1]

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
        close_above_rma200 = close_5m_curr > rma200_curr
        close_below_rma200 = close_5m_curr < rma200_curr
        srsi_cross_up_50 = (smooth_rsi_prev <= 50) and (smooth_rsi_curr > 50)
        srsi_cross_down_50 = (smooth_rsi_prev >= 50) and (smooth_rsi_curr < 50)

        ist = pytz.timezone('Asia/Kolkata')
        current_dt = datetime.now(ist)
        formatted_time = current_dt.strftime('%d-%m-%Y @ %H:%M IST')
        price = df_15m['close'].iloc[-1]
        current_state = None

        # ====== ALERT LOGIC WITH MOMENTUM FILTER ======
        # BUY
        if (ppo_cross_up and 
            ppo_below_020 and
            close_above_rma50 and 
            close_above_rma200 and 
            upw_curr and (not dnw_curr) and 
            strong_bullish_close and
            magical_momentum_hist_curr > 0):
            current_state = "buy"
            debug_log(f"
🟢 BUY SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "buy":
                message = (
                    f"🟢 {pair_name} - BUY
"
                    f"PPO - SIGNAL Crossover (PPO: {ppo_curr:.2f})
"
                    f"Price: ${price:,.2f}
"
                    f"{formatted_time}" 
                )
                send_telegram_alert(message)
            else:
                debug_log(f"BUY already alerted for {pair_name}, skipping duplicate")

        # SELL
        elif (ppo_cross_down and 
            ppo_above_minus020 and 
            close_below_rma50 and 
            close_below_rma200 and 
            dnw_curr and (not upw_curr) and 
            strong_bearish_close and
            magical_momentum_hist_curr < 0):
            current_state = "sell"
            debug_log(f"
🔴 SELL SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "sell":
                message = (
                    f"🔴 {pair_name} - SELL
"
                    f"PPO - SIGNAL Crossunder (PPO: {ppo_curr:.2f})
"
                    f"Price: ${price:,.2f}
"
                    f"{formatted_time}"
                )
                send_telegram_alert(message)
            else:
                debug_log(f"SELL already alerted for {pair_name}, skipping duplicate")

        # SRSI BUY ALERT (SRSI CROSS UP 50)
        elif (srsi_cross_up_50 and 
              ppo_above_signal and 
              ppo_below_030 and 
              close_above_rma50 and 
              close_above_rma200 and 
              upw_curr and (not dnw_curr) and 
              strong_bullish_close and
              magical_momentum_hist_curr > 0):
            current_state = "buy_srsi50"
            debug_log(f"
⬆️ BUY (SRSI 50) SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "buy_srsi50":
                message = (
                    f"⬆️ {pair_name} - BUY (SRSI 50)
"
                    f"SRSI 15m Cross Up 50 ({smooth_rsi_curr:.2f})
"
                    f"Price: ${price:,.2f}
"
                    f"{formatted_time}" 
                )
                send_telegram_alert(message)
            else:
                debug_log(f"BUY (SRSI 50) already alerted for {pair_name}, skipping duplicate")

        # SRSI SELL ALERT (SRSI CROSS DOWN 50)
        elif (srsi_cross_down_50 and 
                ppo_below_signal and 
                ppo_above_minus030 and 
                close_below_rma50 and 
                close_below_rma200 and 
                dnw_curr and (not upw_curr) and 
                strong_bearish_close and
                magical_momentum_hist_curr < 0):
            current_state = "sell_srsi50"
            debug_log(f"
⬇️ SELL (SRSI 50) SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "sell_srsi50":
                message = (
                    f"⬇️ {pair_name} - SELL (SRSI 50)
"
                    f"SRSI 15m Cross Down 50 ({smooth_rsi_curr:.2f})
"
                    f"Price: ${price:,.2f}
"
                    f"{formatted_time}"
                )
                send_telegram_alert(message)
            else:
                debug_log(f"SELL (SRSI 50) already alerted for {pair_name}, skipping duplicate")

        # LONG (0)
        elif (ppo_cross_above_zero and 
            ppo_above_signal and 
            close_above_rma50 and 
            close_above_rma200 and 
            upw_curr and (not dnw_curr) and 
            strong_bullish_close and
            magical_momentum_hist_curr > 0):
            current_state = "long_zero"
            debug_log(f"
🟢 LONG (0) SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "long_zero":
                message = (
                    f"🟢 {pair_name} - LONG
"
                    f"PPO crossing above 0 ({ppo_curr:.2f})
"
                    f"Price: ${price:,.2f}
"
                    f"{formatted_time}"
                )
                send_telegram_alert(message)
            else:
                debug_log(f"LONG (0) already alerted for {pair_name}, skipping duplicate")

        # LONG (0.11)
        elif (ppo_cross_above_011 and 
            ppo_above_signal and 
            close_above_rma50 and 
            close_above_rma200 and 
            upw_curr and (not dnw_curr) and 
            strong_bullish_close and
            magical_momentum_hist_curr > 0):
            current_state = "long_011"
            debug_log(f"
🟢 LONG (0.11) SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "long_011":
                message = (
                    f"🟢 {pair_name} - LONG
"
                    f"PPO crossing above 0.11 ({ppo_curr:.2f})
"
                    f"Price: ${price:,.2f}
"
                    f"{formatted_time}" 
                )
                send_telegram_alert(message)
            else:
                debug_log(f"LONG (0.11) already alerted for {pair_name}, skipping duplicate")

        # SHORT (0)
        elif (ppo_cross_below_zero and 
                ppo_below_signal and 
                close_below_rma50 and 
                close_below_rma200 and 
                dnw_curr and (not upw_curr) and 
                strong_bearish_close and
                magical_momentum_hist_curr < 0):
            current_state = "short_zero"
            debug_log(f"
🔴 SHORT (0) SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "short_zero":
                message = (
                    f"🔴 {pair_name} - SHORT
"
                    f"PPO crossing below 0 ({ppo_curr:.2f})
"
                    f"Price: ${price:,.2f}
"
                    f"{formatted_time}"
                )
                send_telegram_alert(message)
            else:
                debug_log(f"SHORT (0) already alerted for {pair_name}, skipping duplicate")

        # SHORT (-0.11)
        elif (ppo_cross_below_minus011 and 
            ppo_below_signal and 
            close_below_rma50 and 
            close_below_rma200 and 
            dnw_curr and (not upw_curr) and 
            strong_bearish_close and
            magical_momentum_hist_curr < 0):
            current_state = "short_011"
            debug_log(f"
🔴 SHORT (-0.11) SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "short_011":
                message = (
                    f"🔴 {pair_name} - SHORT
"
                    f"PPO crossing below -0.11 ({ppo_curr:.2f})
"
                    f"Price: ${price:,.2f}
"
                    f"{formatted_time}" 
                )
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
                    last_alerts[pair_name] = new_state
                    alerts_sent += 1
            except Exception as e:
                print(f"Error processing {pair_name} in thread: {e}")
                if DEBUG_MODE:
                    traceback.print_exc()
                continue
    save_state(last_alerts)
    end_time = datetime.now(ist)
    elapsed = (end_time - start_time).total_seconds()
    print(f"✓ Check complete. {alerts_sent} state updates processed. ({elapsed:.1f}s)")
    print("=" * 50)

if __name__ == "__main__":
    main()
