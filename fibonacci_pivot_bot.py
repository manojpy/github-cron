import requests
import pandas as pd
import numpy as np
import time
import os
import json
import pytz
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============ CONFIGURATION ============
# Telegram settings - reads from environment variables (GitHub Secrets)
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '8462496498:AAHYZ4xDIHvrVRjmCmZyoPhupCjRaRgiITc')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '203813932')

# Enable debug mode - set to True to see detailed logs
DEBUG_MODE = os.environ.get('DEBUG_MODE', 'True').lower() == 'true'

# Send test message on startup
SEND_TEST_MESSAGE = os.environ.get('SEND_TEST_MESSAGE', 'True').lower() == 'true'

# Reset state flag - set to 'True' to clear the alert_state.json file on startup
RESET_STATE = os.environ.get('RESET_STATE', 'False').lower() == 'true'

# Delta Exchange API
DELTA_API_BASE = "https://api.delta.exchange"

# Trading pairs to monitor
PAIRS = {
    "BTCUSD": None, "ETHUSD": None, 
    "SOLUSD": None, "AVAXUSD": None,
    "BCHUSD": None, "XRPUSD": None, "BNBUSD": None, "LTCUSD": None,
    "DOTUSD": None, "ADAUSD": None, "SUIUSD": None, "AAVEUSD": None
}

# Special data requirements for pairs with limited history
SPECIAL_PAIRS = {
    # Added min_required_5m to ensure enough data for RMA 200
    "SOLUSD": {"limit_15m": 210, "min_required": 74, "limit_5m": 450, "min_required_5m": 201}
}

# Indicator settings
# PPO settings
PPO_FAST = 7
PPO_SLOW = 16
PPO_SIGNAL = 5
PPO_USE_SMA = False

# Smoothed RSI (SRSI) settings
SRSI_RSI_LEN = 21
SRSI_KALMAN_LEN = 5

# Cirrus Cloud settings
X1 = 22
X2 = 9
X3 = 15
X4 = 5

# Pivot settings
PIVOT_LOOKBACK_PERIOD = 15 # Lookback in days for daily high/low/close

STATE_FILE = 'alert_state.json' 

# ============ UTILITY FUNCTIONS ============

def load_state():
    """Load previous alert state from file"""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                if DEBUG_MODE:
                    print(f"[DEBUG] Loaded state: {state}")
                return state
    except Exception as e:
        print(f"Error loading state: {e}")
    if DEBUG_MODE:
        print("[DEBUG] No previous state found, starting fresh")
    return {}

def save_state(state):
    """Save alert state to file"""
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
        if DEBUG_MODE:
            print(f"[DEBUG] Saved state: {state}")
    except Exception as e:
        print(f"Error saving state: {e}")

def send_telegram_alert(message):
    """Send alert message via Telegram"""
    try:
        if DEBUG_MODE:
            print(f"[DEBUG] Attempting to send message: {message[:100]}...")
        
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
            print(f"❌ Telegram error: {response_data}")
            return False
        
    except Exception as e:
        print(f"❌ Error sending Telegram message: {e}")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
        return False

def send_test_message():
    """Send a test message to verify Telegram connectivity"""
    ist = pytz.timezone('Asia/Kolkata')
    current_dt = datetime.now(ist)
    formatted_time = current_dt.strftime('%d-%m-%Y @ %H:%M IST')
    
    test_msg = f"🔔 Fibonacci Pivot Bot Started\nTest message from Fibonacci Pivot Bot\nTime: {formatted_time}\nDebug Mode: {'ON' if DEBUG_MODE else 'OFF'}"
    
    print("\n" + "="*50)
    print("SENDING TEST MESSAGE")
    print("="*50)
    success = send_telegram_alert(test_msg)
    
    if success:
        print("✓ Test message sent successfully!")
    else:
        print("❌ Test message failed - check your bot token and chat ID")
    
    print("="*50 + "\n")
    return success

def get_product_ids():
    """Fetch all product IDs from Delta Exchange and populate PAIRS dict"""
    try:
        if DEBUG_MODE:
            print("[DEBUG] Fetching product IDs from Delta Exchange...")
        response = requests.get(f"{DELTA_API_BASE}/v2/products", timeout=10)
        data = response.json()
        
        if data.get('success'):
            products = data['result']
            if DEBUG_MODE:
                print(f"[DEBUG] Received {len(products)} products from API")
            
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
                            if DEBUG_MODE:
                                print(f"[DEBUG] Matched {pair_name} -> {product['symbol']} (ID: {product['id']})")
            return True
        else:
            print(f"API Error: {data}")
            return False
            
    except Exception as e:
        print(f"Error fetching products: {e}")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
        return False

def get_candles(product_id, resolution="15", limit=150):
    """Fetch OHLCV candles from Delta Exchange"""
    try:
        to_time = int(time.time())
        # resolution is in minutes, except for "D" (Daily)
        if resolution == "D":
            from_time = to_time - (limit * 24 * 60 * 60) # Approx
        else:
            from_time = to_time - (limit * int(resolution) * 60)
        
        url = f"{DELTA_API_BASE}/v2/chart/history"
        params = {
            'resolution': resolution,
            'symbol': product_id,
            'from': from_time,
            'to': to_time
        }
        
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
            return df
        else:
            return None
            
    except Exception as e:
        print(f"Exception fetching candles for {product_id}: {e}")
        return None

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    return data.rolling(window=period).mean()

def calculate_ppo(df, fast=7, slow=16, signal=5, use_sma=False):
    """Calculate PPO (Percentage Price Oscillator) - matches Pine Script"""
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
    """Implements smoothrngX1 from Pine Script"""
    wper = t * 2 - 1
    avrng = calculate_ema(np.abs(x.diff()), t)
    smoothrng = calculate_ema(avrng, wper) * m
    return smoothrng

def rngfilt(x, r):
    """Implements rngfiltx1x1 from Pine Script using robust array iteration."""
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
    """Calculate Cirrus Cloud Upw (Green) and Dnw (Red) conditions."""
    close = df['close'].copy()
    
    smrngx1x = smoothrng(close, X1, X2)
    smrngx1x2 = smoothrng(close, X3, X4)
    
    filtx1 = rngfilt(close, smrngx1x)
    filtx12 = rngfilt(close, smrngx1x2)
    
    # Upw (Green) is True when filter 1 line is BELOW filter 2 line.
    upw = filtx1 < filtx12 
    # Dnw (Red) is True when filter 1 line is ABOVE filter 2 line.
    dnw = filtx1 > filtx12 
    
    return upw, dnw, filtx1, filtx12 

def calculate_rma(data, period):
    """Calculate RMA (Smoothed Moving Average) - matches Pine Script's ta.rma"""
    # Uses alpha=1/period for closer match to Pine ta.rma
    return data.ewm(alpha=1/period, adjust=False).mean()

def kalman_filter(src, length, R = 0.01, Q = 0.1):
    """Implements the kalman_filter function from Pine Script"""
    result_list = []
    
    estimate = np.nan
    error_est = 1.0
    
    error_meas = R * length
    Q_div_length = Q / length

    # Use a loop to mimic bar-by-bar 'var' behavior in Pine Script
    for i in range(len(src)):
        current_src = src.iloc[i]
        
        if np.isnan(current_src):
            result_list.append(np.nan)
            continue
        
        if np.isnan(estimate):
            if i > 0 and not np.isnan(src.iloc[i-1]):
                estimate = src.iloc[i-1]
            else:
                result_list.append(np.nan)
                continue

        prediction = estimate
        
        kalman_gain = error_est / (error_est + error_meas)
        
        estimate = prediction + kalman_gain * (current_src - prediction)
        
        error_est = (1 - kalman_gain) * error_est + Q_div_length
        
        result_list.append(estimate)

    # Pad with NaNs at the beginning to ensure series is the same length
    num_nan_pad = len(src) - len(result_list)
    final_list = [np.nan] * num_nan_pad + result_list
    
    return pd.Series(final_list, index=src.index)

def calculate_smooth_rsi(df, rsi_len=SRSI_RSI_LEN, kalman_len=SRSI_KALMAN_LEN):
    """Calculate Smoothed RSI using Kalman Filter"""
    close = df['close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Use RMA for Wilder's smoothing
    avg_gain = calculate_rma(gain, rsi_len)
    # Use replace to handle division by zero safely
    avg_loss = calculate_rma(loss, rsi_len) 
    
    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, 1e-9) 
    rsi_value = 100 - (100 / (1 + rs))
    
    # Smooth RSI using Kalman Filter
    smooth_rsi = kalman_filter(rsi_value, kalman_len)
    
    return smooth_rsi

# ============ FIBONACCI PIVOT FUNCTIONS ============

def get_previous_day_ohlc(product_id, days_back_limit=15):
    """Fetch 1-day candles to get the previous day's H, L, C for pivot calculation."""
    df_daily = get_candles(product_id, resolution="D", limit=days_back_limit + 5)
    
    if df_daily is None or len(df_daily) < 2:
        return None
    
    prev_day_candle = df_daily.iloc[-2]
    
    return {
        'high': prev_day_candle['high'],
        'low': prev_day_candle['low'],
        'close': prev_day_candle['close']
    }

def calculate_fibonacci_pivots(h, l, c):
    """Calculate Fibonacci Pivots for the next day based on H, L, C of the previous day."""
    pivot = (h + l + c) / 3
    
    diff = h - l
    
    # Resistance levels
    r3 = pivot + (diff * 1.000)
    r2 = pivot + (diff * 0.618)
    r1 = pivot + (diff * 0.382)

    # Support levels
    s1 = pivot - (diff * 0.382)
    s2 = pivot - (diff * 0.618)
    s3 = pivot - (diff * 1.000)
    
    return {
        'P': pivot, 'R1': r1, 'R2': r2, 'R3': r3, 
        'S1': s1, 'S2': s2, 'S3': s3
    }
    

def check_pair(pair_name, pair_info, last_alerts):
    """Check Fibonacci Pivot conditions for a pair and return new state and log output."""
    
    # --- 1. Internal Log Collector ---
    thread_log = []
    
    def log(message):
        if DEBUG_MODE:
            thread_log.append(f"[DEBUG] {message}")

    try:
        if pair_info is None:
            return last_alerts.get(pair_name), "" 
        
        log(f"\n{'='*60}")
        log(f"Checking {pair_name} for Fibonacci Pivot Alerts")
        log(f"{'='*60}")
        
        # --- 1. Get Pivot Data (Previous Day) ---
        prev_day_ohlc = get_previous_day_ohlc(pair_info['symbol'], PIVOT_LOOKBACK_PERIOD)
        if prev_day_ohlc is None:
            print(f"Skipping {pair_name}: Failed to get previous day OHLC data.") 
            return last_alerts.get(pair_name), '\n'.join(thread_log)
            
        pivots = calculate_fibonacci_pivots(
            prev_day_ohlc['high'], prev_day_ohlc['low'], prev_day_ohlc['close']
        )
        log(f"Daily Pivots: P={pivots['P']:.2f}, R1={pivots['R1']:.2f}, S1={pivots['S1']:.2f}")

        # --- 2. Get 15-Minute Candles and Indicators ---
        limit_15m = SPECIAL_PAIRS.get(pair_name, {}).get("limit_15m", 210)
        min_required = max(SPECIAL_PAIRS.get(pair_name, {}).get("min_required", 200), 65)

        df_15m = get_candles(pair_info['symbol'], "15", limit=limit_15m)
    
        if df_15m is None or len(df_15m) < min_required:
            print(f"Not enough 15m data for {pair_name}. Need {min_required}, got {len(df_15m) if df_15m is not None else 0}.")
            return last_alerts.get(pair_name), '\n'.join(thread_log) 

        # Calculate indicators
        ppo, _ = calculate_ppo(df_15m, PPO_FAST, PPO_SLOW, PPO_SIGNAL, PPO_USE_SMA)
        upw, dnw, _, _ = calculate_cirrus_cloud(df_15m)
        smooth_rsi = calculate_smooth_rsi(df_15m)
        
        # Calculate 15m RMA 50
        rma_50_15m = calculate_rma(df_15m['close'], 50)
        
        if len(ppo) < 3 or len(smooth_rsi) < 3 or len(rma_50_15m) < 3:
            print(f"Skipping {pair_name}: Indicators did not produce enough data (need >= 3).")
            return last_alerts.get(pair_name), '\n'.join(thread_log)


        # Get values for the last closed 15m candle (index -2)
        open_prev = df_15m['open'].iloc[-2]
        close_prev = df_15m['close'].iloc[-2]
        high_prev = df_15m['high'].iloc[-2]
        low_prev = df_15m['low'].iloc[-2]

        
        ppo_curr = ppo.iloc[-2] 
        smooth_rsi_curr = smooth_rsi.iloc[-2]
        rma_50_15m_curr = rma_50_15m.iloc[-2]

        log(f"15m PPO: {ppo_curr:.4f}, SRSI: {smooth_rsi_curr:.2f}, RMA 50: {rma_50_15m_curr:.2f}")
        upw_curr = upw.iloc[-2]
        dnw_curr = dnw.iloc[-2]
        
        # --- Get 5-Minute Candles and RMA 200 ---
        limit_5m = SPECIAL_PAIRS.get(pair_name, {}).get("limit_5m", 450)
        min_required_5m = max(SPECIAL_PAIRS.get(pair_name, {}).get("min_required_5m", 201), 201)
        
        rma_200_5m_curr = np.nan
        
        df_5m = get_candles(pair_info['symbol'], "5", limit=limit_5m)
        
        if df_5m is None or len(df_5m) < min_required_5m:
            log(f"Not enough 5m data for {pair_name}. RMA 200 5m check will be ignored (Need {min_required_5m}, got {len(df_5m) if df_5m is not None else 0}).")
        else:
            rma_200_5m = calculate_rma(df_5m['close'], 200)
            
            if len(rma_200_5m) >= 2 and not np.isnan(rma_200_5m.iloc[-2]):
                rma_200_5m_curr = rma_200_5m.iloc[-2]
                log(f"5m RMA 200: {rma_200_5m_curr:.4f}")
            else:
                log("5m RMA 200 calculation failed to produce a valid value.")
        # --- END 5M RMA 200 BLOCK ---
        
        
        # --- 3. Define Alert Conditions ---
        srsi_above_50 = smooth_rsi_curr > 50
        srsi_below_50 = smooth_rsi_curr < 50
        
        is_green = close_prev > open_prev
        is_red = close_prev < open_prev
        
        # New MA conditions
        rma_long_ok = rma_50_15m_curr < close_prev and rma_200_5m_curr < close_prev
        rma_short_ok = rma_50_15m_curr > close_prev and rma_200_5m_curr > close_prev

        # Wick Checks
        candle_range = high_prev - low_prev
    
        if candle_range <= 0:
            log("Candle range is zero or negative, skipping wick checks.")
            upper_wick_check = False
            lower_wick_check = False
        else:
            upper_wick_length = high_prev - max(open_prev, close_prev)
            lower_wick_length = min(open_prev, close_prev) - low_prev
            upper_wick_check = (upper_wick_length / candle_range) < 0.20
            lower_wick_check = (lower_wick_length / candle_range) < 0.20
        
        # --- 4. Pivot Crossover Logic ---
        long_pivot_lines = {'P': pivots['P'], 'R1': pivots['R1'], 'R2': pivots['R2'], 'S1': pivots['S1'], 'S2': pivots['S2']}
        long_crossover_line = False
        long_crossover_name = None
 
        if is_green:
            for name, line in long_pivot_lines.items():
                if open_prev <= line and close_prev > line: 
                    long_crossover_line = line
                    long_crossover_name = name
                    break

        short_pivot_lines = {'P': pivots['P'], 'S1': pivots['S1'], 'S2': pivots['S2'], 
            'R1': pivots['R1'], 'R2': pivots['R2']}
        short_crossover_line = False
        short_crossover_name = None
        if is_red:
            for name, line in short_pivot_lines.items():
                if open_prev >= line and close_prev < line:
                    short_crossover_line = line
                    short_crossover_name = name
                    break
        
        # --- 5. Signal Detection and State Management ---
        current_signal = None
        updated_state = last_alerts.get(pair_name) 
        
        ist = pytz.timezone('Asia/Kolkata')
        formatted_time = datetime.now(ist).strftime('%d-%m-%Y @ %H:%M IST')
        price = close_prev
        
        # === ADVANCED STATE RESET LOGIC ===
        if updated_state and updated_state.startswith('fib_'):
            try:
                alert_type, pivot_name = updated_state.split('_')[-2:]
                pivot_value = pivots.get(pivot_name)
                
                if pivot_value is not None:
                    if alert_type == "long":
                        if pivot_name != 'R3' and close_prev < pivot_value:
                            updated_state = None
                            log(f"\nALERT STATE RESET: {pair_name} Long (Close ${close_prev:,.2f} < {pivot_name} ${pivot_value:,.2f})")
                 
                    elif alert_type == "short":
                        if pivot_name != 'S3' and close_prev > pivot_value:
                            updated_state = None
                            log(f"\nALERT STATE RESET: {pair_name} Short (Close ${close_prev:,.2f} > {pivot_name} ${pivot_value:,.2f})")
            except Exception as e:
                log(f"Error parsing saved state {updated_state}: {e}")
                
        # 🟢 FINAL LONG SIGNAL CHECK (RMA 50 & RMA 200 added)
        if (upw_curr and (not dnw_curr) and 
            srsi_above_50 and 
            (ppo_curr < 0.20) and 
            long_crossover_line and 
            upper_wick_check and
            rma_long_ok): # <-- NEW CONDITION
        
            current_signal = f"fib_long_{long_crossover_name}"
            log(f"\n🟢 FIB LONG SIGNAL DETECTED for {pair_name}!")
            
            if updated_state != current_signal:
                message = (
                    f"🟢 {pair_name} - **FIB LONG**\n"
                    f"Crossed & Closed Above **{long_crossover_name}** (${long_crossover_line:,.2f})\n"
                    f"PPO 15m: {ppo_curr:.2f}\n" 
                    f"RMA 50 (15m): ${rma_50_15m_curr:,.2f}\n" # <-- NEW INFO
                    f"RMA 200 (5m): ${rma_200_5m_curr:,.2f}\n" # <-- NEW INFO
                    f"Price: ${price:,.2f}\n"
                    f"{formatted_time}"
                )
                send_telegram_alert(message)
                
            updated_state = current_signal
            

        # 🔴 FINAL SHORT SIGNAL CHECK (RMA 50 & RMA 200 added)
        elif (dnw_curr and (not upw_curr) and 
              srsi_below_50 and 
              (ppo_curr > -0.20) and 
              short_crossover_line and 
              lower_wick_check and
              rma_short_ok): # <-- NEW CONDITION
        
            current_signal = f"fib_short_{short_crossover_name}"
            log(f"\n🔴 FIB SHORT SIGNAL DETECTED for {pair_name}!")
            
            if updated_state != current_signal:
                message = (
                    f"🔴 {pair_name} - **FIB SHORT**\n"
                    f"Crossed & Closed Below **{short_crossover_name}** (${short_crossover_line:,.2f})\n"
                    f"PPO 15m: {ppo_curr:.2f}\n"
                    f"RMA 50 (15m): ${rma_50_15m_curr:,.2f}\n" # <-- NEW INFO
                    f"RMA 200 (5m): ${rma_200_5m_curr:,.2f}\n" # <-- NEW INFO
                    f"Price: ${price:,.2f}\n"
                    f"{formatted_time}"
                )
                send_telegram_alert(message)
                
            updated_state = current_signal
                
        else:
            log(f"No Fibonacci Pivot signal conditions met for {pair_name}")
        
        # --- RETURN BOTH STATE AND LOG ---
        return updated_state, '\n'.join(thread_log)

    except Exception as e:
        print(f"Error checking {pair_name}: {e}")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
        
        return last_alerts.get(pair_name), '\n'.join(thread_log) 


def main():
 
    """Main function - runs once per GitHub Actions execution"""
    print("=" * 50)
    ist = pytz.timezone('Asia/Kolkata')
    start_time = datetime.now(ist)
    print(f"Fibonacci Pivot Alert Bot - {start_time.strftime('%d-%m-%Y @ %H:%M IST')}")
    print(f"Debug Mode: {'ON' if DEBUG_MODE else 'OFF'}")
    print("=" * 50)
 
    
    # Send test message if enabled
    if SEND_TEST_MESSAGE:
        send_test_message()

    # === APPLIED RESET STATE LOGIC ===
    if RESET_STATE and os.path.exists(STATE_FILE):
        print(f"ATTENTION: \nRESET_STATE is True. Deleting {STATE_FILE} to clear previous alerts.")
        os.remove(STATE_FILE)
    # =================================
    
    # Load previous state
    last_alerts = load_state()
    
    # Fetch product IDs
    if not get_product_ids():
        print("Failed to fetch products. Exiting.")
        return
    
    found_count = sum(1 for v in PAIRS.values() if v is not None)
    print(f"✓ Monitoring {found_count} pairs")
 
    
    if found_count == 0:
        print("No valid pairs found. Exiting.")
        return
    
    # Check all pairs in parallel
    updates_processed = 0
    pair_logs = [] # Initialize list to hold all collected logs
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        
        future_to_pair = {}
    
        for pair_name, pair_info in PAIRS.items():
            if pair_info is not None:
                # Submit the task. check_pair now returns (new_state, log_output)
                future = executor.submit(check_pair, pair_name, pair_info, last_alerts) 
                
                future_to_pair[future] = pair_name

        for future in as_completed(future_to_pair):
            pair_name = future_to_pair[future]
            
            try:
                # --- CAPTURE BOTH RETURN VALUES ---
                new_state, log_output = future.result() 
                pair_logs.append(log_output) # Store the clean log output for sequential printing
                
                # Only increment updates if the state has changed.
                if new_state != last_alerts.get(pair_name):
                    last_alerts[pair_name] = new_state
                    updates_processed += 1
                    
            except Exception as e:
                # This catches errors in the executor framework itself, not the logic inside check_pair
                print(f"Error processing {pair_name} in thread: {e}") 
                if DEBUG_MODE:
                    import traceback
                    traceback.print_exc()
                continue
            
    # --- PRINT COLLECTED LOGS SEQUENTIALLY (Fixes Interleaving) ---
    if DEBUG_MODE:
        print("\n\n" + "="*50)
        print("SEQUENTIAL DEBUG LOGS START")
        print("="*50)
        for log_output in pair_logs:
            # Print the full, non-interleaved block of logs for this pair
            print(log_output, end='') 
        print("\n" + "="*50)
        print("SEQUENTIAL DEBUG LOGS END")
        print("="*50)
        
    # Save state for next run
    save_state(last_alerts)
 
    
    end_time = datetime.now(ist)
    elapsed = (end_time - start_time).total_seconds()
    print(f"✓ Check complete. {updates_processed} state updates processed. ({elapsed:.1f}s)")
    print("=" * 50)

if __name__ == "__main__":
    main()
