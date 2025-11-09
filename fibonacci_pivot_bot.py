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
    # Adjusted for pairs with limited historical data
    "SOLUSD": {"limit_15m": 210, "min_required": 140, "limit_5m": 450, "min_required_5m": 220}
}

# Indicator settings
# PPO settings
PPO_FAST = 7
PPO_SLOW = 16
PPO_SIGNAL = 5
PPO_USE_SMA = False

# Cirrus Cloud settings
X1 = 22
X2 = 9
X3 = 15
X4 = 5

# Pivot settings
PIVOT_LOOKBACK_PERIOD = 15 
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
            print(f"✗ Telegram error: {response_data}")
            return False
        
    except Exception as e:
        print(f"✗ Error sending Telegram message: {e}")
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
        print("✗ Test message failed - check your bot token and chat ID")
    
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
            print(f"❌ {pair_name}: Failed to fetch previous day OHLC data") 
            return last_alerts.get(pair_name), '\n'.join(thread_log)
            
        pivots = calculate_fibonacci_pivots(
            prev_day_ohlc['high'], prev_day_ohlc['low'], prev_day_ohlc['close']
        )
        log(f"Daily Pivots: P={pivots['P']:.2f}, R1={pivots['R1']:.2f}, R2={pivots['R2']:.2f}, R3={pivots['R3']:.2f}")
        log(f"             S1={pivots['S1']:.2f}, S2={pivots['S2']:.2f}, S3={pivots['S3']:.2f}")

        # --- 2. Get 15-Minute Candles and Indicators ---
        limit_15m = SPECIAL_PAIRS.get(pair_name, {}).get("limit_15m", 250)  # Increased
        min_required = max(SPECIAL_PAIRS.get(pair_name, {}).get("min_required", 150), 150)  # Increased

        df_15m = get_candles(pair_info['symbol'], "15", limit=limit_15m)
    
        if df_15m is None:
            print(f"❌ {pair_name}: Failed to fetch 15m candle data")
            return last_alerts.get(pair_name), '\n'.join(thread_log)
            
        if len(df_15m) < min_required:
            print(f"⚠️  {pair_name}: Insufficient 15m data (need {min_required}, got {len(df_15m)})")
            return last_alerts.get(pair_name), '\n'.join(thread_log)

        # Calculate 15m indicators
        ppo, _ = calculate_ppo(df_15m, PPO_FAST, PPO_SLOW, PPO_SIGNAL, PPO_USE_SMA)
        upw, dnw, _, _ = calculate_cirrus_cloud(df_15m)
        rma_50_15m = calculate_rma(df_15m['close'], 50)
        
        # Validate indicator lengths (FIXED: Added upw/dnw checks)
        if (len(ppo) < 3 or len(rma_50_15m) < 3 or 
            len(upw) < 3 or len(dnw) < 3):
            print(f"⚠️  {pair_name}: Indicators produced insufficient data (<3 points)")
            return last_alerts.get(pair_name), '\n'.join(thread_log)

        # Get values for the last closed 15m candle (index -2)
        open_prev = df_15m['open'].iloc[-2]
        close_prev = df_15m['close'].iloc[-2]
        high_prev = df_15m['high'].iloc[-2]
        low_prev = df_15m['low'].iloc[-2]
        
        ppo_curr = ppo.iloc[-2] 
        rma_50_15m_curr = rma_50_15m.iloc[-2]
        upw_curr = upw.iloc[-2]
        dnw_curr = dnw.iloc[-2]
        
        # Validate for NaN values (ADDED)
        if (np.isnan(ppo_curr) or np.isnan(rma_50_15m_curr) or 
            pd.isna(upw_curr) or pd.isna(dnw_curr)):
            log(f"Skipping {pair_name}: NaN values detected in 15m indicators")
            return last_alerts.get(pair_name), '\n'.join(thread_log)
        
        # Validate price data (ADDED)
        if close_prev <= 0 or open_prev <= 0:
            log(f"Skipping {pair_name}: Invalid price data")
            return last_alerts.get(pair_name), '\n'.join(thread_log)

        log(f"15m PPO: {ppo_curr:.4f}, RMA 50: {rma_50_15m_curr:.2f}")
        
        # --- 3. Get 5-Minute Candles and RMA 200 ---
        limit_5m = SPECIAL_PAIRS.get(pair_name, {}).get("limit_5m", 500)  # Increased
        min_required_5m = max(SPECIAL_PAIRS.get(pair_name, {}).get("min_required_5m", 250), 250)  # Increased
        
        rma_200_5m_curr = np.nan
        rma_200_available = False
        
        df_5m = get_candles(pair_info['symbol'], "5", limit=limit_5m)
      
        if df_5m is None or len(df_5m) < min_required_5m:
            log(f"⚠️  5m data insufficient for {pair_name} (need {min_required_5m}, got {len(df_5m) if df_5m is not None else 0})")
            log("RMA 200 5m check will be skipped - using only RMA 50 15m")
        else:
            rma_200_5m = calculate_rma(df_5m['close'], 200)
            
            if len(rma_200_5m) >= 2 and not np.isnan(rma_200_5m.iloc[-2]):
                rma_200_5m_curr = rma_200_5m.iloc[-2]
                rma_200_available = True
                log(f"5m RMA 200: {rma_200_5m_curr:.4f}")
            else:
                log("5m RMA 200 calculation produced NaN - will be skipped")
        
        # --- 4. Define Alert Conditions ---
        is_green = close_prev > open_prev
        is_red = close_prev < open_prev
        
        # FIXED: Handle RMA 200 unavailability properly
        if rma_200_available:
            rma_long_ok = rma_50_15m_curr < close_prev and rma_200_5m_curr < close_prev
            rma_short_ok = rma_50_15m_curr > close_prev and rma_200_5m_curr > close_prev
            log(f"Using both RMA checks (50 15m + 200 5m)")
        else:
            rma_long_ok = rma_50_15m_curr < close_prev
            rma_short_ok = rma_50_15m_curr > close_prev
            log(f"Using only RMA 50 15m (RMA 200 5m unavailable)")
    
        # Wick Checks
        candle_range = high_prev - low_prev
    
        if candle_range <= 0:
            log("Candle range is zero or negative, skipping wick checks")
            upper_wick_check = False
            lower_wick_check = False
        else:
            upper_wick_length = high_prev - max(open_prev, close_prev)
            lower_wick_length = min(open_prev, close_prev) - low_prev
            upper_wick_check = (upper_wick_length / candle_range) < 0.20
            lower_wick_check = (lower_wick_length / candle_range) < 0.20
        
        # Optional: Add volatility filter (ADDED)
        price_change_pct = abs((close_prev - open_prev) / open_prev) * 100
        if price_change_pct > 10:  # More than 10% move in 15m
            log(f"⚠️  Extreme volatility detected ({price_change_pct:.2f}%), skipping signal")
            return last_alerts.get(pair_name), '\n'.join(thread_log)
        
        # --- 5. Pivot Crossover Logic ---
        # NEW: Block signals if candle closes above R3 or below S3
        if close_prev > pivots['R3']:
            log(f"⚠️  Price closed above R3 (${pivots['R3']:.2f}), blocking LONG signal")
            # Allow state to continue but don't generate new signals
            return last_alerts.get(pair_name), '\n'.join(thread_log)
            
        if close_prev < pivots['S3']:
            log(f"⚠️  Price closed below S3 (${pivots['S3']:.2f}), blocking SHORT signal")
            # Allow state to continue but don't generate new signals
            return last_alerts.get(pair_name), '\n'.join(thread_log)
        
        long_pivot_lines = {'P': pivots['P'], 'R1': pivots['R1'], 'R2': pivots['R2'], 
                           'S1': pivots['S1'], 'S2': pivots['S2']}
        long_crossover_line = False
        long_crossover_name = None
 
        if is_green:
            # Check if candle crossed above any pivot (open below, close above)
            for name, line in long_pivot_lines.items():
                if open_prev <= line and close_prev > line: 
                    long_crossover_line = line
                    long_crossover_name = name
                    break  # Only take first crossover

        short_pivot_lines = {'P': pivots['P'], 'S1': pivots['S1'], 'S2': pivots['S2'], 
                            'R1': pivots['R1'], 'R2': pivots['R2']}
        short_crossover_line = False
        short_crossover_name = None
        
        if is_red:
            # Check if candle crossed below any pivot (open above, close below)
            for name, line in short_pivot_lines.items():
                if open_prev >= line and close_prev < line:
                    short_crossover_line = line
                    short_crossover_name = name
                    break  # Only take first crossover
   
        # --- 6. Signal Detection and State Management ---
        current_signal = None
        updated_state = last_alerts.get(pair_name) 
        
        if updated_state:
            log(f"Previous state loaded: {updated_state}")
        
        ist = pytz.timezone('Asia/Kolkata')
        formatted_time = datetime.now(ist).strftime('%d-%m-%Y @ %H:%M IST')
        price = close_prev
        
        # IMPROVED: State Reset Logic - Check and reset BEFORE evaluating new signals
        if updated_state and updated_state.startswith('fib_'):
            try:
                parts = updated_state.split('_')
                if len(parts) >= 3:
                    alert_type = parts[1]  # 'long' or 'short'
                    pivot_name = parts[2]  # 'R1', 'S1', etc.
                    pivot_value = pivots.get(pivot_name)
                    
                    log(f"Checking state reset: type={alert_type}, pivot={pivot_name} (${pivot_value}), current_price=${close_prev:,.2f}")
                    
                    if pivot_value is not None:
                        if alert_type == "long":
                            # Reset long position if price falls below the original pivot
                            if close_prev < pivot_value:
                                updated_state = None
                                log(f"🔄 STATE RESET: {pair_name} Long - Price ${close_prev:,.2f} fell below {pivot_name} ${pivot_value:,.2f}")
                            else:
                                log(f"State maintained: Price ${close_prev:,.2f} still above {pivot_name} ${pivot_value:,.2f}")
                     
                        elif alert_type == "short":
                            # Reset short position if price rises above the original pivot
                            if close_prev > pivot_value:
                                updated_state = None
                                log(f"🔄 STATE RESET: {pair_name} Short - Price ${close_prev:,.2f} rose above {pivot_name} ${pivot_value:,.2f}")
                            else:
                                log(f"State maintained: Price ${close_prev:,.2f} still below {pivot_name} ${pivot_value:,.2f}")
                    else:
                        # Pivot name not found in current pivots, reset state
                        log(f"⚠️  Invalid pivot name '{pivot_name}' in state, resetting")
                        updated_state = None
            except Exception as e:
                log(f"Error parsing saved state '{updated_state}': {e}")
                updated_state = None  # Reset on parse error
    
        # --- 7. Final Signal Checks ---
        # 🟢 LONG SIGNAL
        if (upw_curr and (not dnw_curr) and 
            (ppo_curr < 0.20) and 
            long_crossover_line and 
            upper_wick_check and
            rma_long_ok):
        
            current_signal = f"fib_long_{long_crossover_name}"
            log(f"\n🟢 FIB LONG SIGNAL DETECTED for {pair_name}!")
            
            if updated_state != current_signal:
                message = (
                    f"🟢 {pair_name} - **FIB LONG**\n"
                    f"Crossed & Closed Above **{long_crossover_name}** (${long_crossover_line:,.2f})\n"
                    f"PPO 15m: {ppo_curr:.2f}\n" 
                    f"Price: ${price:,.2f}\n"
                    f"{formatted_time}"
                )
                send_telegram_alert(message)
                
            updated_state = current_signal

        # 🔴 SHORT SIGNAL
        elif (dnw_curr and (not upw_curr) and 
              (ppo_curr > -0.20) and 
              short_crossover_line and 
              lower_wick_check and
              rma_short_ok):
        
            current_signal = f"fib_short_{short_crossover_name}"
            log(f"\n🔴 FIB SHORT SIGNAL DETECTED for {pair_name}!")
            
            if updated_state != current_signal:
                message = (
                    f"🔴 {pair_name} - **FIB SHORT**\n"
                    f"Crossed & Closed Below **{short_crossover_name}** (${short_crossover_line:,.2f})\n"
                    f"PPO 15m: {ppo_curr:.2f}\n"
                    f"Price: ${price:,.2f}\n"
                    f"{formatted_time}"
                )
                send_telegram_alert(message)
                
            updated_state = current_signal
                
        else:
            log(f"No Fibonacci Pivot signal conditions met for {pair_name}")
        
        return updated_state, '\n'.join(thread_log)

    except Exception as e:
        print(f"❌ Error checking {pair_name}: {e}")
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

    # --- STATE CLEANUP: Purge old/invalid state traces for a clean slate ---
    # Remove old state formats: 'srsi', 'buy', 'sell', 'short_zero', etc.
    # Only keep valid 'fib_long_X' or 'fib_short_X' formats
    keys_to_purge = []
    for key, value in list(last_alerts.items()):
        if value:
            value_str = str(value).lower()
            # Purge if it doesn't match the 'fib_long_' or 'fib_short_' pattern
            if not (value_str.startswith('fib_long_') or value_str.startswith('fib_short_')):
                keys_to_purge.append(key)
    
    if keys_to_purge:
        print(f"\n[INFO] 🧹 Purging old/invalid states for: {', '.join(keys_to_purge)}")
        for key in keys_to_purge:
            last_alerts[key] = None  # Set to None/clean state
    # -------------------------------------------------------------------
    
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
