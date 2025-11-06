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

# Delta Exchange API
DELTA_API_BASE = "https://api.delta.exchange"

# Trading pairs to monitor
PAIRS = {
    "BTCUSD": None, "ETHUSD": None, "SOLUSD": None, "AVAXUSD": None,
    "BCHUSD": None, "XRPUSD": None, "BNBUSD": None, "LTCUSD": None,
    "DOTUSD": None, "ADAUSD": None, "SUIUSD": None, "AAVEUSD": None
}

# Special data requirements for pairs with limited history (Reused from macd_bot)
SPECIAL_PAIRS = {
    "SOLUSD": {"limit_15m": 150, "min_required": 74, "limit_5m": 250, "min_required_5m": 183}
}

# Indicator settings (Reused from macd_bot)
PPO_FAST = 7
PPO_SLOW = 16
PPO_SIGNAL = 5
PPO_USE_SMA = False

MACD_F = 100
MACD_S = 200
MACD_SG = 50

# Cirrus Cloud settings (Reused from macd_bot)
X1 = 22
X2 = 9
X3 = 15
X4 = 5

# Volume and Pivot settings (New/Specific)
VOLUME_SMA_PERIOD = 20
PIVOT_LOOKBACK_PERIOD = 15 # Lookback in days for daily high/low/close

# File to store last alert state (separate file for this bot)
STATE_FILE = 'alert_state.json' 

# ============ UTILITY FUNCTIONS (Extracted from macd_bot.txt) ============

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
    
    test_msg = f"🔔 Fibonacci Bot Started\nTest message from Fibonacci Pivot Bot\nTime: {formatted_time}\nDebug Mode: {'ON' if DEBUG_MODE else 'OFF'}"
    
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

def calculate_macd(df, fast=100, slow=200, signal=50):
    """Calculate MACD"""
    close = df['close']
    
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    
    return macd_line, signal_line

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

# ============ FIBONACCI PIVOT FUNCTIONS (New) ============

def get_previous_day_ohlc(product_id, days_back_limit=15):
    """Fetch 1-day candles to get the previous day's H, L, C for pivot calculation."""
    # Fetch enough data to ensure the last complete daily candle is captured
    df_daily = get_candles(product_id, resolution="D", limit=days_back_limit + 5)
    
    if df_daily is None or len(df_daily) < 2:
        return None
    
    # The last row is often the current, incomplete day. We need the one before it.
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
    
def calculate_volume_sma(df, period):
    """Calculate Simple Moving Average of Volume"""
    return df['volume'].rolling(window=period).mean()

def check_pair(pair_name, pair_info, last_alerts):
    """Check Fibonacci Pivot conditions for a pair"""
    try:
        if pair_info is None:
            return None
        
        debug_log(f"\n{'='*60}")
        debug_log(f"Checking {pair_name} for Fibonacci Pivot Alerts")
        debug_log(f"{'='*60}")
        
        # --- 1. Get Pivot Data (Previous Day) ---
        prev_day_ohlc = get_previous_day_ohlc(pair_info['symbol'], PIVOT_LOOKBACK_PERIOD)
        if prev_day_ohlc is None:
            print(f"Skipping {pair_name}: Failed to get previous day OHLC data.")
            return None
            
        pivots = calculate_fibonacci_pivots(
            prev_day_ohlc['high'], prev_day_ohlc['low'], prev_day_ohlc['close']
        )
        debug_log(f"Daily Pivots: P={pivots['P']:.2f}, R1={pivots['R1']:.2f}, S1={pivots['S1']:.2f}")

        # --- 2. Get 15-Minute Candles and Indicators ---
        limit_15m = SPECIAL_PAIRS.get(pair_name, {}).get("limit_15m", 210)
        min_required = SPECIAL_PAIRS.get(pair_name, {}).get("min_required", 200)

        df_15m = get_candles(pair_info['symbol'], "15", limit=limit_15m)
        if df_15m is None or len(df_15m) < min_required:
            print(f"Not enough 15m data for {pair_name}.")
            return None

        # Calculate indicators
        # Note: We use .iloc[-2] for all indicators and candle data to check the last fully CLOSED 15m candle.
        ppo, _ = calculate_ppo(df_15m, PPO_FAST, PPO_SLOW, PPO_SIGNAL, PPO_USE_SMA)
        macd, macd_signal = calculate_macd(df_15m, MACD_F, MACD_S, MACD_SG)
        upw, dnw, _, _ = calculate_cirrus_cloud(df_15m) # Upw is Green, Dnw is Red
        vol_sma = calculate_volume_sma(df_15m, VOLUME_SMA_PERIOD)

        # Get values for the last closed 15m candle (index -2)
        open_prev = df_15m['open'].iloc[-2]
        close_prev = df_15m['close'].iloc[-2]
        high_prev = df_15m['high'].iloc[-2]
        low_prev = df_15m['low'].iloc[-2]
        volume_prev = df_15m['volume'].iloc[-2]
        vol_sma_prev = vol_sma.iloc[-2]
        
        ppo_curr = ppo.iloc[-2] 
        macd_curr = macd.iloc[-2]
        macd_signal_curr = macd_signal.iloc[-2]
        upw_curr = upw.iloc[-2]
        dnw_curr = dnw.iloc[-2]

        debug_log(f"PPO: {ppo_curr:.4f}, MACD: {macd_curr:.4f}, MACD Signal: {macd_signal_curr:.4f}")
        
        # --- 3. Define Alert Conditions ---
        
        # Candle Characteristics
        is_green = close_prev > open_prev
        is_red = close_prev < open_prev
        
        # Check 5: Volume > 20 SMA of Volume
        vol_check = volume_prev > vol_sma_prev

        # Wick Checks
        candle_range = high_prev - low_prev
        if candle_range <= 0:
            debug_log("Candle range is zero or negative, skipping wick checks.")
            upper_wick_check = False
            lower_wick_check = False
        else:
            upper_wick_length = high_prev - max(open_prev, close_prev)
            lower_wick_length = min(open_prev, close_prev) - low_prev
            
            # Check 6 (Long): Upper wick < 20% of total candle length 
            upper_wick_check = (upper_wick_length / candle_range) < 0.20 
            # Check 6 (Short): Lower wick < 20% of total candle length 
            lower_wick_check = (lower_wick_length / candle_range) < 0.20 
        
        # --- 4. Pivot Crossover Logic ---
        
        # LONG Crossover Check: Green candle crosses and closes ABOVE a pivot line (P, R1, R2, S1, S2)
        long_pivot_lines = {'P': pivots['P'], 'R1': pivots['R1'], 'R2': pivots['R2'], 'S1': pivots['S1'], 'S2': pivots['S2']}
        long_crossover_line = False
        long_crossover_name = None
        if is_green:
            for name, line in long_pivot_lines.items():
                # Candle opens below the pivot line AND closes above the pivot line
                if open_prev <= line and close_prev > line: 
                    long_crossover_line = line
                    long_crossover_name = name
                    break

        # SHORT Crossover Check: Red candle crosses and closes BELOW a pivot line (P, S1, S2, R1, R2)
        short_pivot_lines = {'P': pivots['P'], 'S1': pivots['S1'], 'S2': pivots['S2'], 'R1': pivots['R1'], 'R2': pivots['R2']}
        short_crossover_line = False
        short_crossover_name = None
        if is_red:
            for name, line in short_pivot_lines.items():
                # Candle opens above the pivot line AND closes below the pivot line
                if open_prev >= line and close_prev < line:
                    short_crossover_line = line
                    short_crossover_name = name
                    break
        
        # --- 5. Signal Detection ---
        
        current_state = None
        ist = pytz.timezone('Asia/Kolkata')
        formatted_time = datetime.now(ist).strftime('%d-%m-%Y @ %H:%M IST')
        price = close_prev
        
        
        # 🟢 FINAL LONG SIGNAL CHECK
        # 1. Cirrus Cloud is Green (upw_curr)
        # 2. Macd is above signal (macd_curr > macd_signal_curr)
        # 3. PPO is below 0 (ppo_curr < 0)
        # 4. Long Crossover (long_crossover_line is not False)
        # 5. Volume check (vol_check)
        # 6. Upper wick check (upper_wick_check)
        if (upw_curr and 
            (macd_curr > macd_signal_curr) and 
            (ppo_curr < 0) and 
            long_crossover_line and 
            vol_check and 
            upper_wick_check):
            
            current_state = "fib_long"
            debug_log(f"\n🟢 FIB LONG SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "fib_long":
                message = (
                    f"🟢 {pair_name} - FIB LONG\n"
                    f"Crossed & Closed Above {long_crossover_name} (${long_crossover_line:,.2f})\n"
                    f"PPO: {ppo_curr:.2f}\n"
                    f"Price: ${price:,.2f}\n"
                    f"{formatted_time}"
                )
                send_telegram_alert(message)
            else:
                debug_log(f"FIB LONG already alerted for {pair_name}, skipping duplicate")


        # 🔴 FINAL SHORT SIGNAL CHECK
        # 1. Cirrus Cloud is Red (dnw_curr)
        # 2. Macd is below signal (macd_curr < macd_signal_curr)
        # 3. PPO is above 0 (ppo_curr > 0)
        # 4. Short Crossover (short_crossover_line is not False)
        # 5. Volume check (vol_check)
        # 6. Lower wick check (lower_wick_check)
        elif (dnw_curr and 
              (macd_curr < macd_signal_curr) and 
              (ppo_curr > 0) and 
              short_crossover_line and 
              vol_check and 
              lower_wick_check):
              
            current_state = "fib_short"
            debug_log(f"\n🔴 FIB SHORT SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "fib_short":
                message = (
                    f"🔴 {pair_name} - FIB SHORT\n"
                    f"Crossed & Closed Below {short_crossover_name} (${short_crossover_line:,.2f})\n"
                    f"PPO: {ppo_curr:.2f}\n"
                    f"Price: ${price:,.2f}\n"
                    f"{formatted_time}"
                )
                send_telegram_alert(message)
            else:
                debug_log(f"FIB SHORT already alerted for {pair_name}, skipping duplicate")
                
        else:
            debug_log(f"No Fibonacci Pivot signal conditions met for {pair_name}")
        
        return current_state
        
    except Exception as e:
        print(f"Error checking {pair_name}: {e}")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
        return None

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
    alerts_sent = 0
    
    # Use a ThreadPoolExecutor to run all 'check_pair' calls in parallel.
    with ThreadPoolExecutor(max_workers=10) as executor:
        
        future_to_pair = {}
        
        for pair_name, pair_info in PAIRS.items():
            if pair_info is not None:
                future = executor.submit(check_pair, pair_name, pair_info, last_alerts)
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
                    import traceback
                    traceback.print_exc()
                continue
            
    # Save state for next run
    save_state(last_alerts)
    
    end_time = datetime.now(ist)
    elapsed = (end_time - start_time).total_seconds()
    print(f"✓ Check complete. {alerts_sent} alerts sent. ({elapsed:.1f}s)")
    print("=" * 50)

if __name__ == "__main__":
    main()
