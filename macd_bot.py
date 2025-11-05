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
# You can set this to 'False' after the next successful run
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
    "SOLUSD": {"limit_15m": 150, "min_required": 74, "limit_5m": 250, "min_required_5m": 183}
}

# Indicator settings
# PPO settings
PPO_FAST = 7
PPO_SLOW = 16
PPO_SIGNAL = 5
PPO_USE_SMA = False  # False = use EMA (as per your script)

# MACD settings
MACD_F = 100
MACD_S = 200
MACD_SG = 50

# RMA settings
RMA_50_PERIOD = 50   # RMA50 on 15min
RMA_200_PERIOD = 200   # RMA200 on 5min

# Cirrus Cloud settings
CIRRUS_CLOUD_ENABLED = True
X1 = 22
X2 = 9
X3 = 15
X4 = 5

# File to store last alert state
STATE_FILE = 'alert_state.json'

# ============ FUNCTIONS ============

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
        import traceback
        if DEBUG_MODE:
            traceback.print_exc()
        return False

def send_test_message():
    """Send a test message to verify Telegram connectivity"""
    ist = pytz.timezone('Asia/Kolkata')
    current_dt = datetime.now(ist)
    formatted_time = current_dt.strftime('%d-%m-%Y @ %H:%M IST')
    
    test_msg = f"🔔 Bot Started\nTest message from MACD Bot\nTime: {formatted_time}\nDebug Mode: {'ON' if DEBUG_MODE else 'OFF'}"
    
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
            import traceback
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
            return df
        else:
            print(f"Error fetching candles for {product_id}: {data.get('message', 'No message')}")
            return None
            
    except Exception as e:
        print(f"Exception fetching candles for {product_id}: {e}")
        return None

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    # Uses adjust=False for closer match to Pine ta.ema
    return data.ewm(span=period, adjust=False).mean()

def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    return data.rolling(window=period).mean()

def calculate_rma(data, period):
    """Calculate RMA (Smoothed Moving Average) - same as ta.rma in Pine Script"""
    # Uses alpha=1/period for closer match to Pine ta.rma
    return data.ewm(alpha=1/period, adjust=False).mean()

def calculate_ppo(df, fast=7, slow=16, signal=5, use_sma=False):
    """Calculate PPO (Percentage Price Oscillator) - matches Pine Script"""
    close = df['close']
    
    # Calculate fast and slow MAs
    if use_sma:
        fast_ma = calculate_sma(close, fast)
        slow_ma = calculate_sma(close, slow)
    else:
        fast_ma = calculate_ema(close, fast)
        slow_ma = calculate_ema(close, slow)
    
    # Calculate PPO
    ppo = (fast_ma - slow_ma) / slow_ma * 100
    
    # Calculate signal line
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
    # avrng = ta.ema(math.abs(x - x[1]), t)
    avrng = calculate_ema(np.abs(x.diff()), t)
    # smoothrng = ta.ema(avrng, wper) * m
    smoothrng = calculate_ema(avrng, wper) * m
    return smoothrng

def rngfilt(x, r):
    """
    Implements rngfiltx1x1 from Pine Script using robust array iteration.
    This is the complex, self-referential filter logic.
    """
    # Use a list to store the results, starting with the first value
    # Pine: rngfiltx1x1 = x (Initialization for the first bar)
    result_list = [x.iloc[0]] 
    
    for i in range(1, len(x)):
        # Previous filtered value (nz(rngfiltx1x1[1]))
        prev_f = result_list[-1]
        curr_x = x.iloc[i] # Current close price (x)
        curr_r = r.iloc[i] # Current smoothed range (r)
        
        f = 0.0 # Initialize current filter value

        # Pine: x > nz(rngfiltx1x1[1]) 
        if curr_x > prev_f:
            # Pine: f := x - r < f ? f : x - r
            if curr_x - curr_r < prev_f:
                f = prev_f
            else:
                f = curr_x - curr_r
        else:
            # Pine: x + r > f ? f : x + r
            if curr_x + curr_r > prev_f:
                f = prev_f
            else:
                f = curr_x + curr_r
        
        result_list.append(f)
        
    # Convert the list back to a Pandas Series, matching the original index
    return pd.Series(result_list, index=x.index)


def calculate_cirrus_cloud(df):
    """
    Calculate Cirrus Cloud Upw and Dnw conditions.
    *** FINAL CONFIRMED LOGIC: Based on Pine Script source code (`filtx1 < filtx12 ? colorUp : colorDown`) ***
    """
    close = df['close'].copy()
    
    # Calculate smoothed ranges
    smrngx1x = smoothrng(close, X1, X2)
    smrngx1x2 = smoothrng(close, X3, X4)
    
    # Apply range filter
    filtx1 = rngfilt(close, smrngx1x)
    filtx12 = rngfilt(close, smrngx1x2)
    
    # The Pine Script color logic: filtx1 < filtx12 ? colorUp : colorDown
    # Upw (Green) is True when filter 1 line is BELOW filter 2 line.
    upw = filtx1 < filtx12 
    # Dnw (Red) is True when filter 1 line is ABOVE filter 2 line.
    dnw = filtx1 > filtx12 
    
    # Return filter lines for better debugging
    return upw, dnw, filtx1, filtx12 

def check_pair(pair_name, pair_info, last_alerts):
    """Check PPO and MACD crossover conditions for a pair"""
    try:
        if pair_info is None:
            return None
        
        debug_log(f"\n{'='*60}")
        debug_log(f"Checking {pair_name}")
        debug_log(f"{'='*60}")
        
        # Check if this pair has special requirements
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
        
        # Fetch 15-minute candles for PPO, MACD, EMA100
        df_15m = get_candles(pair_info['symbol'], "15", limit=limit_15m)
        
        # Fetch 5-minute candles for RMA200
        df_5m = get_candles(pair_info['symbol'], "5", limit=limit_5m)
        
        if df_15m is None or len(df_15m) < min_required:
            print(f"Not enough 15m data for {pair_name} ({len(df_15m) if df_15m is not None else 0}/{min_required})")
            return None
            
        if df_5m is None or len(df_5m) < min_required_5m:
            print(f"Not enough 5m data for {pair_name} ({len(df_5m) if df_5m is not None else 0}/{min_required_5m})")
            return None
        
        # Calculate indicators on 15min timeframe
        ppo, ppo_signal = calculate_ppo(df_15m, PPO_FAST, PPO_SLOW, PPO_SIGNAL, PPO_USE_SMA)
        macd, macd_signal = calculate_macd(df_15m, MACD_F, MACD_S, MACD_SG)
        rma_50 = calculate_rma(df_15m['close'], RMA_50_PERIOD)
        
        # Calculate RMA200 on 5min timeframe
        rma_200 = calculate_rma(df_5m['close'], RMA_200_PERIOD)
        
        # Calculate Cirrus Cloud on 15min timeframe (now returns filtx1/filtx12 for debug)
        upw, dnw, filtx1, filtx12 = calculate_cirrus_cloud(df_15m)
        
        # Get latest values from 15min
        ppo_curr = ppo.iloc[-1]
        ppo_prev = ppo.iloc[-2]
        ppo_signal_curr = ppo_signal.iloc[-1]
        ppo_signal_prev = ppo_signal.iloc[-2]
        
        macd_curr = macd.iloc[-1]
        macd_signal_curr = macd_signal.iloc[-1]
        
        close_curr = df_15m['close'].iloc[-1]
        rma50_curr = rma_50.iloc[-1]
        
        upw_curr = upw.iloc[-1]
        upw_prev = upw.iloc[-2]
        dnw_curr = dnw.iloc[-1]
        dnw_prev = dnw.iloc[-2]
        
        # Get latest values from 5min
        close_5m_curr = df_5m['close'].iloc[-1]
        rma200_curr = rma_200.iloc[-1]
        
        # Debug: Print all indicator values
        debug_log(f"Price: ${close_curr:,.2f}")
        debug_log(f"PPO: {ppo_curr:.4f} (prev: {ppo_prev:.4f})")
        debug_log(f"PPO Signal: {ppo_signal_curr:.4f} (prev: {ppo_signal_prev:.4f})")
        debug_log(f"MACD: {macd_curr:.4f}, Signal: {macd_signal_curr:.4f}")
        debug_log(f"RMA50 (15m): {rma50_curr:.2f}, Close: {close_curr:.2f}")
        debug_log(f"RMA200 (5m): {rma200_curr:.2f}, Close: {close_5m_curr:.2f}")
        
        # *** DEBUG LINES: Print raw filter values for diagnostics ***
        debug_log(f"Cirrus Filter 1 (filtx1): {filtx1.iloc[-1]:.4f}") 
        debug_log(f"Cirrus Filter 2 (filtx12): {filtx12.iloc[-1]:.4f}") 
        
        debug_log(f"Cirrus Cloud - Upw: {upw_curr}, Dnw: {dnw_curr}")
        
        # Detect PPO crossovers
        ppo_cross_up = (ppo_prev <= ppo_signal_prev) and (ppo_curr > ppo_signal_curr)
        ppo_cross_down = (ppo_prev >= ppo_signal_prev) and (ppo_curr < ppo_signal_curr)
        
        # Detect PPO zero-line crossovers
        ppo_cross_above_zero = (ppo_prev <= 0) and (ppo_curr > 0)
        ppo_cross_below_zero = (ppo_prev >= 0) and (ppo_curr < 0)
        ppo_cross_above_011 = (ppo_prev <= 0.11) and (ppo_curr > 0.11)
        ppo_cross_below_minus011 = (ppo_prev >= -0.11) and (ppo_curr < -0.11)
        
        # PPO value conditions
        ppo_below_020 = ppo_curr < 0.20
        ppo_above_minus020 = ppo_curr > -0.20
        ppo_above_signal = ppo_curr > ppo_signal_curr
        ppo_below_signal = ppo_curr < ppo_signal_curr
        
        # MACD conditions
        macd_above_signal = macd_curr > macd_signal_curr
        macd_below_signal = macd_curr < macd_signal_curr
        
        # RMA conditions
        close_above_rma50 = close_curr > rma50_curr
        close_below_rma50 = close_curr < rma50_curr
        close_above_rma200 = close_5m_curr > rma200_curr
        close_below_rma200 = close_5m_curr < rma200_curr
        
        # Debug: Print crossover detections
        debug_log(f"\nCrossover Checks:")
        debug_log(f"  PPO cross up: {ppo_cross_up}")
        debug_log(f"  PPO cross down: {ppo_cross_down}")
        debug_log(f"  PPO above 0: {ppo_cross_above_zero}")
        debug_log(f"  PPO below 0: {ppo_cross_below_zero}")
        debug_log(f"  PPO above 0.11: {ppo_cross_above_011}")
        debug_log(f"  PPO below -0.11: {ppo_cross_below_minus011}")
        
        debug_log(f"\nCondition Checks:")
        debug_log(f"  PPO < 0.20: {ppo_below_020}")
        debug_log(f"  PPO > -0.20: {ppo_above_minus020}")
        debug_log(f"  PPO > Signal: {ppo_above_signal}")
        debug_log(f"  MACD > Signal: {macd_above_signal}")
        debug_log(f"  Close > RMA50: {close_above_rma50}")
        debug_log(f"  Close > RMA200: {close_above_rma200}")
        debug_log(f"  Upw (Cirrus): {upw_curr}") # Now means GREEN
        debug_log(f"  Dnw (Cirrus): {dnw_curr}") # Now means RED
        
        current_state = None
        
        # Get IST time in correct format
        ist = pytz.timezone('Asia/Kolkata')
        current_dt = datetime.now(ist)
        formatted_time = current_dt.strftime('%d-%m-%Y @ %H:%M IST')
        price = df_15m['close'].iloc[-1]
        
        # BUY: PPO crosses up AND PPO < 0.20 AND MACD > Signal AND Close > RMA50 AND Close > RMA200 AND Cirrus Cloud is GREEN (Upw)
        if ppo_cross_up and ppo_below_020 and macd_above_signal and close_above_rma50 and close_above_rma200 and upw_curr:
            current_state = "buy"
            debug_log(f"\n🟢 BUY SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "buy":
                message = f"🟢 {pair_name} - BUY\nPPO - SIGNAL Crossover (PPO: {ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
                send_telegram_alert(message)
            else:
                debug_log(f"BUY already alerted for {pair_name}, skipping duplicate")
        
        # SELL: PPO crosses down AND PPO > -0.20 AND MACD < Signal AND Close < RMA50 AND Close < RMA200 AND Cirrus Cloud is RED (Dnw)
        elif ppo_cross_down and ppo_above_minus020 and macd_below_signal and close_below_rma50 and close_below_rma200 and dnw_curr:
            current_state = "sell"
            debug_log(f"\n🔴 SELL SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "sell":
                message = f"🔴 {pair_name} - SELL\nPPO - SIGNAL Crossunder (PPO: {ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
                send_telegram_alert(message)
            else:
                debug_log(f"SELL already alerted for {pair_name}, skipping duplicate")
        
        # LONG: PPO > Signal AND PPO crosses above 0 AND Cirrus Cloud is GREEN (Upw)
        elif ppo_cross_above_zero and ppo_above_signal and macd_above_signal and close_above_rma50 and close_above_rma200 and upw_curr:
            current_state = "long_zero"
            debug_log(f"\n🟢 LONG (0) SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "long_zero":
                message = f"🟢 {pair_name} - LONG\nPPO crossing above 0 ({ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
                send_telegram_alert(message)
            else:
                debug_log(f"LONG (0) already alerted for {pair_name}, skipping duplicate")
        
        # LONG: PPO > Signal AND PPO crosses above 0.11 AND Cirrus Cloud is GREEN (Upw)
        elif ppo_cross_above_011 and ppo_above_signal and macd_above_signal and close_above_rma50 and close_above_rma200 and upw_curr:
            current_state = "long_011"
            debug_log(f"\n🟢 LONG (0.11) SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "long_011":
                message = f"🟢 {pair_name} - LONG\nPPO crossing above 0.11 ({ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
                send_telegram_alert(message)
            else:
                debug_log(f"LONG (0.11) already alerted for {pair_name}, skipping duplicate")
        
        # SHORT: PPO < Signal AND PPO crosses below 0 AND Cirrus Cloud is RED (Dnw)
        elif ppo_cross_below_zero and ppo_below_signal and macd_below_signal and close_below_rma50 and close_below_rma200 and dnw_curr:
            current_state = "short_zero"
            debug_log(f"\n🔴 SHORT (0) SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "short_zero":
                message = f"🔴 {pair_name} - SHORT\nPPO crossing below 0 ({ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
                send_telegram_alert(message)
            else:
                debug_log(f"SHORT (0) already alerted for {pair_name}, skipping duplicate")
        
        # SHORT: PPO < Signal AND PPO crosses below -0.11 AND Cirrus Cloud is RED (Dnw)
        elif ppo_cross_below_minus011 and ppo_below_signal and macd_below_signal and close_below_rma50 and close_below_rma200 and dnw_curr:
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
            import traceback
            traceback.print_exc()
        return None

def main():
    """Main function - runs once per GitHub Actions execution"""
    print("=" * 50)
    ist = pytz.timezone('Asia/Kolkata')
    start_time = datetime.now(ist)
    print(f"PPO/MACD/Cirrus Cloud Alert Bot - {start_time.strftime('%d-%m-%Y @ %H:%M IST')}")
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
