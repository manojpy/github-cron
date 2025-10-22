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
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '8462496498:AAFEfw5DU1YtZ_D4nTNbuV5RIdL2K_DqgE0')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '203813932')

# DEBUG MODE - Set to True to see detailed logs for specific pairs
DEBUG_MODE = True
DEBUG_PAIRS = ["SOLUSD"]  # Add pairs you want to debug, or leave empty for all pairs

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
    "SOLUSD": {"limit_15m": 150, "min_required": 74}  # SOLUSD only returns ~74 candles max
}

# Indicator settings
# PPO settings
PPO_FAST = 7
PPO_SLOW = 16
PPO_SIGNAL = 5
PPO_USE_SMA = False  # False = use EMA (as per your script)

# MACD settings
MACD_FAST = 100
MACD_SLOW = 200
MACD_SIGNAL = 50

# EMA/RMA settings
EMA_100_PERIOD = 100   # EMA100 on 15min
RMA_200_PERIOD = 200   # RMA200 on 5min

# File to store last alert state
STATE_FILE = 'alert_state.json'

# ============ FUNCTIONS ============

def debug_log(pair_name, message):
    """Print debug message if DEBUG_MODE is enabled and pair matches"""
    if DEBUG_MODE and (not DEBUG_PAIRS or pair_name in DEBUG_PAIRS):
        print(f"[DEBUG {pair_name}] {message}")

def load_state():
    """Load previous alert state from file"""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading state: {e}")
    return {}

def save_state(state):
    """Save alert state to file"""
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        print(f"Error saving state: {e}")

def send_telegram_alert(message):
    """Send alert message via Telegram"""
    try:
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
            print(f"Telegram error: {response_data}")
            return False
        
    except Exception as e:
        print(f"Error sending Telegram message: {e}")
        return False

def send_test_message():
    """Send a test message to verify Telegram connectivity"""
    test_message = (
        f"🧪 TEST MESSAGE\n"
        f"Your MACD Bot is working!\n"
        f"Time: {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%d-%m-%Y @ %H:%M IST')}"
    )
    if send_telegram_alert(test_message):
        print("✓ Test message sent successfully!")
        return True
    else:
        print("✗ Failed to send test message")
        return False

def get_product_ids():
    """Fetch all product IDs from Delta Exchange"""
    try:
        response = requests.get(f"{DELTA_API_BASE}/v2/products", timeout=10)
        data = response.json()
        
        if data.get('success'):
            products = data['result']
            
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
                            debug_log(pair_name, f"Found product - ID: {product['id']}, Symbol: {product['symbol']}")
            
            return True
        else:
            print(f"API Error: {data}")
            return False
            
    except Exception as e:
        print(f"Error fetching products: {e}")
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
            print(f"Error fetching candles for {product_id}: {data.get('message', 'No message')}")
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

def calculate_rma(data, period):
    """Calculate RMA (Smoothed Moving Average) - same as ta.rma in Pine Script"""
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

def check_pair(pair_name, pair_info, last_alerts):
    """Check PPO and MACD crossover conditions for a pair"""
    try:
        if pair_info is None:
            debug_log(pair_name, "Pair info is None - skipping")
            return None
        
        # Check if this pair has special requirements
        if pair_name in SPECIAL_PAIRS:
            limit_15m = SPECIAL_PAIRS[pair_name]["limit_15m"]
            min_required = SPECIAL_PAIRS[pair_name]["min_required"]
        else:
            limit_15m = 210
            min_required = 200
        
        debug_log(pair_name, f"Fetching 15m candles (limit: {limit_15m})")
        # Fetch 15-minute candles for PPO, MACD, EMA100
        df_15m = get_candles(pair_info['symbol'], "15", limit=limit_15m)
        
        debug_log(pair_name, f"Fetching 5m candles")
        # Fetch 5-minute candles for RMA200
        df_5m = get_candles(pair_info['symbol'], "5", limit=210)
        
        if df_15m is None or len(df_15m) < min_required:
            debug_log(pair_name, f"Not enough 15m data: {len(df_15m) if df_15m is not None else 0}/{min_required}")
            print(f"Not enough 15m data for {pair_name} ({len(df_15m) if df_15m is not None else 0}/{min_required})")
            return None
            
        if df_5m is None or len(df_5m) < 200:
            debug_log(pair_name, f"Not enough 5m data: {len(df_5m) if df_5m is not None else 0}/200")
            print(f"Not enough 5m data for {pair_name} ({len(df_5m) if df_5m is not None else 0}/200)")
            return None
        
        debug_log(pair_name, "Calculating indicators...")
        
        # Calculate indicators on 15min timeframe
        ppo, ppo_signal = calculate_ppo(df_15m, PPO_FAST, PPO_SLOW, PPO_SIGNAL, PPO_USE_SMA)
        macd, macd_signal = calculate_macd(df_15m, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        ema_100 = calculate_ema(df_15m['close'], EMA_100_PERIOD)
        
        # Calculate RMA200 on 5min timeframe
        rma_200 = calculate_rma(df_5m['close'], RMA_200_PERIOD)
        
        # Get latest values from 15min
        ppo_curr = ppo.iloc[-1]
        ppo_prev = ppo.iloc[-2]
        ppo_signal_curr = ppo_signal.iloc[-1]
        ppo_signal_prev = ppo_signal.iloc[-2]
        
        macd_curr = macd.iloc[-1]
        macd_signal_curr = macd_signal.iloc[-1]
        
        close_curr = df_15m['close'].iloc[-1]
        ema100_curr = ema_100.iloc[-1]
        
        # Get latest values from 5min
        close_5m_curr = df_5m['close'].iloc[-1]
        rma200_curr = rma_200.iloc[-1]
        
        # Debug output
        debug_log(pair_name, f"Price (15m): {close_curr:.2f}")
        debug_log(pair_name, f"PPO: {ppo_curr:.4f} | Signal: {ppo_signal_curr:.4f} | Prev PPO: {ppo_prev:.4f}")
        debug_log(pair_name, f"MACD: {macd_curr:.4f} | Signal: {macd_signal_curr:.4f}")
        debug_log(pair_name, f"EMA100: {ema100_curr:.2f}")
        debug_log(pair_name, f"RMA200 (5m): {rma200_curr:.2f}")
        
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
        
        # EMA/RMA conditions
        close_above_ema100 = close_curr > ema100_curr
        close_below_ema100 = close_curr < ema100_curr
        close_above_rma200 = close_5m_curr > rma200_curr
        close_below_rma200 = close_5m_curr < rma200_curr
        
        # Debug condition checks
        debug_log(pair_name, f"PPO Cross Up: {ppo_cross_up} | Cross Down: {ppo_cross_down}")
        debug_log(pair_name, f"PPO < 0.20: {ppo_below_020} | PPO > -0.20: {ppo_above_minus020}")
        debug_log(pair_name, f"MACD > Signal: {macd_above_signal} | Close > EMA100: {close_above_ema100}")
        debug_log(pair_name, f"Close > RMA200: {close_above_rma200}")
        
        current_state = None
        
        # Get IST time in correct format
        ist = pytz.timezone('Asia/Kolkata')
        current_dt = datetime.now(ist)
        formatted_time = current_dt.strftime('%d-%m-%Y @ %H:%M IST')
        price = df_15m['close'].iloc[-1]
        
        # BUY: PPO crosses up AND PPO < 0.20 AND MACD > Signal AND Close > EMA100 AND Close > RMA200
        if ppo_cross_up and ppo_below_020 and macd_above_signal and close_above_ema100 and close_above_rma200:
            current_state = "buy"
            debug_log(pair_name, "BUY signal triggered!")
            if last_alerts.get(pair_name) != "buy":
                message = (
                    f"🟢 {pair_name} - BUY\n"
                    f"PPO - SIGNAL Crossover (PPO: {ppo_curr:.2f})\n"
                    f"Price: ${price:,.2f}\n"
                    f"{formatted_time}"
                )
                send_telegram_alert(message)
        
        # SELL: PPO crosses down AND PPO > -0.20 AND MACD < Signal AND Close < EMA100 AND Close < RMA200
        elif ppo_cross_down and ppo_above_minus020 and macd_below_signal and close_below_ema100 and close_below_rma200:
            current_state = "sell"
            debug_log(pair_name, "SELL signal triggered!")
            if last_alerts.get(pair_name) != "sell":
                message = (
                    f"🔴 {pair_name} - SELL\n"
                    f"PPO - SIGNAL Crossunder (PPO: {ppo_curr:.2f})\n"
                    f"Price: ${price:,.2f}\n"
                    f"{formatted_time}"
                )
                send_telegram_alert(message)
        
        # LONG: PPO > Signal AND PPO crosses above 0
        elif ppo_cross_above_zero and ppo_above_signal and macd_above_signal and close_above_ema100 and close_above_rma200:
            current_state = "long_zero"
            debug_log(pair_name, "LONG (zero) signal triggered!")
            if last_alerts.get(pair_name) != "long_zero":
                message = (
                    f"🟢 {pair_name} - LONG\n"
                    f"PPO crossing above 0 ({ppo_curr:.2f})\n"
                    f"Price: ${price:,.2f}\n"
                    f"{formatted_time}"
                )
                send_telegram_alert(message)
        
        # LONG: PPO > Signal AND PPO crosses above 0.11
        elif ppo_cross_above_011 and ppo_above_signal and macd_above_signal and close_above_ema100 and close_above_rma200:
            current_state = "long_011"
            debug_log(pair_name, "LONG (0.11) signal triggered!")
            if last_alerts.get(pair_name) != "long_011":
                message = (
                    f"🟢 {pair_name} - LONG\n"
                    f"PPO crossing above 0.11 ({ppo_curr:.2f})\n"
                    f"Price: ${price:,.2f}\n"
                    f"{formatted_time}"
                )
                send_telegram_alert(message)
        
        # SHORT: PPO < Signal AND PPO crosses below 0
        elif ppo_cross_below_zero and ppo_below_signal and macd_below_signal and close_below_ema100 and close_below_rma200:
            current_state = "short_zero"
            debug_log(pair_name, "SHORT (zero) signal triggered!")
            if last_alerts.get(pair_name) != "short_zero":
                message = (
                    f"🔴 {pair_name} - SHORT\n"
                    f"PPO crossing below 0 ({ppo_curr:.2f})\n"
                    f"Price: ${price:,.2f}\n"
                    f"{formatted_time}"
                )
                send_telegram_alert(message)
        
        # SHORT: PPO < Signal AND PPO crosses below -0.11
        elif ppo_cross_below_minus011 and ppo_below_signal and macd_below_signal and close_below_ema100 and close_below_rma200:
            current_state = "short_011"
            debug_log(pair_name, "SHORT (0.11) signal triggered!")
            if last_alerts.get(pair_name) != "short_011":
                message = (
                    f"🔴 {pair_name} - SHORT\n"
                    f"PPO crossing below -0.11 ({ppo_curr:.2f})\n"
                    f"Price: ${price:,.2f}\n"
                    f"{formatted_time}"
                )
                send_telegram_alert(message)
        
        return current_state
        
    except Exception as e:
        print(f"Error checking {pair_name}: {e}")
        debug_log(pair_name, f"Exception: {e}")
        return None

def main():
    """Main function - runs once per GitHub Actions execution"""
    print("=" * 50)
    ist = pytz.timezone('Asia/Kolkata')
    start_time = datetime.now(ist)
    print(f"PPO/MACD Alert Bot - {start_time.strftime('%d-%m-%Y @ %H:%M IST')}")
    if DEBUG_MODE:
        print(f"DEBUG MODE: ON (Pairs: {DEBUG_PAIRS if DEBUG_PAIRS else 'ALL'})")
    print("=" * 50)
    
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
    # We limit workers to 10 to avoid hitting the API too hard all at once.
    with ThreadPoolExecutor(max_workers=10) as executor:
        
        # Create a dictionary to map a running "future" (thread) to its pair_name
        future_to_pair = {}
        
        for pair_name, pair_info in PAIRS.items():
            if pair_info is not None:
                # Submit the task to the thread pool.
                # The executor runs 'check_pair(pair_name, pair_info, last_alerts)' in the background.
                future = executor.submit(check_pair, pair_name, pair_info, last_alerts)
                future_to_pair[future] = pair_name

        # As each thread finishes, process its result
        for future in as_completed(future_to_pair):
            pair_name = future_to_pair[future]
            try:
                # Get the return value from the check_pair function
                new_state = future.result() 
                if new_state:
                    last_alerts[pair_name] = new_state
                    alerts_sent += 1
            except Exception as e:
                # Catch any error that happened inside the thread
                print(f"Error processing {pair_name} in thread: {e}")
                continue
            
    # Save state for next run
    save_state(last_alerts)
    
    end_time = datetime.now(ist)
    elapsed = (end_time - start_time).total_seconds()
    print(f"\n✓ Check complete. {alerts_sent} alerts sent. ({elapsed:.1f}s)")
    print("=" * 50)

if __name__ == "__main__":
    main()