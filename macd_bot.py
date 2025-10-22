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

# Delta Exchange API
DELTA_API_BASE = "https://api.delta.exchange"

# Trading pairs to monitor
PAIRS = {
    "BTCUSD": None, "ETHUSD": None, "SOLUSD": None, "AVAXUSD": None,
    "BCHUSD": None, "XRPUSD": None, "BNBUSD": None, "LTCUSD": None,
    "DOTUSD": None, "ADAUSD": None, "SUIUSD": None, "AAVEUSD": None
}

# Special data requirements for pairs with limited history
SPECIAL_PAIRS = {
    "SOLUSD": {"limit_15m": 100, "min_required": 90}
}

# Indicator settings
PPO_FAST, PPO_SLOW, PPO_SIGNAL = 7, 16, 5
PPO_USE_SMA = False
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 100, 200, 50
EMA_100_PERIOD = 100
RMA_200_PERIOD = 200
STATE_FILE = 'alert_state.json'

# ============ FUNCTIONS ============

def load_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading state: {e}")
    return {}

def save_state(state):
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        print(f"Error saving state: {e}")

def send_telegram_alert(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        response = requests.post(url, data=data, timeout=10)
        response_data = response.json()
        if response_data.get('ok'):
            print(f"✓ Alert sent successfully for message starting with: {message.splitlines()[0]}")
            return True
        else:
            print(f"Telegram error: {response_data}")
            return False
    except Exception as e:
        print(f"Error sending Telegram message: {e}")
        return False

def get_product_ids():
    """Fetch all product IDs from Delta Exchange and log successes/failures."""
    print("Fetching product IDs...")
    try:
        response = requests.get(f"{DELTA_API_BASE}/v2/products", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('success'):
            products = data['result']
            product_map = {p['symbol']: p for p in products if p.get('contract_type') == 'perpetual_futures'}
            
            for pair_name in PAIRS.keys():
                # Attempt to find a matching symbol in a few common formats
                potential_symbols = [
                    pair_name, 
                    pair_name.replace('USD', '_USDT'), 
                    pair_name.replace('USD', 'USDT')
                ]
                found = False
                for symbol in potential_symbols:
                    if symbol in product_map:
                        product = product_map[symbol]
                        PAIRS[pair_name] = {
                            'id': product['id'],
                            'symbol': product['symbol'], # Store the *actual* symbol used by the API
                            'contract_type': product['contract_type']
                        }
                        found = True
                        break # Found a match, move to the next pair_name
                
            # --- NEW: Check which pairs were NOT found ---
            for pair_name, info in PAIRS.items():
                if info is None:
                    print(f"⚠️ Warning: Could not find product ID for {pair_name}")
            return True
        else:
            print(f"API Error when fetching products: {data}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching products: {e}")
        return False

def get_candles(api_symbol, resolution="15", limit=150):
    """Fetch OHLCV candles from Delta Exchange with better error logging."""
    try:
        to_time = int(time.time())
        from_time = to_time - (limit * int(resolution) * 60)
        url = f"{DELTA_API_BASE}/v2/chart/history"
        params = {'resolution': resolution, 'symbol': api_symbol, 'from': from_time, 'to': to_time}
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if data.get('success'):
            result = data['result']
            df = pd.DataFrame(result)
            df.rename(columns={'t':'timestamp', 'o':'open', 'h':'high', 'l':'low', 'c':'close', 'v':'volume'}, inplace=True)
            return df
        else:
            # --- NEW: Log the specific API error ---
            print(f"API Error fetching candles for {api_symbol}: {data.get('message', 'No message')}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Exception fetching candles for {api_symbol}: {e}")
        return None

def calculate_ema(data, period): return data.ewm(span=period, adjust=False).mean()
def calculate_rma(data, period): return data.ewm(alpha=1/period, adjust=False).mean()

def calculate_ppo(df, fast=PPO_FAST, slow=PPO_SLOW, signal=PPO_SIGNAL, use_sma=PPO_USE_SMA):
    close = df['close']
    ma_func = pd.Series.rolling if use_sma else calculate_ema
    fast_ma = ma_func(close, fast)
    slow_ma = ma_func(close, slow)
    ppo = (fast_ma - slow_ma) / slow_ma * 100
    ppo_signal = ma_func(ppo, signal)
    return ppo, ppo_signal

def calculate_macd(df, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    close = df['close']
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    return macd_line, signal_line

def check_pair(pair_name, pair_info, last_alerts):
    """Check crossover conditions for a pair with better logging."""
    if pair_info is None:
        # This will now be preceded by the "Warning: Could not find..." message from get_product_ids
        return None
    
    try:
        api_symbol = pair_info['symbol'] # Use the actual symbol from the API
        
        # Determine data requirements
        reqs = SPECIAL_PAIRS.get(pair_name, {"limit_15m": 210, "min_required": 200})
        limit_15m, min_required_15m = reqs["limit_15m"], reqs["min_required"]
        limit_5m, min_required_5m = 210, 200
        
        # Fetch candle data
        df_15m = get_candles(api_symbol, "15", limit=limit_15m)
        df_5m = get_candles(api_symbol, "5", limit=limit_5m)
        
        # --- NEW: Improved data validation and logging ---
        if df_15m is None or len(df_15m) < min_required_15m:
            print(f"Skipping {pair_name}: Not enough 15m data ({len(df_15m) if df_15m is not None else 'API error'}/{min_required_15m})")
            return None
        if df_5m is None or len(df_5m) < min_required_5m:
            print(f"Skipping {pair_name}: Not enough 5m data ({len(df_5m) if df_5m is not None else 'API error'}/{min_required_5m})")
            return None
        
        # Calculate indicators
        ppo, ppo_signal = calculate_ppo(df_15m)
        macd, macd_signal = calculate_macd(df_15m)
        ema_100 = calculate_ema(df_15m['close'], EMA_100_PERIOD)
        rma_200 = calculate_rma(df_5m['close'], RMA_200_PERIOD)
        
        # Get latest values
        ppo_curr, ppo_prev = ppo.iloc[-1], ppo.iloc[-2]
        ppo_signal_curr, ppo_signal_prev = ppo_signal.iloc[-1], ppo_signal.iloc[-2]
        macd_curr, macd_signal_curr = macd.iloc[-1], macd_signal.iloc[-1]
        close_curr, ema100_curr = df_15m['close'].iloc[-1], ema_100.iloc[-1]
        close_5m_curr, rma200_curr = df_5m['close'].iloc[-1], rma_200.iloc[-1]

        # Define conditions
        ppo_cross_up = (ppo_prev <= ppo_signal_prev) and (ppo_curr > ppo_signal_curr)
        ppo_cross_down = (ppo_prev >= ppo_signal_prev) and (ppo_curr < ppo_signal_curr)
        # ... (rest of the conditions are the same)
        
        # (The entire alert logic block remains unchanged here)
        ist = pytz.timezone('Asia/Kolkata')
        formatted_time = datetime.now(ist).strftime('%d-%m-%Y @ %H:%M IST')
        price = df_15m['close'].iloc[-1]
        
        alert_triggered = False
        current_state = None
        message = ""

        # BUY
        if ppo_cross_up and ppo_curr < 0.20 and macd_curr > macd_signal_curr and close_curr > ema100_curr and close_5m_curr > rma200_curr:
            current_state, message = "buy", f"🟢 {pair_name} - BUY\nPPO - SIGNAL Crossover (PPO: {ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
        # SELL
        elif ppo_cross_down and ppo_curr > -0.20 and macd_curr < macd_signal_curr and close_curr < ema100_curr and close_5m_curr < rma200_curr:
            current_state, message = "sell", f"🔴 {pair_name} - SELL\nPPO - SIGNAL Crossunder (PPO: {ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
        
        if current_state and last_alerts.get(pair_name) != current_state:
            send_telegram_alert(message)
            return current_state

    except Exception as e:
        print(f"Unhandled error in check_pair for {pair_name}: {e}")
    return None

def main():
    """Main function - runs once per GitHub Actions execution"""
    start_time = time.time()
    ist = pytz.timezone('Asia/Kolkata')
    print("=" * 50)
    print(f"PPO/MACD Alert Bot - {datetime.now(ist).strftime('%d-%m-%Y @ %H:%M IST')}")
    print("=" * 50)
    
    last_alerts = load_state()
    
    if not get_product_ids():
        print("Failed to fetch products. Exiting.")
        return
    
    found_count = sum(1 for v in PAIRS.values() if v is not None)
    print(f"✓ Found IDs for {found_count}/{len(PAIRS)} pairs.")
    if found_count == 0:
        print("No valid pairs found. Exiting.")
        return
        
    updated_alerts = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_pair = {executor.submit(check_pair, name, info, last_alerts): name for name, info in PAIRS.items() if info}
        
        for future in as_completed(future_to_pair):
            pair_name = future_to_pair[future]
            try:
                new_state = future.result()
                if new_state:
                    updated_alerts[pair_name] = new_state
            except Exception as e:
                print(f"Error processing {pair_name} in thread: {e}")
    
    if updated_alerts:
        last_alerts.update(updated_alerts)
        save_state(last_alerts)
    
    elapsed = time.time() - start_time
    print(f"✓ Check complete. {len(updated_alerts)} state changes recorded. ({elapsed:.1f}s)")
    print("=" * 50)

if __name__ == "__main__":
    main()
