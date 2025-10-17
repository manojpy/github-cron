import requests
import pandas as pd
import numpy as np
import time
import os
import json
from datetime import datetime

# ============ CONFIGURATION ============
# Telegram settings - reads from environment variables (GitHub Secrets)
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '8462496498:AAGK9D5IWHmrYVI8zk6TLoKikNzHZSzSJns')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '203813932')

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

# Indicator settings
MACD_FAST = 20
MACD_SLOW = 43
MACD_SIGNAL = 15
EMA_SHORT = 40
EMA_LONG = 100

# File to store last alert state
STATE_FILE = 'alert_state.json'

# ============ FUNCTIONS ============

def load_state():
    """Load previous alert state from file"""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
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
            "parse_mode": "HTML"
        }
        
        response = requests.post(url, data=data, timeout=10)
        response_data = response.json()
        
        if response_data.get('ok'):
            print(f"✓ Telegram alert sent successfully")
            return True
        else:
            print(f"Telegram error: {response_data}")
            return False
        
    except Exception as e:
        print(f"Error sending Telegram message: {e}")
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
                            print(f"✓ Found {pair_name}: {product['symbol']}")
            
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
        from_time = to_time - (limit * 15 * 60)
        
        url = f"{DELTA_API_BASE}/v2/chart/history"
        params = {
            'resolution': resolution,
            'symbol': product_id,
            'from': from_time,
            'to': to_time
        }
        
        response = requests.get(url, params=params, timeout=10)
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
            print(f"Candle API Error: {data}")
            return None
            
    except Exception as e:
        print(f"Error fetching candles: {e}")
        return None

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_macd(df):
    """Calculate MACD and Signal line"""
    ema_fast = calculate_ema(df['close'], MACD_FAST)
    ema_slow = calculate_ema(df['close'], MACD_SLOW)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, MACD_SIGNAL)
    return macd_line, signal_line

def check_pair(pair_name, pair_info, last_alerts):
    """Check MACD crossover conditions for a pair"""
    try:
        if pair_info is None:
            return None
        
        df = get_candles(pair_info['symbol'], "15", limit=150)
        
        if df is None or len(df) < 100:
            print(f"Insufficient data for {pair_name}")
            return None
        
        macd_line, signal_line = calculate_macd(df)
        ema_40 = calculate_ema(df['close'], EMA_SHORT)
        ema_100 = calculate_ema(df['close'], EMA_LONG)
        
        macd_curr = macd_line.iloc[-1]
        macd_prev = macd_line.iloc[-2]
        signal_curr = signal_line.iloc[-1]
        signal_prev = signal_line.iloc[-2]
        ema40_curr = ema_40.iloc[-1]
        ema100_curr = ema_100.iloc[-1]
        
        bullish_cross = (macd_prev <= signal_prev) and (macd_curr > signal_curr)
        bearish_cross = (macd_prev >= signal_prev) and (macd_curr < signal_curr)
        
        ema_bullish = ema40_curr > ema100_curr
        ema_bearish = ema40_curr < ema100_curr
        
        current_state = None
        
        if bullish_cross and ema_bullish:
            current_state = "bullish"
            if last_alerts.get(pair_name) != "bullish":
                price = df['close'].iloc[-1]
                message = (
                    f"🟢 <b>{pair_name} - Bullish MACD Crossover</b>\n\n"
                    f"MACD crossed above Signal line\n"
                    f"EMA40 &gt; EMA100 ✓\n"
                    f"Price: ${price:,.4f}\n"
                    f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                send_telegram_alert(message)
                print(f"✓ Bullish alert sent for {pair_name}")
                
        elif bearish_cross and ema_bearish:
            current_state = "bearish"
            if last_alerts.get(pair_name) != "bearish":
                price = df['close'].iloc[-1]
                message = (
                    f"🔴 <b>{pair_name} - Bearish MACD Crossover</b>\n\n"
                    f"MACD crossed below Signal line\n"
                    f"EMA40 &lt; EMA100 ✓\n"
                    f"Price: ${price:,.4f}\n"
                    f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                send_telegram_alert(message)
                print(f"✓ Bearish alert sent for {pair_name}")
        
        return current_state
        
    except Exception as e:
        print(f"Error checking {pair_name}: {e}")
        return None

def main():
    """Main function - runs once per GitHub Actions execution"""
    print("=" * 50)
    print(f"MACD Alert Bot - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Load previous state
    last_alerts = load_state()
    print(f"Loaded previous state: {len(last_alerts)} pairs tracked")
    
    # Fetch product IDs
    print("\nFetching product information...")
    if not get_product_ids():
        print("Failed to fetch products. Exiting.")
        return
    
    found_count = sum(1 for v in PAIRS.values() if v is not None)
    print(f"✓ Found {found_count}/{len(PAIRS)} pairs\n")
    
    if found_count == 0:
        print("No valid pairs found. Exiting.")
        return
    
    # Check all pairs
    print("Checking for MACD crossovers...")
    alerts_sent = 0
    
    for pair_name, pair_info in PAIRS.items():
        if pair_info is not None:
            new_state = check_pair(pair_name, pair_info, last_alerts)
            if new_state:
                last_alerts[pair_name] = new_state
                alerts_sent += 1
            time.sleep(1)  # Small delay between API calls
    
    # Save state for next run
    save_state(last_alerts)
    
    print(f"\n✓ Check complete. {alerts_sent} crossover alerts sent.")
    print("=" * 50)

if __name__ == "__main__":
    main()