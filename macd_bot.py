import requests
import pandas as pd
import numpy as np
import time
import os
import json
from datetime import datetime
import pytz

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
SMI_K_PERIOD = 30      # %K period
SMI_K_SMOOTHING = 5    # %K smoothing
SMI_D_SMOOTHING = 5    # %D smoothing
EMA_40_PERIOD = 40     # EMA40 on 15min
EMA_100_PERIOD = 100   # EMA100 on 15min
EMA_200_PERIOD = 200   # EMA200 on 10min

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

def calculate_smi(df, k_period=30, d_period=5, ema_period=5):
    """Calculate Stochastic Momentum Index (SMI) - matches TradingView Pine Script"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate highest high and lowest low over k_period
    highest_high = high.rolling(window=k_period).max()
    lowest_low = low.rolling(window=k_period).min()
    
    # Calculate range
    highest_lowest_range = highest_high - lowest_low
    
    # Calculate relative range
    relative_range = close - (highest_high + lowest_low) / 2
    
    # Double EMA function (EMA of EMA)
    def ema_ema(series, period):
        ema1 = series.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        return ema2
    
    # Calculate SMI
    smi = 200 * (ema_ema(relative_range, d_period) / ema_ema(highest_lowest_range, d_period))
    
    # Calculate signal line (EMA of SMI)
    smi_signal = smi.ewm(span=ema_period, adjust=False).mean()
    
    return smi, smi_signal

def check_pair(pair_name, pair_info, last_alerts):
    """Check MACD crossover conditions for a pair"""
    try:
        if pair_info is None:
            return None
        
        print(f"\n--- Checking {pair_name} ---")
        
        # Fetch 15-minute candles for MACD and EMA100
        df_15m = get_candles(pair_info['symbol'], "15", limit=150)
        
        # Fetch 10-minute candles for EMA200
        df_10m = get_candles(pair_info['symbol'], "10", limit=200)
        
        if df_15m is None or len(df_15m) < 100:
            print(f"Insufficient 15min data for {pair_name}")
            return None
            
        if df_10m is None or len(df_10m) < 200:
            print(f"Insufficient 10min data for {pair_name}")
            return None
        
        # Calculate indicators on 15min timeframe
        smi, smi_signal = calculate_smi(df_15m, SMI_K_PERIOD, SMI_K_SMOOTHING, SMI_D_SMOOTHING)
        ema_40 = calculate_ema(df_15m['close'], EMA_40_PERIOD)
        ema_100 = calculate_ema(df_15m['close'], EMA_100_PERIOD)
        
        # Calculate EMA200 on 10min timeframe
        ema_200 = calculate_ema(df_10m['close'], EMA_200_PERIOD)
        
        # Get latest values from 15min
        smi_curr = smi.iloc[-1]
        smi_prev = smi.iloc[-2]
        smi_signal_curr = smi_signal.iloc[-1]
        smi_signal_prev = smi_signal.iloc[-2]
        close_curr = df_15m['close'].iloc[-1]
        ema40_curr = ema_40.iloc[-1]
        ema100_curr = ema_100.iloc[-1]
        
        # Get latest values from 10min
        close_10m_curr = df_10m['close'].iloc[-1]
        ema200_curr = ema_200.iloc[-1]
        
        bullish_cross = (smi_prev <= smi_signal_prev) and (smi_curr > smi_signal_curr)
        bearish_cross = (smi_prev >= smi_signal_prev) and (smi_curr < smi_signal_curr)
        
        # New conditions: EMA40 vs EMA100 (15min) AND Close vs EMA200 (10min)
        ema40_above_ema100 = ema40_curr > ema100_curr
        ema40_below_ema100 = ema40_curr < ema100_curr
        close_above_ema200 = close_10m_curr > ema200_curr
        close_below_ema200 = close_10m_curr < ema200_curr
        
        # Debug logging
        print(f"Price: ${close_curr:,.4f}")
        print(f"SMI: {smi_curr:.2f}, Signal: {smi_signal_curr:.2f}")
        print(f"SMI prev: {smi_prev:.2f}, Signal prev: {smi_signal_prev:.2f}")
        print(f"Bullish cross: {bullish_cross}, Bearish cross: {bearish_cross}")
        print(f"EMA40: {ema40_curr:.2f}, EMA100: {ema100_curr:.2f}")
        print(f"EMA40 > EMA100: {ema40_above_ema100}, EMA40 < EMA100: {ema40_below_ema100}")
        print(f"Close(10m): {close_10m_curr:.2f}, EMA200(10m): {ema200_curr:.2f}")
        print(f"Close > EMA200: {close_above_ema200}, Close < EMA200: {close_below_ema200}")
        print(f"Last alert state: {last_alerts.get(pair_name, 'None')}")
        
        current_state = None
        
        # Bullish: SMI cross up AND EMA40 > EMA100(15m) AND close > EMA200(10m)
        if bullish_cross and ema40_above_ema100 and close_above_ema200:
            current_state = "bullish"
            if last_alerts.get(pair_name) != "bullish":
                price = df_15m['close'].iloc[-1]
                # Get IST time
                ist = pytz.timezone('Asia/Kolkata')
                current_time = datetime.now(ist).strftime('%d-%m-%Y %H:%M:%S IST')
                
                message = (
                    f"🟢 <b>{pair_name} - Bullish SMI Crossover</b>\n\n"
                    f"SMI crossed above Signal line\n"
                    f"Price: ${price:,.4f}\n"
                    f"Time: {current_time}"
                )
                send_telegram_alert(message)
                print(f"✓ Bullish alert sent for {pair_name}")
                
        # Bearish: SMI cross down AND EMA40 < EMA100(15m) AND close < EMA200(10m)
        elif bearish_cross and ema40_below_ema100 and close_below_ema200:
            current_state = "bearish"
            if last_alerts.get(pair_name) != "bearish":
                price = df_15m['close'].iloc[-1]
                # Get IST time
                ist = pytz.timezone('Asia/Kolkata')
                current_time = datetime.now(ist).strftime('%d-%m-%Y %H:%M:%S IST')
                
                message = (
                    f"🔴 <b>{pair_name} - Bearish SMI Crossover</b>\n\n"
                    f"SMI crossed below Signal line\n"
                    f"Price: ${price:,.4f}\n"
                    f"Time: {current_time}"
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
    print(f"SMI Alert Bot - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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