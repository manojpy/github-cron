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
    "SOLUSD": {"limit_15m": 150, "min_required": 74, "limit_5m": 150, "min_required_5m": 74},
    "SUIUSD": {"limit_15m": 150, "min_required": 74, "limit_5m": 150, "min_required_5m": 74}
}

# PPO Parameters
PPO_FAST = 12
PPO_SLOW = 26
PPO_SIGNAL = 9
PPO_USE_SMA = False # If True, uses SMA instead of EMA for calculation

# RMA Parameters
RMA_50_PERIOD = 50
RMA_200_PERIOD = 200

# Smoothed RSI Parameters
SRSI_LENGTH = 14
SRSI_SMOOTHING = 5
SRSI_RMA = True # Use RMA instead of standard EMA for smoothing

# Cirrus Cloud Parameters
CIRRUS_RSI_PERIOD = 14
CIRRUS_RMA_PERIOD = 10
CIRRUS_MULTIPLIER = 3

# State file location
STATE_FILE = 'last_alerts.json'

# ============ UTILITY FUNCTIONS ============

def debug_log(message):
    """Prints a message only if DEBUG_MODE is True."""
    if DEBUG_MODE:
        print(message)

def send_telegram_alert(message):
    """Sends a message to the Telegram chat."""
    if TELEGRAM_BOT_TOKEN == 'xxxx' or TELEGRAM_CHAT_ID == 'cccc':
        debug_log("WARNING: Telegram credentials not configured. Skipping alert.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error sending Telegram alert: {e}")
        if DEBUG_MODE:
            traceback.print_exc()

def get_candles(symbol, interval, limit=200):
    """Fetches historical candles from Delta Exchange."""
    try:
        url = f"{DELTA_API_BASE}/v2/public/charts/candles"
        params = {
            'symbol': symbol,
            'resolution': interval,
            'limit': limit
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Check if the response contains candles
        if not data or 'candles' not in data or not data['candles']:
            debug_log(f"No candles returned for {symbol} on {interval}")
            return None
        
        df = pd.DataFrame(data['candles'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Convert necessary columns to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.sort_index()

    except requests.exceptions.RequestException as e:
        print(f"API Error for {symbol} ({interval}): {e}")
        if DEBUG_MODE:
            traceback.print_exc()
        return None
    except Exception as e:
        print(f"Data processing error for {symbol} ({interval}): {e}")
        if DEBUG_MODE:
            traceback.print_exc()
        return None

def calculate_ppo(df, fast_period, slow_period, signal_period, use_sma=False):
    """Calculates Percentage Price Oscillator (PPO) and its signal line."""
    if use_sma:
        fast_ma = df['close'].rolling(window=fast_period).mean()
        slow_ma = df['close'].rolling(window=slow_period).mean()
    else:
        # Use EWM for EMA-like calculation
        fast_ma = df['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ma = df['close'].ewm(span=slow_period, adjust=False).mean()

    # PPO calculation: (Fast MA - Slow MA) / Slow MA * 100
    ppo = ((fast_ma - slow_ma) / slow_ma) * 100
    
    # PPO signal line (EMA of PPO)
    ppo_signal = ppo.ewm(span=signal_period, adjust=False).mean()
    
    return ppo.dropna(), ppo_signal.dropna()

def calculate_rma(series, period):
    """Calculates the Running Moving Average (RMA)."""
    alpha = 1 / period
    # EWM is used here with 'adjust=False' to approximate the RMA formula:
    # RMA = alpha * current_close + (1 - alpha) * previous_RMA
    return series.ewm(alpha=alpha, adjust=False).mean().dropna()

def calculate_rsi(series, length):
    """Calculates the Relative Strength Index (RSI) using RMA."""
    delta = series.diff()
    up = delta.where(delta > 0, 0)
    down = -delta.where(delta < 0, 0)
    
    # Calculate RMA of gains and losses
    up_rma = calculate_rma(up, length)
    down_rma = calculate_rma(down, length)
    
    # Avoid division by zero
    rs = up_rma / down_rma.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_smooth_rsi(df):
    """Calculates the Smoothed RSI (SRSI) as defined by parameters."""
    rsi = calculate_rsi(df['close'], SRSI_LENGTH)
    
    if SRSI_RMA:
        # Smoothed RSI uses RMA for smoothing
        smooth_rsi = calculate_rma(rsi, SRSI_SMOOTHING)
    else:
        # Or EMA if SRSI_RMA is False
        smooth_rsi = rsi.ewm(span=SRSI_SMOOTHING, adjust=False).mean()
        
    return smooth_rsi.dropna()

def calculate_cirrus_cloud(df):
    """Calculates the Cirrus Cloud indicator components."""
    # 1. Calculate RSI (using RMA approach)
    rsi = calculate_rsi(df['close'], CIRRUS_RSI_PERIOD)
    
    # 2. Calculate the filters (RMA of RSI and RMA of filter 1)
    filtx1 = calculate_rma(rsi, CIRRUS_RMA_PERIOD)
    filtx12 = calculate_rma(filtx1, CIRRUS_RMA_PERIOD)
    
    # 3. Calculate the main cloud components
    # The cloud logic uses the difference between filtx1 and filtx12
    diff = filtx1 - filtx12
    
    # Upw: Difference > Multiplier * (ATR/Range-based value, simplified here)
    # The original Cirrus uses ATR; here, we approximate the logic for color state:
    # Upw (Green/Bullish): filtx1 > filtx12 AND filtx1 > threshold
    upw = (filtx1 > filtx12) & (diff.abs() > diff.abs().mean() * 0.1) # Simplified threshold
    
    # Dnw (Red/Bearish): filtx1 < filtx12 AND filtx1 < threshold
    dnw = (filtx1 < filtx12) & (diff.abs() > diff.abs().mean() * 0.1) # Simplified threshold

    # Shift for consistency with lookback logic
    return upw.shift(1).fillna(False), dnw.shift(1).fillna(False), filtx1, filtx12


def load_state():
    """Loads the last known alert state from a file."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading state file: {e}. Starting clean.")
            return {}
    return {}

def save_state(last_alerts):
    """Saves the current alert state to a file."""
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(last_alerts, f, indent=4)
    except Exception as e:
        print(f"Error saving state file: {e}")

# ============ CORE LOGIC ============

def check_pair(pair_name, pair_info, last_alerts):
    """Check PPO, RMA, Cirrus, and SRSI conditions for a pair and send alerts."""
    
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
        
        # Fetch 15-minute candles for PPO, RMA50, Cirrus, SRSI
        df_15m = get_candles(pair_info['symbol'], "15", limit=limit_15m)
        
        # Fetch 5-minute candles for RMA200 (Condition kept)
        df_5m = get_candles(pair_info['symbol'], "5", limit=limit_5m)
        
        if df_15m is None or len(df_15m) < min_required:
            print(f"Not enough 15m data for {pair_name} ({len(df_15m) if df_15m is not None else 0}/{min_required})")
            return None
    
        
        if df_5m is None or len(df_5m) < min_required_5m:
            print(f"Not enough 5m data for {pair_name} ({len(df_5m) if df_5m is not None else 0}/{min_required_5m})")
            return None
        
        # Calculate indicators on 15min timeframe
        ppo, ppo_signal = calculate_ppo(df_15m, PPO_FAST, PPO_SLOW, PPO_SIGNAL, PPO_USE_SMA)
        rma_50 = calculate_rma(df_15m['close'], RMA_50_PERIOD)
        
        # Calculate RMA200 on 5min timeframe (Condition kept)
        rma_200 = calculate_rma(df_5m['close'], RMA_200_PERIOD)
        
        
        # Calculate Cirrus Cloud on 15min timeframe 
        upw, dnw, filtx1, filtx12 = calculate_cirrus_cloud(df_15m)
      
        # Calculate Smoothed RSI on 15min timeframe
        smooth_rsi = calculate_smooth_rsi(df_15m)
        
        # Get latest values from 15min
        ppo_curr = ppo.iloc[-1]
        ppo_prev = ppo.iloc[-2]
        ppo_signal_curr = ppo_signal.iloc[-1]
        
        
        ppo_signal_prev = ppo_signal.iloc[-2]
        
        # Smoothed RSI values
        smooth_rsi_curr = smooth_rsi.iloc[-1]
        srsi_prev = smooth_rsi.iloc[-2] # Added SRSI previous value
        
        close_curr = df_15m['close'].iloc[-1]
        rma50_curr = rma_50.iloc[-1]
        
        upw_curr = upw.iloc[-1]
        upw_prev = upw.iloc[-2]
        dnw_curr = dnw.iloc[-1]
      
        dnw_prev = dnw.iloc[-2]
        
        # Get latest values from 5min for RMA200
        close_5m_curr = df_5m['close'].iloc[-1]
        rma200_curr = rma_200.iloc[-1]
        
        # --- CANDLE STRUCTURE CHECKS ---
        
        open_curr = df_15m['open'].iloc[-1]
        high_curr = df_15m['high'].iloc[-1]
        low_curr = df_15m['low'].iloc[-1]
        
        # Candle Metrics
        total_range = high_curr - low_curr
        # Upper Wick: High - Max(Open, Close)
        upper_wick = high_curr - max(open_curr, close_curr)
      
        # Lower Wick: Min(Open, Close) - Low
        
        lower_wick = min(open_curr, close_curr) - low_curr
        
        # Basic candle type check
        bullish_candle = close_curr > open_curr
        bearish_candle = close_curr < open_curr

        # 20% Wick conditions
        wick_check_valid = total_range > 0
    
        # Strong Bullish Close: Bullish Candle AND Upper Wick < 20% of Total Range
        strong_bullish_close = False
        # Strong Bearish Close: Bearish Candle AND Lower Wick < 20% of Total Range
        strong_bearish_close = False

        if wick_check_valid:
            strong_bullish_close = bullish_candle and (upper_wick / total_range) < 0.20
     
            strong_bearish_close = bearish_candle and (lower_wick / total_range) < 0.20
        
        
        # Debug logs for new variables
        debug_log(f"\nCandle Metrics (15m):")
        debug_log(f"  O:{open_curr:.2f} H:{high_curr:.2f} L:{low_curr:.2f} C:{close_curr:.2f}")
        debug_log(f"  Range: {total_range:.2f}, UW: {upper_wick:.2f}, LW: {lower_wick:.2f}")
        
        if wick_check_valid:
 
            debug_log(f"  Strong Bullish Close (20% Rule): {strong_bullish_close}")
            
            debug_log(f"  Strong Bearish Close (20% Rule): {strong_bearish_close}")
        else:
            debug_log("  Candle range is zero, skipping wick checks.")
        # --- END CANDLE STRUCTURE CHECKS ---
        
 
        # Debug: Print all indicator values
        debug_log(f"Price: ${close_curr:,.2f}")
        debug_log(f"PPO: {ppo_curr:.4f} (prev: {ppo_prev:.4f})\n")
        debug_log(f"PPO Signal: {ppo_signal_curr:.4f} (prev: {ppo_signal_prev:.4f})")
      
        # 5m PPO Debug Logs (Removed)

        debug_log(f"RMA50 (15m): {rma50_curr:.2f}, Close: {close_curr:.2f}")
        debug_log(f"RMA200 (5m): {rma200_curr:.2f}, Close: {close_5m_curr:.2f}")
        
        # Smoothed RSI Debug Log
        debug_log(f"Smoothed RSI (15m): {smooth_rsi_curr:.2f}") 

     
        # *** DEBUG LINES: Print raw filter values for diagnostics ***
        debug_log(f"Cirrus Filter 1 (filtx1): {filtx1.iloc[-1]:.4f}") 
        debug_log(f"Cirrus Filter 2 (filtx12): {filtx12.iloc[-1]:.4f}") 
        
        
        debug_log(f"Cirrus Cloud - Upw: {upw_curr}, Dnw: {dnw_curr}")
        
        # Detect PPO crossovers (15m)
        ppo_cross_up = (ppo_prev <= ppo_signal_prev) and (ppo_curr > ppo_signal_curr)
        ppo_cross_down = (ppo_prev >= ppo_signal_prev) and (ppo_curr < ppo_signal_curr)
        
        # Detect PPO zero-line crossovers (15m)
        
        ppo_cross_above_zero = (ppo_prev <= 0) and (ppo_curr > 0)
        ppo_cross_below_zero = (ppo_prev >= 0) and (ppo_curr < 0)
        ppo_cross_above_011 = (ppo_prev <= 0.11) and (ppo_curr > 0.11)
      
        ppo_cross_below_minus011 = (ppo_prev >= -0.11) and (ppo_curr < -0.11)
        
        # PPO value conditions (15m)
        ppo_below_020 = ppo_curr < 0.20
        ppo_above_minus020 = ppo_curr > -0.20
        ppo_above_signal = ppo_curr > ppo_signal_curr
        ppo_below_signal = ppo_curr < ppo_signal_curr
        
        # New PPO 15m Thresholds for new SRSI alerts
        ppo_below_030 = ppo_curr < 0.30
        ppo_above_minus030 = ppo_curr > -0.30
        
        # Removed 5m PPO Crossover and Value Conditions
        
        # RMA conditions
        close_above_rma50 = close_curr > rma50_curr
        close_below_rma50 = close_curr < rma50_curr
        close_above_rma200 = close_5m_curr > rma200_curr
        close_below_rma200 = close_5m_curr < rma200_curr
        
        # Smoothed RSI conditions
        srsi_above_50 = smooth_rsi_curr > 50
      
        srsi_below_50 = smooth_rsi_curr < 50
        
        # New SRSI crossover checks
        srsi_cross_up_55 = (srsi_prev <= 55) and (smooth_rsi_curr > 55)
        srsi_cross_down_45 = (srsi_prev >= 45) and (smooth_rsi_curr < 45)
        
        # Debug: Print crossover detections
        debug_log(f"\nCrossover Checks:")
        
        debug_log(f"  PPO 15m cross up: {ppo_cross_up}")
        debug_log(f"  PPO 15m cross down: {ppo_cross_down}")
        # Removed 5m PPO cross debug logs
        
        debug_log(f"\nCondition Checks:")
        
        debug_log(f"  PPO 15m < 0.20: {ppo_below_020}")
        debug_log(f"  PPO 15m > -0.20: {ppo_above_minus020}")
        debug_log(f"  PPO 15m < 0.30: {ppo_below_030}")
        debug_log(f"  PPO 15m > -0.30: {ppo_above_minus030}")
        debug_log(f"  PPO 15m > Signal: {ppo_above_signal}")
        debug_log(f"  Close > RMA50: {close_above_rma50}")
        debug_log(f"  Close > RMA200: {close_above_rma200}")
        debug_log(f"  Upw (Cirrus): {upw_curr}") # Now means GREEN
        
        debug_log(f"  Dnw (Cirrus): {dnw_curr}") # Now means RED
        debug_log(f"  SRSI > 50: {srsi_above_50}") # New debug log
        debug_log(f"  SRSI < 50: {srsi_below_50}") # New debug log
        debug_log(f"  SRSI Cross Up 55: {srsi_cross_up_55}")
        debug_log(f"  SRSI Cross Down 45: {srsi_cross_down_45}")
        
 
        current_state = None
        
        # Get IST time in correct format
        ist = pytz.timezone('Asia/Kolkata')
        current_dt = datetime.now(ist)
        formatted_time = current_dt.strftime('%d-%m-%Y @ %H:%M IST')
        price = df_15m['close'].iloc[-1]
        
 
        # 1. ORIGINAL BUY: PPO crosses up AND PPO < 0.20 AND ... (RMA200 Kept)
        if (ppo_cross_up and 
            ppo_below_020 and 
            close_above_rma50 and 
            close_above_rma200 and 
            upw_curr and (not dnw_curr) and 
            strong_bullish_close and
           
            srsi_above_50): 
            current_state = "buy"
            debug_log(f"\n🟢 BUY SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "buy":
            
                message = f"🟢 {pair_name} - BUY\nPPO - SIGNAL Crossover (PPO: {ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
             
                send_telegram_alert(message)
            else:
                debug_log(f"BUY already alerted for {pair_name}, skipping duplicate")
        
        # 2. ORIGINAL SELL: PPO crosses down AND PPO > -0.20 AND ... (RMA200 Kept)
        
        elif (ppo_cross_down and 
             
              ppo_above_minus020 and 
              close_below_rma50 and 
              close_below_rma200 and 
              dnw_curr and (not upw_curr) and 
              strong_bearish_close and
              srsi_below_50): 
            current_state = "sell"
      
            debug_log(f"\n🔴 SELL SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "sell":
                message = f"🔴 {pair_name} - SELL\nPPO - SIGNAL Crossunder (PPO: {ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
            
                send_telegram_alert(message)
            else:
       
                debug_log(f"SELL already alerted for {pair_name}, skipping duplicate")

        # 3. === NEW BUY ALERT (SRSI Cross Up 55) ===
        elif (srsi_cross_up_55 and 
              ppo_above_signal and # 15m PPO should be above signal
              ppo_below_030 and # PPO should be less than 0.30
              close_above_rma50 and 
              close_above_rma200 and 
              upw_curr and (not dnw_curr) and 
              strong_bullish_close): 
            current_state = "buy_srsi_cross"
            debug_log(f"\n🟢 BUY (SRSI Cross 55) SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "buy_srsi_cross":
                message = f"🟢 {pair_name} - BUY (SRSI)\nSRSI Cross Up 55 ({smooth_rsi_curr:.2f})\n15m PPO: {ppo_curr:.2f}\nPrice: ${price:,.2f}\n{formatted_time}"
                send_telegram_alert(message)
            else:
                debug_log(f"BUY (SRSI Cross 55) already alerted for {pair_name}, skipping duplicate")
            
        # 4. === NEW SELL ALERT (SRSI Cross Down 45) ===
        elif (srsi_cross_down_45 and 
              ppo_below_signal and # 15m PPO should be below signal
              ppo_above_minus030 and # PPO should be greater than -0.30
              close_below_rma50 and 
              close_below_rma200 and 
              dnw_curr and (not upw_curr) and 
              strong_bearish_close): 
            current_state = "sell_srsi_cross"
            debug_log(f"\n🔴 SELL (SRSI Cross 45) SIGNAL DETECTED for {pair_name}!")
            
            if last_alerts.get(pair_name) != "sell_srsi_cross":
                message = f"🔴 {pair_name} - SELL (SRSI)\nSRSI Cross Down 45 ({smooth_rsi_curr:.2f})\n15m PPO: {ppo_curr:.2f}\nPrice: ${price:,.2f}\n{formatted_time}"
                send_telegram_alert(message)
            else:
                debug_log(f"SELL (SRSI Cross 45) already alerted for {pair_name}, skipping duplicate")
        
        # 5. LONG (0): PPO > Signal AND PPO crosses above 0 AND ... (RMA200 Kept)
        elif (ppo_cross_above_zero and 
              ppo_above_signal and 
              close_above_rma50 and 
              close_above_rma200 and 
              upw_curr and (not dnw_curr) and 
              strong_bullish_close and
              srsi_above_50): 
            current_state = "long_zero"
            debug_log(f"\n🟢 LONG (0) SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "long_zero":
            
                message = f"🟢 {pair_name} - LONG\nPPO crossing above 0 ({ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
                send_telegram_alert(message)
            else:
                debug_log(f"LONG (0) already alerted for {pair_name}, skipping duplicate")
       
        # 6. LONG (0.11): PPO > Signal AND PPO crosses above 0.11 AND ... (RMA200 Kept)
        
        elif (ppo_cross_above_011 and 
              ppo_above_signal and 
              close_above_rma50 and 
              close_above_rma200 and 
              upw_curr and (not dnw_curr) and 
              strong_bullish_close and
              srsi_above_50): 
            current_state = "long_011"
            debug_log(f"\n🟢 LONG (0.11) SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "long_011":
            
                message = f"🟢 {pair_name} - LONG\nPPO crossing above 0.11 ({ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
            
                send_telegram_alert(message)
            else:
                debug_log(f"LONG (0.11) already alerted for {pair_name}, skipping duplicate")
        
        # 7. SHORT (0): PPO < Signal AND PPO crosses below 0 AND ... (RMA200 Kept)
        elif (ppo_cross_below_zero and 
              ppo_below_signal and 
              close_below_rma50 and 
              close_below_rma200 and 
              dnw_curr and (not upw_curr) and 
             
              strong_bearish_close and
              srsi_below_50): 
            
            current_state = "short_zero"
            debug_log(f"\n🔴 SHORT (0) SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "short_zero":
                message = f"🔴 {pair_name} - SHORT\nPPO crossing below 0 ({ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
                send_telegram_alert(message)
            else:
            
                debug_log(f"SHORT (0) already alerted for {pair_name}, skipping duplicate")
        
        # 8. SHORT (-0.11): PPO < Signal AND PPO crosses below -0.11 AND ... (RMA200 Kept)
   
        elif (ppo_cross_below_minus011 and 
              ppo_below_signal and 
              close_below_rma50 and 
              close_below_rma200 and 
              dnw_curr and (not upw_curr) and 
              strong_bearish_close and
          
              srsi_below_50): 
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
        print(f"Unhandled error in check_pair for {pair_name}: {e}")
        if DEBUG_MODE:
            traceback.print_exc()
        return None

def main():
    """Main function to orchestrate the check."""
    ist = pytz.timezone('Asia/Kolkata')
    
    # 1. Load instrument list (only once)
    try:
        url = f"{DELTA_API_BASE}/v2/public/tickers"
        response = requests.get(url)
        response.raise_for_status()
        tickers = response.json()
        
        # Map pair names to instrument symbols
        for ticker in tickers:
            symbol = ticker.get('symbol')
            # Only consider symbols in our PAIRS list
            if symbol in PAIRS:
                # Add all necessary info to the PAIRS dictionary
                PAIRS[symbol] = {
                    'symbol': symbol,
                    'last_price': ticker.get('last_price'),
                    'underlying_asset': ticker.get('underlying_asset'),
                    'type': ticker.get('type')
                }
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching instrument data: {e}. Exiting.")
        if DEBUG_MODE:
            traceback.print_exc()
        return
    except Exception as e:
        print(f"Error processing instrument data: {e}. Exiting.")
        if DEBUG_MODE:
            traceback.print_exc()
        return


    # 2. Send test message on first run if enabled
    if SEND_TEST_MESSAGE and not os.path.exists(STATE_FILE):
        send_telegram_alert("🤖 **Delta Bot Initializing**\nSuccessfully started monitoring pairs.")
    
    # 3. Load previous state
    last_alerts = load_state()

    # 4. Check all pairs using threads
    MAX_WORKERS = 5
    alerts_sent = 0
    start_time = datetime.now(ist)

    # Use ThreadPoolExecutor for concurrent checks
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_pair = {}
        for pair_name, pair_info in PAIRS.items():
            if pair_info is None:
                debug_log(f"Skipping {pair_name}: Instrument data not found.")
                continue
            
            # IMPORTANT: Pass a copy of last_alerts for thread safety 
            # (although check_pair only reads from it for the initial state).
            future = executor.submit(check_pair, pair_name, pair_info, last_alerts.copy())
            
            future_to_pair[future] = pair_name

        for future in as_completed(future_to_pair):
            pair_name = future_to_pair[future]
            try:
                new_state = future.result() 
                # Only update the state if a new signal was detected
                if new_state: 
                    # new_state is the signal 

                    # type (e.g., "buy", "short_011", "buy_trend")
                    last_alerts[pair_name] = new_state
                    # The alert is sent inside check_pair if the state changes, 
                    # so 
                    # we just track that an update was processed here.
                    alerts_sent += 1 
            except Exception as e:
                print(f"Error processing {pair_name} in thread: {e}")
                if DEBUG_MODE:
                    
                    traceback.print_exc()
                continue
            
    # Save state for next run
    save_state(last_alerts)
    
    end_time = datetime.now(ist)
    elapsed = (end_time - start_time).total_seconds()
    print(f"✓ Check complete. {alerts_sent} state updates processed. ({elapsed:.1f}s)")
    print("=" * 50)

if __name__ == "__main__":
    main()
