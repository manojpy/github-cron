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
PPO_USE_SMA = False  # False = use EMA

# RMA settings
RMA_50_PERIOD = 50   # RMA50 on 15min
RMA_200_PERIOD = 200 # RMA200 on 5min 

# Cirrus Cloud settings
CIRRUS_CLOUD_ENABLED = True
X1 = 22
X2 = 9
X3 = 15
X4 = 5

# Smoothed RSI (SRSI) settings
SRSI_RSI_LEN = 21
SRSI_KALMAN_LEN = 5
SRSI_EMA_LEN = 5 

# Magical Momentum Indicator settings
MOMENTUM_RESPONSIVENESS = 0.9
MOMENTUM_PERIOD = 144
MOMENTUM_STDEV_PERIOD = 50

# File to store last alert state
STATE_FILE = 'alert_state.json'

# ============ UTILITY FUNCTIONS ============

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
            # traceback.print_exc() # Removed import for brevity
            pass
        return False

def send_test_message():
    """Send a test message to verify Telegram connectivity"""
    ist = pytz.timezone('Asia/Kolkata')
    current_dt = datetime.now(ist)
    formatted_time = current_dt.strftime('%d-%m-%Y @ %H:%M IST')
    
    test_msg = f"🔔 Bot Started\nTest message from PPO Bot\nTime: {formatted_time}\nDebug Mode: {'ON' if DEBUG_MODE else 'OFF'}"
    
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
            # traceback.print_exc() # Removed import for brevity
            pass
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
    """
    close = df['close'].copy()
    
    # Calculate smoothed ranges
    smrngx1x = smoothrng(close, X1, X2)
    smrngx1x2 = smoothrng(close, X3, X4)
    
    # Apply range filter
    filtx1 = rngfilt(close, smrngx1x)
    filtx12 = rngfilt(close, smrngx1x2)
    
    # Upw (Green) is True when filter 1 line is BELOW filter 2 line.
    upw = filtx1 < filtx12 
    # Dnw (Red) is True when filter 1 
    dnw = filtx1 > filtx12 
    
    # Return filter lines for better debugging
    return upw, dnw, filtx1, filtx12 

# Kalman Filter implementation

def kalman_filter(src, length, R = 0.01, Q = 0.1):
    """Implements the kalman_filter function from Pine Script"""
    result_list = []
    
    # Initialize 'var' variables outside the loop as per Pine Script logic
    estimate = np.nan
    error_est = 1.0
    
    # Pre-calculate constants
    error_meas = R * length
    Q_div_length = Q / length
    
    for i in range(len(src)):
        current_src = src.iloc[i]
        
        if np.isnan(estimate):
            # We initialize estimate to the first available source value
            if i > 0:
                estimate = src.iloc[i-1]
            else:
                result_list.append(np.nan)
                continue
                
        prediction = estimate
        
        # kalman_gain := error_est / (error_est 
        kalman_gain = error_est / (error_est + error_meas)
        
        # estimate := prediction + kalman_gain * (src - prediction)
        estimate = prediction + kalman_gain * (current_src - prediction)
        
        # error_est := (1 - kalman_gain) * error_est + Q / (length)
        error_est = (1 - kalman_gain) * error_est + Q_div_length
 
        
        result_list.append(estimate)

    # Prepend NaNs to match original Series length and index alignment
    nans_to_add = len(src) - len(result_list)
    padded_results = [np.nan] * nans_to_add + result_list
    
    return pd.Series(padded_results, index=src.index)

# Function to calculate Smoothed RSI
def calculate_smooth_rsi(df, rsi_len=SRSI_RSI_LEN, kalman_len=SRSI_KALMAN_LEN):
    """Calculate Smoothed RSI using Kalman Filter"""
    # 1. Calculate RSI
    close = df['close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Use RMA (Smoothed Moving Average) for Wilder's smoothing
    avg_gain = calculate_rma(gain, rsi_len)
    avg_loss = calculate_rma(loss, rsi_len)
    
    # Handle division by zero
    rs = avg_gain.divide(avg_loss.replace(0, np.nan)) 
    rsi_value = 100 - (100 / (1 + rs))
    
    # 2. Smooth RSI using Kalman Filter
    smooth_rsi = kalman_filter(rsi_value, kalman_len)
    
    return smooth_rsi

# === CORRECTED Magical Momentum Indicator ===
def calculate_magical_momentum(df, period=MOMENTUM_PERIOD, responsiveness=MOMENTUM_RESPONSIVENESS, stdev_period=MOMENTUM_STDEV_PERIOD):
    """
    Implements the Magical Momentum Indicator logic from Pine Script, corrected
    for initialization and recursive variable handling.
    """
    close = df['close'].copy()
    n = len(close)
    if n == 0:
        return pd.Series([], dtype=float)

    # --- Pass 1: Worm and Raw Momentum Initialization ---
    sd_series = close.rolling(window=stdev_period).std() * responsiveness
    ma_series = calculate_sma(close, period)

    worm_list = []
    raw_momentum_list = []
    
    # Initialize 'var worm' to the first 'source' (close) value
    worm_prev = close.iloc[0] 

    # --- Pass 1: Recursive Worm Calculation ---
    for i in range(n):
        current_close = close.iloc[i]
        current_ma = ma_series.iloc[i]
        current_sd = sd_series.iloc[i]
        
        if i == 0:
            new_worm = current_close
            worm_list.append(new_worm)
        else:
            # Pine: diff = source - worm[1]
            diff = current_close - worm_prev
            
            # Pine: delta = math.abs(diff) > sd ? math.sign(diff) * sd : diff
            if np.isnan(current_sd) or current_sd == 0.0:
                delta = diff
            elif np.abs(diff) > current_sd:
                delta = np.sign(diff) * current_sd
            else:
                delta = diff
                
            # Pine: worm := worm[1] + delta
            new_worm = worm_prev + delta
            worm_list.append(new_worm)
        
        worm_prev = new_worm
        
        # Pine: raw_momentum = (worm - ma) / worm
        if np.isnan(new_worm) or new_worm == 0.0 or np.isnan(current_ma):
            raw_momentum = np.nan
        else:
            raw_momentum = (new_worm - current_ma) / new_worm
        raw_momentum_list.append(raw_momentum)

    raw_momentum_series = pd.Series(raw_momentum_list, index=close.index)
    
    # --- Post-Pass 1 calculations ---
    current_med = raw_momentum_series
    min_med = current_med.rolling(window=period).min()
    max_med = current_med.rolling(window=period).max()
    
    # Pine: temp = (current_med - min_med) / (max_med - min_med)
    diff_med = max_med - min_med
    # Replace division by zero with NaN, then fill NaNs (usually where min=max) with 0.5
    temp_series = (current_med - min_med).divide(diff_med.replace(0, np.nan))
    temp_series = temp_series.fillna(0.5).replace([np.inf, -np.inf], 0.5)

    # --- Pass 2: Recursive Value and Momentum ---
    value_list = []
    momentum_list = []
    
    # Pine: var value = 0.5 * 2 = 1.0 (Initialization for the first bar)
    value_prev = 0.0 # Emulates nz(value[1]) for the first bar
    momentum_prev = 0.0 # Emulates nz(momentum[1]) for the first bar

    for i in range(n):
        current_temp = temp_series.iloc[i]
        
        if np.isnan(current_temp):
            value_list.append(np.nan)
            momentum_list.append(np.nan)
            continue
    
        # --- 7. Value (Recursive) ---
        
        # Pine: value := value * (temp - .5 + .5 * nz(value[1]))
        if i == 0:
            # First bar initialization: value = 1.0. 
            base_multiplier = 1.0 
        else:
            # Subsequent bars: value := value[1] * ...
            base_multiplier = value_list[-1] 
        
        # nz(value[1]) emulation: value_prev is the previous *final* value
        new_value = base_multiplier * (current_temp - 0.5 + 0.5 * value_prev)

        # Apply limits
        new_value = min(0.9999, max(-0.9999, new_value))
        
        value_list.append(new_value)
        value_prev = new_value # Update previous value for the next iteration

        # --- 8. Momentum (Recursive) ---
        
        if np.abs(new_value) >= 1.0:
            new_momentum = np.nan
        else:
            # Pine: temp2 = (1 + value) / (1 - value)
            temp2 = (1 + new_value) / (1 - new_value)
            
            # Pine: momentum = .25 * math.log(temp2)
            momentum_unfiltered = 0.25 * np.log(temp2)
            
            # Pine: momentum := momentum + .5 * nz(momentum[1])
            # nz(momentum[1]) emulation: momentum_prev is the previous final momentum
            new_momentum = momentum_unfiltered + 0.5 * momentum_prev
            
        momentum_list.append(new_momentum)
        
        # Update previous momentum for the next iteration
        momentum_prev = new_momentum
    
    # The final output is 'hist' / 'momentum'
    hist = pd.Series(momentum_list, index=close.index)
    
    return hist
# === END CORRECTED Magical Momentum Indicator ===


def check_pair(pair_name, pair_info, last_alerts):
    """Check PPO and RMA/Cirrus/SRSI/Momentum conditions for a pair"""
    
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
        
        # Fetch 15-minute candles for PPO, RMA50, Cirrus, SRSI, Momentum
        required_15m_limit = max(limit_15m, MOMENTUM_PERIOD + MOMENTUM_STDEV_PERIOD + 2) 
        df_15m = get_candles(pair_info['symbol'], "15", limit=required_15m_limit)
        
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
        rma_50 = calculate_rma(df_15m['close'], RMA_50_PERIOD)
        
        # Calculate RMA200 on 5min timeframe
        rma_200 = calculate_rma(df_5m['close'], RMA_200_PERIOD)
        
        
        # Calculate Cirrus Cloud on 15min timeframe 
        upw, dnw, filtx1, filtx12 = calculate_cirrus_cloud(df_15m)
      
        # Calculate Smoothed RSI on 15min timeframe
        smooth_rsi = calculate_smooth_rsi(df_15m)
        
        # Calculate Magical Momentum Indicator
        magical_momentum_hist = calculate_magical_momentum(df_15m)
        
        # Get latest values from 15min
        ppo_curr = ppo.iloc[-1]
        ppo_prev = ppo.iloc[-2]
        ppo_signal_curr = ppo_signal.iloc[-1]
        ppo_signal_prev = ppo_signal.iloc[-2]
        
        # Smoothed RSI values
        smooth_rsi_curr = smooth_rsi.iloc[-1]
        smooth_rsi_prev = smooth_rsi.iloc[-2] 
        
        # Magical Momentum Hist value
        magical_hist_curr = magical_momentum_hist.iloc[-1]
        
        close_curr = df_15m['close'].iloc[-1]
        rma50_curr = rma_50.iloc[-1]
        
        upw_curr = upw.iloc[-1]
        dnw_curr = dnw.iloc[-1]
        
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
            debug_log("Candle range is zero, skipping wick checks.")
        # --- END CANDLE STRUCTURE CHECKS ---
        
 
        # Get latest values from 5min
        close_5m_curr = df_5m['close'].iloc[-1]
        rma200_curr = rma_200.iloc[-1]
        
 
        
        # Debug: Print all indicator values
        debug_log(f"Price: ${close_curr:,.2f}")
        debug_log(f"PPO: {ppo_curr:.4f} (prev: {ppo_prev:.4f})")
        debug_log(f"PPO Signal: {ppo_signal_curr:.4f} (prev: {ppo_signal_prev:.4f})")
      

        debug_log(f"RMA50 (15m): {rma50_curr:.2f}, Close: {close_curr:.2f}")
        debug_log(f"RMA200 (5m): {rma200_curr:.2f}, Close: {close_5m_curr:.2f}")
        
        # Smoothed RSI Debug Log
        debug_log(f"Smoothed RSI (15m): {smooth_rsi_curr:.2f} (prev: {smooth_rsi_prev:.2f})") 
        
        # Magical Momentum Hist Debug Log
        debug_log(f"Magical Momentum Hist (15m): {magical_hist_curr:.4f}") 

     
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
        
        # Added: PPO conditions for new SRSI alerts
        ppo_below_030 = ppo_curr < 0.30
        ppo_above_minus030 = ppo_curr > -0.30
        
        ppo_15m_above_020 = ppo_curr > 0.20 
        ppo_15m_below_minus020 = ppo_curr < -0.20 
        
        # RMA conditions
        close_above_rma50 = close_curr > rma50_curr
        close_below_rma50 = close_curr < rma50_curr
        close_above_rma200 = close_5m_curr > rma200_curr
        close_below_rma200 = close_5m_curr < rma200_curr
        
        # Smoothed RSI conditions (REMOVED FROM LOGIC, KEPT FOR DEBUG)
        srsi_above_50 = smooth_rsi_curr > 50
        srsi_below_50 = smooth_rsi_curr < 50
        
        # Modified: Smoothed RSI Crossover Conditions
        srsi_cross_up_50 = (smooth_rsi_prev <= 50) and (smooth_rsi_curr > 50)
        srsi_cross_down_50 = (smooth_rsi_prev >= 50) and (smooth_rsi_curr < 50)
        
        # Magical Momentum Hist conditions
        magical_hist_bullish = magical_hist_curr > 0 
        magical_hist_bearish = magical_hist_curr < 0 

        # Debug: Print crossover detections
        debug_log(f"\nCrossover Checks:")
        
        debug_log(f"  PPO 15m cross up: {ppo_cross_up}")
        debug_log(f"  PPO 15m cross down: {ppo_cross_down}")
        debug_log(f"  SRSI cross up 50 (Modified): {srsi_cross_up_50}") 
        debug_log(f"  SRSI cross down 50 (Modified): {srsi_cross_down_50}") 
        debug_log(f"  PPO 15m above 0.20: {ppo_15m_above_020}")
        debug_log(f"  PPO 15m below -0.20: {ppo_15m_below_minus020}")
        
        debug_log(f"\nCondition Checks:")
        
        debug_log(f"  PPO 15m < 0.20: {ppo_below_020}")
        debug_log(f"  PPO 15m > -0.20: {ppo_above_minus020}")
        debug_log(f"  PPO 15m < 0.30: {ppo_below_030}")
        debug_log(f"  PPO 15m > -0.30: {ppo_above_minus030}")
        debug_log(f"  PPO 15m > Signal: {ppo_above_signal}")
        debug_log(f"  Close > RMA50: {close_above_rma50}")
        debug_log(f"  Close > RMA200: {close_above_rma200}")
        debug_log(f"  Upw (Cirrus): {upw_curr}") 
        
        debug_log(f"  Dnw (Cirrus): {dnw_curr}") 
        debug_log(f"  SRSI > 50: {srsi_above_50}") 
        debug_log(f"  SRSI < 50: {srsi_below_50}") 
        debug_log(f"  Magical Hist > 0 (NEW): {magical_hist_bullish}") 
        debug_log(f"  Magical Hist < 0 (NEW): {magical_hist_bearish}") 
        
 
        current_state = None
        
        # Get IST time in correct format
        ist = pytz.timezone('Asia/Kolkata')
        current_dt = datetime.now(ist)
        formatted_time = current_dt.strftime('%d-%m-%Y @ %H:%M IST')
        price = df_15m['close'].iloc[-1]
        
 
        # 1. ORIGINAL BUY: PPO crosses up AND PPO < 0.20 AND ... (Hist > 0 added)
        if (ppo_cross_up and 
            ppo_below_020 and 
            close_above_rma50 and 
            close_above_rma200 and 
            upw_curr and (not dnw_curr) and 
            strong_bullish_close and
            magical_hist_bullish): 
            current_state = "buy"
            debug_log(f"\n🟢 BUY SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "buy":
            
                message = f"🟢 {pair_name} - BUY\nPPO - SIGNAL Crossover (PPO: {ppo_curr:.2f})\nMagical Hist: {magical_hist_curr:.4f}\nPrice: ${price:,.2f}\n{formatted_time}" 
                send_telegram_alert(message)
            else:
                
                debug_log(f"BUY already alerted for {pair_name}, skipping duplicate")
        
        # 2. ORIGINAL SELL: PPO crosses down AND PPO > -0.20 AND ... (Hist < 0 added)
        
        elif (ppo_cross_down and 
              ppo_above_minus020 and 
              close_below_rma50 and 
              close_below_rma200 and 
              dnw_curr and (not upw_curr) and 
              strong_bearish_close and
              magical_hist_bearish): 
            current_state = "sell"
            debug_log(f"\n🔴 SELL SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "sell":
    
                message = f"🔴 {pair_name} - SELL\nPPO - SIGNAL Crossunder (PPO: {ppo_curr:.2f})\nMagical Hist: {magical_hist_curr:.4f}\nPrice: ${price:,.2f}\n{formatted_time}" 
            
                send_telegram_alert(message)
            else:
                debug_log(f"SELL already alerted for {pair_name}, skipping duplicate")
               
        # 3. NEW BUY ALERT (SRSI CROSS UP 50) (Hist > 0 added)
        elif (srsi_cross_up_50 and 
              ppo_above_signal and 
              ppo_below_030 and 
              close_above_rma50 and 
              close_above_rma200 and 
              upw_curr and (not dnw_curr) and 
              strong_bullish_close and
              magical_hist_bullish): 
            current_state = "buy_srsi50" 
            debug_log(f"\n⬆️ BUY (SRSI 50) SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "buy_srsi50": 
                message = f"⬆️ {pair_name} - BUY (SRSI 50)\nSRSI 15m Cross Up 50 ({smooth_rsi_curr:.2f})\nMagical Hist: {magical_hist_curr:.4f}\nPrice: ${price:,.2f}\n{formatted_time}" 
                send_telegram_alert(message)
            else:
                debug_log(f"BUY (SRSI 50) already alerted for {pair_name}, skipping duplicate") 
                
        # 4. NEW SELL ALERT (SRSI CROSS DOWN 50) (Hist < 0 added)
        elif (srsi_cross_down_50 and 
              ppo_below_signal and 
              ppo_above_minus030 and 
              close_below_rma50 and 
              close_below_rma200 and 
              dnw_curr and (not upw_curr) and 
              strong_bearish_close and
              magical_hist_bearish): 
            current_state = "sell_srsi50" 
            debug_log(f"\n⬇️ SELL (SRSI 50) SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "sell_srsi50": 
                message = f"⬇️ {pair_name} - SELL (SRSI 50)\nSRSI 15m Cross Down 50 ({smooth_rsi_curr:.2f})\nMagical Hist: {magical_hist_curr:.4f}\nPrice: ${price:,.2f}\n{formatted_time}" 
              
                send_telegram_alert(message)
            else:
                debug_log(f"SELL (SRSI 50) already alerted for {pair_name}, skipping duplicate") 

        
        # 5. LONG (0): PPO > Signal AND PPO crosses above 0 AND ... (Hist > 0 added)
        elif (ppo_cross_above_zero and 
              ppo_above_signal and 
              close_above_rma50 and 
              close_above_rma200 and 
              upw_curr and (not dnw_curr) and 
              strong_bullish_close and
              magical_hist_bullish): 
            current_state = "long_zero"
            debug_log(f"\n🟢 LONG (0) SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "long_zero":
            
                message = f"🟢 {pair_name} - LONG\nPPO crossing above 0 ({ppo_curr:.2f})\nMagical Hist: {magical_hist_curr:.4f}\nPrice: ${price:,.2f}\n{formatted_time}" 
                send_telegram_alert(message)
            else:
        
                debug_log(f"LONG (0) already alerted for {pair_name}, skipping duplicate")
       
        # 6. LONG (0.11): PPO > Signal AND PPO crosses above 0.11 AND ... (Hist > 0 added)
        
        elif (ppo_cross_above_011 and 
              ppo_above_signal and 
              close_above_rma50 and 
              close_above_rma200 and 
              upw_curr and (not dnw_curr) and 
              strong_bullish_close and
              magical_hist_bullish): 
            current_state = "long_011"
            debug_log(f"\n🟢 LONG (0.11) SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "long_011":
            
                message = f"🟢 {pair_name} - LONG\nPPO crossing above 0.11 ({ppo_curr:.2f})\nMagical Hist: {magical_hist_curr:.4f}\nPrice: ${price:,.2f}\n{formatted_time}" 
            
                send_telegram_alert(message)
            else:
                
                debug_log(f"LONG (0.11) already alerted for {pair_name}, skipping duplicate")
        
        # 7. SHORT (0): PPO < Signal AND PPO crosses below 0 AND ... (Hist < 0 added)
        elif (ppo_cross_below_zero and 
              ppo_below_signal and 
              close_below_rma50 and 
              close_below_rma200 and 
              dnw_curr and (not upw_curr) and 
              strong_bearish_close and
              magical_hist_bearish): 
            
            current_state = "short_zero"
            debug_log(f"\n🔴 SHORT (0) SIGNAL DETECTED for {pair_name}!")
            
            if last_alerts.get(pair_name) != "short_zero":
                message = f"🔴 {pair_name} - SHORT\nPPO crossing below 0 ({ppo_curr:.2f})\nMagical Hist: {magical_hist_curr:.4f}\nPrice: ${price:,.2f}\n{formatted_time}" 
                send_telegram_alert(message)
            else:
            
                debug_log(f"SHORT (0) already alerted for {pair_name}, skipping duplicate")
        
        
        # 8. SHORT (-0.11): PPO < Signal AND PPO crosses below -0.11 AND ... (Hist < 0 added)
        elif (ppo_cross_below_minus011 and 
              ppo_below_signal and 
              close_below_rma50 and 
              close_below_rma200 and 
              dnw_curr and (not upw_curr) and 
              strong_bearish_close and
              magical_hist_bearish): 
            current_state = "short_011"
            debug_log(f"\n🔴 SHORT (-0.11) SIGNAL DETECTED for {pair_name}!")
            if last_alerts.get(pair_name) != "short_011":
                message = f"🔴 {pair_name} - SHORT\nPPO crossing below -0.11 ({ppo_curr:.2f})\nMagical Hist: {magical_hist_curr:.4f}\nPrice: ${price:,.2f}\n{formatted_time}" 
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

# ... (rest of the script: main function) ...

def main():
    """Main function - runs once per GitHub Actions execution"""
    print("=" * 50)
    ist = pytz.timezone('Asia/Kolkata')
    start_time = datetime.now(ist)
    print(f"PPO/Cirrus Cloud Alert Bot - {start_time.strftime('%d-%m-%Y @ %H:%M IST')}")
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
    
    # Use a ThreadPoolExecutor to run all 'check_pair' 
    # calls in parallel.
    with ThreadPoolExecutor(max_workers=10) as executor:
        
        future_to_pair = {}
        
        for pair_name, pair_info in PAIRS.items():
            if pair_info is not None:
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
                    # type (e.g., "buy", "short_011", "buy_srsi55")
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
