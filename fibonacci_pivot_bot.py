import requests
import pandas as pd
import numpy as np
import time
import os
import json
import pytz
import tempfile
import shutil
import logging
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============ CONFIGURATION ============
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '8462496498:AAHYZ4xDIHvrVRjmCmZyoPhupCjRaRgiITc')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '203813932')

DEBUG_MODE = os.environ.get('DEBUG_MODE', 'True').lower() == 'true'
SEND_TEST_MESSAGE = os.environ.get('SEND_TEST_MESSAGE', 'True').lower() == 'true'
RESET_STATE = os.environ.get('RESET_STATE', 'False').lower() == 'true'

DELTA_API_BASE = "https://api.india.delta.exchange"

PAIRS = {
    "BTCUSD": None, "ETHUSD": None,
    "SOLUSD": None, "AVAXUSD": None,
    "BCHUSD": None, "XRPUSD": None, "BNBUSD": None, "LTCUSD": None,
    "DOTUSD": None, "ADAUSD": None, "SUIUSD": None, "AAVEUSD": None
}

SPECIAL_PAIRS = {
    "SOLUSD": {"limit_15m": 210, "min_required": 180, "limit_5m": 300, "min_required_5m": 200}
}

# Indicator settings
RMA_PERIOD = 200
FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

# State file paths (NEW/IMPROVED)
STATE_FILE = os.environ.get("STATE_FILE_PATH", "fib_state.json")
STATE_FILE_BAK = STATE_FILE + ".bak"

# ============ LOGGING SETUP ============
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(ch)

# ============ UTILITY FUNCTIONS (IMPROVED FOR ROBUSTNESS) ============

def create_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

SESSION = create_session()

def load_state():
    """
    Load previous alert state from file with backup recovery.
    Initializes a 'NO_SIGNAL' state for all pairs if files are missing.
    """
    
    def _initialize_default_state():
        """Returns a clean state for all monitored pairs."""
        return {pair: {"state": "NO_SIGNAL", "ts": 0} for pair in PAIRS.keys()}

    def _safe_load(filepath):
        """Attempts to load and validate state from a specific path."""
        try:
            if not os.path.exists(filepath):
                return None
            
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            if not isinstance(state, dict):
                raise ValueError("State file is not a dict")

            # Normalize schema to {pair: {"state": str, "ts": int}} and filter for known PAIRS
            normalized = {}
            for k, v in state.items():
                if k in PAIRS:
                    if isinstance(v, dict):
                        s = v.get("state")
                        ts = v.get("ts", 0)
                        if isinstance(s, str):
                            normalized[k] = {"state": s, "ts": int(ts)}
                    elif isinstance(v, str):
                        normalized[k] = {"state": v, "ts": 0}
            return normalized
        except Exception:
            logger.exception(f"Error reading state file {filepath}")
            return None

    # 1. Try primary file
    loaded_state = _safe_load(STATE_FILE)
    if loaded_state is not None:
        logger.debug(f"Loaded primary state (partial): {loaded_state}")
    
    # 2. Try backup file if primary failed
    if loaded_state is None:
        loaded_state = _safe_load(STATE_FILE_BAK)
        if loaded_state is not None:
            logger.info("Recovered state from backup.")
            logger.debug(f"Loaded backup state (partial): {loaded_state}")

    # 3. Combine with default state for all PAIRS
    final_state = _initialize_default_state()
    if loaded_state is not None:
        # Merge loaded state over default state, filling in missing pairs with default
        final_state.update(loaded_state)
        
    if loaded_state is None:
         logger.debug("No previous state found or recovery failed, starting fresh (NO_SIGNAL)")

    return final_state

def save_state(state):
    """Save alert state to file atomically with backup creation and fsync."""
    try:
        temp_file = STATE_FILE + '.tmp'
        
        # 1. Write the new state to a temporary file
        with open(temp_file, 'w') as f:
            json.dump(state, f, indent=4)
            f.flush()
            # Ensure data is written to the physical disk (important for robustness)
            os.fsync(f.fileno()) 

        # 2. Rename existing state file to backup (atomic backup)
        if os.path.exists(STATE_FILE):
            try:
                # os.replace is atomic on most modern OS filesystems (like rename)
                os.replace(STATE_FILE, STATE_FILE_BAK) 
            except Exception:
                # If rename fails, log a warning but proceed
                logger.warning("Could not rename primary file to backup.")
        
        # 3. Atomically replace the old state file (or backup) with the new temporary file
        os.replace(temp_file, STATE_FILE)
        
        logger.debug(f"Saved state: {state}")
    except Exception:
        logger.exception("Error saving state")

def send_telegram_alert(message):
    """Send alert message via Telegram"""
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == 'xxxx' or not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID == 'xxxx':
        logger.warning("Telegram not configured: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        return False
    try:
        logger.debug(f"Attempting to send message: {message[:100]}...")
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": None 
        }
        response = SESSION.post(url, data=data, timeout=10)
        response_data = response.json()
        if response_data.get('ok'):
            logger.info(f"Alert sent successfully")
            return True
        else:
            logger.error(f"Telegram error: {response_data}")
            return False
    except Exception:
        logger.exception("Error sending Telegram message")
        return False

def send_test_message():
    """Send a test message to verify Telegram connectivity"""
    ist = pytz.timezone('Asia/Kolkata')
    current_dt = datetime.now(ist)
    formatted_time = current_dt.strftime('%d-%m-%Y @ %H:%M IST')
    test_msg = f"🔔 Bot Started\nTest message from Fibonacci Pivot Bot\nTime: {formatted_time}\nDebug Mode: {'ON' if DEBUG_MODE else 'OFF'}"
    logger.info("="*50)
    logger.info("SENDING TEST MESSAGE")
    logger.info("="*50)
    success = send_telegram_alert(test_msg)
    if success:
        logger.info("Test message sent successfully!")
    else:
        logger.error("Test message failed - check your bot token and chat ID")
    logger.info("="*50)
    return success

def get_product_ids():
    """Fetch all product IDs from Delta Exchange"""
    try:
        logger.debug("Fetching product IDs from Delta Exchange...")
        response = SESSION.get(f"{DELTA_API_BASE}/v2/products", timeout=10)
        data = response.json()
        if data.get('success'):
            products = data['result']
            logger.debug(f"Received {len(products)} products from API")
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
                            logger.debug(f"Matched {pair_name} -> {product['symbol']} (ID: {product['id']})")
            return True
        else:
            logger.error(f"API Error: {data}")
            return False
    except Exception:
        logger.exception("Error fetching products")
        return False

def get_candles(product_id, resolution="15", limit=150):
    """Fetch OHLCV candles from Delta Exchange with volume check"""
    # [Implementation of get_candles remains the same, using logger]
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
        logger.debug(f"Fetching {resolution}m candles for {product_id}, limit={limit}")
        response = SESSION.get(url, params=params, timeout=15)
        data = response.json()

        if data.get('success'):
            result = data['result']
            arrays = [result.get('t', []), result.get('o', []), result.get('h', []), result.get('l', []), result.get('c', []), result.get('v', [])]
            min_len = min(map(len, arrays)) if arrays else 0
            if min_len == 0:
                logger.debug(f"No candles returned for {product_id}")
                return None
            df = pd.DataFrame({
                'timestamp': result.get('t', [])[:min_len],
                'open': result.get('o', [])[:min_len],
                'high': result.get('h', [])[:min_len],
                'low': result.get('l', [])[:min_len],
                'close': result.get('c', [])[:min_len],
                'volume': result.get('v', [])[:min_len]
            })

            df = df.sort_values('timestamp').drop_duplicates(subset='timestamp').reset_index(drop=True)

            if df['close'].iloc[-1] <= 0:
                logger.debug(f"Invalid price (≤0) for {product_id}, skipping")
                return None

            # Volume Check: If the latest volume is too low or zero, it might be stale/bad data
            latest_volume = df['volume'].iloc[-1]
            if latest_volume < 1:
                 logger.warning(f"Low/Zero volume for {product_id} ({resolution}m): {latest_volume}. Check data.")

            logger.debug(f"Received {len(df)} candles for {product_id} ({resolution}m)")

            # Check data freshness
            last_candle_time = df['timestamp'].iloc[-1]
            time_diff = time.time() - last_candle_time
            max_age_soft = int(resolution) * 60 * 6  # 6 candles soft threshold
            if time_diff > max_age_soft:
                logger.warning(f"Stale data for {product_id} ({resolution}m) - {time_diff/60:.1f} min old. Skipping.")
                return None

            return df
        else:
            logger.error(f"Error fetching candles for {product_id}: {data.get('message', 'No message')}")
            return None
    except Exception:
        logger.exception(f"Exception fetching candles for {product_id}")
        return None

def calculate_rma(data, period):
    r = data.ewm(alpha=1/period, adjust=False).mean()
    return r.bfill().ffill()

def calculate_pivots(df):
    """
    Calculate Daily Fibonacci Pivots using the previous day's High, Low, Close.
    Requires data that spans the close of the *previous* day.
    """
    # Assuming df is 1D or 15m data with enough history
    if len(df) < 2:
        return None

    # Get the HLC of the last *closed* 1D candle, which represents "Yesterday"
    # This requires special handling if input df is not 1D. 
    # Since the bot is using 15m data, we need to find the HLC of the last 24h period ending at midnight UTC (or 5:30 IST)
    # The original bot seems to assume the last full candle is the "Daily" one.

    # Simpler: use the HLC of the last *full* candle for P/R/S calculation
    high = df['high'].iloc[-2]  # Yesterday's High/Last Closed High
    low = df['low'].iloc[-2]    # Yesterday's Low/Last Closed Low
    close = df['close'].iloc[-2] # Yesterday's Close/Last Closed Close
    
    # Range
    price_range = high - low
    
    # Pivot Point (P)
    P = (high + low + close) / 3
    
    # Fibonacci Levels
    pivots = {'P': P}
    
    # Resistance
    pivots['R3'] = P + (price_range * 1.0)
    pivots['R2'] = P + (price_range * 0.618)
    pivots['R1'] = P + (price_range * 0.382)

    # Support
    pivots['S1'] = P - (price_range * 0.382)
    pivots['S2'] = P - (price_range * 0.618)
    pivots['S3'] = P - (price_range * 1.0)
    
    # Sort levels by price for easier comparison
    sorted_levels = sorted(pivots.items(), key=lambda item: item[1])
    pivots['sorted_levels'] = sorted_levels
    
    return pivots

def check_pair(pair_name, pair_info, last_alerts, logger_ref, debug_output):
    """
    Check Fibonacci Pivot and RMA conditions for a pair.
    Returns (new_state, log_output)
    """
    
    # Redirect print() to the log stream capture
    def local_print(*args, **kwargs):
        debug_output.append(f"{' '.join(map(str, args))}\n")

    logger_ref.debug(f"\n{'='*60}")
    logger_ref.debug(f"Checking {pair_name}")
    logger_ref.debug(f"{'='*60}")

    # Prepare last state info
    now_ts = int(time.time())
    last_state_for_pair = last_alerts.get(pair_name, {"state": "NO_SIGNAL", "ts": 0})
    last_signal_state = last_state_for_pair.get("state", "NO_SIGNAL")
    
    current_signal = 'NO_SIGNAL'
    send_message = None

    try:
        # Data fetching based on SPECIAL_PAIRS config
        if pair_name in SPECIAL_PAIRS:
            limit_15m = SPECIAL_PAIRS[pair_name]["limit_15m"]
            min_required = SPECIAL_PAIRS[pair_name]["min_required"]
        else:
            limit_15m = 210
            min_required = 200

        df_15m = get_candles(pair_info['symbol'], "15", limit=limit_15m)

        if df_15m is None or len(df_15m) < (min_required + 2):
            local_print(f"⚠️ Insufficient 15m data for {pair_name}: {len(df_15m) if df_15m is not None else 0}/{min_required + 2} candles.")
            return last_state_for_pair, "".join(debug_output)
        
        # --- ROBUST INDEXING LOGIC ---
        # 1. Determine closed index for 15m (RMA, Pivots)
        resolution_sec_15m = 15 * 60
        current_15m_interval_start_ts = now_ts - (now_ts % resolution_sec_15m)

        is_last_15m_candle_incomplete = df_15m['timestamp'].iloc[-1] >= current_15m_interval_start_ts

        # Set indices based on 15m candle completeness
        if is_last_15m_candle_incomplete:
            # The last candle (index -1) is incomplete. Use index -2 as the signal candle.
            last_i = -2
            prev_i = -3
            logger_ref.debug(f"Current 15m candle (idx -1) is incomplete. Using closed candle at idx {last_i}")
        else:
            # The last candle (index -1) is the last closed one.
            last_i = -1
            prev_i = -2
            logger_ref.debug(f"Last fetched 15m candle (idx -1) is closed. Using it for signal.")

        if len(df_15m) < abs(prev_i): 
            local_print(f"⚠️ Insufficient 15m data for {pair_name} after adjusting for incomplete candle. Needs {abs(prev_i)} rows.")
            return last_state_for_pair, "".join(debug_output)
        # --- END ROBUST INDEXING LOGIC ---


        rma_200 = calculate_rma(df_15m['close'], RMA_PERIOD)
        pivots = calculate_pivots(df_15m)

        if pivots is None:
            local_print(f"⚠️ Failed to calculate pivots for {pair_name}.")
            return last_state_for_pair, "".join(debug_output)

        # Extract last closed values
        close_curr = float(df_15m['close'].iloc[last_i])
        close_prev = float(df_15m['close'].iloc[prev_i])

        rma200_curr = float(rma_200.iloc[last_i])
        if pd.isna(rma200_curr):
            local_print(f"⚠️ NaN values in RMA200 for {pair_name}, skipping.")
            return last_state_for_pair, "".join(debug_output)

        # Pivot checks
        pivot_breaks = []
        for level_name, level_price in pivots['sorted_levels']:
            if level_name != 'P': # Skip P in the level name for now
                if (close_prev < level_price) and (close_curr >= level_price):
                    pivot_breaks.append(("UP", level_name, level_price))
                elif (close_prev > level_price) and (close_curr <= level_price):
                    pivot_breaks.append(("DOWN", level_name, level_price))

        # Check for bounce off P
        p_price = pivots['P']
        p_bounce_long = (close_prev < p_price) and (close_curr >= p_price)
        p_bounce_short = (close_prev > p_price) and (close_curr <= p_price)
        
        # Current time & formatting
        ist = pytz.timezone('Asia/Kolkata')
        current_dt = datetime.now(ist)
        formatted_time = current_dt.strftime('%d-%m-%Y @ %H:%M IST')
        price = close_curr

        local_print(f"\nIndicator Values:")
        local_print(f"Price (15m): ${close_curr:,.2f} (prev: ${close_prev:,.2f})")
        local_print(f"RMA200 (15m): {rma200_curr:.2f}")
        local_print(f"Pivot P: {p_price:.2f}, R1: {pivots['R1']:.2f}, S1: {pivots['S1']:.2f}")
        local_print(f"Last State: {last_signal_state}")
        
        # --- ALERT LOGIC ---
        
        # 1. Major Breakout/Breakdown (R3/S3) with RMA confirmation
        if pivot_breaks:
            direction, level_name, level_price = pivot_breaks[0] # Focus on the first detected break
            
            if direction == "UP" and close_curr > rma200_curr:
                current_signal = f"LONG_BREAK_{level_name}"
                send_message = f"🟢 {pair_name} - LONG\nMajor Breakout: Price crossed ABOVE {level_name} (${level_price:,.2f})\nConfirmed by RMA200 ({rma200_curr:,.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
            
            elif direction == "DOWN" and close_curr < rma200_curr:
                current_signal = f"SHORT_BREAK_{level_name}"
                send_message = f"🔴 {pair_name} - SHORT\nMajor Breakdown: Price crossed BELOW {level_name} (${level_price:,.2f})\nConfirmed by RMA200 ({rma200_curr:,.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
            
            # 2. Pivot Bounce (Counter-Trend or Continuation)
            # Only consider P, R1, S1 for bounces when not breaking R3/S3
            elif direction == "UP" and level_name in ['R1', 'S1', 'P']:
                 # Break above R1/S1/P on 15m (Continuation or Trend Change)
                if last_signal_state not in ["LONG_R1_BREAK", "LONG_S1_BREAK", "LONG_P_BOUNCE"]:
                    current_signal = f"LONG_{level_name}_BREAK"
                    send_message = f"⬆️ {pair_name} - LONG\nPrice broke ABOVE {level_name} (${level_price:,.2f})\nPrice: ${price:,.2f}\n{formatted_time}"

            elif direction == "DOWN" and level_name in ['R1', 'S1', 'P']:
                # Break below R1/S1/P on 15m (Continuation or Trend Change)
                if last_signal_state not in ["SHORT_R1_BREAK", "SHORT_S1_BREAK", "SHORT_P_BOUNCE"]:
                    current_signal = f"SHORT_{level_name}_BREAK"
                    send_message = f"⬇️ {pair_name} - SHORT\nPrice broke BELOW {level_name} (${level_price:,.2f})\nPrice: ${price:,.2f}\n{formatted_time}"

        # 3. Simple RMA Cross (If no pivot break)
        if current_signal == 'NO_SIGNAL':
            if (close_prev < rma200_curr) and (close_curr >= rma200_curr):
                current_signal = "LONG_RMA_CROSS"
                send_message = f"⬆️ {pair_name} - LONG\nPrice crossed ABOVE RMA200 ({rma200_curr:,.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
            elif (close_prev > rma200_curr) and (close_curr <= rma200_curr):
                current_signal = "SHORT_RMA_CROSS"
                send_message = f"⬇️ {pair_name} - SHORT\nPrice crossed BELOW RMA200 ({rma200_curr:,.2f})\nPrice: ${price:,.2f}\n{formatted_time}"

        logger_ref.debug(f"Current Signal: {current_signal}, Last Signal: {last_signal_state}")
        
        # --- IDEMPOTENCY AND STATE TRANSITION CHECK ---
        
        # 1. No signal was detected
        if current_signal == 'NO_SIGNAL':
            
            # Reset state if we were previously in a signal state
            if last_signal_state != 'NO_SIGNAL':
                # Transition from a signal state back to NO_SIGNAL
                new_state = {"state": "NO_SIGNAL", "ts": now_ts}
                logger_ref.debug(f"Transitioning {pair_name} to NO_SIGNAL.")
                return new_state, "".join(debug_output)
            
            # Otherwise, return the existing state (no change needed)
            return last_state_for_pair, "".join(debug_output)

        # 2. A signal (current_signal) was detected
        
        # Idempotency Check: If the new signal is the same as the last recorded state, skip the alert.
        if current_signal == last_signal_state:
            logger_ref.debug(f"Idempotency check: {pair_name} signal is still {current_signal}. Skipping alert.")
            # Do not send alert, and return the existing state
            return last_state_for_pair, "".join(debug_output)
        
        # 3. NEW Signal Detected: Send Alert and Prepare New State
        if send_message:
            send_telegram_alert(send_message)

        # Return structured state for saving
        new_state = {"state": current_signal, "ts": now_ts}
        return new_state, "".join(debug_output)

    except Exception:
        logger_ref.exception(f"Error checking {pair_name}")
        # Return existing state on error to avoid losing state during an exception
        return last_state_for_pair, "".join(debug_output)

def run_with_jitter(fn, *args, **kwargs):
    time.sleep(np.random.uniform(0.1, 0.7))
    return fn(*args, **kwargs)

def main():
    if RESET_STATE:
        try:
            os.remove(STATE_FILE)
            os.remove(STATE_FILE_BAK)
            logger.warning(f"State files reset successfully due to RESET_STATE=True.")
        except FileNotFoundError:
            pass # Ignore if files don't exist
        except Exception:
            logger.exception("Failed to delete state files.")

    logger.info("=" * 50)
    ist = pytz.timezone('Asia/Kolkata')
    start_time = datetime.now(ist)
    logger.info(f"Fibonacci Pivot Bot - {start_time.strftime('%d-%m-%Y @ %H:%M IST')}")
    logger.info(f"Debug Mode: {'ON' if DEBUG_MODE else 'OFF'}")
    logger.info("=" * 50)

    if SEND_TEST_MESSAGE:
        send_test_message()
    elif SEND_TEST_MESSAGE:
        logger.warning("Skipping test message: Telegram not configured.")

    last_alerts = load_state()

    if not get_product_ids():
        logger.error("Failed to fetch products. Exiting.")
        return

    found_count = sum(1 for v in PAIRS.values() if v is not None)
    logger.info(f"Monitoring {found_count} pairs")
    if found_count == 0:
        logger.error("No valid pairs found. Exiting.")
        return

    updates_processed = 0
    pair_logs = []

    # Use ThreadPoolExecutor to run check_pair for all pairs concurrently
    max_workers = min(4, found_count)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pair = {}
        for pair_name, pair_info in PAIRS.items():
            if pair_info is not None:
                # check_pair now takes logger_ref and debug_output list
                debug_output = []
                future = executor.submit(check_pair, pair_name, pair_info, last_alerts, logger, debug_output)
                future_to_pair[future] = pair_name
        
        for future in as_completed(future_to_pair):
            pair_name = future_to_pair[future]
            try:
                # new_state is the dictionary {"state": str, "ts": int} or the old state on error
                new_state, log_output = future.result() 
                pair_logs.append(log_output)
                
                # Check for state transition
                if new_state.get('state') != last_alerts.get(pair_name, {}).get('state') or new_state.get('ts') != last_alerts.get(pair_name, {}).get('ts'):
                    last_alerts[pair_name] = new_state
                    updates_processed += 1
                
            except Exception as e:
                logger.exception(f"Error processing {pair_name} in thread: {e}")
                continue
    
    # Dump sequential debug logs after all processing is done
    if DEBUG_MODE and pair_logs:
        logger.debug("SEQUENTIAL DEBUG LOGS START")
        for log_output in pair_logs:
            if log_output:
                print(log_output, end='')
        logger.debug("SEQUENTIAL DEBUG LOGS END")

    # Always save, even if no updates, to ensure file exists and is tracked
    save_state(last_alerts)
    end_time = datetime.now(ist)
    elapsed = (end_time - start_time).total_seconds()
    logger.info(f"✓ Check complete. {updates_processed} state updates processed. ({elapsed:.1f}s)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Fatal error in main execution: {e}")
