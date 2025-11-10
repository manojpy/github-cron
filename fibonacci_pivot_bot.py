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
    "SOLUSD": {"limit_15m": 210, "min_required": 140, "limit_5m": 450, "min_required_5m": 220}
}

# Indicator settings
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
    """Save alert state to file atomically"""
    try:
        temp_file = STATE_FILE + '.tmp'
        with open(temp_file, 'w') as f:
            json.dump(state, f)
        os.replace(temp_file, STATE_FILE)
        if DEBUG_MODE:
            print(f"[DEBUG] Saved state: {state}")
    except Exception as e:
        print(f"Error saving state: {e}")

def send_telegram_alert(message):
    """Send alert message via Telegram using HTML mode"""
    try:
        if DEBUG_MODE:
            print(f"[DEBUG] Attempting to send message: {message[:100]}...")
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
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
    """Fetch all product IDs from Delta Exchange and populate PAIRS dict with robust symbol matching"""
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
                if product.get('contract_type') != 'perpetual_futures':
                    continue
                raw_symbol = product['symbol']
                # Normalize Delta symbols to our naming convention (e.g., SOLUSD_PERP -> SOLUSD)
                if raw_symbol.endswith('_PERP'):
                    normalized = raw_symbol[:-5]  # Remove '_PERP'
                elif raw_symbol.endswith('USD'):
                    normalized = raw_symbol
                elif '_USD' in raw_symbol:
                    normalized = raw_symbol.replace('_USD', 'USD')
                else:
                    normalized = raw_symbol  # Fallback

                for pair_name in PAIRS.keys():
                    if normalized == pair_name:
                        PAIRS[pair_name] = {
                            'id': product['id'],
                            'symbol': product['symbol'],  # Use original symbol for API calls
                            'contract_type': product['contract_type']
                        }
                        if DEBUG_MODE:
                            print(f"[DEBUG] Matched {pair_name} ← {product['symbol']} (ID: {product['id']})")
                        break
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
        if resolution == "D":
            from_time = to_time - (limit * 24 * 60 * 60)
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
    return data.ewm(span=period, adjust=False).mean()

def calculate_sma(data, period):
    return data.rolling(window=period).mean()

def calculate_ppo(df, fast=7, slow=16, signal=5, use_sma=False):
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
    wper = t * 2 - 1
    avrng = calculate_ema(np.abs(x.diff()), t)
    smoothrng = calculate_ema(avrng, wper) * m
    return smoothrng

def rngfilt(x, r):
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
    close = df['close'].copy()
    smrngx1x = smoothrng(close, X1, X2)
    smrngx1x2 = smoothrng(close, X3, X4)
    filtx1 = rngfilt(close, smrngx1x)
    filtx12 = rngfilt(close, smrngx1x2)
    upw = filtx1 < filtx12 
    dnw = filtx1 > filtx12 
    return upw, dnw, filtx1, filtx12 

def calculate_rma(data, period):
    return data.ewm(alpha=1/period, adjust=False).mean()

# ============ MAGICAL MOMENTUM HISTOGRAM ============
def calculate_magical_momentum_hist(df, period=144, responsiveness=0.9):
    if len(df) < period + 50:
        return pd.Series([np.nan] * len(df), index=df.index)
    close = df['close'].astype(float)
    responsiveness = max(0.00001, responsiveness)
    sd = close.rolling(window=50).std() * responsiveness
    sd = sd.bfill().fillna(0.001)
    worm = close.copy()
    for i in range(1, len(close)):
        diff = close.iloc[i] - worm.iloc[i - 1]
        abs_diff = abs(diff)
        if abs_diff > sd.iloc[i]:
            delta = np.sign(diff) * sd.iloc[i]
        else:
            delta = diff
        worm.iloc[i] = worm.iloc[i - 1] + delta
    ma = close.rolling(window=period).mean()
    raw_momentum = (worm - ma) / worm.replace(0, np.nan)
    raw_momentum = raw_momentum.fillna(0)
    min_med = raw_momentum.rolling(window=period).min()
    max_med = raw_momentum.rolling(window=period).max()
    rng = max_med - min_med
    temp = pd.Series(0.0, index=df.index)
    valid = rng > 1e-10
    temp.loc[valid] = (raw_momentum.loc[valid] - min_med.loc[valid]) / rng.loc[valid]
    value = pd.Series(0.0, index=df.index)
    value.iloc[0] = 0.0
    for i in range(1, len(temp)):
        v_prev = value.iloc[i - 1]
        v_new = 1.0 * (temp.iloc[i] - 0.5 + 0.5 * v_prev)
        value.iloc[i] = max(-0.9999, min(0.9999, v_new))
    temp2 = (1 + value) / (1 - value)
    temp2 = temp2.replace([np.inf, -np.inf], np.nan)
    log_temp2 = np.log(temp2)
    momentum = 0.25 * log_temp2
    momentum = momentum.fillna(0)
    hist = pd.Series(0.0, index=df.index)
    hist.iloc[0] = momentum.iloc[0]
    for i in range(1, len(momentum)):
        hist.iloc[i] = momentum.iloc[i] + 0.5 * hist.iloc[i - 1]
    return hist

# ============ FIBONACCI PIVOT FUNCTIONS ============
def get_previous_day_ohlc(product_id, days_back_limit=15):
    """Fetch 1-day candles and return true prior trading day's H, L, C (IST-aware)."""
    df_daily = get_candles(product_id, resolution="D", limit=days_back_limit + 5)
    if df_daily is None or len(df_daily) < 2:
        return None

    ist = pytz.timezone('Asia/Kolkata')
    df_daily['datetime'] = pd.to_datetime(df_daily['timestamp'], unit='s', utc=True).dt.tz_convert(ist)
    df_daily = df_daily.set_index('datetime')

    today_ist = datetime.now(ist).date()
    yesterday_ist = today_ist - pd.Timedelta(days=1)

    prev_day_data = df_daily[df_daily.index.date == yesterday_ist]
    if len(prev_day_data) == 0:
        prev_day_data = df_daily.iloc[[-1]]
        if DEBUG_MODE:
            print(f"[DEBUG] Yesterday ({yesterday_ist}) not found. Using latest daily candle: {prev_day_data.index[-1].date()}")
    
    candle = prev_day_data.iloc[-1]
    return {
        'high': candle['high'],
        'low': candle['low'],
        'close': candle['close']
    }

def calculate_fibonacci_pivots(h, l, c):
    pivot = (h + l + c) / 3
    diff = h - l
    r3 = pivot + (diff * 1.000)
    r2 = pivot + (diff * 0.618)
    r1 = pivot + (diff * 0.382)
    s1 = pivot - (diff * 0.382)
    s2 = pivot - (diff * 0.618)
    s3 = pivot - (diff * 1.000)
    return {
        'P': pivot, 'R1': r1, 'R2': r2, 'R3': r3, 
        'S1': s1, 'S2': s2, 'S3': s3
    }

def check_pair(pair_name, pair_info, last_alerts):
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
        prev_day_ohlc = get_previous_day_ohlc(pair_info['symbol'], PIVOT_LOOKBACK_PERIOD)
        if prev_day_ohlc is None:
            print(f"❌ {pair_name}: Failed to fetch previous day OHLC data") 
            return last_alerts.get(pair_name), '\n'.join(thread_log)
        pivots = calculate_fibonacci_pivots(
            prev_day_ohlc['high'], prev_day_ohlc['low'], prev_day_ohlc['close']
        )
        log(f"Daily Pivots: P={pivots['P']:.2f}, R1={pivots['R1']:.2f}, R2={pivots['R2']:.2f}, R3={pivots['R3']:.2f}")
        log(f"             S1={pivots['S1']:.2f}, S2={pivots['S2']:.2f}, S3={pivots['S3']:.2f}")
        limit_15m = SPECIAL_PAIRS.get(pair_name, {}).get("limit_15m", 250)
        min_required = max(SPECIAL_PAIRS.get(pair_name, {}).get("min_required", 150), 150)
        df_15m = get_candles(pair_info['symbol'], "15", limit=limit_15m)
        if df_15m is None:
            print(f"❌ {pair_name}: Failed to fetch 15m candle data")
            return last_alerts.get(pair_name), '\n'.join(thread_log)
        if len(df_15m) < min_required:
            print(f"⚠️  {pair_name}: Insufficient 15m data (need {min_required}, got {len(df_15m)})")
            return last_alerts.get(pair_name), '\n'.join(thread_log)
        ppo, _ = calculate_ppo(df_15m, PPO_FAST, PPO_SLOW, PPO_SIGNAL, PPO_USE_SMA)
        upw, dnw, _, _ = calculate_cirrus_cloud(df_15m)
        rma_50_15m = calculate_rma(df_15m['close'], 50)
        if (len(ppo) < 3 or len(rma_50_15m) < 3 or 
            len(upw) < 3 or len(dnw) < 3):
            print(f"⚠️  {pair_name}: Indicators produced insufficient data (<3 points)")
            return last_alerts.get(pair_name), '\n'.join(thread_log)
        open_prev = df_15m['open'].iloc[-2]
        close_prev = df_15m['close'].iloc[-2]
        high_prev = df_15m['high'].iloc[-2]
        low_prev = df_15m['low'].iloc[-2]
        ppo_curr = ppo.iloc[-2] 
        rma_50_15m_curr = rma_50_15m.iloc[-2]
        upw_curr = upw.iloc[-2]
        dnw_curr = dnw.iloc[-2]
        if (np.isnan(ppo_curr) or np.isnan(rma_50_15m_curr) or 
            pd.isna(upw_curr) or pd.isna(dnw_curr)):
            log(f"Skipping {pair_name}: NaN values detected in 15m indicators")
            return last_alerts.get(pair_name), '\n'.join(thread_log)
        if close_prev <= 0 or open_prev <= 0:
            log(f"Skipping {pair_name}: Invalid price data")
            return last_alerts.get(pair_name), '\n'.join(thread_log)
        log(f"15m PPO: {ppo_curr:.4f}, RMA 50: {rma_50_15m_curr:.2f}")
        magical_hist = calculate_magical_momentum_hist(df_15m, period=144, responsiveness=0.9)
        magical_hist_curr = magical_hist.iloc[-2]
        if np.isnan(magical_hist_curr):
            log(f"⚠️  NaN in Magical Momentum Hist for {pair_name}, skipping")
            return last_alerts.get(pair_name), '\n'.join(thread_log)
        log(f"Magical Momentum Hist (15m): {magical_hist_curr:.6f}")
        limit_5m = SPECIAL_PAIRS.get(pair_name, {}).get("limit_5m", 500)
        min_required_5m = max(SPECIAL_PAIRS.get(pair_name, {}).get("min_required_5m", 250), 250)
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
        is_green = close_prev > open_prev
        is_red = close_prev < open_prev
        if rma_200_available:
            rma_long_ok = rma_50_15m_curr < close_prev and rma_200_5m_curr < close_prev
            rma_short_ok = rma_50_15m_curr > close_prev and rma_200_5m_curr > close_prev
            log(f"Using both RMA checks (50 15m + 200 5m)")
        else:
            rma_long_ok = rma_50_15m_curr < close_prev
            rma_short_ok = rma_50_15m_curr > close_prev
            log(f"Using only RMA 50 15m (RMA 200 5m unavailable)")
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
        price_change_pct = abs((close_prev - open_prev) / open_prev) * 100
        if price_change_pct > 10:
            log(f"⚠️  Extreme volatility detected ({price_change_pct:.2f}%), skipping signal")
            return last_alerts.get(pair_name), '\n'.join(thread_log)
        if close_prev > pivots['R3']:
            log(f"⚠️  Price closed above R3 (${pivots['R3']:.2f}), blocking LONG signal")
            return last_alerts.get(pair_name), '\n'.join(thread_log)
        if close_prev < pivots['S3']:
            log(f"⚠️  Price closed below S3 (${pivots['S3']:.2f}), blocking SHORT signal")
            return last_alerts.get(pair_name), '\n'.join(thread_log)
        long_pivot_lines = {'P': pivots['P'], 'R1': pivots['R1'], 'R2': pivots['R2'], 'S1': pivots['S1'], 'S2': pivots['S2']}
        long_crossover_line = False
        long_crossover_name = None
        if is_green:
            for name, line in long_pivot_lines.items():
                if open_prev <= line and close_prev > line: 
                    long_crossover_line = line
                    long_crossover_name = name
                    break
        short_pivot_lines = {'P': pivots['P'], 'S1': pivots['S1'], 'S2': pivots['S2'], 'R1': pivots['R1'], 'R2': pivots['R2']}
        short_crossover_line = False
        short_crossover_name = None
        if is_red:
            for name, line in short_pivot_lines.items():
                if open_prev >= line and close_prev < line:
                    short_crossover_line = line
                    short_crossover_name = name
                    break
        current_signal = None
        updated_state = last_alerts.get(pair_name) 
        if updated_state:
            log(f"Previous state loaded: {updated_state}")
        ist = pytz.timezone('Asia/Kolkata')
        formatted_time = datetime.now(ist).strftime('%d-%m-%Y @ %H:%M IST')
        price = close_prev
        if updated_state and updated_state.startswith('fib_'):
            try:
                parts = updated_state.split('_')
                if len(parts) >= 3:
                    alert_type = parts[1]
                    pivot_name = parts[2]
                    pivot_value = pivots.get(pivot_name)
                    log(f"Checking state reset: type={alert_type}, pivot={pivot_name} (${pivot_value}), current_price=${close_prev:,.2f}")
                    if pivot_value is not None:
                        if alert_type == "long":
                            if close_prev < pivot_value:
                                updated_state = None
                                log(f"🔄 STATE RESET: {pair_name} Long - Price ${close_prev:,.2f} fell below {pivot_name} ${pivot_value:,.2f}")
                            else:
                                log(f"State maintained: Price ${close_prev:,.2f} still above {pivot_name} ${pivot_value:,.2f}")
                        elif alert_type == "short":
                            if close_prev > pivot_value:
                                updated_state = None
                                log(f"🔄 STATE RESET: {pair_name} Short - Price ${close_prev:,.2f} rose above {pivot_name} ${pivot_value:,.2f}")
                            else:
                                log(f"State maintained: Price ${close_prev:,.2f} still below {pivot_name} ${pivot_value:,.2f}")
                    else:
                        log(f"⚠️  Invalid pivot name '{pivot_name}' in state, resetting")
                        updated_state = None
            except Exception as e:
                log(f"Error parsing saved state '{updated_state}': {e}")
                updated_state = None
        # 🟢 LONG SIGNAL
        if (upw_curr and (not dnw_curr) and 
            (ppo_curr < 0.20) and 
            long_crossover_line and 
            upper_wick_check and
            rma_long_ok and
            magical_hist_curr > 0):
            current_signal = f"fib_long_{long_crossover_name}"
            log(f"\n🟢 FIB LONG SIGNAL DETECTED for {pair_name}!")
            if updated_state != current_signal:
                message = (
                    f"🟢 {pair_name} - <b>FIB LONG</b>\n"
                    f"Crossed & Closed Above <b>{long_crossover_name}</b> (${long_crossover_line:,.2f})\n"
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
              rma_short_ok and
              magical_hist_curr < 0):
            current_signal = f"fib_short_{short_crossover_name}"
            log(f"\n🔴 FIB SHORT SIGNAL DETECTED for {pair_name}!")
            if updated_state != current_signal:
                message = (
                    f"🔴 {pair_name} - <b>FIB SHORT</b>\n"
                    f"Crossed & Closed Below <b>{short_crossover_name}</b> (${short_crossover_line:,.2f})\n"
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
    print("=" * 50)
    ist = pytz.timezone('Asia/Kolkata')
    start_time = datetime.now(ist)
    print(f"Fibonacci Pivot Alert Bot - {start_time.strftime('%d-%m-%Y @ %H:%M IST')}")
    print(f"Debug Mode: {'ON' if DEBUG_MODE else 'OFF'}")
    print("=" * 50)
    if SEND_TEST_MESSAGE:
        send_test_message()
    if RESET_STATE and os.path.exists(STATE_FILE):
        print(f"ATTENTION: \nRESET_STATE is True. Deleting {STATE_FILE} to clear previous alerts.")
        os.remove(STATE_FILE)
    last_alerts = load_state()
    keys_to_purge = []
    for key, value in list(last_alerts.items()):
        if value:
            value_str = str(value).lower()
            if not (value_str.startswith('fib_long_') or value_str.startswith('fib_short_')):
                keys_to_purge.append(key)
    if keys_to_purge:
        print(f"\n[INFO] 🧹 Purging old/invalid states for: {', '.join(keys_to_purge)}")
        for key in keys_to_purge:
            last_alerts[key] = None
    if not get_product_ids():
        print("Failed to fetch products. Exiting.")
        return
    found_count = sum(1 for v in PAIRS.values() if v is not None)
    print(f"✓ Monitoring {found_count} pairs")
    if found_count == 0:
        print("No valid pairs found. Exiting.")
        return
    updates_processed = 0
    pair_logs = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_pair = {}
        for pair_name, pair_info in PAIRS.items():
            if pair_info is not None:
                future = executor.submit(check_pair, pair_name, pair_info, last_alerts)
                future_to_pair[future] = pair_name
        for future in as_completed(future_to_pair):
            pair_name = future_to_pair[future]
            try:
                new_state, log_output = future.result()
                pair_logs.append(log_output)
                if new_state != last_alerts.get(pair_name):
                    last_alerts[pair_name] = new_state
                    updates_processed += 1
            except Exception as e:
                print(f"Error processing {pair_name} in thread: {e}")
                if DEBUG_MODE:
                    import traceback
                    traceback.print_exc()
                continue
    if DEBUG_MODE:
        print("\n" + "="*50)
        print("SEQUENTIAL DEBUG LOGS START")
        print("="*50)
        for log_output in pair_logs:
            print(log_output, end='')
        print("\n" + "="*50)
        print("SEQUENTIAL DEBUG LOGS END")
        print("="*50)
    save_state(last_alerts)
    end_time = datetime.now(ist)
    elapsed = (end_time - start_time).total_seconds()
    print(f"✓ Check complete. {updates_processed} state updates processed. ({elapsed:.1f}s)")
    print("=" * 50)

if __name__ == "__main__":
    main()
