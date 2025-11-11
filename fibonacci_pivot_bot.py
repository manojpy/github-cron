import requests
import pandas as pd
import numpy as np
import time
import os
import json
import pytz
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============ CONFIGURATION ============
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '8462496498:AAHYZ4xDIHvrVRjmCmZyoPhupCjRaRgiITc')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '203813932')

if TELEGRAM_BOT_TOKEN == 'xxxx' or TELEGRAM_CHAT_ID == 'xxxx':
    raise RuntimeError("❌ TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in GitHub Secrets.")

DEBUG_MODE = os.environ.get('DEBUG_MODE', 'True').lower() == 'true'
SEND_TEST_MESSAGE = os.environ.get('SEND_TEST_MESSAGE', 'True').lower() == 'true'
RESET_STATE = os.environ.get('RESET_STATE', 'False').lower() == 'true'

DELTA_API_BASE = "https://api.delta.exchange"

PAIRS = {
    "BTCUSD": None, "ETHUSD": None,
    "SOLUSD": None, "AVAXUSD": None,
    "BCHUSD": None, "XRPUSD": None, "BNBUSD": None, "LTCUSD": None,
    "DOTUSD": None, "ADAUSD": None, "SUIUSD": None, "AAVEUSD": None
}

SPECIAL_PAIRS = {
    "SOLUSD": {"limit_15m": 210, "min_required": 140, "limit_5m": 450, "min_required_5m": 220}
}

PPO_FAST = 7
PPO_SLOW = 16
PPO_SIGNAL = 5
PPO_USE_SMA = False

X1 = 22
X2 = 9
X3 = 15
X4 = 5

PIVOT_LOOKBACK_PERIOD = 15
# State file paths (read from environment, fallback to defaults)
STATE_FILE = os.environ.get("STATE_FILE_PATH", "fib_state.json")
STATE_FILE_BAK = STATE_FILE + ".bak"


# ============ HTTP SESSION WITH RETRIES ============
def create_session_with_retries():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


# ============ VWAP WITH DAILY RESET (00:00 UTC = 5:30 IST) ============
def calculate_vwap_daily_reset(df):
    """
    Calculate VWAP reset at 00:00 UTC each day using hlc3.
    Matches Pine Script behavior and aligns with 5:30 AM IST reset.
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df['date'] = df['datetime'].dt.date
    hlc3 = (df['high'] + df['low'] + df['close']) / 3.0
    df['hlc3_vol'] = hlc3 * df['volume']
    df['cum_vol'] = df.groupby('date')['volume'].cumsum()
    df['cum_hlc3_vol'] = df.groupby('date')['hlc3_vol'].cumsum()
    df['vwap'] = df['cum_hlc3_vol'] / df['cum_vol'].replace(0, np.nan)
    return df['vwap']


# ============ UTILITY FUNCTIONS ============
def load_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                content = f.read().strip()
                if not content:
                    if DEBUG_MODE:
                        print("[DEBUG] State file is empty. Starting fresh.")
                    return {}
                state = json.loads(content)
                if DEBUG_MODE:
                    print(f"[DEBUG] Loaded state: {state}")
                return state
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WARN] State file corrupted or unreadable: {e}. Starting fresh.")
    if DEBUG_MODE:
        print("[DEBUG] No valid previous state found, starting fresh")
    return {}


def save_state(state):
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
        if DEBUG_MODE:
            print(f"[DEBUG] Saved state: {state}")
    except Exception as e:
        print(f"Error saving state: {e}")


def send_telegram_alert(message):
    try:
        if DEBUG_MODE:
            print(f"[DEBUG] Attempting to send message: {message[:100]}...")
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": None
        }
        session = create_session_with_retries()
        response = session.post(url, data=data, timeout=10)
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
    ist = pytz.timezone('Asia/Kolkata')
    current_dt = datetime.now(ist)
    formatted_time = current_dt.strftime('%d-%m-%Y @ %H:%M IST')
    test_msg = f"""🔔 Fibonacci Pivot Bot Started
Test message from Fibonacci Pivot Bot
Time: {formatted_time}
Debug Mode: {'ON' if DEBUG_MODE else 'OFF'}"""
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
    try:
        if DEBUG_MODE:
            print("[DEBUG] Fetching product IDs from Delta Exchange...")
        session = create_session_with_retries()
        response = session.get(f"{DELTA_API_BASE}/v2/products", timeout=10)
        data = response.json()
        if data.get('success'):
            products = data['result']
            if DEBUG_MODE:
                print(f"[DEBUG] Received {len(products)} products from API")
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
                            if DEBUG_MODE:
                                print(f"[DEBUG] Matched {pair_name} -> {product['symbol']} (ID: {product['id']})")
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
        session = create_session_with_retries()
        response = session.get(url, params=params, timeout=15)
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
            df = df.sort_values('timestamp').reset_index(drop=True)
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
        if curr_x > prev_f:
            f = prev_f if (curr_x - curr_r < prev_f) else (curr_x - curr_r)
        else:
            f = prev_f if (curr_x + curr_r > prev_f) else (curr_x + curr_r)
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


def calculate_magical_momentum_hist(df, period=144, responsiveness=0.9):
    if len(df) < period + 50:
        return pd.Series([np.nan] * len(df), index=df.index)
    close = df['close'].astype(float)
    if close.isnull().all():
        return pd.Series([np.nan] * len(df), index=df.index)
    responsiveness = max(0.00001, responsiveness)
    sd = close.rolling(window=50).std() * responsiveness
    sd = sd.bfill().fillna(0.001)
    worm = close.copy()
    worm.iloc[0] = close.iloc[0]
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
    if len(value) > 0:
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
    if len(hist) > 0:
        hist.iloc[0] = momentum.iloc[0]
        for i in range(1, len(momentum)):
            hist.iloc[i] = momentum.iloc[i] + 0.5 * hist.iloc[i - 1]
    return hist


def get_previous_day_ohlc(product_id, days_back_limit=15):
    df_daily = get_candles(product_id, resolution="D", limit=days_back_limit + 5)
    if df_daily is None or len(df_daily) < 2:
        return None

    df_daily['date'] = pd.to_datetime(df_daily['timestamp'], unit='s', utc=True).dt.date
    today = datetime.now(timezone.utc).date()
    yesterday = today - pd.Timedelta(days=1)
    prev_day_row = df_daily[df_daily['date'] == yesterday]
    if not prev_day_row.empty:
        prev_day_candle = prev_day_row.iloc[-1]
        return {
            'high': prev_day_candle['high'],
            'low': prev_day_candle['low'],
            'close': prev_day_candle['close']
        }
    else:
        return {
            'high': df_daily.iloc[-2]['high'],
            'low': df_daily.iloc[-2]['low'],
            'close': df_daily.iloc[-2]['close']
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
        min_required = SPECIAL_PAIRS.get(pair_name, {}).get("min_required", 150)
        df_15m = get_candles(pair_info['symbol'], "15", limit=limit_15m)
        if df_15m is None:
            print(f"❌ {pair_name}: Failed to fetch 15m candle data")
            return last_alerts.get(pair_name), '\n'.join(thread_log)

        if len(df_15m) < min_required:
            print(f"⚠️  {pair_name}: Insufficient 15m data (need {min_required}, got {len(df_15m)})")
            return last_alerts.get(pair_name), '\n'.join(thread_log)

        # ========== VWAP with DAILY RESET (00:00 UTC = 5:30 IST) ==========
        vwap_15m = calculate_vwap_daily_reset(df_15m)
        vwap_curr = vwap_15m.iloc[-2]
        if np.isnan(vwap_curr):
            log("⚠️  VWAP is NaN, skipping VBuy/VSel signals")
            vwap_curr = None
        else:
            log(f"15m VWAP (UTC-reset = 5:30 AM IST): {vwap_curr:.4f}")

        ppo, _ = calculate_ppo(df_15m, PPO_FAST, PPO_SLOW, PPO_SIGNAL, PPO_USE_SMA)
        upw, dnw, _, _ = calculate_cirrus_cloud(df_15m)
        rma_50_15m = calculate_rma(df_15m['close'], 50)

        if (len(ppo) < 3 or len(rma_50_15m) < 3 or len(upw) < 3 or len(dnw) < 3):
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

        if (np.isnan(ppo_curr) or np.isnan(rma_50_15m_curr) or pd.isna(upw_curr) or pd.isna(dnw_curr)):
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
        min_required_5m = SPECIAL_PAIRS.get(pair_name, {}).get("min_required_5m", 250)
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
        long_crossover_line = None
        long_crossover_name = None
        if is_green:
            for name, line in long_pivot_lines.items():
                if open_prev <= line and close_prev > line:
                    long_crossover_line = line
                    long_crossover_name = name
                    break

        short_pivot_lines = {'P': pivots['P'], 'S1': pivots['S1'], 'S2': pivots['S2'], 'R1': pivots['R1'], 'R2': pivots['R2']}
        short_crossover_line = None
        short_crossover_name = None
        if is_red:
            for name, line in short_pivot_lines.items():
                if open_prev >= line and close_prev < line:
                    short_crossover_line = line
                    short_crossover_name = name
                    break

        base_long_ok = (
            upw_curr and (not dnw_curr) and
            upper_wick_check and
            rma_long_ok and
            magical_hist_curr > 0 and
            is_green
        )
        base_short_ok = (
            dnw_curr and (not upw_curr) and
            lower_wick_check and
            rma_short_ok and
            magical_hist_curr < 0 and
            is_red
        )

        # 🔔 VWAP SIGNALS — REQUIRE CROSSOVER + CLOSE (Pine Script style)
        vbuy_conditions_met = False
        vsell_conditions_met = False

        if vwap_curr is not None and len(df_15m) >= 3:
            close_prev2 = df_15m['close'].iloc[-3]  # previous closed candle
            vwap_prev = vwap_15m.iloc[-3]           # VWAP of that candle

            if not (np.isnan(close_prev2) or np.isnan(vwap_prev)):
                # VBuy: crossed UP through VWAP and closed above
                if base_long_ok and (close_prev2 <= vwap_prev) and (close_prev > vwap_curr):
                    vbuy_conditions_met = True
                # VSell: crossed DOWN through VWAP and closed below
                if base_short_ok and (close_prev2 >= vwap_prev) and (close_prev < vwap_curr):
                    vsell_conditions_met = True

        # 🔔 Original Fib signals (with crossover)
        fib_long_met = base_long_ok and long_crossover_line is not None
        fib_short_met = base_short_ok and short_crossover_line is not None

        current_signal = None
        updated_state = last_alerts.get(pair_name)
        if updated_state:
            log(f"Previous state loaded: {updated_state}")

        ist = pytz.timezone('Asia/Kolkata')
        formatted_time = datetime.now(ist).strftime('%d-%m-%Y @ %H:%M IST')
        price = close_prev

        # 🔁 State reset logic with validation
        if updated_state:
            if updated_state.startswith('fib_'):
                try:
                    parts = updated_state.split('_')
                    if len(parts) >= 3 and parts[2] in ['P', 'R1', 'R2', 'S1', 'S2']:
                        alert_type = parts[1]
                        pivot_name = parts[2]
                        pivot_value = pivots.get(pivot_name)
                        if pivot_value is not None:
                            if alert_type == "long" and close_prev < pivot_value:
                                updated_state = None
                                log(f"🔄 STATE RESET: {pair_name} Long - Price fell below {pivot_name}")
                            elif alert_type == "short" and close_prev > pivot_value:
                                updated_state = None
                                log(f"🔄 STATE RESET: {pair_name} Short - Price rose above {pivot_name}")
                        else:
                            updated_state = None
                except Exception as e:
                    log(f"Error parsing fib state: {e}")
                    updated_state = None
            elif updated_state == 'vbuy' and close_prev < vwap_curr:
                updated_state = None
                log(f"🔄 STATE RESET: {pair_name} VBuy - Price closed below VWAP")
            elif updated_state == 'vsell' and close_prev > vwap_curr:
                updated_state = None
                log(f"🔄 STATE RESET: {pair_name} VSell - Price closed above VWAP")

        # 🔔 Handle signals
        if vbuy_conditions_met:
            current_signal = 'vbuy'
            log(f"\n🔵 VBuy SIGNAL DETECTED for {pair_name}!")
            if updated_state != current_signal:
                diagnostic = f"[O:{open_prev:.5f},C:{close_prev:.5f},VWAP:{vwap_curr:.5f}]"
                message = (
                    f"🔵 {pair_name} - **VBuy**\n"
                    f"Crossed & Closed Above VWAP (${vwap_curr:,.2f}) {diagnostic}\n"
                    f"PPO 15m: {ppo_curr:.2f}\n"
                    f"Price: ${price:,.2f}\n"
                    f"{formatted_time}"
                )
                send_telegram_alert(message)
            updated_state = current_signal

        elif vsell_conditions_met:
            current_signal = 'vsell'
            log(f"\n🟠 VSell SIGNAL DETECTED for {pair_name}!")
            if updated_state != current_signal:
                diagnostic = f"[O:{open_prev:.5f},C:{close_prev:.5f},VWAP:{vwap_curr:.5f}]"
                message = (
                    f"🟠 {pair_name} - **VSell**\n"
                    f"Crossed & Closed Below VWAP (${vwap_curr:,.2f}) {diagnostic}\n"
                    f"PPO 15m: {ppo_curr:.2f}\n"
                    f"Price: ${price:,.2f}\n"
                    f"{formatted_time}"
                )
                send_telegram_alert(message)
            updated_state = current_signal

        elif fib_long_met:
            current_signal = f"fib_long_{long_crossover_name}"
            log(f"\n🟢 FIB LONG SIGNAL DETECTED for {pair_name}!")
            if updated_state != current_signal:
                diagnostic = f"[O:{open_prev:.5f},C:{close_prev:.5f},R2:{pivots['R2']:.5f}]"
                message = (
                    f"🟢 {pair_name} - **FIB LONG**\n"
                    f"Crossed & Closed Above **{long_crossover_name}** (${long_crossover_line:,.2f}) {diagnostic}\n"
                    f"PPO 15m: {ppo_curr:.2f}\n"
                    f"Price: ${price:,.2f}\n"
                    f"{formatted_time}"
                )
                send_telegram_alert(message)
            updated_state = current_signal

        elif fib_short_met:
            current_signal = f"fib_short_{short_crossover_name}"
            log(f"\n🔴 FIB SHORT SIGNAL DETECTED for {pair_name}!")
            if updated_state != current_signal:
                diagnostic = f"[O:{open_prev:.5f},C:{close_prev:.5f},S2:{pivots['S2']:.5f}]"
                message = (
                    f"🔴 {pair_name} - **FIB SHORT**\n"
                    f"Crossed & Closed Below **{short_crossover_name}** (${short_crossover_line:,.2f}) {diagnostic}\n"
                    f"PPO 15m: {ppo_curr:.2f}\n"
                    f"Price: ${price:,.2f}\n"
                    f"{formatted_time}"
                )
                send_telegram_alert(message)
            updated_state = current_signal

        else:
            log(f"No signal conditions met for {pair_name}")

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
            if not (value_str.startswith('fib_long_') or value_str.startswith('fib_short_') or value_str in ['vbuy', 'vsell']):
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
    with ThreadPoolExecutor(max_workers=6) as executor:  # reduced from 10
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
