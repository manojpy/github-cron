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
STATE_FILE = os.environ.get("STATE_FILE_PATH", "fib_state.json")
STATE_FILE_BAK = STATE_FILE + ".bak"

# ============ LOGGING ============
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ============ HTTP SESSION WITH RETRIES ============
def create_session_with_retries():
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(['GET', 'POST'])
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# ============ VWAP WITH DAILY RESET (00:00 UTC = 5:30 IST) ============
def calculate_vwap_daily_reset(df):
    """
    Calculate VWAP reset at 00:00 UTC each day using hlc3.
    Matches Pine Script behavior and aligns with 5:30 AM IST reset.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    df = df.copy()
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df['date'] = df['datetime'].dt.date
    hlc3 = (df['high'] + df['low'] + df['close']) / 3.0
    df['hlc3_vol'] = hlc3 * df['volume']
    df['cum_vol'] = df.groupby('date')['volume'].cumsum()
    df['cum_hlc3_vol'] = df.groupby('date')['hlc3_vol'].cumsum()
    vwap = df['cum_hlc3_vol'] / df['cum_vol'].replace(0, np.nan)

    # ✅ Updated to use ffill/bfill instead of deprecated fillna(method=...)
    return vwap.ffill().bfill()


# ============ STATE MANAGEMENT ============
def load_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                content = f.read().strip()
                if not content:
                    logger.debug("State file empty. Starting fresh.")
                    return {}
                state = json.loads(content)
                logger.debug(f"Loaded state: {state}")
                return state
    except Exception as e:
        logger.warning(f"State file corrupted/unreadable: {e}. Attempting backup.")
        try:
            if os.path.exists(STATE_FILE_BAK):
                with open(STATE_FILE_BAK, 'r') as f:
                    return json.loads(f.read())
        except Exception as e2:
            logger.warning(f"Backup state also invalid: {e2}. Starting fresh.")
    return {}

def save_state(state):
    try:
        with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
            json.dump(state, tmp)
            temp_name = tmp.name
        shutil.move(temp_name, STATE_FILE)
        try:
            shutil.copy(STATE_FILE, STATE_FILE_BAK)
        except Exception as e:
            logger.debug(f"Could not write backup state: {e}")
        logger.debug(f"Saved state: {state}")
    except Exception as e:
        logger.error(f"Error saving state: {e}")

# ============ TELEGRAM ALERTS ============
def send_telegram_alert(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        session = create_session_with_retries()
        response = session.post(url, data=data, timeout=10)
        try:
            response_data = response.json()
        except ValueError:
            logger.error(f"Telegram non-JSON response: {response.text}")
            return False
        if response_data.get('ok'):
            logger.info("Alert sent successfully")
            time.sleep(0.8)  # gentle rate-limiting
            return True
        logger.error(f"Telegram error: {response_data}")
        return False
    except Exception as e:
        logger.exception(f"Error sending Telegram message: {e}")
        return False

def send_test_message():
    ist = pytz.timezone('Asia/Kolkata')
    current_dt = datetime.now(ist)
    formatted_time = current_dt.strftime('%d-%m-%Y @ %H:%M IST')
    test_msg = f"""🔔 Fibonacci Pivot Bot Started
Test message from Fibonacci Pivot Bot
Time: {formatted_time}
Debug Mode: {'ON' if DEBUG_MODE else 'OFF'}"""
    logger.info("Sending test message...")
    success = send_telegram_alert(test_msg)
    if not success:
        logger.warning("Test message failed - check bot token and chat ID")
    return success

# ============ API HELPERS ============
def safe_json(response, context=""):
    try:
        return response.json()
    except ValueError:
        logger.error(f"Invalid JSON response in {context}: {getattr(response, 'text', '')[:200]}")
        return {}

def get_product_ids():
    try:
        logger.debug("Fetching product IDs from Delta Exchange...")
        session = create_session_with_retries()
        response = session.get(f"{DELTA_API_BASE}/v2/products", timeout=10)
        data = safe_json(response, context="get_product_ids")
        if data.get('success'):
            products = data.get('result', [])
            logger.debug(f"Received {len(products)} products from API")
            for product in products:
                symbol = product.get('symbol', '').replace('_USDT', 'USD').replace('USDT', 'USD')
                if product.get('contract_type') == 'perpetual_futures':
                    for pair_name in PAIRS.keys():
                        if symbol == pair_name or symbol.replace('_', '') == pair_name:
                            PAIRS[pair_name] = {
                                'id': product.get('id'),
                                'symbol': product.get('symbol'),
                                'contract_type': product.get('contract_type')
                            }
                            logger.debug(f"Matched {pair_name} -> {product.get('symbol')} (ID: {product.get('id')})")
            return True
        logger.error(f"API Error in get_product_ids: {data}")
        return False
    except Exception as e:
        logger.exception(f"Error fetching products: {e}")
        return False

def get_candles(product_id, resolution="15", limit=150):
    try:
        if not product_id:
            return None
        to_time = int(time.time())
        if resolution == "D":
            from_time = to_time - (limit * 24 * 60 * 60)
        else:
            res_minutes = int(resolution)
            from_time = to_time - (limit * res_minutes * 60)
        url = f"{DELTA_API_BASE}/v2/chart/history"
        params = {'resolution': resolution, 'symbol': product_id, 'from': from_time, 'to': to_time}
        session = create_session_with_retries()
        response = session.get(url, params=params, timeout=15)
        data = safe_json(response, context=f"get_candles({product_id},{resolution})")
        if data.get('success') and data.get('result'):
            result = data['result']
            # Validate keys exist
            for k in ['t', 'o', 'h', 'l', 'c', 'v']:
                if k not in result:
                    logger.error(f"Missing key {k} in candle result for {product_id}")
                    return None
            df = pd.DataFrame({
                'timestamp': result['t'],
                'open': result['o'],
                'high': result['h'],
                'low': result['l'],
                'close': result['c'],
                'volume': result['v']
            })
            df = df.sort_values('timestamp').reset_index(drop=True)
            # Basic sanity checks
            if df.empty or df[['open', 'high', 'low', 'close']].isnull().any().any():
                logger.warning(f"NaNs in candle data for {product_id} {resolution}m")
            return df
        logger.warning(f"No candle data for {product_id} {resolution}m: {data}")
        return None
    except Exception as e:
        logger.exception(f"Exception fetching candles for {product_id}: {e}")
        return None

# ============ INDICATORS ============
def calculate_ema(data, period):
    if len(data) < 1 or period <= 0:
        return pd.Series([np.nan] * len(data), index=data.index)
    return data.ewm(span=period, adjust=False).mean()

def calculate_sma(data, period):
    if len(data) < period or period <= 0:
        return pd.Series([np.nan] * len(data), index=data.index)
    return data.rolling(window=period, min_periods=period).mean()

def calculate_ppo(df, fast=7, slow=16, signal=5, use_sma=False):
    close = df['close'].astype(float)
    fast_ma = calculate_sma(close, fast) if use_sma else calculate_ema(close, fast)
    slow_ma = calculate_sma(close, slow) if use_sma else calculate_ema(close, slow)
    with np.errstate(divide='ignore', invalid='ignore'):
        ppo = (fast_ma - slow_ma) / slow_ma * 100
    ppo = ppo.replace([np.inf, -np.inf], np.nan)
    ppo_signal = calculate_sma(ppo, signal) if use_sma else calculate_ema(ppo, signal)
    return ppo, ppo_signal

def smoothrng(x, t, m):
    wper = t * 2 - 1
    avrng = calculate_ema(np.abs(x.diff()), t)
    val = calculate_ema(avrng, wper) * m

    # ✅ Updated to use ffill/bfill instead of deprecated fillna(method=...)
    return val.bfill().ffill()

def rngfilt(x, r):
    if len(x) == 0:
        return pd.Series(dtype=float)
    result_list = [x.iloc[0]]
    for i in range(1, len(x)):
        prev_f = result_list[-1]
        curr_x = x.iloc[i]
        curr_r = r.iloc[i] if not np.isnan(r.iloc[i]) else 0
        if curr_x > prev_f:
            f = prev_f if (curr_x - curr_r < prev_f) else (curr_x - curr_r)
        else:
            f = prev_f if (curr_x + curr_r > prev_f) else (curr_x + curr_r)
        result_list.append(f)
    return pd.Series(result_list, index=x.index)

def calculate_cirrus_cloud(df):
    close = df['close'].astype(float).copy()
    smrngx1x = smoothrng(close, X1, X2)
    smrngx1x2 = smoothrng(close, X3, X4)
    filtx1 = rngfilt(close, smrngx1x)
    filtx12 = rngfilt(close, smrngx1x2)
    upw = filtx1 < filtx12
    dnw = filtx1 > filtx12
    return upw, dnw, filtx1, filtx12

def calculate_rma(data, period):
    if period <= 0:
        return pd.Series([np.nan] * len(data), index=data.index)
    return data.ewm(alpha=1/period, adjust=False).mean()

def calculate_magical_momentum_hist(df, period=144, responsiveness=0.9):
    if len(df) < period + 50:
        return pd.Series([np.nan] * len(df), index=df.index)
    close = df['close'].astype(float)
    if close.isnull().all():
        return pd.Series([np.nan] * len(df), index=df.index)
    responsiveness = max(0.00001, responsiveness)
    sd = close.rolling(window=50, min_periods=1).std() * responsiveness
    sd = sd.bfill().fillna(0.001)
    worm = close.copy()
    worm.iloc[0] = close.iloc[0]
    for i in range(1, len(close)):
        diff = close.iloc[i] - worm.iloc[i - 1]
        abs_diff = abs(diff)
        delta = np.sign(diff) * sd.iloc[i] if abs_diff > sd.iloc[i] else diff
        worm.iloc[i] = worm.iloc[i - 1] + delta
    ma = close.rolling(window=period, min_periods=1).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        raw_momentum = (worm - ma) / worm.replace(0, np.nan)
    raw_momentum = raw_momentum.fillna(0)
    min_med = raw_momentum.rolling(window=period, min_periods=1).min()
    max_med = raw_momentum.rolling(window=period, min_periods=1).max()
    rng = max_med - min_med
    temp = pd.Series(0.0, index=df.index)
    valid = rng > 1e-10
    temp.loc[valid] = (raw_momentum.loc[valid] - min_med.loc[valid]) / rng.loc[valid]
    value = pd.Series(0.0, index=df.index)
    if len(value) > 0:
        value.iloc[0] = 0.0
        for i in range(1, len(temp)):
            v_prev = value.iloc[i - 1]
            v_new = (temp.iloc[i] - 0.5 + 0.5 * v_prev)
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

# ============ PIVOTS ============
def get_previous_day_ohlc(product_id, days_back_limit=15):
    df_daily = get_candles(product_id, resolution="D", limit=days_back_limit + 5)
    if df_daily is None or len(df_daily) < 2:
        return None
    df_daily['date'] = pd.to_datetime(df_daily['timestamp'], unit='s', utc=True).dt.date
    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)
    prev_day_row = df_daily[df_daily['date'] == yesterday]
    if not prev_day_row.empty:
        prev_day_candle = prev_day_row.iloc[-1]
    else:
        prev_day_candle = df_daily.iloc[-2]
    return {
        'high': float(prev_day_candle['high']),
        'low': float(prev_day_candle['low']),
        'close': float(prev_day_candle['close'])
    }

def calculate_fibonacci_pivots(h, l, c):
    pivot = (h + l + c) / 3.0
    diff = h - l
    r3 = pivot + (diff * 1.000)
    r2 = pivot + (diff * 0.618)
    r1 = pivot + (diff * 0.382)
    s1 = pivot - (diff * 0.382)
    s2 = pivot - (diff * 0.618)
    s3 = pivot - (diff * 1.000)
    return {'P': pivot, 'R1': r1, 'R2': r2, 'R3': r3, 'S1': s1, 'S2': s2, 'S3': s3}

# ============ CORE CHECK ============
def check_pair(pair_name, pair_info, last_alerts):
    thread_log = []

    def log(message):
        if DEBUG_MODE:
            thread_log.append(f"[DEBUG] {message}")

    try:
        if pair_info is None:
            return last_alerts.get(pair_name), ""

        log("=" * 60)
        log(f"Checking {pair_name} for Fibonacci Pivot Alerts")
        log("=" * 60)

        prev_day_ohlc = get_previous_day_ohlc(pair_info['symbol'], PIVOT_LOOKBACK_PERIOD)
        if prev_day_ohlc is None:
            logger.warning(f"{pair_name}: Failed to fetch previous day OHLC data")
            return last_alerts.get(pair_name), '\n'.join(thread_log)

        pivots = calculate_fibonacci_pivots(prev_day_ohlc['high'], prev_day_ohlc['low'], prev_day_ohlc['close'])
        log(f"Daily Pivots: P={pivots['P']:.2f}, R1={pivots['R1']:.2f}, R2={pivots['R2']:.2f}, R3={pivots['R3']:.2f}")
        log(f"             S1={pivots['S1']:.2f}, S2={pivots['S2']:.2f}, S3={pivots['S3']:.2f}")

        limit_15m = SPECIAL_PAIRS.get(pair_name, {}).get("limit_15m", 250)
        min_required = SPECIAL_PAIRS.get(pair_name, {}).get("min_required", 150)
        df_15m = get_candles(pair_info['symbol'], "15", limit=limit_15m)
        if df_15m is None:
            logger.warning(f"{pair_name}: Failed to fetch 15m candle data")
            return last_alerts.get(pair_name), '\n'.join(thread_log)

        if len(df_15m) < min_required:
            logger.warning(f"{pair_name}: Insufficient 15m data (need {min_required}, got {len(df_15m)})")
            return last_alerts.get(pair_name), '\n'.join(thread_log)

        # VWAP with DAILY RESET
        vwap_15m = calculate_vwap_daily_reset(df_15m)
        vwap_curr = vwap_15m.iloc[-2] if len(vwap_15m) >= 2 else np.nan
        if np.isnan(vwap_curr):
            log("⚠️  VWAP is NaN, skipping VBuy/VSell signals")
            vwap_curr = None
        else:
            log(f"15m VWAP (UTC-reset = 5:30 AM IST): {vwap_curr:.4f}")

        ppo, _ = calculate_ppo(df_15m, PPO_FAST, PPO_SLOW, PPO_SIGNAL, PPO_USE_SMA)
        upw, dnw, _, _ = calculate_cirrus_cloud(df_15m)
        rma_50_15m = calculate_rma(df_15m['close'], 50)

        if (len(ppo) < 3 or len(rma_50_15m) < 3 or len(upw) < 3 or len(dnw) < 3):
            logger.warning(f"{pair_name}: Indicators produced insufficient data (<3 points)")
            return last_alerts.get(pair_name), '\n'.join(thread_log)

        open_prev = float(df_15m['open'].iloc[-2])
        close_prev = float(df_15m['close'].iloc[-2])
        high_prev = float(df_15m['high'].iloc[-2])
        low_prev = float(df_15m['low'].iloc[-2])
        ppo_curr = float(ppo.iloc[-2]) if not np.isnan(ppo.iloc[-2]) else np.nan
        rma_50_15m_curr = float(rma_50_15m.iloc[-2]) if not np.isnan(rma_50_15m.iloc[-2]) else np.nan
        upw_curr = bool(upw.iloc[-2])
        dnw_curr = bool(dnw.iloc[-2])

        if (np.isnan(ppo_curr) or np.isnan(rma_50_15m_curr)):
            log(f"Skipping {pair_name}: NaN values detected in 15m indicators")
            return last_alerts.get(pair_name), '\n'.join(thread_log)

        if close_prev <= 0 or open_prev <= 0:
            log(f"Skipping {pair_name}: Invalid price data")
            return last_alerts.get(pair_name), '\n'.join(thread_log)

        log(f"15m PPO: {ppo_curr:.4f}, RMA 50: {rma_50_15m_curr:.2f}")

        magical_hist = calculate_magical_momentum_hist(df_15m, period=144, responsiveness=0.9)
        magical_hist_curr = magical_hist.iloc[-2] if len(magical_hist) >= 2 else np.nan
        if np.isnan(magical_hist_curr):
            log(f"⚠️  NaN in Magical Momentum Hist for {pair_name}, skipping")
            return last_alerts.get(pair_name), '\n'.join(thread_log)

        log(f"Magical Momentum Hist (15m): {float(magical_hist_curr):.6f}")

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
                rma_200_5m_curr = float(rma_200_5m.iloc[-2])
                rma_200_available = True
                log(f"5m RMA 200: {rma_200_5m_curr:.4f}")
            else:
                log("5m RMA 200 calculation produced NaN - will be skipped")

        is_green = close_prev > open_prev
        is_red = close_prev < open_prev

        if rma_200_available:
            rma_long_ok = rma_50_15m_curr < close_prev and rma_200_5m_curr < close_prev
            rma_short_ok = rma_50_15m_curr > close_prev and rma_200_5m_curr > close_prev
            log("Using both RMA checks (50 15m + 200 5m)")
        else:
            rma_long_ok = rma_50_15m_curr < close_prev
            rma_short_ok = rma_50_15m_curr > close_prev
            log("Using only RMA 50 15m (RMA 200 5m unavailable)")

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
            (magical_hist_curr > 0) and
            is_green
        )
        base_short_ok = (
            dnw_curr and (not upw_curr) and
            lower_wick_check and
            rma_short_ok and
            (magical_hist_curr < 0) and
            is_red
        )

        # VWAP signals
        vbuy_conditions_met = False
        vsell_conditions_met = False
        if vwap_curr is not None and len(df_15m) >= 3:
            close_prev2 = float(df_15m['close'].iloc[-3])
            vwap_prev = float(vwap_15m.iloc[-3]) if not np.isnan(vwap_15m.iloc[-3]) else np.nan
            if not (np.isnan(close_prev2) or np.isnan(vwap_prev)):
                if base_long_ok and (close_prev2 <= vwap_prev) and (close_prev > vwap_curr):
                    vbuy_conditions_met = True
                if base_short_ok and (close_prev2 >= vwap_prev) and (close_prev < vwap_curr):
                    vsell_conditions_met = True

        fib_long_met = base_long_ok and long_crossover_line is not None
        fib_short_met = base_short_ok and short_crossover_line is not None

        current_signal = None
        updated_state = last_alerts.get(pair_name)
        if updated_state:
            log(f"Previous state loaded: {updated_state}")

        ist = pytz.timezone('Asia/Kolkata')
        formatted_time = datetime.now(ist).strftime('%d-%m-%Y @ %H:%M IST')
        price = close_prev

        # State reset logic
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
            elif updated_state == 'vbuy' and (vwap_curr is not None and close_prev < vwap_curr):
                updated_state = None
                log(f"🔄 STATE RESET: {pair_name} VBuy - Price closed below VWAP")
            elif updated_state == 'vsell' and (vwap_curr is not None and close_prev > vwap_curr):
                updated_state = None
                log(f"🔄 STATE RESET: {pair_name} VSell - Price closed above VWAP")

        # Handle signals
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
        logger.exception(f"Error checking {pair_name}: {e}")
        return last_alerts.get(pair_name), '\n'.join(thread_log)

# ============ MAIN ============
def main():
    ist = pytz.timezone('Asia/Kolkata')
    start_time = datetime.now(ist)
    logger.info(f"Fibonacci Pivot Alert Bot - {start_time.strftime('%d-%m-%Y @ %H:%M IST')}")
    logger.info(f"Debug Mode: {'ON' if DEBUG_MODE else 'OFF'}")

    if SEND_TEST_MESSAGE:
        send_test_message()

    if RESET_STATE and os.path.exists(STATE_FILE):
        logger.warning(f"RESET_STATE True. Deleting {STATE_FILE} to clear previous alerts.")
        try:
            os.remove(STATE_FILE)
        except Exception as e:
            logger.error(f"Failed to delete state file: {e}")

    last_alerts = load_state()
    keys_to_purge = []
    for key, value in list(last_alerts.items()):
        if value:
            value_str = str(value).lower()
            if not (value_str.startswith('fib_long_') or value_str.startswith('fib_short_') or value_str in ['vbuy', 'vsell']):
                keys_to_purge.append(key)
    if keys_to_purge:
        logger.info(f"🧹 Purging old/invalid states for: {', '.join(keys_to_purge)}")
        for key in keys_to_purge:
            last_alerts[key] = None

    if not get_product_ids():
        logger.error("Failed to fetch products. Exiting.")
        return

    found_count = sum(1 for v in PAIRS.values() if v is not None)
    logger.info(f"✓ Monitoring {found_count} pairs")
    if found_count == 0:
        logger.error("No valid pairs found. Exiting.")
        return

    updates_processed = 0
    pair_logs = []
    try:
        with ThreadPoolExecutor(max_workers=6) as executor:
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
                    logger.exception(f"Error processing {pair_name} in thread: {e}")
                    continue
    finally:
        if DEBUG_MODE and pair_logs:
            logger.debug("SEQUENTIAL DEBUG LOGS START")
            for log_output in pair_logs:
                if log_output:
                    print(log_output, end='')
            logger.debug("SEQUENTIAL DEBUG LOGS END")

    save_state(last_alerts)
    end_time = datetime.now(ist)
    elapsed = (end_time - start_time).total_seconds()
    logger.info(f"✓ Check complete. {updates_processed} state updates processed. ({elapsed:.1f}s)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
