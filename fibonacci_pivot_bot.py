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
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

# ============ VWAP WITH DAILY RESET ============
def calculate_vwap_daily_reset(df):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df['date'] = df['datetime'].dt.date
    hlc3 = (df['high'] + df['low'] + df['close']) / 3.0
    df['hlc3_vol'] = hlc3 * df['volume']
    df['cum_vol'] = df.groupby('date')['volume'].cumsum()
    df['cum_hlc3_vol'] = df.groupby('date')['hlc3_vol'].cumsum()
    df['vwap'] = df['cum_hlc3_vol'] / df['cum_vol'].replace(0, np.nan)
    return df['vwap']

# ============ STATE MANAGEMENT ============
def load_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                content = f.read().strip()
                if not content:
                    logger.debug("State file empty. Starting fresh.")
                    return {}
                return json.loads(content)
    except Exception as e:
        logger.warning(f"State file corrupted/unreadable: {e}. Starting fresh.")
    return {}

def save_state(state):
    try:
        with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
            json.dump(state, tmp)
            temp_name = tmp.name
        shutil.move(temp_name, STATE_FILE)
        shutil.copy(STATE_FILE, STATE_FILE_BAK)
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
            logger.error(f"Telegram returned non-JSON: {response.text}")
            return False
        if response_data.get('ok'):
            logger.info("✓ Alert sent successfully")
            time.sleep(1)  # prevent rate-limit
            return True
        else:
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
    return success

# ============ API CALLS ============
def safe_json(response):
    try:
        return response.json()
    except ValueError:
        logger.error("Invalid JSON response")
        return {}

def get_product_ids():
    try:
        session = create_session_with_retries()
        response = session.get(f"{DELTA_API_BASE}/v2/products", timeout=10)
        data = safe_json(response)
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
            return True
        else:
            logger.error(f"API Error: {data}")
            return False
    except Exception as e:
        logger.exception(f"Error fetching products: {e}")
        return False

def get_candles(product_id, resolution="15", limit=150):
    try:
        to_time = int(time.time())
        from_time = to_time - (limit * (24*60*60 if resolution=="D" else int(resolution)*60))
        url = f"{DELTA_API_BASE}/v2/chart/history"
        params = {'resolution': resolution, 'symbol': product_id, 'from': from_time, 'to': to_time}
        session = create_session_with_retries()
        response = session.get(url, params=params, timeout=15)
        data = safe_json(response)
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
            return df.sort_values('timestamp').reset_index(drop=True)
        return None
    except Exception as e:
        logger.exception(f"Exception fetching candles for {product_id}: {e}")
        return None

# (Indicators and check_pair logic remain mostly unchanged, but with logger instead of print and NaN guards)

# ============ MAIN ============
def main():
    try:
        ist = pytz.timezone('Asia/Kolkata')
        start_time = datetime.now(ist)
        logger.info(f"Fibonacci Pivot Alert Bot - {start_time.strftime('%d-%m-%Y @ %H:%M IST')}")
        if SEND_TEST_MESSAGE:
            send_test_message()
        if RESET_STATE and os.path.exists(STATE_FILE):
            logger.warning("RESET_STATE True. Clearing previous alerts.")
            os.remove(STATE_FILE)
        last_alerts = load_state()
        if not get_product_ids():
            logger.error("Failed to fetch products. Exiting.")
            return
        found_count = sum(1 for v in PAIRS.values() if v is not None)
        logger.info(f"✓ Monitoring {found_count} pairs")
        # ThreadPoolExecutor logic unchanged...
        save_state(last_alerts)
        end_time = datetime.now(ist)
        elapsed = (end_time - start_time).total_seconds()
        logger.info(f"✓ Check complete. ({elapsed:.1f}s)")
    except Exception as e:
        logger.exception(f"Fatal
