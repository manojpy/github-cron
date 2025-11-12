import os
import sys
import json
import time
import asyncio
import random
import logging
import sqlite3
import traceback
import tempfile
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import aiohttp
import pandas as pd
import numpy as np
import pytz
from aiohttp import ClientConnectorError, ClientResponseError
from logging.handlers import RotatingFileHandler

# -------------------------
# DEFAULT CONFIG
# -------------------------
DEFAULT_CONFIG = {
    "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", "8462496498:AAHYZ4xDIHvrVRjmCmZyoPhupCjRaRgiITc"),
    "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID", "203813932"),
    "DEBUG_MODE": os.getenv("DEBUG_MODE", "True").lower() == "true",
    "SEND_TEST_MESSAGE": os.getenv("SEND_TEST_MESSAGE", "True").lower() == "true",
    "RESET_STATE": os.getenv("RESET_STATE", "False").lower() == "true",
    "DELTA_API_BASE": "https://api.india.delta.exchange",
    "PAIRS": ["BTCUSD","ETHUSD","SOLUSD","AVAXUSD","BCHUSD","XRPUSD","BNBUSD","LTCUSD","DOTUSD","ADAUSD","SUIUSD","AAVEUSD"],
    "SPECIAL_PAIRS": {"SOLUSD": {"limit_15m":210,"min_required":180,"limit_5m":300,"min_required_5m":200}},
    "PPO_FAST": 7,
    "PPO_SLOW": 16,
    "PPO_SIGNAL": 5,
    "PPO_USE_SMA": False,
    "X1": 22, "X2": 9, "X3": 15, "X4": 5,
    "PIVOT_LOOKBACK_PERIOD": 15,
    # state DB path (sqlite)
    "STATE_DB_PATH": os.getenv("STATE_DB_PATH", "fib_state.sqlite"),
    "LOG_FILE": os.getenv("LOG_FILE", "fibonacci_pivot_bot.log"),
    "MAX_CONCURRENCY": int(os.getenv("MAX_CONCURRENCY", "6")),
    "HTTP_TIMEOUT": int(os.getenv("HTTP_TIMEOUT", "15")),
    "FETCH_RETRIES": int(os.getenv("FETCH_RETRIES", "3")),
    "FETCH_BACKOFF": float(os.getenv("FETCH_BACKOFF", "1.5")),
    "JITTER_MIN": float(os.getenv("JITTER_MIN", "0.05")),
    "JITTER_MAX": float(os.getenv("JITTER_MAX", "0.6")),
    # pruning & duplicate settings
    "STATE_EXPIRY_DAYS": int(os.getenv("STATE_EXPIRY_DAYS", "30")),  # keep last 30 days
    "DUPLICATE_SUPPRESSION_SECONDS": int(os.getenv("DUPLICATE_SUPPRESSION_SECONDS", str(60*60))),  # 1 hour
    "EXTREME_CANDLE_PCT": float(os.getenv("EXTREME_CANDLE_PCT", "8.0")),  # skip if candle move > 8%
    # schedule defaults
    "RUN_LOOP_INTERVAL": int(os.getenv("RUN_LOOP_INTERVAL", "900"))  # default loop 15 minutes
}

# load config.json overrides if present
CONFIG_FILE = os.getenv("CONFIG_FILE", "config.json")
if Path(CONFIG_FILE).exists():
    try:
        with open(CONFIG_FILE, "r") as f:
            user_cfg = json.load(f)
        DEFAULT_CONFIG.update(user_cfg)
    except Exception as e:
        print(f"Warning: unable to parse config.json: {e}")

cfg = DEFAULT_CONFIG

# -------------------------
# Logger
# -------------------------
logger = logging.getLogger("fibonacci_pivot_bot")
logger.setLevel(logging.DEBUG if cfg["DEBUG_MODE"] else logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

# Console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Rotating file
try:
    file_handler = RotatingFileHandler(cfg["LOG_FILE"], maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
except Exception as e:
    logger.warning(f"Could not create rotating log file: {e}")

# -------------------------
# Utilities
# -------------------------
def now_ts() -> int:
    return int(time.time())

def human_ts() -> str:
    tz = pytz.timezone("Asia/Kolkata")
    return datetime.now(tz).strftime("%d-%m-%Y @ %H:%M IST")

# -------------------------
# SQLite state DB (safe + metadata)
# -------------------------
class StateDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._ensure_tables()

    def _ensure_tables(self):
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS states (
                pair TEXT PRIMARY KEY,
                state TEXT,
                ts INTEGER
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        self._conn.commit()

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute("SELECT pair, state, ts FROM states")
        rows = cur.fetchall()
        return {r[0]: {"state": r[1], "ts": int(r[2] or 0)} for r in rows}

    def get(self, pair: str) -> Optional[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute("SELECT state, ts FROM states WHERE pair = ?", (pair,))
        r = cur.fetchone()
        if not r:
            return None
        return {"state": r[0], "ts": int(r[1] or 0)}

    def set(self, pair: str, state: Optional[str], ts: Optional[int] = None):
        ts = int(ts or now_ts())
        cur = self._conn.cursor()
        if state is None:
            # remove row
            cur.execute("DELETE FROM states WHERE pair = ?", (pair,))
        else:
            cur.execute("INSERT INTO states(pair, state, ts) VALUES (?, ?, ?) "
                        "ON CONFLICT(pair) DO UPDATE SET state=excluded.state, ts=excluded.ts",
                        (pair, state, ts))
        self._conn.commit()

    def get_metadata(self, key: str) -> Optional[str]:
        cur = self._conn.cursor()
        cur.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        r = cur.fetchone()
        return r[0] if r else None

    def set_metadata(self, key: str, value: str):
        cur = self._conn.cursor()
        cur.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", (key, value))
        self._conn.commit()

    def close(self):
        self._conn.close()

# -------------------------
# Prune (smart daily)
# -------------------------
def prune_old_state_records(db_path: str, expiry_days: int = 30, logger_local: logging.Logger = None):
    """
    Delete 'states' rows older than expiry_days.
    Runs at most once per UTC day (metadata.last_prune).
    """
    try:
        if expiry_days <= 0:
            if logger_local:
                logger_local.debug("Pruning disabled (expiry_days <= 0)")
            return

        if not os.path.exists(db_path):
            if logger_local:
                logger_local.info(f"State DB not found at {db_path}; skipping prune.")
            return

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
        cur.execute("SELECT value FROM metadata WHERE key='last_prune'")
        row = cur.fetchone()
        from datetime import timezone
        today = datetime.now(timezone.utc).date()
        if row:
            try:
                last_prune_date = datetime.fromisoformat(row[0]).date()
                if last_prune_date >= today:
                    if logger_local:
                        logger_local.debug("Prune already run today; skipping.")
                    conn.close()
                    return
            except Exception:
                pass

        # ensure states table
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='states'")
        if not cur.fetchone():
            if logger_local:
                logger_local.debug("No states table yet; skipping prune.")
            conn.close()
            return

        cutoff = int(time.time()) - (expiry_days * 86400)
        cur.execute("DELETE FROM states WHERE ts < ?", (cutoff,))
        deleted = cur.rowcount
        conn.commit()

        cur.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('last_prune', ?)", (datetime.utcnow().isoformat(),))
        conn.commit()

        if deleted > 0:
            try:
                cur.execute("VACUUM;")
                conn.commit()
            except Exception as e:
                if logger_local:
                    logger_local.warning(f"VACUUM failed: {e}")

        conn.close()
        if logger_local:
            logger_local.info(f"Pruned {deleted} states older than {expiry_days} days from {db_path}.")
    except Exception as e:
        if logger_local:
            logger_local.warning(f"Prune failed: {e}")
            logger_local.debug(traceback.format_exc())

# -------------------------
# Async HTTP helpers with retries
# -------------------------
async def fetch_json_with_retries(session: aiohttp.ClientSession, url: str, params: dict=None, retries:int=3, backoff:float=1.5, timeout:int=15):
    for attempt in range(1, retries+1):
        try:
            async with session.get(url, params=params, timeout=timeout) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    # log debug
                    logger.debug(f"HTTP {resp.status} {url} {params} - {text[:200]}")
                    raise ClientResponseError(resp.request_info, resp.history, status=resp.status)
                try:
                    return await resp.json()
                except Exception:
                    logger.debug(f"Non-JSON response from {url}: {text[:200]}")
                    return {}
        except (asyncio.TimeoutError, ClientConnectorError, ClientResponseError) as e:
            logger.debug(f"Fetch attempt {attempt} error: {e} for {url}")
            if attempt == retries:
                logger.warning(f"Failed to fetch {url} after {retries} attempts.")
                return None
            await asyncio.sleep(backoff * (attempt ** 1.2) + random.uniform(0, 0.2))
        except Exception as e:
            logger.exception(f"Unexpected fetch error for {url}: {e}")
            return None
    return None

# -------------------------
# DataFetcher (async) and product mapping
# -------------------------
class DataFetcher:
    def __init__(self, base_url: str, max_parallel: int = 6, timeout: int = 15):
        self.base_url = base_url.rstrip("/")
        self.semaphore = asyncio.Semaphore(max_parallel)
        self.timeout = timeout
        self.cache = {}  # per-run cache

    async def fetch_products(self, session: aiohttp.ClientSession):
        url = f"{self.base_url}/v2/products"
        async with self.semaphore:
            return await fetch_json_with_retries(session, url, retries=cfg["FETCH_RETRIES"], backoff=cfg["FETCH_BACKOFF"], timeout=cfg["HTTP_TIMEOUT"])

    async def fetch_candles(self, session: aiohttp.ClientSession, symbol: str, resolution: str, limit: int):
        key = f"candles:{symbol}:{resolution}:{limit}"
        if key in self.cache:
            return self.cache[key][1]
        await asyncio.sleep(random.uniform(cfg["JITTER_MIN"], cfg["JITTER_MAX"]))
        url = f"{self.base_url}/v2/chart/history"
        params = {"resolution": resolution, "symbol": symbol, "from": int(time.time()) - (limit * (int(resolution) if resolution!='D' else 1440) * 60), "to": int(time.time())}
        async with self.semaphore:
            data = await fetch_json_with_retries(session, url, params=params, retries=cfg["FETCH_RETRIES"], backoff=cfg["FETCH_BACKOFF"], timeout=cfg["HTTP_TIMEOUT"])
        self.cache[key] = (time.time(), data)
        return data

def build_products_map(api_products: dict) -> Dict[str, dict]:
    products_map = {}
    if not api_products:
        return products_map
    for p in api_products.get("result", []):
        try:
            symbol = p.get("symbol", "")
            symbol_norm = symbol.replace("_USDT", "USD").replace("USDT", "USD")
            if p.get("contract_type") == "perpetual_futures":
                for pair_name in cfg["PAIRS"]:
                    if symbol_norm == pair_name or symbol_norm.replace("_", "") == pair_name:
                        products_map[pair_name] = {'id': p.get('id'), 'symbol': p.get('symbol'), 'contract_type': p.get('contract_type')}
        except Exception:
            continue
    return products_map

# -------------------------
# Helpers to parse candles into DataFrame (same format as before)
# -------------------------
def parse_candle_response(res: dict) -> Optional[pd.DataFrame]:
    if not res or not isinstance(res, dict):
        return None
    if not res.get("success"):
        return None
    resr = res.get("result", {})
    arrays = [resr.get('t', []), resr.get('o', []), resr.get('h', []), resr.get('l', []), resr.get('c', []), resr.get('v', [])]
    if any(len(a)==0 for a in arrays):
        return None
    min_len = min(map(len, arrays))
    df = pd.DataFrame({
        'timestamp': resr.get('t', [])[:min_len],
        'open': resr.get('o', [])[:min_len],
        'high': resr.get('h', [])[:min_len],
        'low': resr.get('l', [])[:min_len],
        'close': resr.get('c', [])[:min_len],
        'volume': resr.get('v', [])[:min_len]
    })
    df = df.sort_values('timestamp').drop_duplicates(subset='timestamp').reset_index(drop=True)
    if df.empty or df['close'].astype(float).iloc[-1] <= 0:
        return None
    return df

# -------------------------
# Indicator functions (same algorithms)
# -------------------------
def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=max(1, period//3)).mean()

def calculate_rma(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1/period, adjust=False).mean()

def calculate_ppo(df: pd.DataFrame, fast:int, slow:int, signal:int, use_sma:bool=False):
    close = df['close'].astype(float)
    if use_sma:
        fast_ma = calculate_sma(close, fast)
        slow_ma = calculate_sma(close, slow)
    else:
        fast_ma = calculate_ema(close, fast)
        slow_ma = calculate_ema(close, slow)
    slow_ma = slow_ma.replace(0, np.nan).bfill().ffill()
    ppo = ((fast_ma - slow_ma) / slow_ma) * 100
    ppo = ppo.replace([np.inf, -np.inf], np.nan).bfill().ffill()
    ppo_signal = calculate_sma(ppo, signal) if use_sma else calculate_ema(ppo, signal)
    ppo_signal = ppo_signal.replace([np.inf, -np.inf], np.nan).bfill().ffill()
    return ppo, ppo_signal

def smoothrng(x: pd.Series, t: int, m: int) -> pd.Series:
    wper = t * 2 - 1
    avrng = calculate_ema(np.abs(x.diff().fillna(0)), t)
    smoothrng_val = calculate_ema(avrng, max(1, wper)) * m
    return smoothrng_val.clip(lower=1e-8).bfill().ffill()

def rngfilt(x: pd.Series, r: pd.Series) -> pd.Series:
    if len(x) == 0:
        return pd.Series(dtype=float)
    result = [x.iloc[0]]
    for i in range(1, len(x)):
        prev = result[-1]
        curr_x = x.iloc[i]
        curr_r = max(float(r.iloc[i]), 1e-8)
        if curr_x > prev:
            f = prev if (curr_x - curr_r) < prev else (curr_x - curr_r)
        else:
            f = prev if (curr_x + curr_r) > prev else (curr_x + curr_r)
        result.append(f)
    return pd.Series(result, index=x.index)

def calculate_cirrus_cloud(df: pd.DataFrame):
    close = df['close'].astype(float)
    smrngx1x = smoothrng(close, cfg["X1"], cfg["X2"])
    smrngx1x2 = smoothrng(close, cfg["X3"], cfg["X4"])
    filtx1 = rngfilt(close, smrngx1x)
    filtx12 = rngfilt(close, smrngx1x2)
    upw = filtx1 < filtx12
    dnw = filtx1 > filtx12
    return upw, dnw, filtx1, filtx12

def kalman_filter(src: pd.Series, length: int, R=0.01, Q=0.1) -> pd.Series:
    result = []
    estimate = np.nan
    error_est = 1.0
    error_meas = R * max(1, length)
    Q_div_length = Q / max(1, length)
    for i in range(len(src)):
        current = src.iloc[i]
        if np.isnan(estimate):
            if i > 0:
                estimate = src.iloc[i-1]
            else:
                result.append(np.nan)
                continue
        prediction = estimate
        kalman_gain = error_est / (error_est + error_meas)
        estimate = prediction + kalman_gain * (current - prediction)
        error_est = (1 - kalman_gain) * error_est + Q_div_length
        result.append(estimate)
    return pd.Series(result, index=src.index)

def calculate_smooth_rsi(df: pd.DataFrame, rsi_len: int, kalman_len: int) -> pd.Series:
    close = df['close'].astype(float)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = calculate_rma(gain, rsi_len)
    avg_loss = calculate_rma(loss, rsi_len).replace(0, np.nan).bfill().ffill().clip(lower=1e-8)
    rs = avg_gain.divide(avg_loss)
    rsi_value = 100 - (100 / (1 + rs))
    rsi_value = rsi_value.replace([np.inf, -np.inf], np.nan).bfill().ffill()
    smooth_rsi = kalman_filter(rsi_value, kalman_len).bfill().ffill()
    return smooth_rsi

def calculate_magical_momentum_hist(df: pd.DataFrame, period: int = 144, responsiveness: float = 0.9) -> pd.Series:
    n = len(df)
    if n == 0:
        return pd.Series([], dtype=float)
    if n < period + 50:
        return pd.Series(np.zeros(n), index=df.index, dtype=float)

    close = df['close'].astype(float).copy()
    sd = close.rolling(window=50, min_periods=10).std() * responsiveness
    sd = sd.bfill().ffill().fillna(0.001).clip(lower=1e-6)

    worm = close.copy()
    for i in range(1, n):
        diff = close.iloc[i] - worm.iloc[i - 1]
        delta = np.sign(diff) * sd.iloc[i] if abs(diff) > sd.iloc[i] else diff
        worm.iloc[i] = worm.iloc[i - 1] + delta

    ma = close.rolling(window=period, min_periods=max(5, period // 3)).mean().bfill().ffill()
    denom = worm.replace(0, np.nan).bfill().ffill().clip(lower=1e-8)

    raw_momentum = ((worm - ma).fillna(0)) / denom
    raw_momentum = raw_momentum.replace([np.inf, -np.inf], 0).fillna(0)

    min_med = raw_momentum.rolling(window=period, min_periods=max(5, period // 3)).min().bfill().ffill()
    max_med = raw_momentum.rolling(window=period, min_periods=max(5, period // 3)).max().bfill().ffill()
    rng = (max_med - min_med).replace(0, np.nan).fillna(1e-8)

    temp = pd.Series(0.0, index=df.index)
    valid = rng > 1e-10
    temp.loc[valid] = (raw_momentum.loc[valid] - min_med.loc[valid]) / rng.loc[valid]
    temp = temp.clip(-1, 1).fillna(0)

    value = pd.Series(0.0, index=df.index)
    for i in range(1, n):
        v_prev = value.iloc[i - 1]
        v_new = (temp.iloc[i] - 0.5 + 0.5 * v_prev)
        value.iloc[i] = max(-0.9999, min(0.9999, v_new))

    temp2 = (1 + value) / (1 - value)
    temp2 = temp2.replace([np.inf, -np.inf], np.nan).clip(lower=1e-6).fillna(1e-6)

    momentum = 0.25 * np.log(temp2)
    momentum = pd.Series(momentum, index=df.index).replace([np.inf, -np.inf], 0).fillna(0)

    hist = pd.Series(0.0, index=df.index)
    for i in range(1, n):
        hist.iloc[i] = momentum.iloc[i] + 0.5 * hist.iloc[i - 1]

    return hist.replace([np.inf, -np.inf], 0).fillna(0)

# -------------------------
# VWAP (daily reset)
# -------------------------
def calculate_vwap_daily_reset(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    df2 = df.copy()
    df2['datetime'] = pd.to_datetime(df2['timestamp'], unit='s', utc=True)
    df2['date'] = df2['datetime'].dt.date
    hlc3 = (df2['high'] + df2['low'] + df2['close']) / 3.0
    df2['hlc3_vol'] = hlc3 * df2['volume']
    df2['cum_vol'] = df2.groupby('date')['volume'].cumsum()
    df2['cum_hlc3_vol'] = df2.groupby('date')['hlc3_vol'].cumsum()
    vwap = df2['cum_hlc3_vol'] / df2['cum_vol'].replace(0, np.nan)
    return vwap.ffill().bfill()

# -------------------------
# Previous day OHLC & Fibonacci pivots
# -------------------------
def get_previous_day_ohlc_from_api(session: aiohttp.ClientSession, symbol: str, days_back_limit:int=15) -> Optional[dict]:
    # fetch daily candles using chart/history resolution="D"
    data = awaitable_fetch_candles_sync(session, symbol, "D", days_back_limit + 5)  # will be awaited in wrapper
    return data

async def awaitable_get_previous_day_ohlc(session: aiohttp.ClientSession, symbol: str, days_back_limit:int=15):
    res = await fetcher.fetch_candles(session, symbol, "D", days_back_limit + 5)
    df = parse_candle_response(res)
    if df is None or len(df) < 2:
        return None
    df['date'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.date
    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)
    prev_day_row = df[df['date'] == yesterday]
    if not prev_day_row.empty:
        prev_day = prev_day_row.iloc[-1]
    else:
        prev_day = df.iloc[-2]
    return {'high': float(prev_day['high']), 'low': float(prev_day['low']), 'close': float(prev_day['close'])}

def calculate_fibonacci_pivots(h: float, l: float, c: float) -> dict:
    pivot = (h + l + c) / 3.0
    diff = h - l
    r3 = pivot + (diff * 1.000)
    r2 = pivot + (diff * 0.618)
    r1 = pivot + (diff * 0.382)
    s1 = pivot - (diff * 0.382)
    s2 = pivot - (diff * 0.618)
    s3 = pivot - (diff * 1.000)
    return {'P': pivot, 'R1': r1, 'R2': r2, 'R3': r3, 'S1': s1, 'S2': s2, 'S3': s3}

# -------------------------
# Helper: cloud state
# -------------------------
def cloud_state_from(upw: pd.Series, dnw: pd.Series, idx: int = -1) -> str:
    u = bool(upw.iloc[idx])
    d = bool(dnw.iloc[idx])
    if u and not d:
        return "green"
    elif d and not u:
        return "red"
    else:
        return "neutral"

# -------------------------
# Duplicate suppression helper
# -------------------------
def should_suppress_duplicate(last_state: Optional[Dict[str, Any]], current_signal: str, suppression_seconds:int) -> bool:
    if not last_state:
        return False
    last_signal = last_state.get("state")
    last_ts = int(last_state.get("ts", 0))
    if last_signal == current_signal and (now_ts() - last_ts) <= suppression_seconds:
        return True
    return False

# -------------------------
# Core pair evaluation (keeps original logic but wrapped async)
# -------------------------
async def evaluate_pair_async(session: aiohttp.ClientSession, fetcher: DataFetcher, products_map: Dict[str, dict], pair_name: str, last_state: Optional[Dict[str, Any]]) -> Optional[Tuple[str, Dict[str,Any]]]:
    """
    Returns (pair_name, new_state_dict) or None.
    new_state_dict: {"state": <str>, "ts": <int>}
    This function preserves your original logic, with added guards:
     - skip on extreme candles (> EXTREME_CANDLE_PCT)
     - skip pairs with insufficient data
     - use closed-candle index logic (timestamp-based)
     - duplicate suppression (don't re-send same signal within suppression_seconds)
    """
    try:
        prod = products_map.get(pair_name)
        if not prod:
            logger.debug(f"No product mapping for {pair_name}")
            return None

        # special pairs params
        sp = cfg["SPECIAL_PAIRS"].get(pair_name, {})
        limit_15m = sp.get("limit_15m", 250)
        min_required_15m = sp.get("min_required", 150)
        limit_5m = sp.get("limit_5m", 500)
        min_required_5m = sp.get("min_required_5m", 250)

        # fetch candles concurrently
        res15 = await fetcher.fetch_candles(session, prod['symbol'], "15", limit_15m)
        res5 = await fetcher.fetch_candles(session, prod['symbol'], "5", limit_5m)  # optional

        df_15m = parse_candle_response(res15)
        df_5m = parse_candle_response(res5) if res5 else None

        if df_15m is None or len(df_15m) < min_required_15m:
            logger.warning(f"{pair_name}: insufficient 15m data ({0 if df_15m is None else len(df_15m)})")
            return None

        # determine closed index for 15m
        nowts = time.time()
        res15s = 15*60
        current_15m_start = nowts - (nowts % res15s)
        last_15_incomplete = df_15m['timestamp'].iloc[-1] >= current_15m_start
        last_i_15m = -2 if last_15_incomplete else -1
        if len(df_15m) < abs(last_i_15m)+1:
            logger.warning(f"{pair_name}: not enough 15m candles after index adjust")
            return None

        # determine closed index for 5m (for RMA200) if available
        last_i_5m = -1
        rma_200_available = False
        if df_5m is not None and len(df_5m) >= 50:
            res5s = 5*60
            current_5m_start = nowts - (nowts % res5s)
            last_5_incomplete = df_5m['timestamp'].iloc[-1] >= current_5m_start
            last_i_5m = -2 if last_5_incomplete else -1
            if len(df_5m) < abs(last_i_5m)+1:
                df_5m = None

        # compute pivots (previous day)
        # fetch daily candles
        resd = await fetcher.fetch_candles(session, prod['symbol'], "D", cfg["PIVOT_LOOKBACK_PERIOD"] + 5)
        df_daily = parse_candle_response(resd)
        if df_daily is None or len(df_daily) < 2:
            logger.warning(f"{pair_name}: failed to fetch daily candles")
            return None
        df_daily['date'] = pd.to_datetime(df_daily['timestamp'], unit='s', utc=True).dt.date
        today = datetime.now(timezone.utc).date()
        yesterday = today - timedelta(days=1)
        prev_day_row = df_daily[df_daily['date'] == yesterday]
        if not prev_day_row.empty:
            prev = prev_day_row.iloc[-1]
        else:
            prev = df_daily.iloc[-2]
        prev_day_ohlc = {'high': float(prev['high']), 'low': float(prev['low']), 'close': float(prev['close'])}
        pivots = calculate_fibonacci_pivots(prev_day_ohlc['high'], prev_day_ohlc['low'], prev_day_ohlc['close'])

        # indicators on 15m
        vwap_15m = calculate_vwap_daily_reset(df_15m)
        ppo, _ = calculate_ppo(df_15m, cfg["PPO_FAST"], cfg["PPO_SLOW"], cfg["PPO_SIGNAL"], cfg["PPO_USE_SMA"])
        upw, dnw, _, _ = calculate_cirrus_cloud(df_15m)
        rma_50_15m = calculate_rma(df_15m['close'], 50)
        magical_hist = calculate_magical_momentum_hist(df_15m, period=144, responsiveness=0.9)

        # extract using last closed index
        open_c = float(df_15m['open'].iloc[last_i_15m])
        close_c = float(df_15m['close'].iloc[last_i_15m])
        high_c = float(df_15m['high'].iloc[last_i_15m])
        low_c = float(df_15m['low'].iloc[last_i_15m])

        # extreme candle guard
        candle_pct = abs((close_c - open_c) / open_c) * 100 if open_c != 0 else 0
        if candle_pct > cfg["EXTREME_CANDLE_PCT"]:
            logger.info(f"{pair_name}: extreme candle {candle_pct:.2f}% > {cfg['EXTREME_CANDLE_PCT']}% - skipping")
            return None

        # basic indicator checks
        idx = last_i_15m
        try:
            ppo_curr = float(ppo.iloc[idx])
            rma50_curr = float(rma_50_15m.iloc[idx])
            magical_curr = float(magical_hist.iloc[idx]) if len(magical_hist) > abs(idx) else np.nan
        except Exception:
            logger.debug(f"{pair_name}: indicators indexing issue - skipping")
            return None

        # 5m RMA200 if available
        rma200_5m_curr = np.nan
        if df_5m is not None:
            rma200 = calculate_rma(df_5m['close'], 200)
            try:
                rma200_5m_curr = float(rma200.iloc[last_i_5m])
                rma_200_available = not np.isnan(rma200_5m_curr)
            except Exception:
                rma_200_available = False

        # clouds
        cloud_state = cloud_state_from(upw, dnw, idx=last_i_15m)

        # candle wicks
        total_range = high_c - low_c
        if total_range <= 0:
            upper_wick_ok = False
            lower_wick_ok = False
        else:
            upper_wick = high_c - max(open_c, close_c)
            lower_wick = min(open_c, close_c) - low_c
            upper_wick_ok = (upper_wick / total_range) < 0.20
            lower_wick_ok = (lower_wick / total_range) < 0.20

        # VWAP checks (if vwaps available)
        vwap_curr = None
        try:
            vwap_curr = vwap_15m.iloc[last_i_15m]
            if np.isnan(vwap_curr):
                vwap_curr = None
        except Exception:
            vwap_curr = None

        # pivot crossovers detection (use same logic)
        is_green = close_c > open_c
        is_red = close_c < open_c

        long_pivot_lines = {'P':pivots['P'],'R1':pivots['R1'],'R2':pivots['R2'],'S1':pivots['S1'],'S2':pivots['S2']}
        long_crossover_line = None; long_crossover_name = None
        if is_green:
            for name,line in long_pivot_lines.items():
                if open_c <= line and close_c > line:
                    long_crossover_line = line; long_crossover_name = name; break

        short_pivot_lines = {'P':pivots['P'],'S1':pivots['S1'],'S2':pivots['S2'],'R1':pivots['R1'],'R2':pivots['R2']}
        short_crossover_line = None; short_crossover_name = None
        if is_red:
            for name,line in short_pivot_lines.items():
                if open_c >= line and close_c < line:
                    short_crossover_line = line; short_crossover_name = name; break

        # base gates
        if rma_200_available:
            rma_long_ok = (rma50_curr < close_c) and (rma200_5m_curr < close_c)
            rma_short_ok = (rma50_curr > close_c) and (rma200_5m_curr > close_c)
        else:
            rma_long_ok = (rma50_curr < close_c)
            rma_short_ok = (rma50_curr > close_c)

        base_long_ok = (cloud_state == "green") and upper_wick_ok and rma_long_ok and (magical_curr > 0) and is_green
        base_short_ok = (cloud_state == "red") and lower_wick_ok and rma_short_ok and (magical_curr < 0) and is_red

        # VWAP crossover: need previous candle for crossover detection
        vbuy = False; vsell = False
        if vwap_curr is not None and len(df_15m) >= abs(last_i_15m)+2:
            prev_close = float(df_15m['close'].iloc[last_i_15m - 1])
            prev_vwap = float(vwap_15m.iloc[last_i_15m - 1])
            if base_long_ok and prev_close <= prev_vwap and close_c > vwap_curr:
                vbuy = True
            if base_short_ok and prev_close >= prev_vwap and close_c < vwap_curr:
                vsell = True

        fib_long = base_long_ok and (long_crossover_line is not None)
        fib_short = base_short_ok and (short_crossover_line is not None)

        # compose signals with triangle emojis (user requested upward green and downward red triangles)
        up_sig = "🟢🔺"  # upward green triangle visual: green circle + triangle
        down_sig = "🔴🔻"  # downward red triangle

        # decide current_signal string identical to earlier naming for state persistence
        current_signal = None
        message = None

        # load last state (dict with state & ts)
        last_state_pair = last_state  # may be None or dict

        # duplicate suppression: if same state recently recorded (within DUPLICATE_SUPPRESSION_SECONDS) -> don't re-alert
        suppress_secs = cfg["DUPLICATE_SUPPRESSION_SECONDS"]

        # check generation order (same priority as your original code)
        if vbuy:
            current_signal = "vbuy"
            if not should_suppress_duplicate(last_state_pair, current_signal, suppress_secs):
                message = (f"{up_sig} {pair_name} - VBuy\n"
                           f"Closed Above VWAP (${vwap_curr:,.2f})\n"
                           f"PPO 15m: {ppo_curr:.2f}\nPrice: ${close_c:,.2f}\n{human_ts()}")
        elif vsell:
            current_signal = "vsell"
            if not should_suppress_duplicate(last_state_pair, current_signal, suppress_secs):
                message = (f"{down_sig} {pair_name} - VSell\n"
                           f"Closed Below VWAP (${vwap_curr:,.2f})\n"
                           f"PPO 15m: {ppo_curr:.2f}\nPrice: ${close_c:,.2f}\n{human_ts()}")
        elif fib_long:
            current_signal = f"fib_long_{long_crossover_name}"
            if not should_suppress_duplicate(last_state_pair, current_signal, suppress_secs):
                message = (f"{up_sig} {pair_name} - FIB LONG\n"
                           f"Closed Above {long_crossover_name} (${long_crossover_line:,.2f})\n"
                           f"PPO 15m: {ppo_curr:.2f}\nPrice: ${close_c:,.2f}\n{human_ts()}")
        elif fib_short:
            current_signal = f"fib_short_{short_crossover_name}"
            if not should_suppress_duplicate(last_state_pair, current_signal, suppress_secs):
                message = (f"{down_sig} {pair_name} - FIB SHORT\n"
                           f"Closed Below {short_crossover_name} (${short_crossover_line:,.2f})\n"
                           f"PPO 15m: {ppo_curr:.2f}\nPrice: ${close_c:,.2f}\n{human_ts()}")

        # If no signal found, maybe transition to NO_SIGNAL (same behavior preserved)
        if current_signal is None:
            # if previously had a non-none state, move to NO_SIGNAL
            prev_state_val = last_state_pair.get("state") if last_state_pair else None
            if prev_state_val and prev_state_val != "NO_SIGNAL":
                return pair_name, {"state":"NO_SIGNAL", "ts": now_ts()}
            return None

        # check if message exists (not suppressed)
        if message:
            # send the alert async (we will rely on outer caller to handle the session)
            # But to keep modularity: perform a POST here
            # We'll create a small session for posting (shortlived)
            try:
                # send async
                await send_telegram_async(cfg["TELEGRAM_BOT_TOKEN"], cfg["TELEGRAM_CHAT_ID"], message)
                logger.info(f"Alert sent for {pair_name}: {current_signal}")
            except Exception as e:
                logger.exception(f"Failed to send alert for {pair_name}: {e}")

        # persist state (whether message sent or suppressed we update the state to avoid re-sending in same hour)
        return pair_name, {"state": current_signal, "ts": now_ts()}

    except Exception as e:
        logger.exception(f"Error evaluating pair {pair_name}: {e}")
        if cfg["DEBUG_MODE"]:
            logger.debug(traceback.format_exc())
        return None

# -------------------------
# Async Telegram sender
# -------------------------
async def send_telegram_async(token: str, chat_id: str, text: str, session: Optional[aiohttp.ClientSession]=None) -> bool:
    if not token or token == "xxxx" or not chat_id or chat_id == "xxxx":
        logger.warning("Telegram not configured; skipping.")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": text}
    own_session = False
    if session is None:
        session = aiohttp.ClientSession()
        own_session = True
    try:
        async with session.post(url, data=data, timeout=10) as resp:
            try:
                js = await resp.json(content_type=None)
            except Exception:
                js = {"ok": False, "status": resp.status, "text": await resp.text()}
            ok = js.get("ok", False)
            if ok:
                # gentle pacing (if invoked in loop it's okay)
                await asyncio.sleep(0.2)
                return True
            else:
                logger.warning(f"Telegram API returned error: {js}")
                return False
    except Exception as e:
        logger.exception(f"Telegram send failed: {e}")
        return False
    finally:
        if own_session:
            await session.close()

# -------------------------
# Runner orchestration (main)
# -------------------------
async def run_once(send_test: bool = True):
    logger.info("Starting fibonacci_pivot_bot_improved run_once")

    # daily prune
    try:
        prune_old_state_records(cfg["STATE_DB_PATH"], cfg["STATE_EXPIRY_DAYS"], logger)
    except Exception as e:
        logger.warning(f"Prune error: {e}")

    # state DB
    state_db = StateDB(cfg["STATE_DB_PATH"])
    last_alerts = state_db.load_all()

    # optional: reset state
    if cfg["RESET_STATE"]:
        logger.warning("RESET_STATE requested: clearing states table")
        for p in cfg["PAIRS"]:
            state_db.set(p, None)

    # send test message
    if send_test and cfg["SEND_TEST_MESSAGE"] and cfg["TELEGRAM_BOT_TOKEN"] != "xxxx" and cfg["TELEGRAM_CHAT_ID"] != "xxxx":
        test_msg = f"🔔 Fibonacci Pivot Bot started\nTime: {human_ts()}\nDebug: {'ON' if cfg['DEBUG_MODE'] else 'OFF'}"
        await send_telegram_async(cfg["TELEGRAM_BOT_TOKEN"], cfg["TELEGRAM_CHAT_ID"], test_msg)

    # fetch products & build map
    fetcher_local = DataFetcher(cfg["DELTA_API_BASE"], max_parallel=cfg["MAX_CONCURRENCY"], timeout=cfg["HTTP_TIMEOUT"])
    async with aiohttp.ClientSession() as session:
        prod_resp = await fetcher_local.fetch_products(session)
        if not prod_resp:
            logger.error("Failed to fetch products from API")
            state_db.close()
            return
        products_map = build_products_map(prod_resp)
        found = len(products_map)
        logger.info(f"Found {found} tradable pairs mapped to config.")
        if found == 0:
            logger.error("No products mapped; exiting")
            state_db.close()
            return

        # schedule and run async tasks with concurrency control
        sem = asyncio.Semaphore(cfg["MAX_CONCURRENCY"])
        tasks = []
        for pair in cfg["PAIRS"]:
            if pair not in products_map:
                continue
            last_state = last_alerts.get(pair)
            async def run_one(pair_name=pair, last_s=last_state):
                async with sem:
                    await asyncio.sleep(random.uniform(cfg["JITTER_MIN"], cfg["JITTER_MAX"]))
                    return await evaluate_pair_async(session, fetcher_local, products_map, pair_name, last_s)
            tasks.append(asyncio.create_task(run_one()))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        updates = 0
        for r in results:
            if isinstance(r, Exception):
                logger.exception(f"Task exception: {r}")
                continue
            if not r:
                continue
            pair_name, new_state = r
            if not isinstance(new_state, dict):
                continue
            prev = state_db.get(pair_name)
            if prev != new_state:
                state_db.set(pair_name, new_state.get("state"), new_state.get("ts"))
                updates += 1

        logger.info(f"Run complete. {updates} state updates applied.")
        state_db.close()

# -------------------------
# CLI + graceful stop
# -------------------------
stop_requested = False
def request_stop(signum, frame):
    global stop_requested
    logger.info(f"Signal {signum} received. Stopping after current run.")
    stop_requested = True

import signal
signal.signal(signal.SIGINT, request_stop)
signal.signal(signal.SIGTERM, request_stop)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fibonacci Pivot Bot (improved)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--once", action="store_true", help="Run once and exit")
    group.add_argument("--loop", type=int, metavar="SECONDS", help="Run in loop every N seconds")
    args = parser.parse_args()

    if args.once:
        asyncio.run(run_once())
    elif args.loop:
        interval = args.loop
        logger.info(f"Loop mode: interval={interval}s")
        while not stop_requested:
            start = time.time()
            try:
                asyncio.run(run_once())
            except Exception:
                logger.exception("Unhandled in run_once")
            elapsed = time.time() - start
            to_sleep = max(0, interval - elapsed)
            if to_sleep > 0:
                time.sleep(to_sleep)
    else:
        asyncio.run(run_once())

if __name__ == "__main__":
    main()
