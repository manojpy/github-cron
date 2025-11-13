import os
import sys
import json
import time
import asyncio
import random
import logging
import sqlite3
import signal
import traceback
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
import pandas as pd
import numpy as np
import pytz
from aiohttp import ClientConnectorError, ClientResponseError
from logging.handlers import RotatingFileHandler

# -------------------------
# Default configuration
# -------------------------
DEFAULT_CONFIG = {
    "TELEGRAM_BOT_TOKEN": os.environ.get("TELEGRAM_BOT_TOKEN", "8462496498:AAHYZ4xDIHvrVRjmCmZyoPhupCjRaRgiITc"),
    "TELEGRAM_CHAT_ID": os.environ.get("TELEGRAM_CHAT_ID", "203813932"),
    "DEBUG_MODE": os.environ.get("DEBUG_MODE", "True").lower() == "true",
    "SEND_TEST_MESSAGE": os.environ.get("SEND_TEST_MESSAGE", "True").lower() == "true",
    "DELTA_API_BASE": "https://api.india.delta.exchange",
    "PAIRS": ["BTCUSD", "ETHUSD", "SOLUSD", "AVAXUSD", "BCHUSD", "XRPUSD", "BNBUSD", "LTCUSD", "DOTUSD", "ADAUSD", "SUIUSD", "AAVEUSD"],
    "SPECIAL_PAIRS": {
        "SOLUSD": {"limit_15m": 210, "min_required": 180, "limit_5m": 300, "min_required_5m": 200}
    },
    "PPO_FAST": 7,
    "PPO_SLOW": 16,
    "PPO_SIGNAL": 5,
    "PPO_USE_SMA": False,
    "RMA_50_PERIOD": 50,
    "RMA_200_PERIOD": 200,
    "CIRRUS_CLOUD_ENABLED": True,
    "X1": 22, "X2": 9, "X3": 15, "X4": 5,
    "SRSI_RSI_LEN": 21, "SRSI_KALMAN_LEN": 5, "SRSI_EMA_LEN": 5,
    "STATE_DB_PATH": os.environ.get("STATE_DB_PATH", "macd_state.sqlite"),
    "LOG_FILE": os.environ.get("LOG_FILE", "macd_bot.log"),
    "MAX_PARALLEL_FETCH": 8,
    "HTTP_TIMEOUT": 15,
    "CANDLE_FETCH_RETRIES": 3,
    "CANDLE_FETCH_BACKOFF": 1.5,
    "JITTER_MIN": 0.05,
    "JITTER_MAX": 0.6,
    # daily prune: number of days to retain (0 disables pruning)
    "STATE_EXPIRY_DAYS": int(os.environ.get("STATE_EXPIRY_DAYS", "30"))
}

# Load config.json (optional) and overlay defaults
CONFIG_FILE = os.getenv("CONFIG_FILE", "config.json")

config = DEFAULT_CONFIG.copy()

if Path(CONFIG_FILE).exists():
    try:
        with open(CONFIG_FILE, "r") as f:
            user_cfg = json.load(f)
        config.update(user_cfg)  # JSON baseline
        # FIX: Ensure f-string is on one line to prevent SyntaxError
        print(f"Loaded configuration from {CONFIG_FILE}") 
    except Exception as e:
        print(f"⚠️ Warning: unable to parse {CONFIG_FILE}: {e}")
else:
    print(f"⚠️ Warning: config file {CONFIG_FILE} not found, using defaults.")

# Now override with environment variables (YAML wins)
config["DEBUG_MODE"] = os.getenv("DEBUG_MODE", str(config.get("DEBUG_MODE", True))).lower() == "true"
config["SEND_TEST_MESSAGE"] = os.getenv("SEND_TEST_MESSAGE", str(config.get("SEND_TEST_MESSAGE", True))).lower() == "true"
config["STATE_DB_PATH"] = os.getenv("STATE_DB_PATH", config.get("STATE_DB_PATH", "state.sqlite"))
config["LOG_FILE"] = os.getenv("LOG_FILE", config.get("LOG_FILE", "bot.log"))

cfg = config

# -------------------------
# Logger setup
# -------------------------
logger = logging.getLogger("macd_bot")
logger.setLevel(logging.DEBUG if cfg["DEBUG_MODE"] else logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG if cfg["DEBUG_MODE"] else logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# Rotating file handler
try:
    fh = RotatingFileHandler(cfg["LOG_FILE"], maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    # FIX: Ensure proper indentation for the try block body to prevent SyntaxError
    fh.setLevel(logging.DEBUG) 
    fh.setFormatter(formatter)
    logger.addHandler(fh)
except Exception as e:
    logger.warning(f"Could not set up rotating log file: {e}")

# -------------------------
# SQLite state store (StateDB)
# -------------------------
class StateDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        # Ensure directory exists if path includes directories
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        # FIX: Ensure proper indentation for __init__ body to prevent IndentationError
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._ensure_table()

    def _ensure_table(self):
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS states (
                pair TEXT PRIMARY KEY,
                state TEXT,
                ts INTEGER
            )
        """)
        # metadata table uses for pruning schedule
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

    def set(self, pair: str, state: str, ts: Optional[int] = None):
        ts = int(ts or time.time())
        cur = self._conn.cursor()
        cur.execute("INSERT INTO states(pair, state, ts) VALUES (?, ?, ?) "
                    "ON CONFLICT(pair) DO UPDATE SET state=excluded.state, ts=excluded.ts",
                    (pair, state, ts))
        self._conn.commit()

    def close(self):
        self._conn.close()

    # metadata helpers
    def get_metadata(self, key: str) -> Optional[str]:
        cur = self._conn.cursor()
        cur.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        r = cur.fetchone()
        return r[0] if r else None

    def set_metadata(self, key: str, value: str):
        cur = self._conn.cursor()
        cur.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", 
                    (key, value))
        self._conn.commit()

# -------------------------
# Pruning logic (daily)
# -------------------------
def prune_old_state_records(db_path: str, expiry_days: int = 30, logger_local: logging.Logger = None):
    """
    Run pruning once per UTC day. Uses metadata.last_prune ISO date to prevent
    repeated pruning within same day. Deletes rows where ts < cutoff.
    VACUUM only if deleted > 0.
    """
    try:
        if expiry_days <= 0:
            if logger_local:
                logger_local.debug("STATE_EXPIRY_DAYS <= 0 -> pruning disabled.")
            return

        if not os.path.exists(db_path):
            if logger_local:
                logger_local.info(f"State DB not found at {db_path}, skipping prune.")
            return

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        # ensure metadata table exists
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
                        logger_local.debug("Daily prune already completed — skipping.")
                    conn.close()
                    return
            except Exception:
                # malformed metadata -> continue and overwrite
                pass

        # ensure states table exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='states';")
        if not cur.fetchone():
            if logger_local:
                logger_local.debug("No 'states' table yet — skipping prune.")
            conn.close()
            return

        cutoff = int(time.time()) - (expiry_days * 86400)
        cur.execute("DELETE FROM states WHERE ts < ?", (cutoff,))
        deleted = cur.rowcount
        conn.commit()

        # write last_prune metadata
        cur.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('last_prune', ?)", (datetime.utcnow().isoformat(),))
        conn.commit()

        if deleted > 0:
            try:
                cur.execute("VACUUM;")
                conn.commit()
            except Exception as e:
                if logger_local:
                    logger_local.warning(f"VACUUM failed or skipped: {e}")

        conn.close()
        if logger_local:
            logger_local.info(f"Pruned {deleted} old state entries (> {expiry_days} days) from {db_path}.")
    except Exception as e:
        if logger_local:
            logger_local.warning(f"Error pruning old state records: {e}")
            if cfg.get("DEBUG_MODE"):
                logger_local.debug(traceback.format_exc())

# -------------------------
# HTTP helpers (async)
# -------------------------
async def async_fetch_json(session: aiohttp.ClientSession, url: str, params: dict = None,
                           retries: int = 3, backoff: float = 1.5, timeout: int = 15) -> Optional[dict]:
    """Fetch JSON with retry/backoff and error handling."""
    for attempt in range(1, retries + 1):
        try:
            async with session.get(url, params=params, timeout=timeout) as resp:
                text = await resp.text()
                try:
                    data = await resp.json()
                except Exception:
                    data = None
                if resp.status >= 400:
                    logger.debug(f"HTTP {resp.status} for {url} (attempt {attempt}): {text[:200]}")
                    raise ClientResponseError(resp.request_info, resp.history, status=resp.status)
                return data
        except (asyncio.TimeoutError, ClientConnectorError, ClientResponseError) as e:
            logger.debug(f"Fetch error {e} for {url} (attempt {attempt})")
            if attempt == retries:
                logger.warning(f"Failed to fetch {url} after {retries} attempts.")
                return None
            await asyncio.sleep(backoff * (attempt ** 1.2) + random.uniform(0, 0.2))
        except Exception as e:
            logger.exception(f"Unexpected error fetching {url}: {e}")
            return None
    return None

# -------------------------
# DataFetcher (async)
# -------------------------
class DataFetcher:
    def __init__(self, api_base: str, max_parallel: int = 8, timeout: int = 15):
        self.api_base = api_base.rstrip("/")
        self.semaphore = asyncio.Semaphore(max_parallel)
        self.timeout = timeout
        self._cache = {}  # simple per-run cache

    async def fetch_products(self, session: aiohttp.ClientSession) -> Optional[dict]:
        url = f"{self.api_base}/v2/products"
        async with self.semaphore:
            return await async_fetch_json(session, url, retries=3, backoff=1.5, timeout=self.timeout)

    async def fetch_candles(self, session: aiohttp.ClientSession, symbol: str, resolution: str, limit: int):
        """
        symbol: product symbol string as from product['symbol'] (e.g. "BTC_USDT")
        resolution: "15" or "5"
        limit: number of candles
        """
        key = f"candles:{symbol}:{resolution}:{limit}"
        if key in self._cache:
            return self._cache[key][1]

        # jitter to reduce thundering herd
        await asyncio.sleep(random.uniform(cfg["JITTER_MIN"], cfg["JITTER_MAX"]))

        url = f"{self.api_base}/v2/chart/history"
        params = {"resolution": resolution, "symbol": symbol, "from": int(time.time()) - (limit * int(resolution) * 60), "to": int(time.time())}
        async with self.semaphore:
            data = await async_fetch_json(session, url, params=params, retries=cfg["CANDLE_FETCH_RETRIES"], backoff=cfg["CANDLE_FETCH_BACKOFF"], timeout=cfg["HTTP_TIMEOUT"])
        self._cache[key] = (time.time(), data)
        return data

# -------------------------
# Indicator functions (preserved)
# -------------------------
def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    return data.ewm(span=period, adjust=False).mean()

def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    return data.rolling(window=period, min_periods=max(2, period//3)).mean()

def calculate_rma(data: pd.Series, period: int) -> pd.Series:
    r = data.ewm(alpha=1/period, adjust=False).mean()
    return r.bfill().ffill()

def calculate_ppo(df: pd.DataFrame, fast: int, slow: int, signal: int, use_sma: bool = False) -> Tuple[pd.Series, pd.Series]:
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
    result_list = [x.iloc[0]]
    for i in range(1, len(x)):
        prev_f = result_list[-1]
        curr_x = x.iloc[i]
        curr_r = max(float(r.iloc[i]), 1e-8)
        if curr_x > prev_f:
            f = prev_f if (curr_x - curr_r) < prev_f else (curr_x - curr_r)
        else:
            f = prev_f if (curr_x + curr_r) > prev_f else (curr_x + curr_r)
        result_list.append(f)
    return pd.Series(result_list, index=x.index)

def calculate_cirrus_cloud(df: pd.DataFrame):
    close = df['close'].copy()
    smrngx1x = smoothrng(close, cfg["X1"], cfg["X2"])
    smrngx1x2 = smoothrng(close, cfg["X3"], cfg["X4"])
    filtx1 = rngfilt(close, smrngx1x)
    filtx12 = rngfilt(close, smrngx1x2)
    upw = filtx1 < filtx12
    dnw = filtx1 > filtx12
    return upw, dnw, filtx1, filtx12

def kalman_filter(src: pd.Series, length: int, R=0.01, Q=0.1) -> pd.Series:
    result_list = []
    estimate = np.nan
    error_est = 1.0
    error_meas = R * max(1, length)
    Q_div_length = Q / max(1, length)
    for i in range(len(src)):
        current_src = src.iloc[i]
        if np.isnan(estimate):
            if i > 0:
                estimate = src.iloc[i-1]
            else:
                result_list.append(np.nan)
                continue
        prediction = estimate
        kalman_gain = error_est / (error_est + error_meas)
        estimate = prediction + kalman_gain * (current_src - prediction)
        error_est = (1 - kalman_gain) * error_est + Q_div_length
        result_list.append(estimate)
    return pd.Series(result_list, index=src.index)

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
    responsiveness = max(1e-5, float(responsiveness))
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
# Validation helpers
# -------------------------
def parse_candles_result(result: dict) -> Optional[pd.DataFrame]:
    """Convert Delta API candle response to pandas DataFrame, with safety checks."""
    if not result or not isinstance(result, dict):
        return None
    if not result.get("success"):
        return None
    res = result.get("result", {})
    arrays = [res.get('t', []), res.get('o', []), res.get('h', []), res.get('l', []), res.get('c', []), res.get('v', [])]
    if not arrays or any(len(arr) == 0 for arr in arrays):
        return None
    min_len = min(map(len, arrays))
    df = pd.DataFrame({
        'timestamp': res.get('t', [])[:min_len],
        'open': res.get('o', [])[:min_len],
        'high': res.get('h', [])[:min_len],
        'low': res.get('l', [])[:min_len],
        'close': res.get('c', [])[:min_len],
        'volume': res.get('v', [])[:min_len]
    })
    df = df.sort_values('timestamp').drop_duplicates(subset='timestamp').reset_index(drop=True)
    # basic check
    if df.empty or df['close'].astype(float).iloc[-1] <= 0:
        return None
    return df

# -------------------------
# Timestamp / closed-index (replaced original determine_closed_indices)
# -------------------------
def get_last_closed_indices(df_15m: pd.DataFrame, df_5m: pd.DataFrame) -> Tuple[int, int, int]:
    """Return safe indices for last closed candles across 15m and 5m timeframes."""
    now_ts = int(time.time())

    def closed_index_for(df: pd.DataFrame, resolution_min: int) -> int:
        if df is None or df.empty:
            return -1
        # Last candle timestamp from API is usually the start of that candle's interval
        last_ts = int(df['timestamp'].iloc[-1])
        current_interval_start = now_ts - (now_ts % (resolution_min * 60))
        # If latest candle timestamp is within current open interval, it's incomplete
        if last_ts >= current_interval_start:
            return -2  # use -2 to refer to previous closed candle
        return -1  # most recent candle is closed; use -1

    last_i = closed_index_for(df_15m, 15)
    prev_i = last_i - 1
    last_i_5m = closed_index_for(df_5m, 5)
    return last_i, prev_i, last_i_5m

# -------------------------
# Evaluation logic (modular)
# -------------------------
def determine_closed_indices(df_15m: pd.DataFrame, df_5m: pd.DataFrame) -> Tuple[int, int, int]:
    # kept for backward compatibility if some modules call it; delegate to new function
    return get_last_closed_indices(df_15m, df_5m)

def evaluate_pair_logic(pair_name: str, df_15m: pd.DataFrame, df_5m: pd.DataFrame, last_state_for_pair: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Evaluate indicators and conditions; return new state dict if changed (with optional message)."""
    try:
        last_i, prev_i, last_i_5m = get_last_closed_indices(df_15m, df_5m)

        magical_hist = calculate_magical_momentum_hist(df_15m, period=144, responsiveness=0.9)
        # Check for enough data points for MMH reversal logic (at least last_i, last_i-1, last_i-2, last_i-3)
        if magical_hist.empty or len(magical_hist) < abs(last_i) + 4:
            logger.debug(f"MMH empty or insufficient data ({len(magical_hist)}) -> skip")
            return None
        
        # Get MMH values for last 4 closed candles
        # Note: indices are relative to the end of the series. If last_i is -1, then last_i-3 is -4 (4th last candle)
        mmh_curr = float(magical_hist.iloc[last_i]) # MMH[-1] or MMH[-2]
        mmh_prev1 = float(magical_hist.iloc[last_i - 1]) # MMH[-2] or MMH[-3]
        mmh_prev2 = float(magical_hist.iloc[last_i - 2]) # MMH[-3] or MMH[-4]
        mmh_prev3 = float(magical_hist.iloc[last_i - 3]) # MMH[-4] or MMH[-5]
        
        # Existing indicator calculations
        ppo, ppo_signal = calculate_ppo(df_15m, cfg["PPO_FAST"], cfg["PPO_SLOW"], cfg["PPO_SIGNAL"], cfg["PPO_USE_SMA"])
        rma_50 = calculate_rma(df_15m['close'], cfg["RMA_50_PERIOD"])
        rma_200 = calculate_rma(df_5m['close'], cfg["RMA_200_PERIOD"])
        upw, dnw, filtx1, filtx12 = calculate_cirrus_cloud(df_15m)
        smooth_rsi = calculate_smooth_rsi(df_15m, cfg["SRSI_RSI_LEN"], cfg["SRSI_KALMAN_LEN"])

        # extract last closed values
        ppo_curr = float(ppo.iloc[last_i]); ppo_prev = float(ppo.iloc[prev_i])
        ppo_signal_curr = float(ppo_signal.iloc[last_i]); ppo_signal_prev = float(ppo_signal.iloc[prev_i])
        smooth_rsi_curr = float(smooth_rsi.iloc[last_i]); smooth_rsi_prev = float(smooth_rsi.iloc[prev_i])

        close_curr = float(df_15m['close'].iloc[last_i])
        open_curr = float(df_15m['open'].iloc[last_i])
        high_curr = float(df_15m['high'].iloc[last_i])
        low_curr = float(df_15m['low'].iloc[last_i])

        rma50_curr = float(rma_50.iloc[last_i])
        rma200_curr = float(rma_200.iloc[last_i_5m])

        # safety checks (added mmh values to safety check)
        if any(pd.isna(x) for x in [ppo_curr, ppo_prev, ppo_signal_curr, ppo_signal_prev, smooth_rsi_curr, smooth_rsi_prev, rma50_curr, rma200_curr, mmh_curr, mmh_prev1, mmh_prev2, mmh_prev3]):
            logger.debug(f"NaN in core indicators for {pair_name}, skipping")
            return None

        # candle metrics
        total_range = high_curr - low_curr
        if total_range <= 0:
            upper_wick = 0.0; lower_wick = 0.0; strong_bullish_close = False; strong_bearish_close = False
        else:
            upper_wick = max(0.0, high_curr - max(open_curr, close_curr))
            lower_wick = max(0.0, min(open_curr, close_curr) - low_curr)
            bullish_candle = close_curr > open_curr
            bearish_candle = close_curr < open_curr
            strong_bullish_close = bullish_candle and (upper_wick / total_range) < 0.20
            strong_bearish_close = bearish_candle and (lower_wick / total_range) < 0.20

        # cross/band logic (PPO-Signal crossover/crossunder removed)
        # ppo_cross_up = (ppo_prev <= ppo_signal_prev) and (ppo_curr > ppo_signal_curr) # REMOVED
        # ppo_cross_down = (ppo_prev >= ppo_signal_prev) and (ppo_curr < ppo_signal_curr) # REMOVED
        ppo_cross_above_zero = (ppo_prev <= 0) and (ppo_curr > 0)
        ppo_cross_below_zero = (ppo_prev >= 0) and (ppo_curr < 0)
        ppo_cross_above_011 = (ppo_prev <= 0.11) and (ppo_curr > 0.11)
        ppo_cross_below_minus011 = (ppo_prev >= -0.11) and (ppo_curr < -0.11)

        ppo_below_020 = ppo_curr < 0.20
        ppo_above_minus020 = ppo_curr > -0.20
        ppo_above_signal = ppo_curr > ppo_signal_curr
        ppo_below_signal = ppo_curr < ppo_signal_curr

        ppo_below_030 = ppo_curr < 0.30
        ppo_above_minus030 = ppo_curr > -0.30

        close_above_rma50 = close_curr > rma50_curr
        close_below_rma50 = close_curr < rma50_curr
        close_above_rma200 = close_curr > rma200_curr
        close_below_rma200 = close_curr < rma200_curr
        
        # New conditions for MMH reversal (last 4 closed candles)
        # 3 falling (prev3 > prev2 > prev1) then 1 rising (curr > prev1)
        mmh_reversal_buy = (mmh_prev3 > mmh_prev2 and mmh_prev2 > mmh_prev1 and mmh_curr > mmh_prev1) 
        # 3 rising (prev3 < prev2 < prev1) then 1 falling (curr < prev1)
        mmh_reversal_sell = (mmh_prev3 < mmh_prev2 and mmh_prev2 < mmh_prev1 and mmh_curr < mmh_prev1) 

        srsi_cross_up_50 = (smooth_rsi_prev <= 50) and (smooth_rsi_curr > 50)
        srsi_cross_down_50 = (smooth_rsi_prev >= 50) and (smooth_rsi_curr < 50)

        cloud_state = "neutral"
        if cfg["CIRRUS_CLOUD_ENABLED"]:
            cloud_state = ("green" if (bool(upw.iloc[last_i]) and not bool(dnw.iloc[last_i]))
                           else "red" if (bool(dnw.iloc[last_i]) and not bool(upw.iloc[last_i]))
                           else "neutral")

        # condition sets (modified to remove ppo cross and add mmh reversal logic)
        
        # New MMH BUY Reversal Condition Set
        # "Rma50 for 15 minutes below close" -> close_above_rma50
        # "Rma200 for 5 minutes below close" -> close_above_rma200
        buy_mmh_reversal_conds_corrected = {
            "mmh_reversal_buy": mmh_reversal_buy,
            "close_above_rma50": close_above_rma50,
            "close_above_rma200": close_above_rma200,
            "magical_hist_curr>0": mmh_curr > 0,
            "cloud_green": (cloud_state == "green"),
            "strong_bullish_close": strong_bullish_close,
        }

        # New MMH SELL Reversal Condition Set
        # "Rma50 for 15 minutes above close" -> close_below_rma50
        # "Rma200 for 5 minutes above close" -> close_below_rma200
        sell_mmh_reversal_conds_corrected = {
            "mmh_reversal_sell": mmh_reversal_sell,
            "close_below_rma50": close_below_rma50,
            "close_below_rma200": close_below_rma200,
            "magical_hist_curr<0": mmh_curr < 0,
            "cloud_red": (cloud_state == "red"),
            "strong_bearish_close": strong_bearish_close,
        }
        
        # Keeping existing SRSI and PPO(0, 0.11) conditions
        buy_srsi_conds = {
            "srsi_cross_up_50": srsi_cross_up_50,
            "ppo_above_signal": ppo_above_signal,
            "ppo_below_030": ppo_below_030,
            "close_above_rma50": close_above_rma50,
            "close_above_rma200": close_above_rma200,
            "cloud_green": (cloud_state == "green"),
            "strong_bullish_close": strong_bullish_close,
            "magical_hist_curr>0": mmh_curr > 0,
        }
        sell_srsi_conds = {
            "srsi_cross_down_50": srsi_cross_down_50,
            "ppo_below_signal": ppo_below_signal,
            "ppo_above_minus030": ppo_above_minus030,
            "close_below_rma50": close_below_rma50,
            "close_below_rma200": close_below_rma200,
            "cloud_red": (cloud_state == "red"),
            "strong_bearish_close": strong_bearish_close,
            "magical_hist_curr<0": mmh_curr < 0,
        }
        long0_conds = {
            "ppo_cross_above_zero": ppo_cross_above_zero,
            "ppo_above_signal": ppo_above_signal,
            "close_above_rma50": close_above_rma50,
            "close_above_rma200": close_above_rma200,
            "cloud_green": (cloud_state == "green"),
            "strong_bullish_close": strong_bullish_close,
            "magical_hist_curr>0": mmh_curr > 0,
        }
        long011_conds = {
            "ppo_cross_above_011": ppo_cross_above_011,
            "ppo_above_signal": ppo_above_signal,
            "close_above_rma50": close_above_rma50,
            "close_above_rma200": close_above_rma200,
            "cloud_green": (cloud_state == "green"),
            "strong_bullish_close": strong_bullish_close,
            "magical_hist_curr>0": mmh_curr > 0,
        }
        short0_conds = {
            "ppo_cross_below_zero": ppo_cross_below_zero,
            "ppo_below_signal": ppo_below_signal,
            "close_below_rma50": close_below_rma50,
            "close_below_rma200": close_below_rma200,
            "cloud_red": (cloud_state == "red"),
            "strong_bearish_close": strong_bearish_close,
            "magical_hist_curr<0": mmh_curr < 0,
        }
        short011_conds = {
            "ppo_cross_below_minus011": ppo_cross_below_minus011,
            "ppo_below_signal": ppo_below_signal,
            "close_below_rma50": close_below_rma50,
            "close_below_rma200": close_below_rma200,
            "cloud_red": (cloud_state == "red"),
            "strong_bearish_close": strong_bearish_close,
            "magical_hist_curr<0": mmh_curr < 0,
        }

        # state determination & idempotency
        current_state = None
        send_message = None
        price = close_curr
        ist = pytz.timezone('Asia/Kolkata')
        current_dt = datetime.now(ist)
        formatted_time = current_dt.strftime('%d-%m-%Y @ %H:%M IST')

        # --- NEW MMH REVERSAL LOGIC (Using Green/Red Dots) ---
        if all(buy_mmh_reversal_conds_corrected.values()):
            current_state = "buy_mmh_reversal"
            send_message = f"🟢 {pair_name} - BUY (MMH Reversal)\nMMH 15m Reversal Up ({mmh_curr:.5f})\nPrice: ${price:,.2f}\n{formatted_time}"
            
        if current_state is None and all(sell_mmh_reversal_conds_corrected.values()):
            current_state = "sell_mmh_reversal"
            send_message = f"🔴 {pair_name} - SELL (MMH Reversal)\nMMH 15m Reversal Down ({mmh_curr:.5f})\nPrice: ${price:,.2f}\n{formatted_time}"
            
        # --- Remaining existing alerts ---
        if current_state is None and all(buy_srsi_conds.values()):
            current_state = "buy_srsi50"
            send_message = f"⬆️ {pair_name} - BUY (SRSI 50)\nSRSI 15m Cross Up 50 ({smooth_rsi_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
        if current_state is None and all(sell_srsi_conds.values()):
            current_state = "sell_srsi50"
            send_message = f"⬇️ {pair_name} - SELL (SRSI 50)\nSRSI 15m Cross Down 50 ({smooth_rsi_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
 
        if current_state is None and all(long0_conds.values()):
            current_state = "long_zero"
            send_message = f"🟢 {pair_name} - LONG\nPPO crossing above 0 ({ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
        if current_state is None and all(long011_conds.values()):
            current_state = "long_011"
            send_message = f"🟢 {pair_name} - LONG\nPPO crossing above 0.11 ({ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
   
        if current_state is None and all(short0_conds.values()):
            current_state = "short_zero"
            send_message = f"🔴 {pair_name} - SHORT\nPPO crossing below 0 ({ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
        if current_state is None and all(short011_conds.values()):
            current_state = "short_011"
            send_message = f"🔴 {pair_name} - SHORT\nPPO crossing below -0.11 ({ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"

        now_ts = int(time.time())
        last_state_value = None
        if isinstance(last_state_for_pair, dict):
            last_state_value = last_state_for_pair.get("state")
        elif isinstance(last_state_for_pair, str):
            last_state_value = last_state_for_pair

        if current_state is None:
            # transition back to NO_SIGNAL if previously was a signal
            if last_state_value != 'NO_SIGNAL' and last_state_value is not None:
                return {"state": "NO_SIGNAL", "ts": now_ts}
            return last_state_for_pair

        # idempotency check
        if current_state == last_state_value:
            logger.debug(f"Idempotency: {pair_name} signal remains {current_state}.")
            return last_state_for_pair

        # return new signal and message (message popped & sent by caller)
        return {"state": current_state, "ts": now_ts, "message": send_message}

    except Exception as e:
        logger.exception(f"Error evaluating logic for {pair_name}: {e}")
        return None

# -------------------------
# Telegram sender (async)
# -------------------------
async def send_telegram_alert_async(session: aiohttp.ClientSession, message: str) -> bool:
    token = cfg["TELEGRAM_BOT_TOKEN"]
    chat_id = cfg["TELEGRAM_CHAT_ID"]
    if not token or token == 'xxxx' or not chat_id or chat_id == 'xxxx':
        logger.warning("Telegram not configured; skipping alert.")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    try:
        async with session.post(url, data=data, timeout=10) as resp:
            try:
                js = await resp.json(content_type=None)
            except Exception:
                js = {"ok": False, "status_code": resp.status, "text": await resp.text()}
            ok = js.get("ok", False)
            if ok:
                logger.info("Alert sent successfully")
                return True
            else:
                logger.warning(f"Telegram API responded with error: {js}")
                return False
    except Exception as e:
        logger.exception(f"Error sending telegram alert: {e}")
        return False

# -------------------------
# Orchestration: check a single pair
# -------------------------
async def check_pair(session: aiohttp.ClientSession, fetcher: DataFetcher, products_map: Dict[str, dict],
                     pair_name: str, last_state_for_pair: Optional[Dict[str, Any]]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Fetch candles, compute indicators, evaluate conditions, send alert if needed, return new state."""
    try:
        prod = products_map.get(pair_name)
        if not prod:
            logger.debug(f"No product mapping for {pair_name}")
            return None

        sp = cfg["SPECIAL_PAIRS"].get(pair_name, {})
        limit_15m = sp.get("limit_15m", 210)
        min_required = sp.get("min_required", 200)
        limit_5m = sp.get("limit_5m", 300)
        min_required_5m = sp.get("min_required_5m", 250)

        symbol = prod['symbol']
        # fetch both timeframes
        res15_task = asyncio.create_task(fetcher.fetch_candles(session, symbol, "15", limit_15m))
        res5_task = asyncio.create_task(fetcher.fetch_candles(session, symbol, "5", limit_5m))
        res15 = await res15_task
        res5 = await res5_task

        df_15m = parse_candles_result(res15)
        df_5m = parse_candles_result(res5)

        if df_15m is None or len(df_15m) < (min_required + 2):
            logger.warning(f"Insufficient 15m data for {pair_name}: {0 if df_15m is None else len(df_15m)}/{min_required + 2}")
            return None
        if df_5m is None or len(df_5m) < (min_required_5m + 2):
            logger.warning(f"Insufficient 5m data for {pair_name}: {0 if df_5m is None else len(df_5m)}/{min_required_5m + 2}")
            return None

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_15m[col] = pd.to_numeric(df_15m[col], errors='coerce')
            df_5m[col] = pd.to_numeric(df_5m[col], errors='coerce')

        new_state = evaluate_pair_logic(pair_name, df_15m, df_5m, last_state_for_pair)
        if new_state is None:
            return None

        message = new_state.pop("message", None)
        if message:
            await send_telegram_alert_async(session, message)

        return pair_name, new_state

    except Exception as e:
        logger.exception(f"Error in check_pair for {pair_name}: {e}")
        return None

# -------------------------
# Product mapping helper
# -------------------------
def build_products_map_from_api_result(api_products: dict) -> Dict[str, dict]:
    products_map = {}
    if not api_products or not api_products.get("result"):
        return products_map
    for p in api_products['result']:
        try:
            symbol = p.get('symbol', '')
            symbol_norm = symbol.replace('_USDT', 'USD').replace('USDT', 'USD')
            if p.get('contract_type') == 'perpetual_futures':
                for pair_name in cfg["PAIRS"]:
                    if symbol_norm == pair_name or symbol_norm.replace('_', '') == pair_name:
                        products_map[pair_name] = {'id': p.get('id'), 'symbol': p.get('symbol'), 'contract_type': p.get('contract_type')}
        except Exception:
            continue
    return products_map

# -------------------------
# Runner
# -------------------------
async def run_once(send_test: bool = True):
    logger.info("Starting run_once")

    # --- Smart daily pruning before DB open ---
    expiry_days = int(cfg.get("STATE_EXPIRY_DAYS", 0))
    try:
        prune_old_state_records(cfg["STATE_DB_PATH"], expiry_days, logger)
    except Exception as e:
        logger.warning(f"Prune step failed: {e}")

    # Load last alerts
    state_db = StateDB(cfg["STATE_DB_PATH"])
    last_alerts = state_db.load_all()

    fetcher = DataFetcher(cfg["DELTA_API_BASE"], max_parallel=cfg["MAX_PARALLEL_FETCH"], timeout=cfg["HTTP_TIMEOUT"])
    async with aiohttp.ClientSession() as session:
        # startup test message
        if send_test and cfg["SEND_TEST_MESSAGE"] and cfg["TELEGRAM_BOT_TOKEN"] != 'xxxx' and cfg["TELEGRAM_CHAT_ID"] != 'xxxx':
            ist = pytz.timezone('Asia/Kolkata')
            current_dt = datetime.now(ist)
            formatted_time = current_dt.strftime('%d-%m-%Y @ %H:%M IST')
            test_msg = f"🔔 Bot Started\nTest message from PPO Bot\nTime: {formatted_time}\nDebug Mode: {'ON' if cfg['DEBUG_MODE'] else 'OFF'}"
            await send_telegram_alert_async(session, test_msg)
        elif send_test and (cfg["TELEGRAM_BOT_TOKEN"] == 'xxxx' or cfg["TELEGRAM_CHAT_ID"] == 'xxxx'):
            logger.info("Skipping test message: Telegram not configured.")

        # fetch products
        prod_resp = await fetcher.fetch_products(session)
        if not prod_resp:
            logger.error("Failed to fetch products. Aborting run.")
            state_db.close()
            return
        products_map = build_products_map_from_api_result(prod_resp)
        found_count = len(products_map)
        logger.info(f"Found {found_count} tradable pairs mapped to configuration.")
        if found_count == 0:
            logger.error("No pairs found. Exiting.")
            state_db.close()
            return

        # prepare tasks
        tasks = []
        sem = asyncio.Semaphore(cfg["MAX_PARALLEL_FETCH"])
        for pair_name in cfg["PAIRS"]:
            prod_info = products_map.get(pair_name)
            if not prod_info:
                continue
            last_state = last_alerts.get(pair_name)
            async def run_check(pair=pair_name, prod=prod_info, last_state_for_pair=last_state):
                async with sem:
                    await asyncio.sleep(random.uniform(cfg["JITTER_MIN"], cfg["JITTER_MAX"]))
                    return await check_pair(session, fetcher, products_map, pair, last_state_for_pair)
            tasks.append(asyncio.create_task(run_check()))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # persist changes
        updates = 0
        for item in results:
            if isinstance(item, Exception):
                logger.exception(f"Task exception: {item}")
                continue
            if not item:
                continue
            pair_name, new_state = item
            if not isinstance(new_state, dict):
                continue
            prev = state_db.get(pair_name)
            if prev != new_state:
                state_db.set(pair_name, new_state.get("state"), new_state.get("ts"))
                updates += 1

        logger.info(f"Run complete. {updates} state updates applied.")
        state_db.close()

# -------------------------
# CLI loop and graceful shutdown
# -------------------------
stop_requested = False
def request_stop(signum, frame):
    global stop_requested
    logger.info(f"Received signal {signum}, stopping gracefully...")
    stop_requested = True

signal.signal(signal.SIGINT, request_stop)
signal.signal(signal.SIGTERM, request_stop)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PPO/Cirrus alert bot (improved)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--once", action="store_true", help="Run once and exit")
    group.add_argument("--loop", type=int, metavar="SECONDS", help="Run in a loop every N seconds")
    args = parser.parse_args()

    if args.once:
        asyncio.run(run_once())
    elif args.loop:
        interval = args.loop
        logger.info(f"Starting loop mode (interval={interval}s). Ctrl-C to stop.")
        while not stop_requested:
            start = time.time()
            try:
                asyncio.run(run_once())
            except Exception:
                logger.exception("Unhandled exception in run_once")
            elapsed = time.time() - start
            to_sleep = max(0, interval - elapsed)
            if to_sleep > 0:
                logger.debug(f"Sleeping {to_sleep:.1f}s until next run")
                time.sleep(to_sleep)
    else:
        # default: single run
        asyncio.run(run_once())

if __name__ == "__main__":
    main()
