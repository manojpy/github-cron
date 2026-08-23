from __future__ import annotations
import time
import random
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import deque
from typing import Dict, Any, Optional, Tuple, List, ClassVar, Callable
import ssl
import aiohttp
from aiohttp import ClientConnectorError, ClientResponseError, TCPConnector, ClientError
import numpy as np

from bot_config import (
    cfg, logger, logger_main, Constants, json_dumps, json_loads, JSONDecodeError,
    __version__, shutdown_event, format_ist_time, normalize_timestamp, normalize_timestamp_array,
)
from indicators import get_utc_date_key, _prior_leg_direction

class SessionManager:
    _session: ClassVar[Optional[aiohttp.ClientSession]] = None
    _ssl_context: ClassVar[Optional[ssl.SSLContext]] = None
    _lock: ClassVar[Optional[asyncio.Lock]] = None
    _creation_time: ClassVar[float] = 0.0

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    @classmethod
    def _get_ssl_context(cls) -> ssl.SSLContext:
        if cls._ssl_context is None:
            ctx = ssl.create_default_context()
            ctx.check_hostname = True
            ctx.verify_mode = ssl.CERT_REQUIRED
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            cls._ssl_context = ctx
            logger.debug("SSL context created with TLSv1.2+ minimum")
        return cls._ssl_context

    @classmethod
    async def get_session(cls) -> aiohttp.ClientSession:
        old_session_to_close: Optional[aiohttp.ClientSession] = None
        async with cls._get_lock():  
            should_recreate = cls._session is None or cls._session.closed
            if should_recreate:
                if cls._session and not cls._session.closed:
                    old_session_to_close = cls._session

                connector = TCPConnector(
                    limit=max(cfg.TCP_CONN_LIMIT, cfg.MAX_PARALLEL_FETCH),
                    limit_per_host=max(cfg.TCP_CONN_LIMIT_PER_HOST, cfg.MAX_PARALLEL_FETCH),
                    ssl=cls._get_ssl_context(),
                    force_close=False,
                    enable_cleanup_closed=True,
                    ttl_dns_cache=3600,
                    keepalive_timeout=90,
                    family=0,
                )

                timeout = aiohttp.ClientTimeout(
                    total=cfg.HTTP_TIMEOUT,
                    connect=8,
                    sock_read=cfg.HTTP_TIMEOUT,
                )

                cls._session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers={
                        "User-Agent": f"{cfg.BOT_NAME}/{__version__}",
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip, deflate",
                        "Connection": "keep-alive",
                    },
                    raise_for_status=False,
                )
                cls._creation_time = time.monotonic()

                if cfg.DEBUG_MODE:
                    logger.debug("HTTP session created")

            new_session = cls._session

        if old_session_to_close is not None:
            try:
                await old_session_to_close.close()
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Error closing old session: {e}")

        return new_session

    @classmethod
    async def close_session(cls) -> None:
        session_to_close: Optional[aiohttp.ClientSession] = None
        session_age = 0.0
        async with cls._get_lock():
            if cls._session and not cls._session.closed:
                session_to_close = cls._session
                session_age = time.monotonic() - cls._creation_time
                cls._session = None
                cls._creation_time = 0.0
            else:
                logger.debug("Session already closed or not created")

        if session_to_close is not None:
            try:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Closing HTTP session | Age: {session_age:.1f}s"
                    )
                await session_to_close.close()
                await asyncio.sleep(0.1)  # OPTIMIZED: Reduced from 0.25s
                logger.info("HTTP session closed successfully")
            except Exception as e:
                logger.warning(f"Error closing session: {e}")

    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        if cls._session is None:
            return {
                "active": False,
                "age_seconds": 0.0,
            }
        age = time.monotonic() - cls._creation_time if cls._creation_time > 0 else 0.0
        return {
            "active": not cls._session.closed,
            "age_seconds": round(age, 1),
        }

class RetryCategory:
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    API_ERROR = "api_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

def categorize_exception(exc: Exception) -> str:
    if isinstance(exc, asyncio.TimeoutError):
        return RetryCategory.TIMEOUT
    elif isinstance(exc, ClientConnectorError):
        return RetryCategory.NETWORK
    elif isinstance(exc, ClientResponseError):
        if hasattr(exc, "status") and exc.status == 429:
            return RetryCategory.RATE_LIMIT
        return RetryCategory.API_ERROR
    elif isinstance(exc, ClientError):
        return RetryCategory.NETWORK
    return RetryCategory.UNKNOWN

def compute_backoff(base: float, attempt: int, cap: float = 30.0, jitter_range: Tuple[float, float] = (0.1, 0.5)) -> float:
    """Exponential backoff with jitter. base=starting delay in seconds, attempt=1-indexed retry count."""
    base_delay = min(base * (2 ** (attempt - 1)), cap)
    jitter = base_delay * random.uniform(*jitter_range)
    return base_delay + jitter

async def async_fetch_json(url: str, params: Optional[Dict[str, Any]] = None, retries: int = 3, backoff: float = 1.5, timeout: int = 15) -> Optional[Dict[str, Any]]:   
    session = await SessionManager.get_session()    
    retry_stats = {
        RetryCategory.NETWORK: 0,
        RetryCategory.RATE_LIMIT: 0,
        RetryCategory.API_ERROR: 0,
        RetryCategory.TIMEOUT: 0,
        RetryCategory.UNKNOWN: 0
    }
    last_error: Optional[Exception] = None
    
    for attempt in range(1, retries + 1):
        if shutdown_event.is_set():
            logger.debug(f"Shutdown requested, aborting fetch: {url[:80]}")
            return None
        
        try:
            async with session.get(url, params=params, timeout=timeout) as resp:
                if resp.status == 429:
                    retry_after = resp.headers.get('Retry-After')
                    try:
                        retry_val = int(retry_after) if retry_after else 2
                    except (ValueError, TypeError):
                        retry_val = 5
                    wait_sec = min(retry_val, Constants.CIRCUIT_BREAKER_MAX_WAIT)
                    jitter = random.uniform(0.1, 0.5)
                    total_wait = wait_sec + jitter             
                    retry_stats[RetryCategory.RATE_LIMIT] += 1
                    logger.warning(
                        f"Rate limited (429) | URL: {url[:80]} | "
                        f"Retry-After: {retry_after}s | Waiting: {total_wait:.2f}s | "
                        f"Attempt: {attempt}/{retries}"
                    )
                    
                    await asyncio.sleep(total_wait)
                    continue
                
                if resp.status >= 500:
                    retry_stats[RetryCategory.API_ERROR] += 1
                    logger.warning(
                        f"Server error {resp.status} | URL: {url[:80]} | "
                        f"Attempt: {attempt}/{retries}"
                    )          
                    if attempt < retries:
                        total_delay = compute_backoff(backoff, attempt, cap=Constants.CIRCUIT_BREAKER_MAX_WAIT / 10)
                        await asyncio.sleep(total_delay)
                    continue

                if resp.status >= 400:
                    logger.error(
                        f"Client error {resp.status} for {url[:80]} | "
                        f"This usually indicates invalid request - not retrying"
                    )
                    return False
                try:
                    data = await resp.json(loads=json_loads)
                except (JSONDecodeError, TypeError, ValueError) as e:
                    retry_stats[RetryCategory.API_ERROR] += 1
                    logger.warning(
                        f"Malformed JSON on 200 OK (attempt {attempt}/{retries}) | "
                        f"URL: {url[:80]} | Error: {str(e)[:100]}"
                    )
                    if attempt < retries:
                        total_delay = compute_backoff(backoff, attempt, cap=Constants.CIRCUIT_BREAKER_MAX_WAIT / 10)
                        await asyncio.sleep(total_delay)
                    continue         
                if any(retry_stats.values()):
                    logger.info(
                        f"Fetch succeeded after retries | URL: {url[:80]} | "
                        f"Attempts: {attempt} | Stats: {retry_stats}"
                    )
                
                return data
                
        except asyncio.TimeoutError as e:
            last_error = e
            retry_stats[RetryCategory.TIMEOUT] += 1
            logger.warning(
                f"Timeout (attempt {attempt}/{retries}) | "
                f"URL: {url[:80]} | Timeout configured: {timeout}s"
            )
            if attempt < retries:
                total_delay = compute_backoff(backoff, attempt, cap=Constants.CIRCUIT_BREAKER_MAX_WAIT / 10)
                logger.debug(f"Retrying after {total_delay:.2f}s...")
                await asyncio.sleep(total_delay)

        except ClientError as e:
            last_error = e
            category = categorize_exception(e)
            retry_stats[category] = retry_stats.get(category, 0) + 1
            
            logger.warning(
                f"Network error (attempt {attempt}/{retries}) | "
                f"Category: {category} | URL: {url[:80]} | Error: {str(e)[:100]}"
            )
            if attempt < retries:
                total_delay = compute_backoff(backoff, attempt, cap=Constants.CIRCUIT_BREAKER_MAX_WAIT / 10)
                logger.debug(f"Retrying after {total_delay:.2f}s...")
                await asyncio.sleep(total_delay)

        except Exception as e:
            last_error = e
            retry_stats[RetryCategory.UNKNOWN] += 1
            logger.exception(f"Unexpected fetch error for {url[:80]}: {e}")
            break    
    logger.error(
        f"Failed to fetch after {retries} attempts | URL: {url[:80]} | "
        f"Stats: {retry_stats} | Last error: {last_error}"
    )
    return None

class RateLimitedFetcher:
    def __init__(self, max_per_minute: int = 60, concurrency: int = 4):
        self.max_per_minute = max_per_minute
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        self.requests: deque[float] = deque()
        self.lock = asyncio.Lock()
        self.total_waits = 0
        self.total_wait_time = 0.0
        self.last_request_time = 0.0

    async def call(self, func: Callable, *args, **kwargs):
        claimed_ts: Optional[float] = None
        while True:
            sleep_needed = 0.0
            async with self.lock:
                now = time.monotonic()
                while self.requests and now - self.requests[0] > 60.0:
                    self.requests.popleft()
                if len(self.requests) < self.max_per_minute:
                    self.requests.append(now)
                    self.last_request_time = now
                    claimed_ts = now
                    break
                else:
                    oldest_request_age = now - self.requests[0]
                    wait_needed = max(0.0, 60.0 - oldest_request_age)
                    sleep_needed = wait_needed + random.uniform(0.05, 0.2)
                    self.total_waits += 1
                    logger.debug(
                        f"Rate limit reached ({len(self.requests)}/{self.max_per_minute}), "
                        f"sleeping {sleep_needed:.2f}s | Total waits: {self.total_waits}"
                    )
            
            t0 = time.monotonic()
            try:
                await asyncio.sleep(sleep_needed)
            except asyncio.CancelledError:
                self.total_wait_time += max(0.0, time.monotonic() - t0)
                raise
            self.total_wait_time += time.monotonic() - t0

        try:
            async with self.semaphore:
                return await func(*args, **kwargs)
        except asyncio.CancelledError:
            # Give the rate-window slot back — it was claimed before the call started,
            # so a cancellation here shouldn't permanently burn it.
            async with self.lock:
                try:
                    self.requests.remove(claimed_ts)
                except ValueError:
                    pass  # already pruned by the 60s window cleanup above
            raise 

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_waits": self.total_waits,
            "total_wait_time_seconds": round(self.total_wait_time, 2),
            "current_queue_size": len(self.requests),
            "max_per_minute": self.max_per_minute,
            "concurrency_limit": self.concurrency,
            "requests_in_window": len(self.requests),
        }

class APICircuitBreaker:  
    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"
        self.success_count = 0
        self._lock = asyncio.Lock()          # NEW

    async def record_success(self) -> None:  # NEW: async
        async with self._lock:
            if self.state == "HALF_OPEN":
                self.success_count += 1
                if self.success_count >= 2:
                    logger.info("💫 Circuit breaker: Recovered, transitioning to CLOSED")
                    self.state = "CLOSED"
                    self.failures = 0
                    self.success_count = 0
            elif self.state == "CLOSED":
                if self.failures > 0:
                    self.failures = max(0, self.failures - 1)

    async def record_failure(self) -> None:   # NEW: async
        async with self._lock:
            self.failures += 1
            self.last_failure_time = time.time()

            if self.failures >= self.failure_threshold and self.state == "CLOSED":
                logger.warning(
                    f"⚠️ Circuit breaker: OPENED after {self.failures} failures. "
                    f"Blocking requests for {self.recovery_timeout}s"
                )
                self.state = "OPEN"

    async def can_attempt(self) -> Tuple[bool, Optional[str]]:  # NEW: async
        async with self._lock:
            if self.state == "CLOSED":
                return True, None

            if self.state == "OPEN":
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.recovery_timeout:
                    logger.info("🟡 Circuit breaker: Transitioning to HALF_OPEN (testing recovery)")
                    self.state = "HALF_OPEN"
                    self.success_count = 0
                    return True, None
                return False, f"Circuit breaker OPEN (retry in {self.recovery_timeout - elapsed:.0f}s)"
            return True, None

    async def snapshot(self) -> Dict[str, Any]:
        """Serializable copy of current state, for persisting across the
        ephemeral per-run process boundary (see restore())."""
        async with self._lock:
            return {
                "failures": self.failures,
                "last_failure_time": self.last_failure_time,
                "state": self.state,
                "success_count": self.success_count,
            }

    async def restore(self, data: Dict[str, Any]) -> None:
        """Hydrate from a snapshot() dict. Never raises -- malformed or
        missing fields just fall back to CLOSED/fresh, since a bad restore
        should degrade to 'circuit breaker starts closed' (today's existing
        behavior), not crash the run."""
        try:
            state = str(data.get("state", "CLOSED"))
            if state not in ("CLOSED", "OPEN", "HALF_OPEN"):
                state = "CLOSED"
            async with self._lock:
                self.failures = int(data.get("failures", 0))
                self.last_failure_time = float(data.get("last_failure_time", 0.0))
                self.state = state
                self.success_count = int(data.get("success_count", 0))
            logger.info(
                f"🔄 Circuit breaker state restored: {self.state} "
                f"(failures={self.failures})"
            )
        except (TypeError, ValueError) as e:
            logger.warning(f"Circuit breaker restore() got malformed data, staying CLOSED: {e}")

class DataFetcher:
    def __init__(self, api_base: str, *, session: Optional[aiohttp.ClientSession] = None, max_parallel: Optional[int] = None):
        self.api_base = api_base.rstrip("/")
        self._external_session = session
        max_parallel = max_parallel or cfg.MAX_PARALLEL_FETCH
        self.timeout = cfg.HTTP_TIMEOUT
        self.rate_limiter = RateLimitedFetcher(
            max_per_minute=cfg.RATE_LIMIT_PER_MINUTE,
            concurrency=max_parallel,
        )
        self.confirm_rate_limiter = RateLimitedFetcher(
            max_per_minute=cfg.CONFIRM_RATE_LIMIT_PER_MINUTE,
            concurrency=5,
        )
        self.circuit_breaker = APICircuitBreaker(
            failure_threshold=cfg.CB_FAILURE_THRESHOLD,
            recovery_timeout=cfg.CB_RECOVERY_TIMEOUT,
        )
        self.fetch_stats = {
            "products": {"success": 0, "failed": 0},
            "candles": {"success": 0, "failed": 0},
            "circuit_breaker_blocks": 0,
            "rate_limiter_waits": 0,
            "total_wait_time": 0.0,
            "oi_funding_blocks": 0,
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._external_session is not None:
            return self._external_session
        return await SessionManager.get_session()
  
    async def fetch_candles(self, symbol: str, resolution: str, limit: int, reference_time: int, expected_open_15: Optional[int] = None, for_confirmation: bool = False) -> Optional[Dict[str, Any]]:
        can_proceed, reason = await self.circuit_breaker.can_attempt()
        if not can_proceed:
            logger.warning(f"Circuit breaker blocked candles {symbol}: {reason}")
            self.fetch_stats["circuit_breaker_blocks"] += 1
            self.fetch_stats["candles"]["failed"] += 1
            return None

        minutes = int(resolution) if resolution != "D" else 1440
        interval_seconds = minutes * 60

        if minutes == 15 and expected_open_15 is not None:
            expected_open_ts = expected_open_15
        else:
            expected_open_ts = calculate_expected_candle_timestamp(reference_time, minutes)

        buffer_periods = Constants.CANDLE_FETCH_BUFFER_PERIODS
        to_time = reference_time + (interval_seconds * buffer_periods)
        from_time = expected_open_ts - (limit * interval_seconds)

        params = {
            "resolution": resolution,
            "symbol": symbol,
            "from": int(from_time),
            "to": int(to_time),
        }
        url = f"{self.api_base}/v2/chart/history"
        limiter = self.confirm_rate_limiter if for_confirmation else self.rate_limiter

        data = await limiter.call(
            async_fetch_json,
            url,
            params=params,
            retries=cfg.CANDLE_FETCH_RETRIES,
            backoff=cfg.CANDLE_FETCH_BACKOFF,
            timeout=self.timeout,
        )

        if data:
            result = data.get("result", {})
            if result and all(k in result for k in ("t", "o", "h", "l", "c", "v")):
                await self.circuit_breaker.record_success()
                self.fetch_stats["candles"]["success"] += 1

                num_candles = len(result.get("t", []))
                if num_candles > 0:
                    last_open = result["t"][-1]
                    diff = abs(expected_open_ts - last_open)

                    if diff > Constants.API_TIMESTAMP_TOLERANCE_SEC:
                        if last_open < expected_open_ts:
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(
                                    f"⚠️ API DELAY | {symbol} {resolution} | "
                                    f"Expected: {format_ist_time(expected_open_ts)} | "
                                    f"Got: {format_ist_time(last_open)} "
                                    f"(Diff: {diff}s > tolerance {Constants.API_TIMESTAMP_TOLERANCE_SEC}s)"
                                )
                        else:
                            logger.debug(f"API Ahead | {symbol} {resolution} | Diff: {diff}s")
                    else:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                f"✅ Scanned {symbol} {resolution} | "
                                f"Latest: {format_ist_time(last_open)} | Candles: {num_candles}"
                            )
                return data
            else:
                logger.warning(f"Candles response missing fields | Symbol: {symbol}")
                self.fetch_stats["candles"]["failed"] += 1
                await self.circuit_breaker.record_failure()
        else:
            self.fetch_stats["candles"]["failed"] += 1
            if data is False:
                logger.warning(f"Candles fetch got 4xx (not counted against circuit breaker) | Symbol: {symbol}")
            else:
                logger.warning(f"Candles fetch failed | Symbol: {symbol}")
                await self.circuit_breaker.record_failure()

        return None

    async def fetch_daily_cached(self, sdb: "RedisStateStore", symbol: str, limit: int,
                                   reference_time: int) -> Optional[Dict[str, Any]]:
        day_key = get_utc_date_key(reference_time)
        cache_key = f"daily_cache:{symbol}:{day_key}"

        logger.debug(f"📅 Daily cache MISS | {symbol} — fetching live")
        data = await self.fetch_candles(symbol, "D", limit, reference_time)

        if data and data.get("result") and not sdb.degraded:
            try:
                t = data["result"].get("t") or []
                last_ts = int(t[-1]) if t else 0
                if last_ts > 1_000_000_000_000:      # ms → s (same rule as parse_candles_to_numpy)
                    last_ts //= 1000
                yesterday_num = (reference_time // 86400) - 1
                if last_ts and (last_ts // 86400) >= yesterday_num:
                    utc_now = datetime.fromtimestamp(reference_time, tz=timezone.utc)
                    seconds_into_day = (utc_now - utc_now.replace(hour=0, minute=0, second=0, microsecond=0)).seconds
                    ttl = 86400 - seconds_into_day + 300  # +5min buffer past midnight
                    await sdb.set_metadata(cache_key, json_dumps(data), ttl=ttl)
                else:
                    logger_main.info(f"📅 Daily cache WRITE SKIPPED for {symbol}: yesterday's bar not in response yet")
            except Exception as e:
                logger.debug(f"Daily cache write failed for {symbol} (non-fatal): {e}")

        return data

    async def fetch_tickers_batch(self) -> Dict[str, Dict[str, Optional[float]]]:
        self.fetch_stats.setdefault("tickers", {"success": 0, "failed": 0})

        can_proceed, reason = await self.circuit_breaker.can_attempt()
        if not can_proceed:
            logger.warning(f"Circuit breaker blocked tickers fetch: {reason}")
            self.fetch_stats["tickers"]["failed"] += 1
            return {}

        url = f"{self.api_base}/v2/tickers"
        data = await self.rate_limiter.call(
            async_fetch_json,
            url,
            retries=cfg.CANDLE_FETCH_RETRIES,
            backoff=cfg.CANDLE_FETCH_BACKOFF,
            timeout=self.timeout,
        )

        out: Dict[str, Dict[str, Optional[float]]] = {}
        if not data:
            logger.warning("Tickers fetch failed -- OI/funding filter running fail-open this run")
            self.fetch_stats["tickers"]["failed"] += 1
            await self.circuit_breaker.record_failure()
            return out

        for row in (data.get("result") or []):
            if not isinstance(row, dict):
                continue
            symbol = row.get("symbol")
            if not symbol:
                continue

            oi_raw = next(
                (row[k] for k in ("open_interest", "oi", "open_interest_usd", "openInterest") if row.get(k) is not None),
                None,
            )
            oi_value_usd_raw = next(
                (row[k] for k in ("oi_value_usd", "oi_value") if row.get(k) is not None),
                None,
            )
            funding_raw = next(
                (row[k] for k in ("funding_rate", "fundingRate", "funding") if row.get(k) is not None),
                None,
            )
            price_raw = next(
                (row[k] for k in ("mark_price", "markPrice", "close") if row.get(k) is not None),
                None,
            )
            if oi_raw is None and funding_raw is None:
                continue
            try:
                out[symbol] = {
                    "oi": float(oi_raw) if oi_raw is not None else None,
                    "oi_value_usd": float(oi_value_usd_raw) if oi_value_usd_raw is not None else None,
                    "funding": float(funding_raw) if funding_raw is not None else None,
                    "price": float(price_raw) if price_raw is not None else None,
                }
            except (TypeError, ValueError):
                continue

        if out:
            await self.circuit_breaker.record_success()
            self.fetch_stats["tickers"]["success"] += 1
        else:
            logger.warning("Tickers response parsed but no OI/funding fields found -- check Delta API field names")
            self.fetch_stats["tickers"]["failed"] += 1

        return out

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "products": self.fetch_stats["products"].copy(),
            "candles": self.fetch_stats["candles"].copy(),
            "circuit_breaker_blocks": self.fetch_stats["circuit_breaker_blocks"],
            "oi_funding_blocks": self.fetch_stats["oi_funding_blocks"],
            "rate_limiter": self.rate_limiter.get_stats(),
        }     
        total_products = stats["products"]["success"] + stats["products"]["failed"]
        total_candles = stats["candles"]["success"] + stats["candles"]["failed"]
        
        if total_products > 0:
            stats["products"]["success_rate"] = round(
                stats["products"]["success"] / total_products * 100, 1
            )
        
        if total_candles > 0:
            stats["candles"]["success_rate"] = round(
                stats["candles"]["success"] / total_candles * 100, 1
            )        
        return stats

    async def fetch_all_candles_truly_parallel(self, pair_requests: List[Tuple[str, List[Tuple[str, int]]]], reference_time: int) -> Dict[str, Dict[str, Optional[Dict[str, Any]]]]:
        expected_open_15 = calculate_expected_candle_timestamp(reference_time, 15)
        all_tasks = []
        task_metadata = []
        for symbol, resolutions in pair_requests:
            for resolution, limit in resolutions:
                task = self.fetch_candles(
                    symbol, resolution, limit, reference_time, expected_open_15
                )
                all_tasks.append(task)
                task_metadata.append((symbol, resolution))
        results = await asyncio.wait_for(
            asyncio.gather(*all_tasks, return_exceptions=True),
            timeout=cfg.FETCH_PHASE_TIMEOUT_SEC
        )
        output = {}
        success_count = 0
        
        for (symbol, resolution), result in zip(task_metadata, results):
            if symbol not in output: 
                output[symbol] = {}
            if isinstance(result, Exception):
                output[symbol][resolution] = None
            else:
                output[symbol][resolution] = result
                if result: 
                    success_count += 1
        logger.info(f"📏 Parallel fetch complete | Success: {success_count}/{len(all_tasks)}")
        return output

def validate_indicator_values(indicators_dict: Dict[str, float], names: List[str]) -> Tuple[bool, str]:
    for name in names:
        val = indicators_dict.get(name)
        if val is None or np.isnan(val):
            return False, f"{name} is NaN"
    return True, "OK"

def validate_candle_for_alerts(data_15m: Dict[str, np.ndarray], candle_index: int, reference_time: int, pair_name: str, min_wick_ratio: float = 0.20) -> Tuple[bool, bool, Optional[Dict[str, Any]], Optional[str]]:
    try:
        o = float(data_15m["open"][candle_index])
        h = float(data_15m["high"][candle_index])
        l = float(data_15m["low"][candle_index])
        c = float(data_15m["close"][candle_index])
        ts = int(data_15m["timestamp"][candle_index])
        vol = float(data_15m["volume"][candle_index])
    except (IndexError, KeyError, ValueError, TypeError) as e:
        return False, False, None, f"Data access error: {e}"
    
    if any(np.isnan([o, h, l, c])) or any(np.isinf([o, h, l, c])):
        return False, False, None, f"Invalid OHLC: contains NaN or Inf"
    
    if any(x <= 0 for x in [o, h, l, c]):
        return False, False, None, f"Invalid OHLC: non-positive values"
    
    if not (l <= o <= h and l <= c <= h):
        return False, False, None, f"Invalid OHLC: relationships broken (O={o:.4f} H={h:.4f} L={l:.4f} C={c:.4f})"
    
    if vol <= 0:
        return False, False, None, "Zero volume candle — likely exchange placeholder or maintenance window"
    
    interval_seconds = 15 * 60
    candle_age = reference_time - ts
    
    candle_close_time = ts + interval_seconds
    time_since_candle_closed = reference_time - candle_close_time
     
    if not candle_is_stable(ts, reference_time, interval_minutes=15):
        return False, False, None, (
            f"Candle at {format_ist_time(ts)} not stable yet "
            f"(buffer {cfg.CANDLE_MIN_AGE_BUFFER}s, min age {Constants.MIN_CANDLE_AGE_FROM_OPEN}s)"
        )
   
    if candle_age > cfg.MAX_CANDLE_STALENESS_SEC:
        return False, False, None, (
            f"Candle age {candle_age}s from open is > {cfg.MAX_CANDLE_STALENESS_SEC}s. "
            f"This is a stale candle from a previous period! "
            f"(Opened: {format_ist_time(ts)}, Current: {format_ist_time(reference_time)})"
        )
    if cfg.DEBUG_MODE:
        logger.debug(
            f"[{pair_name}] Validating candle at index {candle_index}: "
            f"Open={format_ist_time(ts)}, Age={candle_age}s, "
            f"O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f}"
        )
    if candle_index > 0:
        prev_candle_ts = int(data_15m["timestamp"][candle_index - 1])
        expected_prev_ts = ts - interval_seconds
        if abs(prev_candle_ts - expected_prev_ts) > 60:
            return False, False, None, (
                f"Gap before signal candle: expected prev at {format_ist_time(expected_prev_ts)} "
                f"but found {format_ist_time(prev_candle_ts)} "
                f"(diff={abs(prev_candle_ts - expected_prev_ts)}s). Crossover data unreliable."
            )

    if candle_index + 1 < len(data_15m["timestamp"]):
        next_candle_ts = int(data_15m["timestamp"][candle_index + 1])
        expected_next_ts = ts + interval_seconds
        next_candle_is_still_forming = (next_candle_ts + interval_seconds) > reference_time

        if not next_candle_is_still_forming and abs(next_candle_ts - expected_next_ts) > 60:
            return False, False, None, ( 
                f"Gap detected: Expected next candle at {format_ist_time(expected_next_ts)} " 
                f"but found at {format_ist_time(next_candle_ts)} " f"(diff={abs(next_candle_ts - expected_next_ts)}s). Data may be incomplete." 
            ) 
    candle_range = h - l
    
    if candle_range < 1e-9:
        return False, False, None, f"Zero-range candle (H={h:.4f} L={l:.4f})"
    
    if c > o:
        is_green = True
        is_red = False
        candle_color = "GREEN"
        upper_wick = h - c
        lower_wick = o - l
        body = c - o
    elif c < o:
        is_green = False
        is_red = True
        candle_color = "RED"
        upper_wick = h - o
        lower_wick = c - l
        body = o - c
    else:
        is_green = False
        is_red = False
        candle_color = "DOJI"
        upper_wick = h - o
        lower_wick = c - l
        body = 0.0
    
    calculated_range = upper_wick + body + lower_wick

    if abs(calculated_range - candle_range) > 1e-6 * max(candle_range, 1.0):
        return False, False, None, (
            f"Candle structure error: wicks+body={calculated_range:.6f} "
            f"!= range={candle_range:.6f}"
        )
   
    body_ratio      = body / candle_range
    upper_wick_ratio = upper_wick / candle_range
    lower_wick_ratio = lower_wick / candle_range

    is_valid_for_buy  = (is_green and upper_wick_ratio < min_wick_ratio and body_ratio >= Constants.MIN_BODY_RATIO)
    is_valid_for_sell = (is_red   and lower_wick_ratio < min_wick_ratio and body_ratio >= Constants.MIN_BODY_RATIO)

    candle_info = {
        "timestamp": ts,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": vol,
        "range": candle_range,
        "color": candle_color,
        "is_green": is_green,
        "is_red": is_red,
        "body": body,
        "body_ratio": body_ratio,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
        "upper_wick_ratio": upper_wick_ratio,
        "lower_wick_ratio": lower_wick_ratio,
        "candle_age_seconds": candle_age,
        "time_since_closed": time_since_candle_closed,
        "is_valid_for_buy": is_valid_for_buy,
        "is_valid_for_sell": is_valid_for_sell,
    }
    if not is_valid_for_buy and not is_valid_for_sell:
        if is_green:
            reason = (
                f"GREEN candle rejected: upper wick {upper_wick_ratio*100:.1f}% "
                f"≥ {min_wick_ratio*100:.0f}% or body {body_ratio*100:.1f}% < {Constants.MIN_BODY_RATIO*100:.0f}%"
            )
        elif is_red:
            reason = (
                f"RED candle rejected: lower wick {lower_wick_ratio*100:.1f}% "
                f"≥ {min_wick_ratio*100:.0f}% or body {body_ratio*100:.1f}% < {Constants.MIN_BODY_RATIO*100:.0f}%"
            )
        else:
            reason = f"DOJI candle rejected: body {body_ratio*100:.1f}% < {Constants.MIN_BODY_RATIO*100:.0f}%"
        return False, False, candle_info, reason

    return is_valid_for_buy, is_valid_for_sell, candle_info, None

def calculate_expected_candle_timestamp(reference_time: int, interval_minutes: int) -> int: 
    interval_seconds = interval_minutes * 60
    current_interval_open = (reference_time // interval_seconds) * interval_seconds
    last_closed_candle_open = current_interval_open - interval_seconds
    return last_closed_candle_open

def _unpack_bar_core(open_arr, high_arr, low_arr, close_arr, idx: int):
    """Minimal 11-element tuple — no wick ratios (only needed for signal candle)."""
    o = open_arr[idx]; h = high_arr[idx]; l = low_arr[idx]; c = close_arr[idx]
    if np.isnan(o) or np.isnan(h) or np.isnan(l) or np.isnan(c):
        return None
    rng = h - l
    if rng <= 1e-12:
        return None
    body = abs(c - o)
    return (o, h, l, c, rng, body, body / rng, c > o, c < o, min(o, c), max(o, c))

def detect_reversal_candle_pattern(data_15m: "PriceData", i: int) -> Tuple[bool, bool, str]:
    if i < 1 or i >= len(data_15m.close):
        return False, False, ""

    oa, ha, la, ca = data_15m.open, data_15m.high, data_15m.low, data_15m.close

    # ── signal candle (i) ──
    _m0 = _unpack_bar_core(oa, ha, la, ca, i)
    if _m0 is None:
        return False, False, ""
    (o0, h0, l0, c0, rng0, body0, br0, g0, r0, bl0, bh0) = _m0
    uwr0 = (h0 - bh0) / rng0
    lwr0 = (bl0 - l0) / rng0

    # ── prior candle (i-1) ──
    _m1 = _unpack_bar_core(oa, ha, la, ca, i - 1)
    if _m1 is None:
        return False, False, ""
    (o1, h1, l1, c1, rng1, body1, br1, g1, r1, bl1, bh1) = _m1

    # ═══════════════════════════════════════════════════════════════════════
    # DATA INTEGRITY: Require exactly contiguous 15m candles (900s)
    # ═══════════════════════════════════════════════════════════════════════
    if data_15m.ts[i] - data_15m.ts[i - 1] != 900:
        return False, False, ""
    if i >= 2 and data_15m.ts[i - 1] - data_15m.ts[i - 2] != 900:
        return False, False, ""

    # ── optional 3rd candle (i-2) ──
    _m2 = _unpack_bar_core(oa, ha, la, ca, i - 2) if i >= 2 else None
    if _m2 is not None:
        (o2, h2, l2, c2, rng2, body2, br2, g2, r2, bl2, bh2) = _m2
        mid2 = (bl2 + bh2) * 0.5

    # ── prior leg directions ──
    lk = Constants.REVERSAL_PRIOR_LEG_LOOKBACK
    p1 = _prior_leg_direction(ca, ha, la, i - 1, lk)
    p3 = _prior_leg_direction(ca, ha, la, i - 3, lk) if i >= 3 else 0
    prior_down_1, prior_up_1 = (p1 == -1), (p1 == 1)
    prior_down_3, prior_up_3 = (p3 == -1), (p3 == 1)

    # ── 3-candle patterns ──
    if _m2 is not None:
        if (prior_down_3
                and r2 and br2 >= Constants.REVERSAL_STAR_BIG_BODY_MIN_RATIO
                and br1 <= Constants.REVERSAL_STAR_SMALL_BODY_MAX_RATIO
                and g0 and br0 >= Constants.REVERSAL_STAR_BIG_BODY_MIN_RATIO
                and c0 > mid2):
            return True, False, "Morning Star"

        if (prior_up_3
                and g2 and br2 >= Constants.REVERSAL_STAR_BIG_BODY_MIN_RATIO
                and br1 <= Constants.REVERSAL_STAR_SMALL_BODY_MAX_RATIO
                and r0 and br0 >= Constants.REVERSAL_STAR_BIG_BODY_MIN_RATIO
                and c0 < mid2):
            return False, True, "Evening Star"

        if (prior_down_3
                and g2 and g1 and g0
                and br2 >= Constants.REVERSAL_SOLDIERS_MIN_BODY_RATIO
                and br1 >= Constants.REVERSAL_SOLDIERS_MIN_BODY_RATIO
                and br0 >= Constants.REVERSAL_SOLDIERS_MIN_BODY_RATIO
                and o1 >= bl2 and o1 <= bh2 and c1 > c2
                and o0 >= bl1 and o0 <= bh1 and c0 > c1):
            return True, False, "Three White Soldiers"

        if (prior_up_3
                and r2 and r1 and r0
                and br2 >= Constants.REVERSAL_SOLDIERS_MIN_BODY_RATIO
                and br1 >= Constants.REVERSAL_SOLDIERS_MIN_BODY_RATIO
                and br0 >= Constants.REVERSAL_SOLDIERS_MIN_BODY_RATIO
                and o1 <= bh2 and o1 >= bl2 and c1 < c2
                and o0 <= bh1 and o0 >= bl1 and c0 < c1):
            return False, True, "Three Black Crows"

    if br1 >= Constants.REVERSAL_MIN_PRIOR_BODY_RATIO:
        # ── 2-candle patterns ──
        if prior_down_1 and g0 and r1 and o0 < bl1 and c0 > bh1:
            return True, False, "Bullish Engulfing"
        if prior_up_1 and r0 and g1 and o0 > bh1 and c0 < bl1:
            return False, True, "Bearish Engulfing"

        pen_buy = bl1 + Constants.REVERSAL_PIERCING_MIN_PENETRATION * body1
        pen_sell = bh1 - Constants.REVERSAL_PIERCING_MIN_PENETRATION * body1

        if prior_down_1 and r1 and g0 and o0 <= l1 and c0 > pen_buy and c0 < o1:
            return True, False, "Piercing Line"

        if prior_up_1 and g1 and r0 and o0 >= h1 and c0 < pen_sell and c0 > o1:
            return False, True, "Dark Cloud Cover"

        tol = rng0 * Constants.REVERSAL_TWEEZER_TOLERANCE_PCT
        if prior_down_1 and r1 and g0 and abs(l0 - l1) <= tol:
            return True, False, "Tweezer Bottom"
        if prior_up_1 and g1 and r0 and abs(h0 - h1) <= tol:
            return False, True, "Tweezer Top"

        if (prior_down_1 and r1 and g0
                and br0 <= Constants.REVERSAL_HARAMI_MAX_BODY_RATIO * br1 + 1e-9
                and bl0 >= bl1 and bh0 <= bh1):
            return True, False, "Bullish Harami"
        if (prior_up_1 and g1 and r0
                and br0 <= Constants.REVERSAL_HARAMI_MAX_BODY_RATIO * br1 + 1e-9
                and bl0 >= bl1 and bh0 <= bh1):
            return False, True, "Bearish Harami"

    # ── 1-candle patterns ──
    if br0 >= Constants.REVERSAL_MARUBOZU_BODY_RATIO:
        if prior_down_1 and g0:
            return True, False, "Bullish Marubozu"
        if prior_up_1 and r0:
            return False, True, "Bearish Marubozu"

    if br0 <= Constants.REVERSAL_PINBAR_BODY_MAX_RATIO:
        if (prior_down_1 and g0
                and lwr0 >= Constants.REVERSAL_PINBAR_WICK_RATIO
                and uwr0 < 0.15):
            return True, False, "Bullish Pinbar"
        if (prior_up_1 and r0
                and uwr0 >= Constants.REVERSAL_PINBAR_WICK_RATIO
                and lwr0 < 0.15):
            return False, True, "Bearish Pinbar"

    return False, False, ""

def parse_candles_to_numpy(result: Optional[Dict[str, Any]]) -> Optional[PriceData]:
    try:   
        if not result or not isinstance(result, dict):
            logger.warning("parse_candles_to_numpy: result is None or not dict")
            return None
    
        res = result.get("result", {}) or {}
        required_fields = ("t", "o", "h", "l", "c", "v")
    
        if not all(k in res for k in required_fields):
            missing = [k for k in required_fields if k not in res]
            logger.warning(
                f"parse_candles_to_numpy: Missing required fields: {missing} | "
                f"Available: {list(res.keys())}"
            )
            return None
    
        try:
            data = {
                "timestamp": np.asarray(res["t"], dtype=np.int64),
                "open":      np.asarray(res["o"], dtype=np.float64),
                "high":      np.asarray(res["h"], dtype=np.float64),
                "low":       np.asarray(res["l"], dtype=np.float64),
                "close":     np.asarray(res["c"], dtype=np.float64),
                "volume":    np.asarray(res["v"], dtype=np.float64),
            }
    
        except (ValueError, TypeError) as e:
            logger.error(f"parse_candles_to_numpy: Failed to convert data to arrays: {e}")
            return None
    
        n = len(data["timestamp"])
    
        if n == 0:
            logger.warning("parse_candles_to_numpy: empty candle array (n=0)")
            return None
    
        lengths = {k: len(data[k]) for k in ["open", "high", "low", "close", "volume"]}
        if len(set(lengths.values())) != 1:
            bad = {k: v for k, v in lengths.items() if v != n}
            logger.error(f"Length mismatch: {bad}")
            return None
    
        data["timestamp"] = np.where(data["timestamp"] > 1_000_000_000_000, data["timestamp"] // 1000, data["timestamp"])

        o, h, l, c = data["open"], data["high"], data["low"], data["close"]
    
        error_mask = (
            np.isnan(o) | np.isnan(h) | np.isnan(l) | np.isnan(c) |  # NaN check
            np.isinf(o) | np.isinf(h) | np.isinf(l) | np.isinf(c) |  # Inf check
            ~((l <= o) & (o <= h) & (l <= c) & (c <= h)) |            # Relationship check
             (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)                 # Non-positive check
        )
        error_count = np.sum(error_mask)
    
        if error_count > 0:
            error_indices = np.where(error_mask)[0]
            first_errors = error_indices[:min(5, len(error_indices))]
            logger.error(f"parse_candles_to_numpy: {error_count} invalid candle(s) detected")
            for idx in first_errors:
                logger.error(f"  Index {idx}: O={o[idx]:.2f} H={h[idx]:.2f} L={l[idx]:.2f} C={c[idx]:.2f}")
  
            if cfg.SANITIZE_BAD_CANDLES and error_count < n:
                keep_mask = ~error_mask
                logger.warning(
                    f"parse_candles_to_numpy: SANITIZE_BAD_CANDLES=True — dropping {error_count} "
                    f"bad candle(s), keeping {n - error_count}/{n}"
                )
                for k in data:
                    data[k] = data[k][keep_mask]
                o, h, l, c = data["open"], data["high"], data["low"], data["close"]
                n = len(data["timestamp"])
            else:
                logger.error("parse_candles_to_numpy: Rejecting data due to invalid candles")
                return None

        v = data["volume"]
        volume_error_mask = ~np.isfinite(v) | (v < 0)
        volume_error_count = np.sum(volume_error_mask)

        if volume_error_count > 0:
            logger.error(
                f"parse_candles_to_numpy: Found {volume_error_count} invalid volume value(s) out of {n} "
                f"({volume_error_count / n * 100:.1f}%)"
            )
            vol_error_indices = np.where(volume_error_mask)[0]
            for idx in vol_error_indices[:min(5, len(vol_error_indices))]:
                logger.error(f"  Index {idx}: Volume={v[idx]}")

            if cfg.SANITIZE_BAD_CANDLES and volume_error_count < n:
                vol_keep_mask = ~volume_error_mask
                logger.warning(
                    f"parse_candles_to_numpy: SANITIZE_BAD_CANDLES=True — dropping {volume_error_count} "
                    f"bad volume candle(s), keeping {n - volume_error_count}/{n}"
                )
                for k in data:
                    data[k] = data[k][vol_keep_mask]
                n = len(data["timestamp"])
                if n == 0:
                    logger.error("parse_candles_to_numpy: All candles dropped due to volume errors")
                    return None
                o, h, l, c = data["open"], data["high"], data["low"], data["close"]
                v = data["volume"]
            else:
                logger.error("parse_candles_to_numpy: Rejecting data due to invalid volume")
                return None

        hl_mid = (h + l) / 2.0
        candle_range = h - l
        close_deviation = np.abs(c - hl_mid) / (hl_mid + 1e-9)
        deviation_mask = close_deviation > Constants.HIGH_DEVIATION_THRESHOLD
        deviation_count = np.sum(deviation_mask)
 
        if deviation_count > 0:
            dev_indices = np.where(deviation_mask)[0].tolist()
            logger.warning(
                f"parse_candles_to_numpy: {deviation_count} candle(s) with "
                f"close/price deviation > {Constants.HIGH_DEVIATION_THRESHOLD} "
                f"| Indices: {dev_indices[:5]}"
            )
            if cfg.DEBUG_MODE and deviation_count <= 5:
                for idx in dev_indices:
                    dev_pct = close_deviation[idx] * 100
                    logger.debug(
                        f" Index {idx}: Deviation {dev_pct:.2f}% | "
                        f"Mid={hl_mid[idx]:.2f} Close={c[idx]:.2f}"
                    )
            if cfg.REJECT_HIGH_DEVIATION:
                logger.warning("Rejecting candle data due to high deviation (REJECT_HIGH_DEVIATION=True)")
                return None

        if n > 1:
            ts_diffs = np.diff(data["timestamp"])
            min_diff = np.min(ts_diffs)
            max_diff = np.max(ts_diffs)
        
            if min_diff <= 0:
                bad_idx = np.where(ts_diffs <= 0)[0]
                logger.warning(
                    f"parse_candles_to_numpy: {len(bad_idx)} non-monotonic/duplicate timestamp(s) found "
                    f"(indices {bad_idx[:5].tolist()}) | Min diff: {min_diff}s, Max diff: {max_diff}s | "
                    f"Continuing — get_last_closed_index_from_array will reject if this is near the target candle."
                )      
        
            if cfg.DEBUG_MODE:
                logger.debug(
                    f"parse_candles_to_numpy: Timestamp range | "
                    f"First: {format_ist_time(data['timestamp'][0])} | "
                    f"Last: {format_ist_time(data['timestamp'][-1])} | "
                    f"Count: {n} candles"
                )
    
        if cfg.DEBUG_MODE:
            logger.debug(
                f"parse_candles_to_numpy: SUCCESSFUL | "
                f"Candles: {n} | "
                f"Range: {format_ist_time(data['timestamp'][0])} to {format_ist_time(data['timestamp'][-1])}"
            )
    
        return PriceData.from_dict(data)
    except Exception as e:
        logger.error(
            f"parse_candles_to_numpy: Unexpected exception: {e}",
            exc_info=True
        )
        return None

def candle_is_stable(ts_open: int, reference_time: int, interval_minutes: int = 15) -> bool:
    """Check if a candle is fully closed and past the safety buffer."""
    interval_seconds = interval_minutes * 60
    time_since_closed = reference_time - (ts_open + interval_seconds)
    age_from_open = reference_time - ts_open
    return (
        time_since_closed >= cfg.CANDLE_MIN_AGE_BUFFER
        and age_from_open >= Constants.MIN_CANDLE_AGE_FROM_OPEN
    )

def get_last_closed_index_from_array(timestamps: np.ndarray, interval_minutes: int, 
                                     reference_time: int, 
                                     pair_name: Optional[str] = None) -> Optional[int]:
    if timestamps is None or timestamps.size < 1:
        return None
    reference_time = normalize_timestamp(reference_time)
    interval_seconds = interval_minutes * 60
    
    current_period_start = (reference_time // interval_seconds) * interval_seconds
    expected_ts_open_time = current_period_start - interval_seconds

    candle_close_time = expected_ts_open_time + interval_seconds
    time_since_candle_closed = reference_time - candle_close_time

    try:
        ts_normalized = normalize_timestamp_array(timestamps)
    except Exception as e:
        logger.error("[%s] Timestamp normalization failed: %s", pair_name or "?", e)
        return None

    if ts_normalized.size >= 2 and np.any(np.diff(ts_normalized) <= 0):
        target_area_mask = np.abs(ts_normalized - expected_ts_open_time) <= interval_seconds
        if np.any(np.diff(ts_normalized[target_area_mask]) <= 0):
            logger.warning("[%s] Timestamps corrupted near target; rejecting.", pair_name or "?")
            return None
        else:
            logger.info("[%s] Duplicates exist but not near target.", pair_name or "?")

    matches = np.flatnonzero(np.abs(ts_normalized - expected_ts_open_time) <= 1)
    if matches.size == 0:
        if logger.isEnabledFor(logging.DEBUG):
            last_ts = format_ist_time(ts_normalized[-1]) if ts_normalized.size else 'N/A'
            count = int(ts_normalized.size)
            last5_list = [format_ist_time(t) for t in ts_normalized[-5:]]
            last5_str = str(last5_list)

            logger.debug(
                "[%s] Target %dm open %s not found. last_ts=%s count=%s last5=%s",
                pair_name or "?", int(interval_minutes), format_ist_time(expected_ts_open_time),
                last_ts, count, last5_str
            )
        return None

    last_closed_idx = int(matches[-1])
    actual_candle_open = int(ts_normalized[last_closed_idx])

    if not candle_is_stable(actual_candle_open, reference_time, interval_minutes):
        logger.warning(
            "[%s] Candle %dm actual open %s not stable. Skipping.",
            pair_name or "?",
            int(interval_minutes),
            format_ist_time(actual_candle_open),
        )
        return None

    actual_close = actual_candle_open + interval_seconds
    if reference_time < actual_close:
        logger.error(
            "[%s] LOGIC ERROR: Candle not closed! Closes %s, ref %s",
            pair_name or "?",
            format_ist_time(actual_close),
            format_ist_time(reference_time)
        )
        return None

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[%s] Selected CLOSED %dm candle idx=%d %s-%s (closed %ds ago)",
            pair_name or "?", int(interval_minutes), last_closed_idx,
            format_ist_time(actual_candle_open), format_ist_time(actual_close),
            int(time_since_candle_closed)
        )
    return last_closed_idx

@dataclass(frozen=True)
class CandleSnapshot:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_green: bool
    is_red: bool
    is_valid_for_buy: bool
    is_valid_for_sell: bool
    reversal_pattern_name: str = ""
    reversal_bullish: bool = False
    reversal_bearish: bool = False


@dataclass(slots=True)
class PriceData:
    """Typed replacement for the loose {"timestamp": arr, "open": arr, ...} dict
    returned by parse_candles_to_numpy()."""
    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    @classmethod
    def from_dict(cls, d: Dict[str, np.ndarray]) -> "PriceData":
        return cls(
            ts=d["timestamp"], open=d["open"], high=d["high"],
            low=d["low"], close=d["close"], volume=d["volume"],
        )

    def as_dict(self) -> Dict[str, np.ndarray]:
        """Back-compat shim for call sites not yet migrated off dict-style access."""
        return {"timestamp": self.ts, "open": self.open, "high": self.high,
                "low": self.low, "close": self.close, "volume": self.volume}

    def __len__(self) -> int:
        return len(self.ts)

async def confirm_candle_unchanged(fetcher: DataFetcher, symbol: str, pair_name: str,
    ts_curr: int, cached: CandleSnapshot, reference_time: int, logger_pair: logging.Logger) -> Optional[bool]:
    """Returns True=unchanged, False=confirmed repaint/mismatch, None=inconclusive (fetch/network failure)."""
    try:
        raw = await fetcher.fetch_candles(symbol, "15", 5, reference_time, for_confirmation=True) 
        fresh = parse_candles_to_numpy(raw)
        if fresh is None:
            logger_pair.warning(f"[{pair_name}] Confirmation fetch failed — inconclusive, releasing dedup claim")
            return None
        matches = np.flatnonzero(np.abs(fresh.ts - ts_curr) <= 5)
        if matches.size == 0:
            logger_pair.warning(f"[{pair_name}] Confirmation candle {format_ist_time(ts_curr)} not found — inconclusive, releasing dedup claim")
            return None

        idx = int(matches[-1])
        fo = float(fresh.open[idx])
        fh = float(fresh.high[idx])
        fl = float(fresh.low[idx])
        fc = float(fresh.close[idx])
        fvol = float(fresh.volume[idx])

        # Volume check (matches validate_candle_for_alerts)
        if fvol <= 0:
            logger_pair.warning(f"[{pair_name}] Confirmation candle has zero volume — suppressing")
            return False

        # Color consistency check
        was_green = cached.is_green
        was_red = cached.is_red
        is_now_green = fc > fo
        is_now_red = fc < fo
        if (was_green and not is_now_green) or (was_red and not is_now_red):
            logger_pair.warning(
                f"[{pair_name}] Confirmation candle COLOR changed: "
                f"was {'green' if was_green else 'red'}, now "
                f"{'green' if is_now_green else 'red' if is_now_red else 'doji'}"
            )
            return False

        def _price_match(a: float, b: float) -> bool:
            abs_diff = abs(a - b)
            if abs_diff <= 1e-6:
                return True
            rel_diff = abs_diff / max(abs(a), abs(b), 1e-12)
            return rel_diff <= 1e-6

        if (not _price_match(fo, cached.open) or not _price_match(fh, cached.high) or
            not _price_match(fl, cached.low) or not _price_match(fc, cached.close)):

            logger_pair.warning(
                f"[{pair_name}] 🔁 Candle CHANGED since first fetch — repaint detected, suppressing alert | "
                f"First: O={cached.open:.4f} H={cached.high:.4f} L={cached.low:.4f} C={cached.close:.4f} | "
                f"Now:   O={fo:.4f} H={fh:.4f} L={fl:.4f} C={fc:.4f}"
            )
            return False

        return True
    except Exception as e:
        logger_pair.warning(f"[{pair_name}] Confirmation check errored: {e} — inconclusive, releasing dedup claim")
        return None

def independent_candle_reverify(data_15m: Dict[str, np.ndarray], candle_index: int, cached: CandleSnapshot, min_wick_ratio: float, pair_name: str, logger_pair: logging.Logger) -> bool:
    try:
        raw_o = float(data_15m["open"][candle_index])
        raw_h = float(data_15m["high"][candle_index])
        raw_l = float(data_15m["low"][candle_index])
        raw_c = float(data_15m["close"][candle_index])
        raw_ts = int(data_15m["timestamp"][candle_index])
        raw_vol = float(data_15m["volume"][candle_index])
    except (IndexError, KeyError, TypeError, ValueError) as e:
        logger_pair.error(
            f"[{pair_name}] Independent re-verify: cannot read raw OHLCV at index {candle_index}: {e} — suppressing alert"
        )
        return False

    if any(np.isnan([raw_o, raw_h, raw_l, raw_c])) or any(np.isinf([raw_o, raw_h, raw_l, raw_c])):
        logger_pair.error(
            f"[{pair_name}] Independent re-verify: raw OHLC contains NaN/Inf at index {candle_index} — suppressing alert"
        )
        return False

    if raw_ts != cached.timestamp:
        logger_pair.error(
            f"[{pair_name}] Independent re-verify TIMESTAMP MISMATCH: raw={raw_ts} cached={cached.timestamp} "
            f"— suppressing alert"
        )
        return False

    if raw_vol <= 0:
        logger_pair.error(
            f"[{pair_name}] Independent re-verify: zero/negative volume ({raw_vol}) at dispatch — suppressing alert"
        )
        return False

    def _close_enough(a: float, b: float) -> bool:
        abs_diff = abs(a - b)
        rel_tolerance = 1e-6 * max(abs(a), abs(b), 1.0)
        abs_floor = 1e-8  # noise floor for sub-cent priced coins, e.g. $0.00001
        return abs_diff <= max(rel_tolerance, abs_floor)

    mismatches = [
        tag for tag, a, b in (
            ("open", raw_o, cached.open), ("high", raw_h, cached.high),
            ("low", raw_l, cached.low), ("close", raw_c, cached.close),
        ) if not _close_enough(a, b)
    ]
    if mismatches:
        logger_pair.error(
            f"[{pair_name}] Independent re-verify OHLC MISMATCH on {mismatches} at index {candle_index} "
            f"| raw O={raw_o:.6f} H={raw_h:.6f} L={raw_l:.6f} C={raw_c:.6f} "
            f"| cached O={cached.open:.6f} H={cached.high:.6f} L={cached.low:.6f} C={cached.close:.6f} "
            f"— suppressing alert"
        )
        return False

    raw_range = raw_h - raw_l
    if raw_range < 1e-9:
        logger_pair.error(f"[{pair_name}] Independent re-verify: zero-range candle at dispatch — suppressing alert")
        return False

    raw_is_green = raw_c > raw_o
    raw_is_red = raw_c < raw_o

    if raw_is_green != cached.is_green or raw_is_red != cached.is_red:
        logger_pair.error(
            f"[{pair_name}] Independent re-verify COLOR MISMATCH: "
            f"raw(green={raw_is_green}, red={raw_is_red}) vs cached(green={cached.is_green}, red={cached.is_red}) "
            f"| O={raw_o:.4f} C={raw_c:.4f} — suppressing alert"
        )
        return False

    hi_body = max(raw_o, raw_c)
    lo_body = min(raw_o, raw_c)
    raw_upper_wick = raw_h - hi_body
    raw_lower_wick = lo_body - raw_l
    raw_body = hi_body - lo_body

    raw_body_ratio = raw_body / raw_range
    raw_upper_ratio = raw_upper_wick / raw_range
    raw_lower_ratio = raw_lower_wick / raw_range

    raw_valid_buy = raw_is_green and raw_upper_ratio < min_wick_ratio and raw_body_ratio >= Constants.MIN_BODY_RATIO
    raw_valid_sell = raw_is_red and raw_lower_ratio < min_wick_ratio and raw_body_ratio >= Constants.MIN_BODY_RATIO

    if raw_valid_buy != cached.is_valid_for_buy or raw_valid_sell != cached.is_valid_for_sell:
        logger_pair.error(
            f"[{pair_name}] Independent re-verify VALIDITY MISMATCH: "
            f"raw(buy={raw_valid_buy}, sell={raw_valid_sell}) vs cached(buy={cached.is_valid_for_buy}, sell={cached.is_valid_for_sell}) "
            f"| upper_ratio={raw_upper_ratio:.4f} lower_ratio={raw_lower_ratio:.4f} body_ratio={raw_body_ratio:.4f} "
            f"— suppressing alert"
        )
        return False

    if cached.reversal_pattern_name:
        fresh_price_data = PriceData.from_dict(data_15m)
        fresh_bullish, fresh_bearish, fresh_pattern_name = detect_reversal_candle_pattern(
            fresh_price_data, candle_index
        )
        if (fresh_bullish, fresh_bearish, fresh_pattern_name) != (
            cached.reversal_bullish, cached.reversal_bearish, cached.reversal_pattern_name
        ):
            logger_pair.error(
                f"[{pair_name}] Independent re-verify PATTERN MISMATCH: "
                f"cached='{cached.reversal_pattern_name}' (bull={cached.reversal_bullish}, bear={cached.reversal_bearish}) "
                f"vs fresh='{fresh_pattern_name}' (bull={fresh_bullish}, bear={fresh_bearish}) "
                f"— suppressing alert"
            )
            return False

    return True

def cross_check_15m_against_5m(data_5m: PriceData, ts_curr: int, cached: CandleSnapshot,
                               pair_name: str, logger_pair) -> Optional[bool]:
    mask = (data_5m.ts >= ts_curr) & (data_5m.ts < ts_curr + 900)
    idx = np.flatnonzero(mask)
    if len(idx) < 3:
        return None
    agg_o = float(data_5m.open[idx[0]])
    agg_h = float(np.max(data_5m.high[idx]))
    agg_l = float(np.min(data_5m.low[idx]))
    agg_c = float(data_5m.close[idx[-1]])
    def _eq(a, b): return abs(a - b) <= max(1e-6 * max(abs(a), abs(b), 1.0), 1e-8)
    ok = _eq(agg_o, cached.open) and _eq(agg_h, cached.high) \
         and _eq(agg_l, cached.low) and _eq(agg_c, cached.close)
    if not ok:
        logger_pair.warning(
            f"[{pair_name}] 15m/5m ENDPOINT MISMATCH (data-integrity, not repaint) | "
            f"15m: O={cached.open:.4f} H={cached.high:.4f} L={cached.low:.4f} C={cached.close:.4f} | "
            f"5m-agg: O={agg_o:.4f} H={agg_h:.4f} L={agg_l:.4f} C={agg_c:.4f} — suppressing"
        )
    return ok




