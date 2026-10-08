from __future__ import annotations
import time
import asyncio
import logging
import uuid
from typing import Dict, Any, Optional, Tuple, List, ClassVar, Callable, TYPE_CHECKING, Set, Sequence, Awaitable, Union, cast
import hashlib
import numpy as np

import redis.asyncio as redis  # type: ignore[import-untyped]
from redis.exceptions import ConnectionError as RedisConnectionError, RedisError  # type: ignore[import-untyped]

from plan_replay import outcome_path_fields, outcome_path_stream_fields
from bot_config import cfg, logger, json_dumps, json_loads, JSONDecodeError, CONFIG_OVERRIDE_ALLOWED_FIELDS, CONFIG_OVERRIDE_METADATA_KEY, BRAIN_DISABLED_KEYS_METADATA_KEY, PAIR_THRESHOLDS_METADATA_KEY, _get_session_from_ts
from fetcher import compute_backoff

StreamField = Union[bytes, memoryview, str, int, float]
BRAIN_KEY_HISTORY_METADATA_KEY = "brain_alert_key_history"

# Standing decisions (applied overrides, auto-disabled alert keys, learned
# per-pair floors, ...). With the default 7-day metadata TTL these silently
# lapse -- e.g. an auto-disabled alert re-enables itself after a week of
# quiet. Keep this set in sync with redis_audit.DURABLE_METADATA (a unit test
# enforces it).
DURABLE_METADATA_KEYS = frozenset({
    "config_override",
    "brain_disabled_alert_keys",
    "brain_alert_key_history",
    "pair_confluence_thresholds",
    "dynamic_weights",
    "brain_apply_snapshots",
})
DURABLE_METADATA_TTL_SEC = 90 * 86400

if TYPE_CHECKING:
    from fetcher import PriceData

def _rc(client: "Optional[redis.Redis]") -> "redis.Redis":
    """Narrow an Optional Redis client for mypy at call sites that are
    already guarded by an `if not self._redis: return ...` / `if not
    sdb._redis: return ...` check one scope removed (inside a lambda
    passed to `_safe_redis_op`, a nested function, or right after a
    helper call whose truthy return implies the client is connected).
    No behavior change — the assert only documents an invariant these
    call sites already guarantee at runtime; it should never fire.
    """
    assert client is not None
    return client

def _is_quota_exceeded_error(exc: Optional[BaseException]) -> bool:
    """True for a fatal, retry-proof store error: Upstash's old monthly-
    request-quota error ('ERR max requests limit exceeded'), kept for
    provider portability, plus Aiven Valkey's out-of-memory error ('OOM
    command not allowed when used memory > maxmemory'), which fires once
    the plan's memory limit is hit. Both mean the store won't accept
    writes again until something external changes (Upstash's monthly
    reset, or freeing/raising memory on Aiven) — a connection retry
    cannot fix either, so callers use this to fail fast instead of
    burning more commands on retries."""
    if exc is None:
        return False
    msg = str(exc).lower()
    return "max requests limit exceeded" in msg or "oom command not allowed" in msg

async def _execute_pipeline(pipe: Any) -> Any:
    """Await a redis-py pipeline's execute() in a form mypy accepts.

    redis-py types Pipeline.execute() as `Awaitable[Dict] | Dict` because the
    method is shared with the sync Pipeline class. Passing that union directly
    to asyncio.wait_for fails overload resolution regardless of an inline
    cast. Awaiting it inside a plain coroutine narrows it to a single
    Coroutine[Any, Any, Any], which is unambiguously awaitable.
    """
    result = cast("Awaitable[Any]", pipe.execute())
    return await result

async def _blanket_reset_pair(sdb: RedisStateStore, pair_name: str, logger_pair: logging.Logger) -> int:
    from alert_registry import ALERT_KEYS
    all_keys = list(ALERT_KEYS.values())
    previous_states = await sdb.batch_get_all_alert_states(pair_name, all_keys)
    resets = [
        (f"{pair_name}:{rk}", "INACTIVE", None)
        for rk in all_keys
        if previous_states.get(rk, False)
    ]
    if resets:
        await sdb.atomic_batch_update(resets)
        logger_pair.debug(
            f"[{pair_name}] Blanket reset: {len(resets)} active state(s) cleared"
        )
    return len(resets)

async def _redis_key_inventory(
    sdb: "RedisStateStore", max_keys: int = 20000,
) -> Dict[str, Dict[str, int]]:
    """Read-only Redis audit: key count and no-TTL key count per prefix
    (the text before the first ':'). Bounded by max_keys so it cannot
    run away on a large keyspace."""
    client = sdb._redis
    if client is None:
        return {}
    inventory: Dict[str, Dict[str, int]] = {}
    batch: List[str] = []

    async def _flush() -> None:
        if not batch:
            return
        pipe = client.pipeline()
        for key in batch:
            pipe.ttl(key)
        ttls = await pipe.execute()
        for key, ttl in zip(batch, ttls):
            entry = inventory.setdefault(str(key).split(":", 1)[0], {"keys": 0, "no_ttl": 0})
            entry["keys"] += 1
            if ttl == -1:
                entry["no_ttl"] += 1
        batch.clear()
    scanned = 0
    async for key in client.scan_iter(match="*", count=500):
        batch.append(key)
        scanned += 1
        if len(batch) >= 500:
            await _flush()
        if scanned >= max_keys:
            break
    await _flush()
    return inventory

async def _clear_all_redis_states(
    sdb: RedisStateStore,
    pairs: List[str],
    logger: logging.Logger,
    *,
    clear_active_states: bool = True,
    clear_dedups: bool = True,
    clear_pending_outcomes: bool = True,
    clear_shadow_pending: bool = True,
    clear_alert_stats: bool = False,
    clear_shadow_stats: bool = False,
    clear_outcome_streams: bool = False,
    clear_daily_cache: bool = False,
) -> Tuple[int, int, int, int, int, int, int, int, int]:
    if sdb.degraded or not sdb._redis:
        logger.warning("Redis degraded — skipping mass state purge")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0

    deleted_states = 0
    deleted_dedups = 0
    deleted_pending = 0
    deleted_shadow_pending = 0
    deleted_alert_stats = 0
    deleted_shadow_stats = 0
    deleted_shadow_hiconf = 0
    deleted_streams = 0
    deleted_daily_cache = 0

    async def _scan_keys_with_timeout(match: str, count: int = 100, timeout: float = 10.0) -> List[str]:
        """Safely consume an async scan_iter with a timeout to prevent runaway loops."""
        async def _consume():
            return [k async for k in sdb._redis.scan_iter(match=match, count=count)]
        
        try:
            return await asyncio.wait_for(_consume(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"⏱️ Redis scan for '{match}' timed out after {timeout}s. Aborting scan for this prefix to protect run deadline.")
            return []  # Fail-safe: return empty so we don't block the bot

    async def _batch_unlink(keys: List[str], batch_size: int = 100) -> int:
        """Unlink keys in batches to avoid blocking Redis with massive argument lists."""
        if not keys:
            return 0
        total_deleted = 0
        for i in range(0, len(keys), batch_size):
            batch = keys[i:i + batch_size]
            try:
                # unlink() frees memory in a background thread on the Redis server
                total_deleted += await _rc(sdb._redis).unlink(*batch)
            except Exception as e:
                logger.error(f"Batch unlink failed for {len(batch)} keys: {e}")
        return total_deleted

    try:
        if clear_active_states:
            state_keys = await _scan_keys_with_timeout(f"{RedisKeyPrefix.PAIR_STATE}*", count=100)
            deleted_states = await _batch_unlink(state_keys)

        if clear_dedups:
            dedup_keys = await _scan_keys_with_timeout(f"{RedisKeyPrefix.RECENT_ALERT}*", count=500)
            deleted_dedups = await _batch_unlink(dedup_keys)

        if clear_pending_outcomes:
            pending_keys = await _scan_keys_with_timeout(f"{RedisKeyPrefix.OUTCOME_PENDING}*", count=100)
            deleted_pending = await _batch_unlink(pending_keys)

        if clear_shadow_pending:
            shadow_pending_keys = await _scan_keys_with_timeout(f"{RedisKeyPrefix.SHADOW_PENDING}*", count=100)
            deleted_shadow_pending = await _batch_unlink(shadow_pending_keys)

        if clear_alert_stats:
            alert_stats_keys = await _scan_keys_with_timeout(f"{RedisKeyPrefix.ALERT_STATS}*", count=100)
            deleted_alert_stats = await _batch_unlink(alert_stats_keys)

        if clear_shadow_stats:
            shadow_stats_keys = await _scan_keys_with_timeout(f"{RedisKeyPrefix.SHADOW_STATS}*", count=100)
            deleted_shadow_stats = await _batch_unlink(shadow_stats_keys)
            hiconf_keys = await _scan_keys_with_timeout(f"{RedisKeyPrefix.SHADOW_HICONF_STATS}*", count=100)
            deleted_shadow_hiconf = await _batch_unlink(hiconf_keys)

        if clear_outcome_streams:
            exact_keys = [
                RedisKeyPrefix.OUTCOME_LOG_STREAM,
                RedisKeyPrefix.SHADOW_LOG_STREAM,
                RedisKeyPrefix.BRAIN_RUN_COUNTER,
            ]
            brain_report_keys = await _scan_keys_with_timeout("brain_report:*", count=50)
            all_stream_keys = exact_keys + brain_report_keys
            deleted_streams = await _batch_unlink(all_stream_keys)

        if clear_daily_cache:
            daily_cache_keys = await _scan_keys_with_timeout(
                f"{RedisKeyPrefix.METADATA}daily_cache:*", count=200
            )
            deleted_daily_cache = await _batch_unlink(daily_cache_keys)

        logger.info(
            f"🧹 MASS RESET complete | "
            f"States: {deleted_states} | Dedups: {deleted_dedups} | "
            f"Pending: {deleted_pending} | ShadowPending: {deleted_shadow_pending} | "
            f"AlertStats: {deleted_alert_stats} | ShadowStats: {deleted_shadow_stats} | "
            f"ShadowHiConf: {deleted_shadow_hiconf} | Streams: {deleted_streams} | "
            f"DailyCache: {deleted_daily_cache}"
        )
        return (deleted_states, deleted_dedups, deleted_pending, deleted_shadow_pending,
                deleted_alert_stats, deleted_shadow_stats, deleted_shadow_hiconf, deleted_streams,
                deleted_daily_cache)

    except Exception as e:
        logger.error(f"Mass reset failed: {e}")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0

def build_products_map_from_cfg() -> Dict[str, dict]:
    products_map: Dict[str, dict] = {}
    for pair in cfg.PAIRS:
        products_map[pair] = {
            "id": pair,                 
            "symbol": pair,
            "contract_type": "perpetual_futures"
        }
    logger.info(
        f"📦 Product map built from cfg: {len(products_map)}/{len(cfg.PAIRS)} matched | "
        f"Coverage: {(len(products_map)/len(cfg.PAIRS))*100:.0f}%"
    )
    return products_map

def _compute_win_weight(rr_achieved: float, is_win: bool) -> float:
    """Continuous bonus weighting for a resolved outcome.

    Loss -> 0.0.
    Win at or below R:R target -> 1.0.
    Win between target and bonus R:R -> linearly interpolated from 1.0
        up to cfg.OUTCOME_BONUS_WEIGHT (replaces the old flat step where
        a 2.99R trade got nothing and 3.0R jumped to full bonus).
    Win at or above bonus R:R -> capped at cfg.OUTCOME_BONUS_WEIGHT.
    """
    if not is_win:
        return 0.0
    target = cfg.OUTCOME_RR_TARGET
    bonus = cfg.OUTCOME_BONUS_RR
    cap = cfg.OUTCOME_BONUS_WEIGHT
    if rr_achieved <= target:
        return 1.0
    if rr_achieved >= bonus or bonus <= target:
        return cap
    frac = (rr_achieved - target) / (bonus - target)
    return 1.0 + (cap - 1.0) * frac

class RedisKeyPrefix:
    """Centralized Redis key prefixes"""
    PAIR_STATE = "pair_state:"
    METADATA = "metadata:"
    ALERT = "alert:"
    RECENT_ALERT = "recent_alert:"
    LOCK = "lock:"
    OUTCOME_PENDING = "outcome_pending:"
    ALERT_STATS = "alert_stats:"
    OUTCOME_LOG_STREAM = "outcome_log_stream"
    SHADOW_PENDING = "shadow_pending:"
    SHADOW_STATS = "shadow_stats:"
    SHADOW_LOG_STREAM = "shadow_log_stream"
    SHADOW_HICONF_STATS = "shadow_hiconf:"
    BRAIN_RUN_COUNTER = "brain_run_counter"
    CUSUM_STATE = "brain_cusum:"
    CUSUM_WATERMARK = "brain_cusum_watermark:"
    THRESHOLD_HISTORY = "brain_threshold_history:"
    VOTE_COUNT_HISTORY = "brain_vote_counts:"
    LAST_PROCESSED_CANDLE = "last_processed_candle:"  # NEW
    TRADE_COOLDOWN = "trade_cooldown:"
    TELEGRAM_DLQ = "telegram_dlq:"

class RedisStateStore:
    POOL_MAX_AGE_SECONDS = 3600
    SCRIPT_RELOAD_LOCK_TIMEOUT = 2.0

    _global_pools: ClassVar[Dict[str, Optional[redis.Redis]]] = {}
    _pool_healthy: ClassVar[Dict[str, bool]] = {}
    _pool_created_at: ClassVar[Dict[str, float]] = {}
    _pool_reuse_count: ClassVar[Dict[str, int]] = {}
    _pool_lock: ClassVar[Optional[asyncio.Lock]] = None
    _script_reload_lock: ClassVar[Optional[asyncio.Lock]] = None

    @classmethod
    def _get_pool_lock(cls) -> asyncio.Lock:
        if cls._pool_lock is None:
            cls._pool_lock = asyncio.Lock()
        return cls._pool_lock

    @classmethod
    def _get_script_reload_lock(cls) -> asyncio.Lock:
        if cls._script_reload_lock is None:
            cls._script_reload_lock = asyncio.Lock()
        return cls._script_reload_lock

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._redis: Optional[redis.Redis] = None

        self.state_prefix = RedisKeyPrefix.PAIR_STATE
        self.meta_prefix = RedisKeyPrefix.METADATA
        self.alert_prefix = RedisKeyPrefix.ALERT

        self.expiry_seconds = max(cfg.STATE_EXPIRY_DAYS * 86400 if cfg.STATE_EXPIRY_DAYS > 0 else 0, 7 * 86400)
        self.alert_expiry_seconds = cfg.STATE_EXPIRY_DAYS * 86400
        self.metadata_expiry_seconds = 7 * 86400
        self._pending_outcome_keys_by_pair: Optional[Dict[str, List[str]]] = None
        self._shadow_pending_outcome_keys_by_pair: Optional[Dict[str, List[str]]] = None
        self._dlq_empty_at_start: bool = False
        self._dlq_pushed_this_run: bool = False
        self.trade_close_events: List[Dict[str, Any]] = []
        self._close_event_ids: Set[Tuple[str, str, str, int]] = set()
        self._run_resolved_total: int = 0
        self._run_archived_total: int = 0
        self.degraded = False
        self.degraded_alerted = False
        self._connection_attempts = 0
        self._quota_exhausted: bool = False
        self._last_connect_error: Optional[Exception] = None
        self._last_recovery_attempt_ts: float = 0.0
        self._recovery_lock = asyncio.Lock()
        self.recovery_attempts: int = 0
        self.recovery_successes: int = 0

        if cfg.DEBUG_MODE and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"RedisStateStore initialized | "
                f"State TTL: {cfg.STATE_EXPIRY_DAYS}d | "
                f"Alert TTL: {cfg.STATE_EXPIRY_DAYS}d | "
                f"Metadata TTL: 7d"
            )

    async def _record_redis_failure(self, operation: str, exc: Exception) -> None:
        logger.error(f"Redis operation '{operation}' failed: {exc}")
        if _is_quota_exceeded_error(exc):
            self._quota_exhausted = True
        if self.degraded:
            return
        self.degraded = True
        if self._quota_exhausted:
            # A reconnect can't fix a quota/OOM error — it'll just fail the
            # same way and cost another command. Go straight to degraded mode.
            logger.critical(
                f"Redis store unavailable — quota exhausted or memory limit "
                f"hit (failure in '{operation}') — staying degraded for "
                "remainder of run, skipping reconnect attempt"
            )
            if self._redis:
                try:
                    await self._redis.aclose()
                except Exception:
                    pass
                self._redis = None
            return
        logger.warning(f"Redis marked degraded after failure in '{operation}' — attempting one reconnect")
        try:
            reconnected = await self._attempt_connect(timeout=5.0)
            if reconnected:
                logger.info(f"Redis reconnected after failure in '{operation}'")
                self.degraded = False
            else:
                logger.critical(f"Redis reconnect failed after '{operation}' — staying degraded for remainder of run")
        except Exception as reconnect_exc:
            logger.critical(f"Redis reconnect attempt itself failed: {reconnect_exc} — staying degraded")

    async def maybe_recover_from_degraded(self, cooldown_sec: Optional[float] = None) -> bool:
        """Mid-run health probe. _record_redis_failure only reconnects once,
        at the moment of the failure — if that single attempt doesn't land,
        the store stays degraded (dedup, state persistence, dynamic
        weights, etc. all soft-fail or fail-open/closed) for the rest of an
        8-minute run even if Redis recovers seconds later. Call this
        periodically (e.g. once per pair) from the run loop; it no-ops
        instantly unless currently degraded, and retries at most once every
        `cooldown_sec` even under concurrent callers. Returns True if the
        store is healthy (already, or as of this call).
        """
        if not self.degraded:
            return True
        if self._quota_exhausted:
            return False  # reconnecting can't fix a quota/OOM condition
        if cooldown_sec is None:
            cooldown_sec = float(cfg.REDIS_RECOVERY_COOLDOWN_SEC)
        async with self._recovery_lock:
            if not self.degraded:
                return True
            now = time.time()
            if now - self._last_recovery_attempt_ts < cooldown_sec:
                return False
            self._last_recovery_attempt_ts = now
            self.recovery_attempts += 1
            try:
                reconnected = await self._attempt_connect(timeout=3.0)
            except Exception as exc:
                logger.debug(f"Mid-run Redis recovery probe failed: {exc}")
                return False
            if reconnected:
                self.recovery_successes += 1
                logger.info("♻️ Redis recovered mid-run — degraded mode cleared")
            return reconnected

    async def _attempt_connect(self, timeout: float = 5.0) -> bool:
        try:
            self._redis = redis.from_url(
                self.redis_url,
                socket_connect_timeout=timeout,
                socket_timeout=timeout,
                retry_on_timeout=True,
                max_connections=32,
                decode_responses=True,
            )

            ok = await self._ping_with_retry(timeout)
            if not ok:
                raise RedisConnectionError("ping failed after retries")

            logger.info("Redis connected")
            self.degraded = False
            self.degraded_alerted = False
            self._connection_attempts = 0

            async with RedisStateStore._get_pool_lock():
                existing_pool = RedisStateStore._global_pools.get(self.redis_url)
                pool_is_healthy = False
                if existing_pool:
                    try:
                        pool_is_healthy = await asyncio.wait_for(existing_pool.ping(), timeout=1.0)
                    except Exception:
                        pool_is_healthy = False

                if existing_pool and pool_is_healthy:
                    if self._redis is not existing_pool:
                        await _rc(self._redis).aclose()
                    self._redis = existing_pool
                    logger.debug("Using pool created by another coroutine")
                else:
                    if existing_pool and existing_pool is not self._redis:
                        try:
                            await existing_pool.aclose()
                        except Exception:
                            pass
                    RedisStateStore._global_pools[self.redis_url] = self._redis
                    RedisStateStore._pool_healthy[self.redis_url] = True
                    RedisStateStore._pool_created_at[self.redis_url] = time.time()
                    RedisStateStore._pool_reuse_count[self.redis_url] = 0
                    if cfg.DEBUG_MODE:
                        logger.debug("Redis connection saved to per-URL pool")

                return True
        except Exception as exc:
            logger.error(f"Redis connection attempt failed: {exc}")
            self._last_connect_error = exc
            if _is_quota_exceeded_error(exc):
                self._quota_exhausted = True
            if self._redis:
                try:
                    await self._redis.aclose()
                except Exception:
                    pass
                self._redis = None
            return False

    async def connect(self, timeout: float = 5.0) -> None:
        pool_reused = False

        async with RedisStateStore._get_pool_lock():
            pool = RedisStateStore._global_pools.get(self.redis_url)
            healthy = RedisStateStore._pool_healthy.get(self.redis_url, False)

            if pool and healthy:
                pool_age = time.time() - RedisStateStore._pool_created_at.get(self.redis_url, 0.0)
                if pool_age > self.POOL_MAX_AGE_SECONDS:
                    logger.info(f"Redis pool aged {pool_age:.0f}s, refreshing")
                    RedisStateStore._pool_healthy[self.redis_url] = False
                    try:
                        await pool.aclose()
                    except Exception:
                        pass
                    RedisStateStore._global_pools[self.redis_url] = None
                else:
                    try:
                        self._redis = pool
                        ok = await self._ping_with_retry(timeout)
                        if ok:
                            RedisStateStore._pool_reuse_count[self.redis_url] = \
                                RedisStateStore._pool_reuse_count.get(self.redis_url, 0) + 1
                            self.degraded = False
                            pool_reused = True
                            return
                    except Exception as e:
                        if cfg.DEBUG_MODE:
                            logger.debug(f"Pool health check failed: {e}, creating new pool")
                        RedisStateStore._pool_healthy[self.redis_url] = False
                        pool_reused = False

        if pool_reused:
            return

        for attempt in range(1, cfg.REDIS_CONNECTION_RETRIES + 1):
            if await self._attempt_connect(timeout):
                max_conn = getattr(_rc(self._redis).connection_pool, "max_connections", "?")
                logger.info(f"✅ Redis connected ({max_conn} max)")
                self.degraded = False
                self.degraded_alerted = False
                return

            if self._quota_exhausted:
                logger.critical(
                    "❌ Redis store unavailable (quota exhausted or memory "
                    f"limit hit) — skipping remaining {cfg.REDIS_CONNECTION_RETRIES - attempt} "
                    f"connection retr{'y' if cfg.REDIS_CONNECTION_RETRIES - attempt == 1 else 'ies'} "
                    "(retrying can't help until the underlying limit changes)"
                )
                break

            if attempt < cfg.REDIS_CONNECTION_RETRIES:
                delay = compute_backoff(cfg.REDIS_RETRY_DELAY, attempt)
                logger.warning(f"Retrying Redis connection in {delay:.1f}s...")
                await asyncio.sleep(delay)

        logger.critical("❌ Redis connection failed after all retries")
        self.degraded = True
        if self._redis:
            try:
                await self._redis.aclose()
            except Exception:
                pass
        self._redis = None

        logger.warning("""
    🚨 REDIS DEGRADED MODE ACTIVE:
    - Alert deduplication:  UNAVAILABLE (claims fail closed)
    - State persistence:    DISABLED (alerts reset each run)
    - Trading alerts:       BLOCKED (no dedup claim possible, nothing is sent)
    """)

        if cfg.FAIL_ON_REDIS_DOWN:
            raise RedisConnectionError("Redis unavailable after all retries – FAIL_ON_REDIS_DOWN=true")
      
    async def close(self) -> None:
        self._redis = None

    @classmethod
    async def shutdown_global_pool(cls, redis_url: Optional[str] = None) -> None:
        async with cls._get_pool_lock():
            urls = [redis_url] if redis_url else list(cls._global_pools.keys())
            for url in urls:
                pool = cls._global_pools.get(url)
                if pool:
                    try:
                        pool_age = time.time() - cls._pool_created_at.get(url, 0.0)
                        reuse_count = cls._pool_reuse_count.get(url, 0)
                        logger.debug(f"Shutting down Redis pool | url={url} | Age: {pool_age:.1f}s | Reuses: {reuse_count}")

                        await pool.aclose()
                        await asyncio.sleep(0.25)

                    except Exception as e:
                        logger.error(f"Error shutting down Redis pool {url}: {e}")

                cls._global_pools.pop(url, None)
                cls._pool_healthy.pop(url, None)
                cls._pool_created_at.pop(url, None)
                cls._pool_reuse_count.pop(url, None)
            
    async def _ping_with_retry(self, timeout: float) -> bool:
        result = await self._safe_redis_op(lambda: _rc(self._redis).ping(), timeout, "ping")

        return bool(result)

    async def _safe_redis_op(self, fn: Callable[[], Any], timeout: float, op_name: str, parser: Optional[Callable[[Any], Any]] = None):
        if not self._redis:
            return None
        try:
            coro = fn()
            result = await asyncio.wait_for(coro, timeout=timeout)
            return parser(result) if parser else result
        except (asyncio.TimeoutError, RedisConnectionError, RedisError) as e:
            if _is_quota_exceeded_error(e):
                # Fatal, retry-proof condition (Aiven OOM / provider quota) —
                # route through the central handler so degraded mode and
                # _quota_exhausted engage even when this surfaces through a
                # plain get/set rather than one of the batch operations.
                await self._record_redis_failure(op_name, e)
            else:
                logger.error(f"Redis {op_name} failed: {e}")
            return None
        except Exception as e:
            if _is_quota_exceeded_error(e):
                await self._record_redis_failure(op_name, e)
            else:
                logger.error(f"Failed to {op_name}: {e}")
            return None

    async def get_valkey_usage_snapshot(self) -> Dict[str, Optional[float]]:
        """One-call snapshot of Valkey resource usage: total commands
        processed (server-wide, cumulative — diff two snapshots for a
        per-run delta) and current memory usage vs. the plan's maxmemory.
        Uses a single bare INFO call (no section arg, which returns every
        section) so one snapshot costs exactly one command, not two."""
        snapshot: Dict[str, Optional[float]] = {
            "total_commands_processed": None,
            "used_memory_bytes": None,
            "maxmemory_bytes": None,
            "memory_pct": None,
        }
        if self.degraded or not self._redis:
            return snapshot
        info = await self._safe_redis_op(
            lambda: _rc(self._redis).info(),
            2.0,
            "info",
        )
        if not isinstance(info, dict):
            return snapshot
        total_commands = info.get("total_commands_processed")
        if total_commands is not None:
            try:
                snapshot["total_commands_processed"] = float(total_commands)
            except (TypeError, ValueError):
                pass
        used = info.get("used_memory")
        maxmem = info.get("maxmemory")
        try:
            if used is not None:
                snapshot["used_memory_bytes"] = float(used)
        except (TypeError, ValueError):
            pass
        try:
            if maxmem is not None:
                snapshot["maxmemory_bytes"] = float(maxmem)
        except (TypeError, ValueError):
            pass
        if snapshot["used_memory_bytes"] is not None and snapshot["maxmemory_bytes"]:
            snapshot["memory_pct"] = round(
                (snapshot["used_memory_bytes"] / snapshot["maxmemory_bytes"]) * 100, 1
            )
        return snapshot

    async def get(self, key: str, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
        return await self._safe_redis_op(
            lambda: _rc(self._redis).get(f"{self.state_prefix}{key}"),
            timeout,
            f"get {key}",
            parser=lambda r: json_loads(r) if r else None,
        )

    async def set(self, key: str, state: Optional[Any], ts: Optional[int] = None, timeout: float = 2.0) -> None:
        ts = int(ts or time.time())
        redis_key = f"{self.state_prefix}{key}"
        data = json_dumps({"state": state, "ts": ts})
        await self._safe_redis_op(
            lambda: _rc(self._redis).set(
                redis_key,
                data,
                ex=self.expiry_seconds if self.expiry_seconds > 0 else None,
            ),
            timeout,
            f"set {key}",
        )

    async def get_metadata(self, key: str, timeout: float = 2.0) -> Optional[str]:
        return await self._safe_redis_op(
            lambda: _rc(self._redis).get(f"{self.meta_prefix}{key}"),
            timeout,
            f"get_metadata {key}",
            parser=lambda r: r if r else None,
        )

    async def get_durable_metadata(self, key: str, timeout: float = 2.0) -> Optional[str]:
        """get_metadata for standing decisions: GETEX also pushes the 90-day TTL
        forward on every read, so an override the bot keeps using never lapses."""
        return await self._safe_redis_op(
            lambda: _rc(self._redis).getex(
                f"{self.meta_prefix}{key}", ex=DURABLE_METADATA_TTL_SEC
            ),
            timeout,
            f"getex_metadata {key}",
            parser=lambda r: r if r else None,
        )

    async def set_metadata(self, key: str, value: str, timeout: float = 2.0,
                         ttl: Optional[int] = None) -> None:
        if ttl is None:
            ttl = (
                DURABLE_METADATA_TTL_SEC
                if key in DURABLE_METADATA_KEYS
                else self.metadata_expiry_seconds
            )
        await self._safe_redis_op(
            lambda: _rc(self._redis).set(
                f"{self.meta_prefix}{key}",
                value,
                ex=ttl
            ),
            timeout,
            f"set_metadata {key}",
        )

    async def _read_raw_config_override(self) -> Dict[str, Any]:
        """Parses the config_override metadata blob without applying it.
        Shared by load_config_override (startup-apply) and get_config_override
        (read-only inspection, e.g. brain.py checking a path's current state
        before deciding whether to auto-disable/auto-reinstate it)."""
        if self.degraded or not self._redis:
            return {}
        raw = await self.get_durable_metadata(CONFIG_OVERRIDE_METADATA_KEY)
        if not raw:
            return {}
        try:
            override = json_loads(raw)
        except (JSONDecodeError, TypeError, ValueError) as e:
            logger.warning(f"Ignoring malformed config_override in Redis: {e}")
            return {}
        if not isinstance(override, dict):
            logger.warning("Ignoring config_override in Redis: not a JSON object")
            return {}
        return override

    async def load_config_override(self) -> List[str]:
        override = await self._read_raw_config_override()
        applied = []
        for field, new_value in override.items():
            if field not in CONFIG_OVERRIDE_ALLOWED_FIELDS:
                logger.warning(f"Ignoring config_override field '{field}' — not in the allowed safelist")
                continue
            if not hasattr(cfg, field):
                continue
            old_value = getattr(cfg, field)
            try:
                coerced = type(old_value)(new_value)
                # enforces Field(ge/le) and model validators; a ValidationError (a ValueError) leaves cfg unchanged
                type(cfg).__pydantic_validator__.validate_assignment(cfg, field, coerced)
                applied.append(f"{field}: {old_value} -> {coerced}")
            except (TypeError, ValueError) as e:
                logger.warning(f"Ignoring config_override field '{field}' — could not coerce {new_value!r}: {e}")
        return applied

    async def get_config_override(self) -> Dict[str, Any]:
        """Read-only: current override dict, safelist-filtered, for inspection
        without mutating cfg. Used by brain.py to check whether a path is
        already disabled before deciding to (re)write it."""
        override = await self._read_raw_config_override()
        return {k: v for k, v in override.items() if k in CONFIG_OVERRIDE_ALLOWED_FIELDS}

    async def write_config_override(self, field: str, value: Any) -> bool:
        """Merge one field into the live config_override blob (read-modify-write).
        Returns False (and writes nothing) if field isn't on the safelist —
        callers should not assume success without checking the return value."""
        if field not in CONFIG_OVERRIDE_ALLOWED_FIELDS:
            logger.warning(f"Refusing to write config_override field '{field}' — not in the allowed safelist")
            return False
        override = await self._read_raw_config_override()
        override[field] = value
        try:
            await self.set_metadata(CONFIG_OVERRIDE_METADATA_KEY, json_dumps(override))
        except Exception as e:
            logger.warning(f"Failed to write config_override field '{field}': {e}")
            return False
        return True

    async def remove_config_override_field(self, field: str) -> bool:
        """Delete one field from the live config_override blob (read-modify-write).
        Used by the Brain auto-rollback to restore a field that had NO override
        before a plan wrote one. Returns True if the field is absent afterwards."""
        override = await self._read_raw_config_override()
        if field not in override:
            return True
        override.pop(field, None)
        try:
            await self.set_metadata(CONFIG_OVERRIDE_METADATA_KEY, json_dumps(override))
        except Exception as e:
            logger.warning(f"Failed to remove config_override field '{field}': {e}")
            return False
        return True

    async def get_disabled_alert_keys(self) -> Set[str]:
        raw = await self.get_durable_metadata(BRAIN_DISABLED_KEYS_METADATA_KEY)
        if not raw:
            return set()
        try:
            keys = json_loads(raw)
        except (JSONDecodeError, TypeError, ValueError) as e:
            logger.warning(f"Ignoring malformed {BRAIN_DISABLED_KEYS_METADATA_KEY} in Redis: {e}")
            return set()
        return set(keys) if isinstance(keys, list) else set()

    async def get_alert_key_history(self) -> Dict[str, Dict[str, float]]:
        """{'disabled_at': {alert_key: ts}, 'reenabled_at': {alert_key: ts}}.
        Missing or malformed data returns empty maps."""
        empty: Dict[str, Dict[str, float]] = {"disabled_at": {}, "reenabled_at": {}}
        raw = await self.get_metadata(BRAIN_KEY_HISTORY_METADATA_KEY)
        if not raw:
            return empty
        try:
            data = json_loads(raw)
        except (JSONDecodeError, TypeError, ValueError) as e:
            logger.warning(f"Ignoring malformed {BRAIN_KEY_HISTORY_METADATA_KEY} in Redis: {e}")
            return empty
        if not isinstance(data, dict):
            return empty
        out: Dict[str, Dict[str, float]] = {}
        for bucket in ("disabled_at", "reenabled_at"):
            src = data.get(bucket)
            out[bucket] = (
                {k: float(v) for k, v in src.items() if isinstance(v, (int, float))}
                if isinstance(src, dict) else {}
            )
        return out

    async def set_alert_key_disabled(self, alert_key: str, disabled: bool) -> bool:
        current = await self.get_disabled_alert_keys()
        was_disabled = alert_key in current
        if disabled:
            current.add(alert_key)
        else:
            current.discard(alert_key)
        try:
            await self.set_metadata(BRAIN_DISABLED_KEYS_METADATA_KEY, json_dumps(sorted(current)))
        except Exception as e:
            logger.warning(f"Failed to update disabled-key set for '{alert_key}': {e}")
            return False
        # Best-effort: record the transition time. A failure here must not
        # undo or fail the disable/enable itself.
        try:
            if disabled != was_disabled:
                hist = await self.get_alert_key_history()
                now_ts = time.time()
                if disabled:
                    hist["disabled_at"][alert_key] = now_ts
                    hist["reenabled_at"].pop(alert_key, None)
                else:
                    hist["reenabled_at"][alert_key] = now_ts
                    hist["disabled_at"].pop(alert_key, None)
                await self.set_metadata(BRAIN_KEY_HISTORY_METADATA_KEY, json_dumps(hist))
        except Exception as e:
            logger.warning(
                f"Could not record disable/enable time for '{alert_key}': {e} — "
                "the re-enable probation window (BRAIN_REENABLE_PROBATION_DAYS) "
                "will not apply correctly for this key until this succeeds"
            )
        return True

    async def get_pair_thresholds(self) -> Dict[str, float]:
        """All pair -> confluence-abs-score-floor overrides currently stored,
        as learned/written by the brain. Missing or malformed data returns {}."""

        raw = await self.get_durable_metadata(PAIR_THRESHOLDS_METADATA_KEY)
        if not raw:
            return {}
        try:
            data = json_loads(raw)
        except (JSONDecodeError, TypeError, ValueError) as e:
            logger.warning(f"Ignoring malformed {PAIR_THRESHOLDS_METADATA_KEY} in Redis: {e}")
            return {}
        if not isinstance(data, dict):
            logger.warning(f"Ignoring {PAIR_THRESHOLDS_METADATA_KEY} in Redis: not a JSON object")
            return {}
        return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}

    async def get_pair_threshold(self, pair: str) -> Optional[float]:
        """Single pair's stored abs-score floor, or None if not set — caller
        should fall back to cfg.CONFLUENCE_MIN_ABS_SCORE in that case."""
        thresholds = await self.get_pair_thresholds()
        return thresholds.get(pair)

    async def set_pair_threshold(self, pair: str, value: float) -> bool:
        current = await self.get_pair_thresholds()
        current[pair] = value
        try:
            await self.set_metadata(PAIR_THRESHOLDS_METADATA_KEY, json_dumps(current))
        except Exception as e:
            logger.warning(f"Failed to update pair threshold for '{pair}': {e}")
            return False
        return True

    async def get_dynamic_weights(self) -> Optional[Dict[str, float]]:
        """Load the Brain's optimized CONFLUENCE_WEIGHTS from Redis.
        Returns None if not stored yet (caller should fall back to static config)."""
        if self.degraded or not self._redis:
            return None
        raw = await self.get_metadata("dynamic_weights")
        if not raw:
            return None
        try:
            data = json_loads(raw)
            if not isinstance(data, dict):
                logger.warning("Ignoring malformed dynamic_weights in Redis: not a dict")
                return None
            # Validate: only float values, non-negative
            return {k: float(v) for k, v in data.items() if isinstance(v, (int, float)) and float(v) >= 0}
        except (JSONDecodeError, TypeError, ValueError) as e:
            logger.warning(f"Ignoring malformed dynamic_weights in Redis: {e}")
            return None

    # Only these callers may change live weights when ENFORCE_SINGLE_WEIGHT_PATH is on.
    _WEIGHT_WRITE_SOURCES: ClassVar[frozenset] = frozenset({"promotion", "rollback"})

    async def set_dynamic_weights(
        self, weights: Dict[str, float], ttl: int = 30 * 86400, *, source: str = "direct",
    ) -> bool:
        """Persist the Brain's optimized CONFLUENCE_WEIGHTS to Redis.
        TTL default 30 days — refresh on each Brain report.

        With cfg.ENFORCE_SINGLE_WEIGHT_PATH (default True) only source="promotion"
        (champion/challenger) or source="rollback" (restore) may write; any other
        caller is refused so there is exactly one controlled weight-change path."""
        if getattr(cfg, "ENFORCE_SINGLE_WEIGHT_PATH", True) and source not in self._WEIGHT_WRITE_SOURCES:
            logger.warning(
                f"Refused direct dynamic_weights write (source={source!r}): live weights "
                "change only via challenger promotion or rollback"
            )
            return False
        if self.degraded or not self._redis:
            return False
        try:
            await self.set_metadata("dynamic_weights", json_dumps(weights), ttl=ttl)
            return True
        except Exception as e:
            logger.warning(f"Failed to persist dynamic_weights: {e}")
            return False





    async def clear_dynamic_weights(self) -> bool:
        """Remove stored dynamic weights (revert to static CONFLUENCE_WEIGHTS)."""
        if self.degraded or not self._redis:
            return False
        try:
            await self._redis.delete(f"{self.meta_prefix}dynamic_weights")
            return True
        except Exception as e:
            logger.warning(f"Failed to clear dynamic_weights: {e}")
            return False

    async def set_challenger_weights(
        self, weights: Dict[str, float], meta: Optional[Dict[str, Any]] = None,
        ttl: int = 30 * 86400,
    ) -> bool:
        """Store challenger weights + metadata. Never used as live weights
        unless explicitly promoted."""
        if self.degraded or not self._redis:
            return False
        try:
            payload = {"weights": weights, "meta": meta or {}, "stored_at": int(time.time())}
            await self.set_metadata("challenger_weights", json_dumps(payload), ttl=ttl)
            return True
        except Exception as e:
            logger.warning(f"Failed to persist challenger_weights: {e}")
            return False

    async def get_challenger_weights(self) -> Optional[Dict[str, Any]]:
        if self.degraded or not self._redis:
            return None
        try:
            raw = await self.get_metadata("challenger_weights")
            if not raw:
                return None
            data = json_loads(raw)
            return data if isinstance(data, dict) else None
        except Exception as e:
            logger.warning(f"Failed to load challenger_weights: {e}")
            return None

    async def clear_challenger_weights(self) -> bool:
        if self.degraded or not self._redis:
            return False
        try:
            await self._redis.delete(f"{self.meta_prefix}challenger_weights")
            return True
        except Exception as e:
            logger.warning(f"Failed to clear challenger_weights: {e}")
            return False

    async def promote_challenger_to_champion(self) -> bool:
        """Copy challenger → dynamic_weights (live), then clear challenger.
        Call only after OOS gates pass and operator/plan approves."""
        blob = await self.get_challenger_weights()
        if not blob or not blob.get("weights"):
            return False
        ok = await self.set_dynamic_weights(blob["weights"], source="promotion")
        if ok:
            await self.clear_challenger_weights()
        return ok

    async def batch_get_metadata(self, keys: List[str], timeout: float = 5.0) -> Dict[str, Optional[str]]:
        """Fetch many metadata keys in ONE Redis round-trip (pipeline)."""
        if not self._redis or self.degraded or not keys:
            return {k: None for k in keys}
        try:
            async with self._redis.pipeline() as pipe:
                for k in keys:
                    pipe.get(f"{self.meta_prefix}{k}")         
                values = await asyncio.wait_for(_execute_pipeline(pipe), timeout=timeout)
            return {k: v for k, v in zip(keys, values)}
        except Exception as e:
            logger.error(f"batch_get_metadata failed for {len(keys)} keys: {e}")
            return {k: None for k in keys}

    async def batch_set_metadata(self, items: Dict[str, str], timeout: float = 5.0,
                                   ttl: Optional[int] = None) -> bool:
        """Write many metadata keys in ONE Redis round-trip (pipeline)."""
        if not self._redis or self.degraded or not items:
            return True
        try:
            async with self._redis.pipeline() as pipe:
                for k, v in items.items():
                    pipe.set(f"{self.meta_prefix}{k}", v,
                              ex=ttl if ttl is not None else self.metadata_expiry_seconds)
                await asyncio.wait_for(_execute_pipeline(pipe), timeout=timeout)
            return True
        except Exception as e:
            logger.error(f"batch_set_metadata failed: {e}")
            return False

    async def get_last_processed_candle_ts(self, pair_name: str) -> Optional[int]:
        """Get the last candle timestamp that was fully processed for this pair.
        Returns None if no candle has been processed yet, or if the key expired."""
        if self.degraded or not self._redis:
            return None
        key = f"{RedisKeyPrefix.LAST_PROCESSED_CANDLE}{pair_name}"
        raw = await self._safe_redis_op(
            lambda: _rc(self._redis).get(key),
            2.0,
            f"last_processed_candle_get:{pair_name}",
        )
        if raw is None:
            return None
        try:
            return int(raw)
        except (ValueError, TypeError):
            return None

    async def get_last_processed_candle_ts_bulk(
        self, pair_names: Sequence[str]
    ) -> Dict[str, Optional[int]]:
        """Load the last-processed candle timestamp for all requested pairs
        with one MGET, instead of one GET per pair."""
        if self.degraded or not self._redis or not pair_names:
            return {pair: None for pair in pair_names}
        pairs = list(pair_names)
        keys = [f"{RedisKeyPrefix.LAST_PROCESSED_CANDLE}{pair}" for pair in pairs]
        raw_values = await self._safe_redis_op(
            lambda: _rc(self._redis).mget(keys),
            3.0,
            "last_processed_candle_mget",
        )
        if raw_values is None:
            return {pair: None for pair in pairs}
        result: Dict[str, Optional[int]] = {}
        for pair, raw in zip(pairs, raw_values):
            if raw is None:
                result[pair] = None
                continue
            try:
                result[pair] = int(raw)
            except (ValueError, TypeError):
                result[pair] = None
        return result

    async def set_last_processed_candle_ts(self, pair_name: str, ts: int) -> bool:
        """Mark a candle timestamp as processed for this pair."""
        if self.degraded or not self._redis:
            return False
        key = f"{RedisKeyPrefix.LAST_PROCESSED_CANDLE}{pair_name}"
        result = await self._safe_redis_op(
            lambda: _rc(self._redis).set(key, str(ts), ex=self.expiry_seconds),
            2.0,
            f"last_processed_candle_set:{pair_name}",
        )
        return bool(result)

    LAST_SUCCESSFUL_CANDLES_KEY = "last_successful_candles"

    async def merge_last_successful_candles(self, updates: Dict[str, int]) -> Dict[str, int]:
        """Merge this run's per-pair 'last successfully evaluated candle' into
        the persisted map (1 GET + 1 SET per run, never goes backwards) and
        return the merged map. Falls back to `updates` if Redis is unavailable."""
        merged: Dict[str, int] = {}
        if self.degraded or not self._redis:
            return dict(updates)
        raw = await self.get_metadata(self.LAST_SUCCESSFUL_CANDLES_KEY)
        if raw:
            try:
                data = json_loads(raw)
                if isinstance(data, dict):
                    merged = {str(k): int(v) for k, v in data.items()}
            except (JSONDecodeError, TypeError, ValueError):
                merged = {}
        changed = False
        for pair, ts in updates.items():
            if int(ts) > merged.get(pair, 0):
                merged[pair] = int(ts)
                changed = True
        if changed:
            await self.set_metadata(
                self.LAST_SUCCESSFUL_CANDLES_KEY, json_dumps(merged), ttl=30 * 86400,
            )
        return merged

    async def get_adaptive_dedup_windows(self) -> Dict[str, int]:
        """alert_key -> window seconds from the Brain's inter-arrival analysis.
        Empty unless ENABLE_ADAPTIVE_DEDUP_WINDOWS. Read once per run (cached on
        this per-run instance) and re-clamped to the configured hard bounds."""
        if not getattr(cfg, "ENABLE_ADAPTIVE_DEDUP_WINDOWS", False):
            return {}
        cached = getattr(self, "_adaptive_dedup_cache", None)
        if cached is not None:
            return cached
        out: Dict[str, int] = {}
        if not self.degraded and self._redis:
            raw = await self.get_metadata("adaptive_dedup_windows")
            if raw:
                try:
                    lo, hi = int(cfg.ADAPTIVE_DEDUP_MIN_SEC), int(cfg.ADAPTIVE_DEDUP_MAX_SEC)
                    for ak, v in json_loads(raw).items():
                        out[str(ak)] = int(min(max(int(v["window_sec"]), lo), hi))
                except (JSONDecodeError, TypeError, ValueError, KeyError, AttributeError):
                    out = {}
        self._adaptive_dedup_cache = out
        return out

    async def claim_recent_alert(self, pair: str, alert_key: str, ts: int,
                                 window_sec: Optional[int] = None) -> Optional[bool]:
        """Try to claim the dedup window for pair:alert_key.

        True  = claim taken (safe to send)
        False = key already exists (genuine duplicate)
        None  = Redis degraded or errored; outcome unknown (caller fails closed)
        """
        if self.degraded:
            logger.error(
                f"claim_recent_alert: Redis degraded — failing closed for {pair}:{alert_key} "
                f"(no dedup claim possible, alert blocked)"
            )
            return None
        if not self._redis:
            logger.critical(
                f"claim_recent_alert: degraded=False but _redis is None (state desync) — "
                f"failing closed for {pair}:{alert_key}, this alert will be blocked"
            )
            return None
        recent_key = f"{RedisKeyPrefix.RECENT_ALERT}{pair}:{alert_key}"
        effective_window = window_sec if window_sec is not None else cfg.ALERT_DEDUP_WINDOW_SEC
        for attempt in (1, 2):
            try:
                result = await asyncio.wait_for(
                    self._redis.set(recent_key, str(ts), nx=True, ex=effective_window),
                    timeout=3.0
                )
                should_send = bool(result)
                if cfg.DEBUG_MODE and not should_send:
                    logger.debug(f"Dedup: Skipping duplicate {pair}:{alert_key}")
                return should_send
            except Exception as e:
                logger.error(
                    f"Dedup claim attempt {attempt}/2 FAILED for {pair}:{alert_key}: {e}"
                )
        return None

    async def check_recent_alert(self, pair: str, alert_key: str, ts: int, window_sec: Optional[int] = None) -> bool:
        """Boolean wrapper kept for existing callers: True only when the claim was taken."""
        return (await self.claim_recent_alert(pair, alert_key, ts, window_sec)) is True

    async def batch_check_recent_alerts(self, pair: str, alert_keys: List[str], ts: int,
                                          window_sec: Optional[int] = None,
                                          windows: Optional[Dict[str, int]] = None) -> Dict[str, bool]:
        """Claim dedup windows for several alert keys on one pair in ONE Redis
        round-trip (pipeline). Each SET NX EX is still independently atomic —
        this only batches the network round-trip, not the semantics."""
        if not alert_keys:
            return {}
        if self.degraded:
            logger.error(
                f"batch_check_recent_alerts: Redis degraded — failing closed for {pair} "
                f"(no dedup claim possible, {len(alert_keys)} alert(s) blocked)"
            )
            return {k: False for k in alert_keys}
        if not self._redis:
            logger.critical(
                f"batch_check_recent_alerts: degraded=False but _redis is None (state desync) — "
                f"failing closed for {pair}, these alerts will be blocked"
            )
            return {k: False for k in alert_keys}
        effective_window = window_sec if window_sec is not None else cfg.ALERT_DEDUP_WINDOW_SEC
        try:
            async with self._redis.pipeline() as pipe:
                for alert_key in alert_keys:
                    recent_key = f"{RedisKeyPrefix.RECENT_ALERT}{pair}:{alert_key}"
                    pipe.set(
                        recent_key, str(ts), nx=True,
                        ex=int((windows or {}).get(alert_key, effective_window)),
                    )
                results = await asyncio.wait_for(_execute_pipeline(pipe), timeout=3.0)
            return {k: bool(r) for k, r in zip(alert_keys, results)}
        except Exception as e:
            logger.error(f"batch_check_recent_alerts FAILED for {pair} ({alert_keys}): {e}")
            return {k: False for k in alert_keys}  # fail-closed, same policy as check_recent_alert

    async def release_recent_alert(self, pair: str, alert_key: str) -> None:
        """Undo a dedup claim if the message didn't actually get delivered."""
        if self.degraded:
            return
        recent_key = f"{RedisKeyPrefix.RECENT_ALERT}{pair}:{alert_key}"
        try:
            await asyncio.wait_for(_rc(self._redis).delete(recent_key), timeout=1.0)
        except Exception as e:
            
            logger.warning(f"Failed to release dedup claim for {pair}:{alert_key}: {e}")

    # ── Telegram dead-letter queue ───────────────────────────────────────
    # One Redis key per parked alert: telegram_dlq:{pair}:{candle_ts}:{digest}.
    # The digest makes a re-park of the identical message idempotent.

    async def dlq_push(
        self, pair: str, message: str, ts: int, *,
        dedup_keys: Optional[List[str]] = None, source: str = "",
        state_changes: Optional[List[Any]] = None,
        outcomes: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Park a failed-to-send alert. Returns True if it is (now) stored."""
        if self.degraded or not self._redis:
            return False
        self._dlq_pushed_this_run = True
        digest = hashlib.sha1(message.encode("utf-8")).hexdigest()[:10]
        key = f"{RedisKeyPrefix.TELEGRAM_DLQ}{pair}:{int(ts)}:{digest}"
        entry = {
            "pair": pair, "ts": int(ts), "message": message,
            "dedup_keys": list(dedup_keys or []), "source": source,
            # Side effects owed once the message is finally delivered: ACTIVE
            # alert state and the pending-outcome rows (see replay_telegram_dlq).
            "state_changes": [list(c) for c in (state_changes or [])],
            "outcomes": list(outcomes or []),
            "attempts": 0, "queued_at": int(time.time()),
        }
        ttl = int(cfg.TELEGRAM_DLQ_MAX_AGE_SEC) * 2
        result = await self._safe_redis_op(
            lambda: _rc(self._redis).set(key, json_dumps(entry), nx=True, ex=ttl),
            2.0, f"dlq_push:{pair}",
        )
        if result:
            return True
        # NX failed: either the identical entry is already parked (fine) or Redis failed.
        exists = await self._safe_redis_op(
            lambda: _rc(self._redis).exists(key), 2.0, f"dlq_exists:{pair}",
        )
        return bool(exists)

    async def dlq_list(self, limit: int = 10) -> List[Tuple[str, Dict[str, Any]]]:
        """Oldest-first parked alerts as (redis_key, entry)."""
        if self.degraded or not self._redis:
            return []
        pattern = f"{RedisKeyPrefix.TELEGRAM_DLQ}*"

        async def _scan() -> List[str]:
            return [k async for k in _rc(self._redis).scan_iter(match=pattern, count=2000)]

        keys = await self._safe_redis_op(_scan, 3.0, "dlq_scan")
        if not keys:
            self._dlq_empty_at_start = True
            return []
        raw_values = await self._safe_redis_op(
            lambda: _rc(self._redis).mget(keys), 3.0, "dlq_mget",
        )
        out: List[Tuple[str, Dict[str, Any]]] = []
        for k, raw in zip(keys, raw_values or []):
            if not raw:
                continue
            try:
                entry = json_loads(raw)
            except (JSONDecodeError, TypeError, ValueError):
                continue
            if isinstance(entry, dict) and entry.get("message"):
                out.append((k, entry))
        out.sort(key=lambda kv: int(kv[1].get("ts") or 0))
        return out[: max(1, int(limit))]

    async def dlq_count(self) -> int:
        """Number of parked alerts (0 if Redis is unavailable)."""
        if self.degraded or not self._redis:
            return 0
        if getattr(self, "_dlq_empty_at_start", False) and not getattr(self, "_dlq_pushed_this_run", False):
            return 0  # list was empty at run start and nothing was parked since: skip the SCAN
        pattern = f"{RedisKeyPrefix.TELEGRAM_DLQ}*"

        async def _scan() -> List[str]:
            return [k async for k in _rc(self._redis).scan_iter(match=pattern, count=2000)]

        keys = await self._safe_redis_op(_scan, 3.0, "dlq_count")
        return len(keys or [])

    async def dlq_set_attempts(self, key: str, entry: Dict[str, Any], attempts: int) -> bool:
        """Persist an incremented attempt counter (keeps the remaining TTL)."""
        if self.degraded or not self._redis:
            return False
        entry = dict(entry, attempts=int(attempts))
        result = await self._safe_redis_op(
            lambda: _rc(self._redis).set(key, json_dumps(entry), xx=True, keepttl=True),
            2.0, "dlq_set_attempts",
        )
        return bool(result)

    async def dlq_delete(self, key: str) -> bool:
        if self.degraded or not self._redis:
            return False
        result = await self._safe_redis_op(
            lambda: _rc(self._redis).delete(key), 2.0, "dlq_delete",
        )
        return bool(result)

    VOTE_COUNT_HISTORY_MAX = 500

    async def record_pending_outcome(
        self,
        pair: str,
        alert_key: str,
        direction: str,
        entry_ts: int,
        entry_price: float,
        confluence_score: Optional[float] = None,
        confluence_total: Optional[float] = None,
        confluence_votes: Optional[Dict[str, bool]] = None,
        adx_val: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        signal_price: Optional[float] = None,
        fill_price: Optional[float] = None,
        fees_paid_pct: Optional[float] = None,
        effective_score: Optional[float] = None,
        effective_required: Optional[float] = None,
        macro_multiplier: Optional[float] = None,
        cluster_penalty: Optional[float] = None,
        gate_passed: Optional[bool] = None,
    ) -> None:

        if self.degraded or not cfg.ENABLE_WIN_RATE_FILTER:
            return

        _cd_hit = await self.in_trade_cooldown(pair, entry_ts)
        if _cd_hit is not None:
            logger.info(
                f"[{pair}] Cooldown after target (hit candle {_cd_hit}) — "
                f"not recording {alert_key} @ {entry_ts}"
            )
            return

        key = f"{RedisKeyPrefix.OUTCOME_PENDING}{pair}:{alert_key}:{entry_ts}"
        try:
            payload = json_dumps({
                "direction": direction,
                "entry_ts": entry_ts,
                "entry_price": entry_price,
                "confluence_score": confluence_score,
                "confluence_total": confluence_total,
                "confluence_votes": confluence_votes,
                "adx_val": adx_val,
                "context": context,
                "signal_price": signal_price,
                "fill_price": fill_price,
                "fees_paid_pct": fees_paid_pct,
                "effective_score": effective_score,
                "effective_required": effective_required,
                "macro_multiplier": macro_multiplier,
                "cluster_penalty": cluster_penalty,
                "gate_passed": gate_passed,
            })
        except Exception as e:
            logger.warning(
                f"Failed to serialize pending outcome for {pair}:{alert_key}: {e}"
            )
            return
        ttl = max(
            (cfg.OUTCOME_LOOKAHEAD_CANDLES + 4) * 15 * 60,
            24 * 3600,
        )
        try:
            await asyncio.wait_for(
                _rc(self._redis).set(key, payload, ex=ttl),
                timeout=2.0,
            )
        except Exception as e:
            logger.warning(
                f"Failed to record pending outcome for {pair}:{alert_key}: {e}"
            ) 
        if confluence_votes is not None:
            await self.record_vote_count(alert_key, confluence_votes)

    async def cancel_pending_outcome(
        self,
        pair: str,
        alert_key: str,
        entry_ts: int,
    ) -> bool:
        """Remove a pending real outcome when the alert was never delivered.

        A coalesced alert must not remain in the real-outcome population,
        because no Telegram alert was actually delivered to the user.
        """
        if self.degraded or not self._redis:
            return False

        key = f"{RedisKeyPrefix.OUTCOME_PENDING}{pair}:{alert_key}:{entry_ts}"

        try:
            deleted = await asyncio.wait_for(
                _rc(self._redis).delete(key),
                timeout=2.0,
            )
            return bool(deleted)
        except Exception as e:
            logger.warning(
                f"Failed to cancel pending outcome for "
                f"{pair}:{alert_key}:{entry_ts}: {e}"
            )
            return False

    async def record_shadow_pending_outcome(
        self,
        pair: str,
        alert_key: str,
        direction: str,
        entry_ts: int,
        entry_price: float,
        confluence_score: Optional[float] = None,
        confluence_total: Optional[float] = None,
        confluence_votes: Optional[Dict[str, bool]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:

        if self.degraded or not getattr(cfg, "ENABLE_BRAIN", False):
            return

        _cd_hit = await self.in_trade_cooldown(pair, entry_ts)
        if _cd_hit is not None:
            logger.info(
                f"[{pair}] Cooldown after target (hit candle {_cd_hit}) — "
                f"not shadowing {alert_key} @ {entry_ts}"
            )
            return

        key = f"{RedisKeyPrefix.SHADOW_PENDING}{pair}:{alert_key}:{entry_ts}"
        try:
            payload = json_dumps({
                "direction": direction,
                "entry_ts": entry_ts,
                "entry_price": entry_price,
                "confluence_score": confluence_score,
                "confluence_total": confluence_total,
                "confluence_votes": confluence_votes,
                "context": context,
            })
        except Exception as e:
            logger.warning(
                f"Failed to serialize shadow pending outcome for {pair}:{alert_key}: {e}"
            )
            return

        ttl = max(
            (cfg.OUTCOME_LOOKAHEAD_CANDLES + 4) * 15 * 60,
            24 * 3600,
        )

        try:
            await asyncio.wait_for(
                _rc(self._redis).set(key, payload, ex=ttl),
                timeout=2.0,
            )
        except Exception as e:
            logger.warning(
                f"Failed to record shadow pending outcome for {pair}:{alert_key}: {e}"
            )
    # ── Vote-count history (OOD gate) ────────────────────────────────────────

    async def record_vote_count(self, alert_key: str, votes: Dict[str, bool]) -> None:
        if self.degraded or not self._redis:
            return

        count = sum(1 for v in votes.values() if v)
        key = f"{RedisKeyPrefix.VOTE_COUNT_HISTORY}{alert_key}"

        try:
            async with self._redis.pipeline() as pipe:
                pipe.lpush(key, str(count))
                pipe.ltrim(key, 0, self.VOTE_COUNT_HISTORY_MAX - 1)
                pipe.expire(key, 30 * 86400)
                await self._safe_redis_op(
                    lambda: pipe.execute(),
                    2.0,
                    f"vote_count_save:{alert_key}",
                )
        except Exception as e:
            logger.warning(f"Vote-count history save failed for '{alert_key}': {e}")

    async def get_vote_count_history(self, alert_key: str) -> List[int]:
        if self.degraded or not self._redis:
            return []

        key = f"{RedisKeyPrefix.VOTE_COUNT_HISTORY}{alert_key}"

        try:
            raw_list = await self._safe_redis_op(
                lambda: _rc(self._redis).lrange(
                    key,
                    0,
                    self.VOTE_COUNT_HISTORY_MAX - 1,
                ),
                2.0,
                f"vote_count_load:{alert_key}",
            )

            return [int(x) for x in raw_list] if raw_list else []
        except Exception:
            return []

    async def _fetch_pending_keys(
        self, pair: str, precomputed_attr: str, key_prefix: str,
        logger_pair: logging.Logger, label: str,
    ) -> List[str]:
        """Shared key-lookup for resolve_pending_outcomes / resolve_shadow_pending_outcomes:
        use the run-level pre-scan when available, else fall back to a per-pair scan."""
        precomputed = getattr(self, precomputed_attr, None)
        if precomputed is not None:
            return precomputed.get(pair, [])
        try:
            pattern = f"{key_prefix}{pair}:*"
            return [k async for k in _rc(self._redis).scan_iter(match=pattern, count=2000)]
        except Exception as e:
            logger_pair.debug(f"Failed to scan {label} outcomes for {pair}: {e}")
            return []

    def _parse_pending_outcome_row(
        self, key: str, raw: Optional[str], data_15m: "PriceData", i15: int,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        if raw is None:
            return None, "raced"
        data = json_loads(raw)
        entry_ts = int(data["entry_ts"])
        direction = data["direction"]
        entry_price = float(data["entry_price"])
        conf_score = data.get("confluence_score")
        conf_total = data.get("confluence_total")
        conf_votes = data.get("confluence_votes")
        adx_val = data.get("adx_val")

        if entry_price <= 0:
            return None, "bad_entry_price"
        direction_norm = str(direction).lower()
        if direction_norm in ("buy", "long"):
            is_buy = True
        elif direction_norm in ("sell", "short"):
            is_buy = False
        else:
            return None, "bad_direction"

        entry_idx = int(np.searchsorted(data_15m.ts, entry_ts))
        if entry_idx >= len(data_15m.ts) or data_15m.ts[entry_idx] != entry_ts:
            exact_matches = np.flatnonzero(data_15m.ts == entry_ts)
            if exact_matches.size == 0:
                return None, "ts_mismatch"
            entry_idx = int(exact_matches[-1])

        # ── Simulated executable fill: no live order exists, so use the     
        data.setdefault("signal_price", entry_price)
        fill_delay = max(0, int(getattr(cfg, "OUTCOME_FILL_DELAY_CANDLES", 1)))
        fill_idx = entry_idx + fill_delay
        if fill_delay > 0 and fill_idx < len(data_15m.open):
            data["fill_price"] = float(data_15m.open[fill_idx])
        else:
            data["fill_price"] = data.get("fill_price") or entry_price

        # ── FIX (Priority 1): Use fill_price as the anchor for all
        anchor_price = float(data["fill_price"])
        
        # FIX (Issue 2): Anchor lookahead horizon to the fill candle, not the signal candle.
        # This ensures the trade is evaluated over the full N candles post-execution.
        target_idx = fill_idx + cfg.OUTCOME_LOOKAHEAD_CANDLES
        if target_idx > i15:
            return None, "not_ready"
        future_price = float(data_15m.close[target_idx])
        pct_move = (future_price - anchor_price) / anchor_price * 100.0

        # ── R:R-BASED THRESHOLDS (anchored to fill_price) ──
        risk_pct = cfg.OUTCOME_MAE_LOSS_PCT / 100.0            # e.g. 0.005
        target_pct = risk_pct * cfg.OUTCOME_RR_TARGET           # e.g. 0.010 (1.0%)
        bonus_pct = risk_pct * cfg.OUTCOME_BONUS_RR             # e.g. 0.015 (1.5%)

        # ── METRIC 1: close_win (legacy point-in-time check) ──
        close_win = (
            pct_move >= cfg.OUTCOME_FAVORABLE_MOVE_PCT
            if is_buy
            else pct_move <= -cfg.OUTCOME_FAVORABLE_MOVE_PCT
        )
        # ── MAE / MFE from price path (anchored to fill_price) ──
        # FIX (Issue 1): Include the fill candle in the path. Since the simulated 
        # fill occurs at the open of fill_idx, the remainder of that candle's 
        # high/low is tradable and must be evaluated for TP/SL/MFE/MAE.
        path_start = fill_idx if fill_delay > 0 else entry_idx + 1
        path_end = min(target_idx + 1, len(data_15m.low))
        path_low = data_15m.low[path_start:path_end]
        path_high = data_15m.high[path_start:path_end]
        mae = mfe = None
        if len(path_low) and len(path_high):
            if is_buy:
                mae = max(0.0, (anchor_price - float(np.min(path_low))) / anchor_price)
                mfe = max(0.0, (float(np.max(path_high)) - anchor_price) / anchor_price)
            else:
                mae = max(0.0, (float(np.max(path_high)) - anchor_price) / anchor_price)
                mfe = max(0.0, (anchor_price - float(np.min(path_low))) / anchor_price)

        # ── PATH: candle-by-candle excursions in the TRADE's direction, in % of
        # fill price. Lets the Brain replay any other stop/target/horizon on this
        # exact trade later (see plan_replay.py). Index 0 = fill candle. ──
        path_fav = path_adv = path_close = None
        mfe_candle = mae_candle = None
        if len(path_low) and len(path_high):
            path_close_raw = data_15m.close[path_start:path_end]
            if is_buy:
                _fav = (path_high - anchor_price) / anchor_price * 100.0
                _adv = (anchor_price - path_low) / anchor_price * 100.0
                _cls = (path_close_raw - anchor_price) / anchor_price * 100.0
            else:
                _fav = (anchor_price - path_low) / anchor_price * 100.0
                _adv = (path_high - anchor_price) / anchor_price * 100.0
                _cls = (anchor_price - path_close_raw) / anchor_price * 100.0
            path_fav = [round(float(x), 4) for x in _fav]
            path_adv = [round(float(x), 4) for x in _adv]
            path_close = [round(float(x), 4) for x in _cls]
            mfe_candle = int(np.argmax(_fav))
            mae_candle = int(np.argmax(_adv))

        # ── METRIC 2: mfe_win (TP hit at 1:2 R:R) ──
        mfe_win = mfe is not None and mfe >= target_pct

        # ── METRIC 3: mae_loss (SL hit) ──
        mae_loss = mae is not None and mae >= risk_pct

        # ── BONUS: did price exceed the target R:R? ──
        bonus_win = mfe is not None and mfe >= bonus_pct

        # ── R-MULTIPLE ACHIEVED ──
        rr_achieved = (mfe / risk_pct) if (mfe is not None and risk_pct > 0) else 0.0

        # ── BONUS: tp_first ordering (candle-by-candle, anchored to fill_price) ──
        tp_first: Optional[bool] = None
        tp_hit_idx: Optional[int] = None
        sl_hit_idx: Optional[int] = None
        ambiguous_same_candle = False          # ← initialized BEFORE the loop
        if len(path_low) and len(path_high):
            tp_level = anchor_price * (1 + target_pct) if is_buy else anchor_price * (1 - target_pct)
            sl_level = anchor_price * (1 - risk_pct) if is_buy else anchor_price * (1 + risk_pct)

            for candle_offset in range(len(path_low)):
                idx = path_start + candle_offset

                candle_low = float(data_15m.low[idx])
                candle_high = float(data_15m.high[idx])

                tp_reached = candle_high >= tp_level if is_buy else candle_low <= tp_level
                sl_reached = candle_low <= sl_level if is_buy else candle_high >= sl_level

                if tp_reached and tp_hit_idx is None:
                    tp_hit_idx = candle_offset
                if sl_reached and sl_hit_idx is None:
                    sl_hit_idx = candle_offset

                # Early exit if both found
                if tp_hit_idx is not None and sl_hit_idx is not None:
                    if tp_hit_idx < sl_hit_idx:
                        tp_first = True
                    elif sl_hit_idx < tp_hit_idx:
                        tp_first = False
                    else:
                        tp_first = False
                        ambiguous_same_candle = True
                    break
                elif tp_hit_idx is not None:
                    tp_first = True   # Only TP hit
                elif sl_hit_idx is not None:
                    tp_first = False  # Only SL hit
        # else: neither hit → tp_first stays None

        # ── OUTCOME REASON: human-readable label mirroring tp_first/mfe_win/
        # mae_loss, kept for archive_reader.py and any reporting that wants
        # a single descriptive field instead of the boolean trio ──
        if ambiguous_same_candle:
            outcome_reason = "ambiguous_same_candle"
        elif tp_first is True:
            outcome_reason = "target_hit"
        elif tp_first is False:
            outcome_reason = "stop_hit"
        elif mfe_win and mae_loss:
            outcome_reason = "both_hit"
        elif mfe_win:
            outcome_reason = "target_hit_ever"
        elif mae_loss:
            outcome_reason = "stop_hit_ever"
        else:
            outcome_reason = "no_hit"

        # ── NET P&L (cost-adjusted, fill-aware when available) ──
        # Moved ABOVE win determination so OUTCOME_PRIMARY_METRIC=
        # "net_pnl_pct" can actually use it as the win label.
        fee_pct = getattr(cfg, "BRAIN_FEE_PCT", 0.0006)
        slip_pct = getattr(cfg, "BRAIN_SLIPPAGE_PCT", 0.0003)
        base_cost_pct = (fee_pct * 2 + slip_pct * 2) * 100  # round-trip, in %

        sig_p = data.get("signal_price")
        fill_p = data.get("fill_price")
        if sig_p and fill_p and float(sig_p) > 0:
            sig_p = float(sig_p)
            fill_p = float(fill_p)
            if is_buy:
                entry_slip_pct = (fill_p - sig_p) / sig_p * 100
            else:
                entry_slip_pct = (sig_p - fill_p) / sig_p * 100

            realized_cost = (fee_pct * 2) * 100 + abs(entry_slip_pct) * 2
        else:
            realized_cost = base_cost_pct

        if is_buy:
            net_pnl_hold_pct = pct_move - realized_cost
        else:
            net_pnl_hold_pct = -pct_move - realized_cost

        # ── OUTCOME CLASS + PLAN P&L ──
        # net_pnl_pct now describes the SAME trade the label describes: the
        # stop/target bracket decides the exit; only a true timeout exits at the
        # horizon close. (Before, a trade that hit target on candle 7 but closed
        # below the stop on candle 12 was labelled WIN yet stored a LOSS here.)
        # The old hold-to-horizon figure is kept as net_pnl_hold_pct.
        if tp_first is True:
            outcome_class = "tp"
            net_pnl_pct = target_pct * 100.0 - realized_cost
        elif tp_first is False:
            outcome_class = "sl"
            net_pnl_pct = -risk_pct * 100.0 - realized_cost
        else:
            outcome_class = "timeout"
            net_pnl_pct = net_pnl_hold_pct

        # ── PRIMARY WIN: configurable ──
        primary_metric = getattr(cfg, "OUTCOME_PRIMARY_METRIC", "mfe")
        if primary_metric == "mfe":
            if tp_first is True:
                win = True
            elif tp_first is False:
                win = False
            elif mfe_win and not mae_loss:
                win = True       # hit TP, never hit SL
            elif mae_loss and not mfe_win:
                win = False      # hit SL, never hit TP
            else:
                win = close_win  # neither hit, or ambiguous → fall back to close
        elif primary_metric == "net_pnl_pct":
            win = net_pnl_pct > 0
        else:
            win = close_win

        win_weight = _compute_win_weight(rr_achieved, win)
        return {
            "alert_key": key.split(":")[-2],
            "direction": direction,
            "entry_ts": entry_ts,
            "is_buy": is_buy,
            "pct_move": pct_move,
            "anchor_price": anchor_price,  # NEW: the fill_price used as measurement anchor
            # ── Primary win (used by ALL downstream: threshold_engine, CUSUM, brain) ──
            "win": win,
            # ── Three-metric breakdown (for reporting) ──
            "close_win": close_win,
            "mfe_win": mfe_win,
            "mae_loss": mae_loss,
            "tp_first": tp_first,
            "outcome_reason": outcome_reason,
            "mae": mae,
            "mfe": mfe,
            # ── R:R and Bonus fields ──
            "bonus_win": bonus_win,
            "rr_achieved": round(rr_achieved, 2),
            "win_weight": win_weight,
            "conf_score": conf_score,
            "conf_total": conf_total,
            "conf_votes": conf_votes,
            "adx_val": adx_val,
            "context": data.get("context"),
            "signal_price": data.get("signal_price"),
            "fill_price": data.get("fill_price"),
            "fees_paid_pct": data.get("fees_paid_pct"),
            "net_pnl_pct": round(net_pnl_pct, 6),
            "net_pnl_hold_pct": round(net_pnl_hold_pct, 6),
            "outcome_class": outcome_class,
            "tp_candle": tp_hit_idx,
            "sl_candle": sl_hit_idx,
            "mfe_candle": mfe_candle,
            "mae_candle": mae_candle,
            "plan_sl_pct": round(risk_pct * 100.0, 4),
            "plan_tp_pct": round(target_pct * 100.0, 4),
            "plan_horizon": int(cfg.OUTCOME_LOOKAHEAD_CANDLES),
            "path_fav": path_fav,
            "path_adv": path_adv,
            "path_close": path_close,
            "realized_cost_pct": round(realized_cost, 6),
            "cost_basis": "measured_entry_slippage_plus_assumed_exit" if (sig_p and fill_p) else "flat_estimate",
            "effective_score": data.get("effective_score"),
            "effective_required": data.get("effective_required"),
            "macro_multiplier": data.get("macro_multiplier"),
            "cluster_penalty": data.get("cluster_penalty"),
            "gate_passed": data.get("gate_passed"),
        }, ""

    def _early_close_payload(self, raw: str, data_15m: "PriceData", i15: int) -> Optional[str]:
        """If this pending trade's stop or target has already been touched on a
        closed candle, return its JSON with closed_ts/closed_reason added; else None.
        Uses the same fill candle and R:R levels as the horizon resolution, so
        'closed' means exactly what the final outcome will later say."""
        try:
            data = json_loads(raw)
            if data.get("closed_ts") is not None:
                return None
            entry_ts = int(data["entry_ts"])
            entry_price = float(data["entry_price"])
            if entry_price <= 0:
                return None
            is_buy = str(data["direction"]).lower() in ("buy", "long")
            hits = np.flatnonzero(data_15m.ts == entry_ts)
            if hits.size == 0:
                return None
            entry_idx = int(hits[-1])
            fill_delay = max(0, int(getattr(cfg, "OUTCOME_FILL_DELAY_CANDLES", 1)))
            fill_idx = entry_idx + fill_delay
            if fill_delay > 0:
                if fill_idx > i15 or fill_idx >= len(data_15m.open):
                    return None
                anchor = float(data_15m.open[fill_idx])
                path_start = fill_idx
            else:
                anchor = float(data.get("fill_price") or entry_price)
                path_start = entry_idx + 1
            lows = data_15m.low[path_start:i15 + 1]
            highs = data_15m.high[path_start:i15 + 1]
            if not len(lows):
                return None

            risk = cfg.OUTCOME_MAE_LOSS_PCT / 100.0
            target = risk * cfg.OUTCOME_RR_TARGET
            if is_buy:
                tp_mask = highs >= anchor * (1 + target)
                sl_mask = lows <= anchor * (1 - risk)
            else:
                tp_mask = lows <= anchor * (1 - target)
                sl_mask = highs >= anchor * (1 + risk)
            tp_hit = bool(tp_mask.any())
            sl_hit = bool(sl_mask.any())
            if not (tp_hit or sl_hit):
                return None
            n_path = len(lows)
            tp_idx = int(np.argmax(tp_mask)) if tp_hit else n_path
            sl_idx = int(np.argmax(sl_mask)) if sl_hit else n_path
            if tp_idx < sl_idx:
                reason, hit_idx = "target", tp_idx
            elif sl_idx < tp_idx:
                reason, hit_idx = "stop", sl_idx
            else:
                reason, hit_idx = "both", tp_idx  # both touched inside the same candle
            data["closed_ts"] = int(time.time())
            data["closed_reason"] = reason
            data["closed_candle_ts"] = int(data_15m.ts[path_start + hit_idx])
            return json_dumps(data)
        except Exception:
            return None

    async def get_active_trade(self, pair: str) -> Optional[Dict[str, Any]]:
        """The open recorded trade for this pair, or None.

        Open = a pending-outcome row that exists and is not marked closed.
        Uses the run-level key pre-scan (no extra SCAN); costs one pipelined
        GET batch, and only when the pair has pending keys at all."""
        if self.degraded or not self._redis:
            return None
        try:
            keys = await self._fetch_pending_keys(
                pair, "_pending_outcome_keys_by_pair", RedisKeyPrefix.OUTCOME_PENDING,
                logger, "pending",
            )
            if not keys:
                return None
            async with self._redis.pipeline() as pipe:
                for k in keys:
                    pipe.get(k)
                raws = await asyncio.wait_for(_execute_pipeline(pipe), timeout=2.0)
            best: Optional[Dict[str, Any]] = None
            for k, raw in zip(keys, raws):
                if not raw:
                    continue
                d = json_loads(raw)
                if d.get("closed_ts") is not None:
                    continue
                ets = int(d["entry_ts"])
                if best is None or ets > best["entry_ts"]:
                    key_s = str(k)
                    parts = key_s.split(":")
                    best = {"pair": pair, "alert_key": parts[-2] if len(parts) >= 2 else "",
                            "direction": str(d.get("direction", "")), "entry_ts": ets}
            return best
        except Exception as e:
            logger.warning(f"get_active_trade failed for {pair}: {e}")
            return None

    async def in_trade_cooldown(self, pair: str, entry_ts: int) -> Optional[int]:
        """Return the target-hit candle ts when `entry_ts` is inside the pair's
        re-entry cooldown, else None. Fails open (Redis problem = no cooldown)."""
        n = int(getattr(cfg, "TRADE_CLOSE_COOLDOWN_CANDLES", 0))
        if (
            n <= 0
            or not getattr(cfg, "ENABLE_TRADE_CLOSE_NOTICE", True)
            or self.degraded
            or not self._redis
        ):
            return None
        try:
            raw = await asyncio.wait_for(
                _rc(self._redis).get(f"{RedisKeyPrefix.TRADE_COOLDOWN}{pair}"),
                timeout=2.0,
            )
            if raw is None:
                return None
            hit_ts = int(raw)
        except Exception:
            return None
        return hit_ts if int(entry_ts) <= hit_ts + n * 900 else None

    async def _start_trade_cooldown(self, pair: str, hit_ts: int,
                                    logger_pair: logging.Logger) -> None:
        if self.degraded or not self._redis:
            return
        key = f"{RedisKeyPrefix.TRADE_COOLDOWN}{pair}"
        ttl = int((int(cfg.TRADE_CLOSE_COOLDOWN_CANDLES) + 4) * 900)
        try:
            raw = await asyncio.wait_for(_rc(self._redis).get(key), timeout=2.0)
            if raw is not None and int(raw) >= int(hit_ts):
                return  # never move the cooldown backwards
            await asyncio.wait_for(
                _rc(self._redis).set(key, str(int(hit_ts)), ex=ttl), timeout=2.0
            )
        except Exception as e:
            logger_pair.warning(f"[{pair}] Could not start trade cooldown: {e}")

    async def _publish_close_events(self, pair: str, rows: List[Dict[str, Any]],
                                    source: str, logger_pair: logging.Logger) -> None:
        """Queue 'Target Done / Stop Hit' notices for this run's Telegram update and
        start the re-entry cooldown. Call only AFTER the closed_ts write succeeded,
        so each trade is announced once. `source` is 'Recorded' or 'Shadowed'."""
        if not getattr(cfg, "ENABLE_TRADE_CLOSE_NOTICE", True):
            return
        cooldown_hit_ts: Optional[int] = None
        for d in rows:
            try:
                entry_ts = int(d.get("entry_ts"))
            except (TypeError, ValueError):
                continue
            direction = "buy" if str(d.get("direction", "")).lower() in ("buy", "long") else "sell"
            ident = (pair, source, direction, entry_ts)
            if ident in self._close_event_ids:
                continue
            self._close_event_ids.add(ident)
            reason = str(d.get("closed_reason") or "")
            hit_ts = d.get("closed_candle_ts")
            self.trade_close_events.append({
                "pair": pair,
                "source": source,
                "direction": direction,
                "entry_ts": entry_ts,
                "reason": reason,
                "hit_ts": int(hit_ts) if hit_ts is not None else None,
            })
            starts_cooldown = reason == "target" or (
                reason in ("stop", "both")
                and getattr(cfg, "TRADE_CLOSE_COOLDOWN_ON_STOP", False)
            )
            if starts_cooldown and hit_ts is not None:
                cooldown_hit_ts = max(int(hit_ts), cooldown_hit_ts or 0)
        if cooldown_hit_ts is not None and int(getattr(cfg, "TRADE_CLOSE_COOLDOWN_CANDLES", 0)) > 0:
            await self._start_trade_cooldown(pair, cooldown_hit_ts, logger_pair)

    async def resolve_pending_outcomes(self, pair: str, data_15m: "PriceData", i15: int,
                                       logger_pair: logging.Logger) -> None:
        if self.degraded or not cfg.ENABLE_WIN_RATE_FILTER or not self._redis:
            return

        keys = await self._fetch_pending_keys(
            pair, "_pending_outcome_keys_by_pair", RedisKeyPrefix.OUTCOME_PENDING,
            logger_pair, "pending",
        )
        if not keys:
            return
        try:
            async with self._redis.pipeline() as read_pipe:
                for key in keys:
                    read_pipe.get(key)
                raw_values = await asyncio.wait_for(
                    _execute_pipeline(read_pipe),
                    timeout=2.0,
                )
        except Exception as e:
            logger_pair.warning(f"Failed to batch-fetch pending outcomes for {pair}: {e}")
            return

        resolved_count = 0
        not_ready_count = 0
        ts_mismatch_count = 0
        missing_score_count = 0
        bad_payload_count = 0

        stats_ttl = max(cfg.STATE_EXPIRY_DAYS * 86400, 7 * 86400)
        resolved_for_file: List[Dict[str, Any]] = []
        close_events: List[Dict[str, Any]] = []
        try:
            async with self._redis.pipeline() as write_pipe:
                pending_writes = 0

                for key, raw in zip(keys, raw_values):
                    try:
                        result, skip_reason = self._parse_pending_outcome_row(key, raw, data_15m, i15)
                        if skip_reason == "raced":
                            continue
                        if skip_reason == "bad_entry_price":
                            logger_pair.debug(f"Invalid entry_price for pending outcome {key}; skipping")
                            bad_payload_count += 1
                            continue
                        if skip_reason == "bad_direction":
                            logger_pair.debug(f"Unknown direction for pending outcome {key}; skipping")
                            bad_payload_count += 1
                            continue
                        if skip_reason == "ts_mismatch":
                            ts_mismatch_count += 1
                            if cfg.DEBUG_MODE:
                                logger_pair.debug(
                                    f"[{pair}] Outcome entry_ts not found | "
                                    f"first_ts={data_15m.ts[0] if len(data_15m.ts) else None} | "
                                    f"last_ts={data_15m.ts[-1] if len(data_15m.ts) else None}"
                                )
                            continue

                        if skip_reason == "not_ready":
                            not_ready_count += 1
                            if (cfg.ENABLE_SINGLE_ACTIVE_TRADE or cfg.ENABLE_TRADE_CLOSE_NOTICE) and raw:
                                # Stop or target already hit: the trade is over for the
                                # one-active-trade rule. The row stays until its normal
                                # horizon resolution, so the Brain statistics are unchanged.
                                _closed = self._early_close_payload(raw, data_15m, i15)
                                if _closed is not None:
                                    write_pipe.set(key, _closed, keepttl=True)
                                    pending_writes += 1
                                    try:
                                        _d = json_loads(_closed)
                                        close_events.append(_d)
                                        logger_pair.info(
                                            f"[{pair}] Early-close for one-active-trade | "
                                            f"key={key} | "
                                            f"reason={_d.get('closed_reason')} | "
                                            f"entry_ts={_d.get('entry_ts')} | "
                                            f"closed_ts={_d.get('closed_ts')}"
                                        )
                                    except Exception:
                                        pass
                            continue
                        if result is None:
                            continue

                        alert_key = result["alert_key"]
                        direction = result["direction"]
                        entry_ts = result["entry_ts"]
                        pct_move = result["pct_move"]
                        win = result["win"]
                        mae = result["mae"]
                        mfe = result["mfe"]
                        conf_score = result["conf_score"]
                        conf_total = result["conf_total"]
                        conf_votes = result["conf_votes"]
                        adx_val = result["adx_val"]
                        row_context = result.get("context")
                        signal_price = result.get("signal_price")
                        fill_price = result.get("fill_price")
                        fees_paid_pct = result.get("fees_paid_pct")
                        stats_key = f"{RedisKeyPrefix.ALERT_STATS}{pair}:{alert_key}"
                        write_pipe.hincrby(stats_key, "wins" if win else "losses", 1)
                        write_pipe.expire(stats_key, stats_ttl)
                        session = _get_session_from_ts(entry_ts) if entry_ts else "dead"
                        session_stats_key = f"{stats_key}:{session}"
                        write_pipe.hincrby(session_stats_key, "wins" if win else "losses", 1)
                        write_pipe.expire(session_stats_key, stats_ttl)
                        stream_fields: Optional[Dict[StreamField, StreamField]] = None
                        if conf_score is not None and conf_total is not None:
                            stream_fields = {
                                "pair": str(pair),
                                "alert_key": str(alert_key),
                                "direction": str(direction),
                                "score": str(conf_score),
                                "total": str(conf_total),
                                "pct_move": f"{pct_move:.4f}",
                                "win": "1" if win else "0",
                                "entry_ts": str(entry_ts),
                                "session": session,
                                "mae": f"{mae:.5f}" if mae is not None else "",
                                "mfe": f"{mfe:.5f}" if mfe is not None else "",
                                "close_win": "1" if result.get("close_win", win) else "0",
                                "mfe_win": "1" if result.get("mfe_win", False) else "0",
                                "mae_loss": "1" if result.get("mae_loss", False) else "0",
                                "tp_first": (
                                    "1" if result.get("tp_first") is True
                                    else "0" if result.get("tp_first") is False
                                    else ""
                                ),
                                "outcome_reason": result.get("outcome_reason", "unknown"),
                                # ── R:R and Bonus fields ──
                                "bonus_win": "1" if result.get("bonus_win", False) else "0",
                                "rr_achieved": f"{result.get('rr_achieved', 0):.2f}",
                                "win_weight": f"{result.get('win_weight', 1.0):.2f}",
                                "signal_price": f"{signal_price:.8f}" if signal_price is not None else "",
                                "fill_price": f"{fill_price:.8f}" if fill_price is not None else "",
                                "fees_paid_pct": f"{fees_paid_pct:.6f}" if fees_paid_pct is not None else "",
                                "net_pnl_pct": f"{result.get('net_pnl_pct', 0.0):.6f}",
                                "realized_cost_pct": f"{result.get('realized_cost_pct', 0.0):.6f}",
                                **outcome_path_stream_fields(result),
                                "votes": json_dumps(conf_votes) if conf_votes is not None else "",
                                "adx_val": str(adx_val) if adx_val is not None else "",
                                "context": json_dumps(row_context) if row_context is not None else "",
                            }
                        else:
                            missing_score_count += 1
                            logger_pair.debug(
                                f"[{pair}] Outcome for {alert_key} has no confluence score/total; "
                                f"stats updated but stream entry skipped"
                            )
                        if stream_fields is not None and not getattr(cfg, "BRAIN_USE_FILE_STORAGE", False):
                            write_pipe.xadd(
                                RedisKeyPrefix.OUTCOME_LOG_STREAM,
                                stream_fields,
                                maxlen=2000,
                                approximate=True,
                            )
                        write_pipe.delete(key)
                        pending_writes += 1
                        resolved_count += 1
                        if getattr(cfg, "BRAIN_USE_FILE_STORAGE", False):
                            if conf_score is None or conf_total is None:
                                logger_pair.warning(
                                    f"[{pair}] Archived row for {alert_key} dropped: "
                                    f"score={conf_score} total={conf_total}. "
                                    f"Confluence was not computed for this alert's direction "
                                    f"(check ENABLE_CONFLUENCE_GATE and gate_passed)."
                                )
                            else:
                                resolved_for_file.append({
                                    # Stable per-outcome ID: archive_reader dedups on
                                    # `_stream_id`, so a retry after a partial failure
                                    # (archive written, Redis delete failed) cannot
                                    # double-count the trade.
                                    "_stream_id": f"{pair}:{alert_key}:{entry_ts}",
                                    "pair": str(pair),
                                    "alert_key": str(alert_key),
                                    "direction": str(direction),
                                    "entry_ts": entry_ts,
                                    "score": conf_score,
                                    "total": conf_total,
                                    "win": win,
                                    "pct_move": pct_move,
                                    "mae": mae,
                                    "mfe": mfe,
                                    "session": session,
                                    "votes": conf_votes,
                                    "adx_val": adx_val,
                                    "context": row_context,
                                    # ── Three-metric fields ──
                                    "close_win": result.get("close_win", win),
                                    "mfe_win": result.get("mfe_win", False),
                                    "mae_loss": result.get("mae_loss", False),
                                    "tp_first": result.get("tp_first"),
                                    "outcome_reason": result.get("outcome_reason", "unknown"),
                                    # ── R:R and Bonus fields ──
                                    "bonus_win": result.get("bonus_win", False),
                                    "rr_achieved": result.get("rr_achieved", 0.0),
                                    "win_weight": result.get("win_weight", 1.0),
                                    "signal_price": signal_price,
                                    "fill_price": fill_price,
                                    "fees_paid_pct": fees_paid_pct,
                                    "net_pnl_pct": result.get("net_pnl_pct", 0.0),
                                    "realized_cost_pct": result.get("realized_cost_pct", 0.0),
                                    **outcome_path_fields(result, stream=False),
                                    "effective_score": result.get("effective_score"),
                                    "effective_required": result.get("effective_required"),
                                    "macro_multiplier": result.get("macro_multiplier"),
                                    "cluster_penalty": result.get("cluster_penalty"),
                                    "gate_passed": result.get("gate_passed"),
                                })
                    except Exception as e:
                        logger_pair.debug(f"Failed to resolve pending outcome {key}: {e}")
                        bad_payload_count += 1
                        
                        continue

                if pending_writes:
                    # Archive FIRST, then delete pending + update stats. If the file
                    # write fails we bail out before the Redis pipeline executes, so
                    # the pending outcomes stay in Redis and are retried next run
                    # instead of being lost from the Brain archive.
                    if resolved_for_file and getattr(cfg, "BRAIN_USE_FILE_STORAGE", False):
                        try:                    
                            from outcome_storage import append_outcome_batch
                            await asyncio.to_thread(
                                append_outcome_batch, resolved_for_file, False
                            )
                            self._run_archived_total += len(resolved_for_file)
                        except Exception as e:
                            logger_pair.error(
                                f"[{pair}] File archive write failed — {resolved_count} "
                                f"resolved outcome(s) left PENDING in Redis for retry: {e}"
                            )
                            return
                    await asyncio.wait_for(
                        _execute_pipeline(write_pipe),
                        timeout=2.0,
                    )
        except Exception as e:
            dup_risk = " — resolved_for_file was already archived, so a retry next run may re-append duplicate row(s)" if resolved_for_file else ""
            logger_pair.warning(
                f"[{pair}] Failed to persist resolved outcomes (Redis pipeline): {e}{dup_risk}"
            )
            return

        if close_events:
            await self._publish_close_events(pair, close_events, "Recorded", logger_pair)

        self._run_resolved_total += resolved_count
        logger_pair.debug(
            f"[{pair}] Outcome resolution | "
            f"pending={len(keys)} | "
            f"resolved={resolved_count} | "
            f"not_ready={not_ready_count} | "
            f"ts_mismatch={ts_mismatch_count} | "
            f"missing_score={missing_score_count} | "
            f"bad_payload={bad_payload_count}"
        )
        if bad_payload_count:
            logger_pair.warning(
                f"[{pair}] {bad_payload_count} pending outcome(s) had a malformed "
                "payload and were dropped this run (see prior debug lines for keys)"
            )

    async def resolve_shadow_pending_outcomes(
        self,
        pair: str,
        data_15m: "PriceData",
        i15: int,
        logger_pair: logging.Logger,
    ) -> None:
        """Twin of resolve_pending_outcomes for shadow (rejected) alerts. Same grading logic,
        writes to SHADOW_STATS/SHADOW_LOG_STREAM instead, and additionally pools outcomes whose
        confluence was in the 'rewardable' bucket into SHADOW_HICONF_STATS for override checks."""
        if (
            self.degraded
            or not getattr(cfg, "ENABLE_BRAIN", False)
            or not getattr(cfg, "BRAIN_SHADOW_MODE", True)
            or not self._redis
        ):
            return

        keys = await self._fetch_pending_keys(
            pair,
            "_shadow_pending_outcome_keys_by_pair",
            RedisKeyPrefix.SHADOW_PENDING,
            logger_pair,
            "shadow",
        )
        if not keys:
            return
        try:
            raw_values = await asyncio.wait_for(
                _rc(self._redis).mget(keys),
                timeout=2.0,
            )
        except Exception as e:
            logger_pair.warning(
                f"Failed to batch-fetch shadow pending outcomes for {pair}: {e}"
            )
            return

        resolved_count = 0
        hiconf_pct = getattr(cfg, "BRAIN_REWARDABLE_MIN_CONFLUENCE_PCT", 80.0)

        stats_ttl = max(cfg.STATE_EXPIRY_DAYS * 86400, 7 * 86400)
        resolved_for_file: List[Dict[str, Any]] = []
        close_events: List[Dict[str, Any]] = []

        try:
            async with self._redis.pipeline() as write_pipe:
                pending_writes = 0
                for key, raw in zip(keys, raw_values):
                    try:
                        result, skip_reason = self._parse_pending_outcome_row(
                            key, raw, data_15m, i15
                        )
                        if skip_reason:
                            if skip_reason == "not_ready" and raw and cfg.ENABLE_TRADE_CLOSE_NOTICE:
                                _closed = self._early_close_payload(raw, data_15m, i15)
                                if _closed is not None:
                                    write_pipe.set(key, _closed, keepttl=True)
                                    pending_writes += 1
                                    try:
                                        close_events.append(json_loads(_closed))
                                    except Exception:
                                        pass
                            continue
                        if result is None:
                            continue

                        alert_key = result["alert_key"]
                        direction = result["direction"]
                        entry_ts = result["entry_ts"]
                        pct_move = result["pct_move"]
                        win = result["win"]
                        mae = result["mae"]
                        mfe = result["mfe"]
                        conf_score = result["conf_score"]
                        conf_total = result["conf_total"]
                        conf_votes = result["conf_votes"]
                        row_context = result.get("context") or {}
                        shadow_adx_val = row_context.get("adx_val")
                        shadow_rejection_reason = row_context.get("rejection_reason")
                        shadow_effective_score = row_context.get("effective_score")
                        shadow_effective_required = row_context.get("effective_required")
                        shadow_macro_multiplier = row_context.get("macro_multiplier")
                        shadow_cluster_penalty = row_context.get("cluster_penalty")

                        stats_key = f"{RedisKeyPrefix.SHADOW_STATS}{pair}:{alert_key}"
                        write_pipe.hincrby(stats_key, "wins" if win else "losses", 1)
                        write_pipe.expire(stats_key, stats_ttl)

                        if (
                            conf_score is not None
                            and conf_total is not None
                            and conf_total > 0
                        ):
                            conf_pct = (conf_score / conf_total) * 100.0

                            if not getattr(cfg, "BRAIN_USE_FILE_STORAGE", False):
                                shadow_stream_fields: Dict[StreamField, StreamField] = {
                                    "pair": str(pair),
                                    "alert_key": str(alert_key),
                                    "direction": str(direction),
                                    "score": str(conf_score),
                                    "total": str(conf_total),
                                    "pct_move": f"{pct_move:.4f}",
                                    "win": "1" if win else "0",
                                    "entry_ts": str(entry_ts),
                                    "session": _get_session_from_ts(entry_ts) if entry_ts else "dead",
                                    "mae": f"{mae:.5f}" if mae is not None else "",
                                    "mfe": f"{mfe:.5f}" if mfe is not None else "",
                                    "close_win": "1" if result.get("close_win", win) else "0",
                                    "mfe_win": "1" if result.get("mfe_win", False) else "0",
                                    "mae_loss": "1" if result.get("mae_loss", False) else "0",
                                    "tp_first": (
                                        "1" if result.get("tp_first") is True
                                        else "0" if result.get("tp_first") is False
                                        else ""
                                    ),

                                    "outcome_reason": result.get("outcome_reason", "unknown"),
                                    "net_pnl_pct": f"{result.get('net_pnl_pct', 0.0):.6f}",
                                    "realized_cost_pct": f"{result.get('realized_cost_pct', 0.0):.6f}",
                                    **outcome_path_stream_fields(result),
                                    "votes": json_dumps(conf_votes) if conf_votes is not None else "",
                                    "adx_val": str(shadow_adx_val) if shadow_adx_val is not None else "",
                                    "rejection_reason": shadow_rejection_reason or "",
                                    "effective_score": str(shadow_effective_score) if shadow_effective_score is not None else "",
                                    "effective_required": str(shadow_effective_required) if shadow_effective_required is not None else "",
                                    "macro_multiplier": str(shadow_macro_multiplier) if shadow_macro_multiplier is not None else "",
                                    "cluster_penalty": str(shadow_cluster_penalty) if shadow_cluster_penalty is not None else "",
                                }
                                write_pipe.xadd(
                                    RedisKeyPrefix.SHADOW_LOG_STREAM,
                                    shadow_stream_fields,
                                    maxlen=2000,
                                    approximate=True,
                                )

                            if conf_pct >= hiconf_pct:
                                hiconf_key = f"{RedisKeyPrefix.SHADOW_HICONF_STATS}{alert_key}"
                                write_pipe.hincrby(
                                    hiconf_key,
                                    "wins" if win else "losses",
                                    1,
                                )
                                write_pipe.expire(hiconf_key, stats_ttl)

                        write_pipe.delete(key)
                        pending_writes += 1
                        resolved_count += 1

                        if getattr(cfg, "BRAIN_USE_FILE_STORAGE", False):
                            if conf_score is None or conf_total is None:
                                logger_pair.warning(
                                    f"[{pair}] Shadow archived row for {alert_key} dropped: "
                                    f"score={conf_score} total={conf_total}. "
                                    f"Confluence was not computed for this alert's direction "
                                    f"(check ENABLE_CONFLUENCE_GATE and gate_passed)."
                                )
                            else:
                                resolved_for_file.append({
                                    "pair": str(pair),
                                    "alert_key": str(alert_key),
                                    "direction": str(direction),
                                    "entry_ts": entry_ts,
                                    "score": conf_score,
                                    "total": conf_total,
                                    "win": win,
                                    "pct_move": pct_move,
                                    "mae": mae,
                                    "mfe": mfe,
                                    "session": _get_session_from_ts(entry_ts) if entry_ts else "dead",
                                    "votes": conf_votes,
                                    "shadow": True,
                                    "_stream_id": f"{pair}:{alert_key}:{entry_ts}",
                                    # ─ Three-metric fields ──
                                    "close_win": result.get("close_win", win),
                                    "mfe_win": result.get("mfe_win", False),
                                    "mae_loss": result.get("mae_loss", False),
                                    "tp_first": result.get("tp_first"),
                                    "outcome_reason": result.get("outcome_reason", "unknown"),
                                    # ── R:R and Bonus fields ──
                                    "bonus_win": result.get("bonus_win", False),
                                    "rr_achieved": result.get("rr_achieved", 0.0),
                                    "win_weight": result.get("win_weight", 1.0),
                                    "net_pnl_pct": result.get("net_pnl_pct", 0.0),
                                    "realized_cost_pct": result.get("realized_cost_pct", 0.0),
                                    **outcome_path_fields(result, stream=False),
                                    "adx_val": shadow_adx_val,
                                    "rejection_reason": shadow_rejection_reason,
                                    "effective_score": result.get("effective_score"),
                                    "effective_required": result.get("effective_required"),
                                    "macro_multiplier": result.get("macro_multiplier"),
                                    "cluster_penalty": result.get("cluster_penalty"),
                                    "gate_passed": result.get("gate_passed")
                                })
                    except Exception as e:
                        logger_pair.debug(
                            f"Failed to resolve shadow pending outcome {key}: {e}"
                        )
                        continue

                if pending_writes:
                    # Archive FIRST, then delete pending + update stats — same
                    # safety ordering as the real-outcome path. If the file
                    # write fails we bail out before the Redis pipeline runs,
                    # so the shadow pending outcomes stay in Redis and are
                    # retried next run instead of being lost from the archive.
                    if resolved_for_file and getattr(cfg, "BRAIN_USE_FILE_STORAGE", False):
                        try:
                            from outcome_storage import append_outcome_batch
                            await asyncio.to_thread(
                                append_outcome_batch, resolved_for_file, True
                            )
                        except Exception as e:
                            logger_pair.error(
                                f"[{pair}] Shadow file archive write failed — {resolved_count} "
                                f"resolved shadow outcome(s) left PENDING in Redis for retry: {e}"
                            )
                            return
                    await asyncio.wait_for(_execute_pipeline(write_pipe), timeout=2.0)

        except Exception as e:
            dup_risk = " — resolved_for_file was already archived, so a retry next run may re-append duplicate row(s)" if resolved_for_file else ""
            logger_pair.warning(
                f"[{pair}] Failed to persist resolved shadow outcomes (Redis pipeline): {e}{dup_risk}"
            )
            return

        if close_events:
            await self._publish_close_events(pair, close_events, "Shadowed", logger_pair)

        if resolved_count:
            logger_pair.debug(
                f"[{pair}] Shadow outcome resolution| resolved={resolved_count}"
            )

    async def get_alert_win_rate(self, pair: str, alert_key: str) -> Tuple[Optional[float], int]:
        """Returns (win_rate, sample_size). win_rate is None until MIN_WIN_RATE_SAMPLE is reached."""
        if self.degraded or not cfg.ENABLE_WIN_RATE_FILTER:
            return None, 0
        stats_key = f"{RedisKeyPrefix.ALERT_STATS}{pair}:{alert_key}"
        try:
            
            data = await asyncio.wait_for(cast("Awaitable[dict[Any, Any]]", _rc(self._redis).hgetall(stats_key)), timeout=2.0)
            wins = int(data.get("wins", 0))
            losses = int(data.get("losses", 0))
            total = wins + losses
            if total < cfg.MIN_WIN_RATE_SAMPLE:
                return None, total
            return wins / total, total
        except Exception as e:
            logger.warning(f"Failed to read win rate for {pair}:{alert_key}: {e}")
            return None, 0

    async def get_alert_win_rate_session(self, pair: str, alert_key: str, session: str) -> Tuple[Optional[float], int]:
        """Session-scoped twin of get_alert_win_rate. win_rate is None until
        MIN_WIN_RATE_SESSION_SAMPLE is reached for this pair:alert_key:session combo."""
        if self.degraded or not cfg.ENABLE_WIN_RATE_FILTER or not getattr(cfg, "ENABLE_SESSION_FILTER", False):
            return None, 0
        stats_key = f"{RedisKeyPrefix.ALERT_STATS}{pair}:{alert_key}:{session}"
        try:
            data = await asyncio.wait_for(cast("Awaitable[dict[Any, Any]]", _rc(self._redis).hgetall(stats_key)), timeout=2.0)
            wins = int(data.get("wins", 0))
            losses = int(data.get("losses", 0))
            total = wins + losses
            if total < getattr(cfg, "MIN_WIN_RATE_SESSION_SAMPLE", 15):
                return None, total
            return wins / total, total
        except Exception as e:
            logger.warning(f"Failed to read session win rate for {pair}:{alert_key}:{session}: {e}")
            return None, 0

    async def batch_get_alert_win_rates(self, pair: str, alert_keys: List[str], timeout: float = 3.0) -> Dict[str, Tuple[Optional[float], int]]:
        if self.degraded or not cfg.ENABLE_WIN_RATE_FILTER or not alert_keys:
            return {k: (None, 0) for k in alert_keys}
        try:
            async with _rc(self._redis).pipeline() as pipe:
                for ak in alert_keys:
                    pipe.hgetall(f"{RedisKeyPrefix.ALERT_STATS}{pair}:{ak}")
                raw_results = await asyncio.wait_for(_execute_pipeline(pipe), timeout=timeout)
            out: Dict[str, Tuple[Optional[float], int]] = {}
            for ak, data in zip(alert_keys, raw_results):
                if not data:
                    out[ak] = (None, 0)
                    continue
                wins = int(data.get("wins", 0))
                losses = int(data.get("losses", 0))
                total = wins + losses
                if total < cfg.MIN_WIN_RATE_SAMPLE:
                    out[ak] = (None, total)
                else:
                    out[ak] = (wins / total, total)
            return out
        except Exception as e:
            logger.warning(f"batch_get_alert_win_rates({pair}) failed for {len(alert_keys)} keys: {e}")
            return {k: (None, 0) for k in alert_keys}

    async def batch_get_all_alert_states(self, pair: str, alert_keys: List[str], timeout: float = 3.0) -> Dict[str, bool]:
        if not self._redis or self.degraded or not alert_keys:
            return {k: False for k in alert_keys}

        try:      
            hash_key = f"{self.state_prefix}{pair}"
            hash_data = await asyncio.wait_for(
                cast("Awaitable[dict[Any, Any]]", _rc(self._redis).hgetall(hash_key)),
                timeout=timeout,
            )
            states: Dict[str, bool] = {}
            for key in alert_keys:
                val = hash_data.get(key)

                if val is None:
                    states[key] = False
                    continue

                try:
                    parsed_state = json_loads(val)
                    states[key] = parsed_state.get("state") == "ACTIVE"
                except (JSONDecodeError, TypeError) as e:
                    if cfg.DEBUG_MODE:
                        logger.debug(f"Failed to parse state for {pair}:{key}: {e}")
                    states[key] = False
                except Exception as e:
                    logger.error(f"Unexpected error parsing state for {pair}:{key}: {e}")
                    states[key] = False

            return states
        except asyncio.TimeoutError as e:
            await self._record_redis_failure(f"batch_get_all_alert_states({pair})", e)
            return {k: False for k in alert_keys}
        except Exception as e:
            await self._record_redis_failure(f"batch_get_all_alert_states({pair})", e)
            return {k: False for k in alert_keys}

    async def atomic_batch_update(self, updates: Sequence[Tuple[str, Any, Optional[int]]], deletes: Optional[List[str]] = None, timeout: float = 4.0, _retried: bool = False) -> bool:
        if self.degraded or not self._redis:
            return False

        if not updates and not deletes:
            return True

        try:
            async with self._redis.pipeline() as pipe:
                now = int(time.time())
                touched_hashes: Set[str] = set()

                hash_writes: Dict[str, Dict[str, str]] = {}
                for key, state, custom_ts in (updates or []):
                    pair, sep, field = key.partition(":")
                    if not sep:
                        logger.error(
                            f"Skipping malformed state key (expected 'pair:field'): {key}"
                        )
                        continue
                    ts = custom_ts if custom_ts is not None else now
                    try:
                        data = json_dumps({"state": state, "ts": ts})
                    except Exception as e:
                        logger.error(f"Failed to serialize state for {key}: {e}")
                        continue
                    hash_key = f"{self.state_prefix}{pair}"
                    hash_writes.setdefault(hash_key, {})[field] = data

                for hash_key, mapping in hash_writes.items():
                    pipe.hset(hash_key, mapping=mapping)
                    touched_hashes.add(hash_key)

                hash_deletes: Dict[str, List[str]] = {}
                for key in (deletes or []):
                    if not key:
                        continue
                    raw_key = (
                        key[len(self.state_prefix) :]
                        if key.startswith(self.state_prefix)
                        else key
                    )
                    pair, sep, field = raw_key.partition(":")
                    if not sep:
                        logger.error(
                            f"Skipping malformed delete key (expected 'pair:field'): {key}"
                        )
                        continue
                    hash_key = f"{self.state_prefix}{pair}"
                    hash_deletes.setdefault(hash_key, []).append(field)

                for hash_key, fields in hash_deletes.items():
                    pipe.hdel(hash_key, *fields)
                    touched_hashes.add(hash_key)

                if self.expiry_seconds > 0:
                    for hash_key in touched_hashes:
                        pipe.expire(hash_key, self.expiry_seconds)

                await asyncio.wait_for(pipe.execute(), timeout=timeout)
            return True
        except asyncio.TimeoutError as e:
            if not _retried:
                logger.warning("atomic_batch_update timed out — retrying once before degrading")
                return await self.atomic_batch_update(updates, deletes, timeout, _retried=True)
            await self._record_redis_failure("atomic_batch_update", e)
            return False
        except Exception as e:
            await self._record_redis_failure("atomic_batch_update", e)
            return False

    # ── CUSUM state persistence ────────────────────────────────���────────
    async def load_cusum_state(self, alert_key: str) -> Optional[Dict[str, Any]]:
        """Load persisted CUSUM accumulator for one alert_key."""
        if self.degraded or not self._redis:
            return None
        key = f"{RedisKeyPrefix.CUSUM_STATE}{alert_key}"
        raw = await self._safe_redis_op(
            lambda: _rc(self._redis).get(key), 2.0, f"cusum_load:{alert_key}",
        )
        if raw is None:
            return None
        try:
            return json_loads(raw)
        except Exception:
            return None

    async def save_cusum_state(self, alert_key: str, state: Dict[str, Any]) -> None:
        if self.degraded or not self._redis:
            return
        key = f"{RedisKeyPrefix.CUSUM_STATE}{alert_key}"
        try:
            await self._safe_redis_op(
                lambda: _rc(self._redis).set(key, json_dumps(state), ex=30 * 86400),
                2.0, f"cusum_save:{alert_key}",
            )
        except Exception as e:
            logger.warning(f"CUSUM state save failed for '{alert_key}': {e}")

    async def load_cusum_watermark(self, alert_key: str) -> int:
        """Last entry_ts already fed into this alert_key's CUSUM detector.
        0 means nothing has been fed yet."""
        if self.degraded or not self._redis:
            return 0
        key = f"{RedisKeyPrefix.CUSUM_WATERMARK}{alert_key}"
        try:
            raw = await self._safe_redis_op(
                lambda: _rc(self._redis).get(key), 2.0, f"cusum_watermark_load:{alert_key}",
            )
            return int(raw) if raw else 0
        except Exception:
            return 0

    async def save_cusum_watermark(self, alert_key: str, entry_ts: int) -> None:
        if self.degraded or not self._redis:
            return
        key = f"{RedisKeyPrefix.CUSUM_WATERMARK}{alert_key}"
        try:
            await self._safe_redis_op(
                lambda: _rc(self._redis).set(key, str(entry_ts), ex=30 * 86400),
                2.0, f"cusum_watermark_save:{alert_key}",
            )
        except Exception as e:
            logger.warning(f"CUSUM watermark save failed for '{alert_key}': {e}")

    async def load_cusum_bulk(
        self, alert_keys: List[str],
    ) -> Optional[Dict[str, Tuple[int, Optional[Dict[str, Any]]]]]:
        """(watermark, persisted_state) for every alert_key in ONE pipelined
        round-trip. Returns None when the read fails or Redis is degraded:
        callers must then skip the CUSUM update, because a fabricated 0
        watermark would replay the whole window into the detectors."""
        if not alert_keys:
            return {}
        if self.degraded or not self._redis:
            return None
        try:
            async with self._redis.pipeline() as pipe:
                for ak in alert_keys:
                    pipe.get(f"{RedisKeyPrefix.CUSUM_WATERMARK}{ak}")
                    pipe.get(f"{RedisKeyPrefix.CUSUM_STATE}{ak}")
                raw = await asyncio.wait_for(_execute_pipeline(pipe), timeout=5.0)
        except Exception as e:
            logger.warning(f"CUSUM bulk load failed: {e}")
            return None
        out: Dict[str, Tuple[int, Optional[Dict[str, Any]]]] = {}
        for idx, ak in enumerate(alert_keys):
            wm_raw, st_raw = raw[2 * idx], raw[2 * idx + 1]
            try:
                wm = int(wm_raw) if wm_raw else 0
            except (TypeError, ValueError):
                wm = 0
            try:
                st = json_loads(st_raw) if st_raw else None
            except Exception:
                st = None
            out[ak] = (wm, st)
        return out

    async def save_cusum_bulk(
        self, items: List[Tuple[str, Dict[str, Any], int]],
    ) -> bool:
        """Persist (alert_key, state, watermark) triples in ONE round-trip."""
        if not items or self.degraded or not self._redis:
            return False
        try:
            async with self._redis.pipeline() as pipe:
                for ak, st, wm in items:
                    pipe.set(f"{RedisKeyPrefix.CUSUM_STATE}{ak}", json_dumps(st), ex=30 * 86400)
                    pipe.set(f"{RedisKeyPrefix.CUSUM_WATERMARK}{ak}", str(int(wm)), ex=30 * 86400)
                await asyncio.wait_for(_execute_pipeline(pipe), timeout=5.0)
            return True
        except Exception as e:
            logger.warning(f"CUSUM bulk save failed: {e}")
            return False

    async def load_threshold_history(self, key_suffix: str = "") -> List[float]:
        """key_suffix="" (default) is the existing global CONFLUENCE_MIN_ABS_SCORE
        history, unchanged. Pass a pair name to track that pair's own history
        under a separate list (used by per-pair threshold stability checks)."""
        if self.degraded or not self._redis:
            return []
        key = f"{RedisKeyPrefix.THRESHOLD_HISTORY}{key_suffix}"
        try:
            raw_list = await self._safe_redis_op(
                lambda: _rc(self._redis).lrange(key, 0, 9),
                2.0, f"threshold_history_load:{key_suffix or 'global'}",
            )
            if not raw_list:
                return []
            return [float(x) for x in raw_list]
        except Exception:
            return []

    async def save_threshold_value(self, value: float, key_suffix: str = "") -> None:
        if self.degraded or not self._redis:
            return
        key = f"{RedisKeyPrefix.THRESHOLD_HISTORY}{key_suffix}"
        try:
            async with self._redis.pipeline() as pipe:
                pipe.lpush(key, str(value))
                pipe.ltrim(key, 0, 9)
                pipe.expire(key, 30 * 86400)
                await self._safe_redis_op(
                    lambda: pipe.execute(), 2.0, f"threshold_history_save:{key_suffix or 'global'}",
                )
        except Exception as e:
            logger.warning(f"Threshold-history save failed for '{key_suffix or 'global'}': {e}")

class RedisLock:    
    RELEASE_LUA = """
    if redis.call("GET", KEYS[1]) == ARGV[1] then
        return redis.call("DEL", KEYS[1])
    else
        return 0
    end
    """
    EXTEND_LUA = """
    if redis.call("GET", KEYS[1]) == ARGV[1] then
        return redis.call("EXPIRE", KEYS[1], ARGV[2])
    else
        return 0
    end
    """
    def __init__(self, redis_client: Optional[redis.Redis], lock_key: str, expire: int | None = None):
        self.redis = redis_client
        self.lock_key = f"{RedisKeyPrefix.LOCK}{lock_key}"
        self.expire = expire or cfg.REDIS_LOCK_EXPIRY
        self.token: Optional[str] = None
        self.lost = False
        self.acquired_by_me = False
        self.last_extend_time = time.monotonic() 

    async def acquire(self, timeout: float = 5.0) -> bool:  
        if not self.redis:
            logger.warning("Redis not available; cannot acquire lock")
            return False
        
        try:
            token = str(uuid.uuid4())
            ok = await asyncio.wait_for(
                self.redis.set(self.lock_key, token, nx=True, ex=self.expire),
                timeout=timeout,
            )
            
            if ok:
                self.token = token
                self.acquired_by_me = True
                self.lost = False
                self.last_extend_time = time.monotonic()
                
                logger.info(
                    f"🔐 Lock acquired: {self.lock_key.replace('lock:', '')} ({self.expire}s)"
                )
                return True

            logger.warning(f"Could not acquire Redis lock (held): {self.lock_key}")
            return False
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout acquiring lock {self.lock_key} after {timeout}s")
            return False
        except Exception as e:
            logger.error(f"Redis lock acquisition failed: {e}")
            return False

    async def extend(self, timeout: float = 3.0) -> bool:
        if not self.token or not self.redis or not self.acquired_by_me:
            self.lost = True
            return False
        try:
            result = await asyncio.wait_for(
                cast(Awaitable[Any], self.redis.eval(
                    self.EXTEND_LUA,
                    1,
                    self.lock_key,
                    self.token,
                    str(self.expire),
                )),
                timeout=timeout,
            )
            if result:
                self.last_extend_time = time.monotonic()
                if cfg.DEBUG_MODE:
                    logger.debug(f"Extended Redis lock: {self.lock_key} (now {self.expire}s)")
                return True
            else:
                logger.warning("Lock lost during extend (token mismatch or key missing)")
                self.lost = True
                self.acquired_by_me = False
                return False
                
        except asyncio.TimeoutError:
            logger.error(f"Timeout extending lock {self.lock_key} after {timeout}s")
            self.lost = True
            self.acquired_by_me = False
            return False
        except Exception as e:
            logger.error(f"Error extending Redis lock: {e}")
            self.lost = True
            self.acquired_by_me = False
            return False

    @classmethod
    def get_lock_extend_interval(cls) -> int:    
        extend_at = int(cfg.REDIS_LOCK_EXPIRY * 0.7)
        return max(60, min(extend_at, 540)) 

    def should_extend(self) -> bool:     
        if not self.acquired_by_me or self.lost:
            return False

        extend_threshold = self.__class__.get_lock_extend_interval()       
        elapsed = time.monotonic() - self.last_extend_time 
        should_extend = elapsed >= extend_threshold
        
        if cfg.DEBUG_MODE and should_extend:
            logger.debug(
                f"Lock extension eligible | "
                f"Elapsed: {elapsed:.0f}s | "
                f"Threshold: {extend_threshold}s"
            )
        
        return should_extend
   
    async def release(self, timeout: float = 3.0) -> None:
        if not self.token or not self.redis or not self.acquired_by_me:
            return
        try:
            result = await asyncio.wait_for(
                cast(Awaitable[Any], self.redis.eval(
                    self.RELEASE_LUA, 1, self.lock_key, self.token,
                )),
                timeout=timeout,
            )
            if result:
                logger.info(f"🔏 Lock released: {self.lock_key.replace('lock:', '')}")
                self.acquired_by_me = False
                self.token = None
            else:
                logger.warning(
                    f"Lock release failed (token mismatch): {self.lock_key} | "
                    f"Lock was stolen or lost"
                )
                self.lost = True
                self.acquired_by_me = False
    
        except asyncio.TimeoutError:
            logger.error(f"Timeout releasing lock {self.lock_key} after {timeout}s")
            self.lost = True
            self.acquired_by_me = False
        except Exception as e:
            logger.error(f"Error releasing Redis lock: {e}")
            self.lost = True
            self.acquired_by_me = False
    
        finally:
            self.token = None

    def __repr__(self) -> str:
        status = "HELD" if self.acquired_by_me else ("LOST" if self.lost else "RELEASED")
        token_display = self.token[:8] + "..." if self.token else "None"
        return f"RedisLock({self.lock_key}:{status}:{token_display})"

class TokenBucket:
    def __init__(self, rate: int, burst: int):
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self.lock:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.tokens = min(self.burst, self.tokens + elapsed * (self.rate / 60))
                self.last_update = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait_time = (1 - self.tokens) / (self.rate / 60)
            await asyncio.sleep(wait_time)

