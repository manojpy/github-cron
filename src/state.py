from __future__ import annotations
import time
import asyncio
import logging
import uuid
from typing import Dict, Any, Optional, Tuple, List, ClassVar, Callable, TYPE_CHECKING, Set
import numpy as np
import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError, RedisError

from bot_config import cfg, logger, json_dumps, json_loads, JSONDecodeError, CONFIG_OVERRIDE_ALLOWED_FIELDS, CONFIG_OVERRIDE_METADATA_KEY, BRAIN_DISABLED_KEYS_METADATA_KEY, PAIR_THRESHOLDS_METADATA_KEY, _get_session_from_ts
from fetcher import compute_backoff

if TYPE_CHECKING:
    from fetcher import PriceData

async def _blanket_reset_pair(sdb: RedisStateStore, pair_name: str, logger_pair: logging.Logger) -> int:
    from alerts import ALERT_KEYS
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
) -> Tuple[int, int, int, int, int, int, int, int]:
    if sdb.degraded or not sdb._redis:
        logger.warning("Redis degraded — skipping mass state purge")
        return 0, 0, 0, 0, 0, 0, 0, 0

    deleted_states = 0
    deleted_dedups = 0
    deleted_pending = 0
    deleted_shadow_pending = 0
    deleted_alert_stats = 0
    deleted_shadow_stats = 0
    deleted_shadow_hiconf = 0
    deleted_streams = 0

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
                total_deleted += await sdb._redis.unlink(*batch)
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

        logger.info(
            f"🧹 MASS RESET complete | "
            f"States: {deleted_states} | Dedups: {deleted_dedups} | "
            f"Pending: {deleted_pending} | ShadowPending: {deleted_shadow_pending} | "
            f"AlertStats: {deleted_alert_stats} | ShadowStats: {deleted_shadow_stats} | "
            f"ShadowHiConf: {deleted_shadow_hiconf} | Streams: {deleted_streams}"
        )
        return (deleted_states, deleted_dedups, deleted_pending, deleted_shadow_pending,
                deleted_alert_stats, deleted_shadow_stats, deleted_shadow_hiconf, deleted_streams)

    except Exception as e:
        logger.error(f"Mass reset failed: {e}")
        return 0, 0, 0, 0, 0, 0, 0, 0

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

        self.degraded = False
        self.degraded_alerted = False
        self._connection_attempts = 0

        if cfg.DEBUG_MODE and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"RedisStateStore initialized | "
                f"State TTL: {cfg.STATE_EXPIRY_DAYS}d | "
                f"Alert TTL: {cfg.STATE_EXPIRY_DAYS}d | "
                f"Metadata TTL: 7d"
            )

    async def _record_redis_failure(self, operation: str, exc: Exception) -> None:
        logger.error(f"Redis operation '{operation}' failed: {exc}")
        if self.degraded:
            return
        self.degraded = True
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
                        await self._redis.aclose()
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
                max_conn = getattr(self._redis.connection_pool, "max_connections", "?")
                logger.info(f"✅ Redis connected ({max_conn} max)")
                self.degraded = False
                self.degraded_alerted = False
                return

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
    - Alert deduplication:  DISABLED (may get duplicates)
    - State persistence:    DISABLED (alerts reset each run)
    - Trading alerts:       STILL ACTIVE (core functionality preserved)
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
        result = await self._safe_redis_op(lambda: self._redis.ping(), timeout, "ping")
        return bool(result)

    async def _safe_redis_op(self, fn: Callable[[], Any], timeout: float, op_name: str, parser: Optional[Callable[[Any], Any]] = None):
        if not self._redis:
            return None
        try:
            coro = fn()
            result = await asyncio.wait_for(coro, timeout=timeout)
            return parser(result) if parser else result
        except (asyncio.TimeoutError, RedisConnectionError, RedisError) as e:
            logger.error(f"Redis {op_name} failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to {op_name}: {e}")
            return None

    async def get(self, key: str, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
        return await self._safe_redis_op(
            lambda: self._redis.get(f"{self.state_prefix}{key}"),
            timeout,
            f"get {key}",
            parser=lambda r: json_loads(r) if r else None,
        )

    async def set(self, key: str, state: Optional[Any], ts: Optional[int] = None, timeout: float = 2.0) -> None:
        ts = int(ts or time.time())
        redis_key = f"{self.state_prefix}{key}"
        data = json_dumps({"state": state, "ts": ts})
        await self._safe_redis_op(
            lambda: self._redis.set(
                redis_key,
                data,
                ex=self.expiry_seconds if self.expiry_seconds > 0 else None,
            ),
            timeout,
            f"set {key}",
        )

    async def get_metadata(self, key: str, timeout: float = 2.0) -> Optional[str]:
        return await self._safe_redis_op(
            lambda: self._redis.get(f"{self.meta_prefix}{key}"),
            timeout,
            f"get_metadata {key}",
            parser=lambda r: r if r else None,
        )
    async def set_metadata(self, key: str, value: str, timeout: float = 2.0,
                             ttl: Optional[int] = None) -> None:
        await self._safe_redis_op(
            lambda: self._redis.set(
                f"{self.meta_prefix}{key}",
                value,
                ex=ttl if ttl is not None else self.metadata_expiry_seconds
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
        raw = await self.get_metadata(CONFIG_OVERRIDE_METADATA_KEY)
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
                setattr(cfg, field, coerced)
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

    async def get_disabled_alert_keys(self) -> Set[str]:
        raw = await self.get_metadata(BRAIN_DISABLED_KEYS_METADATA_KEY)
        if not raw:
            return set()
        try:
            keys = json_loads(raw)
        except (JSONDecodeError, TypeError, ValueError) as e:
            logger.warning(f"Ignoring malformed {BRAIN_DISABLED_KEYS_METADATA_KEY} in Redis: {e}")
            return set()
        return set(keys) if isinstance(keys, list) else set()

    async def set_alert_key_disabled(self, alert_key: str, disabled: bool) -> bool:
        current = await self.get_disabled_alert_keys()
        if disabled:
            current.add(alert_key)
        else:
            current.discard(alert_key)
        try:
            await self.set_metadata(BRAIN_DISABLED_KEYS_METADATA_KEY, json_dumps(sorted(current)))
        except Exception as e:
            logger.warning(f"Failed to update disabled-key set for '{alert_key}': {e}")
            return False
        return True

    async def get_pair_thresholds(self) -> Dict[str, float]:
        """All pair -> confluence-abs-score-floor overrides currently stored,
        as learned/written by the brain. Missing or malformed data returns {}."""
        raw = await self.get_metadata(PAIR_THRESHOLDS_METADATA_KEY)
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

    async def set_dynamic_weights(self, weights: Dict[str, float], ttl: int = 30 * 86400) -> bool:
        """Persist the Brain's optimized CONFLUENCE_WEIGHTS to Redis.
        TTL default 30 days — refresh on each Brain report."""
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

    async def batch_get_metadata(self, keys: List[str], timeout: float = 5.0) -> Dict[str, Optional[str]]:
        """Fetch many metadata keys in ONE Redis round-trip (pipeline)."""
        if not self._redis or self.degraded or not keys:
            return {k: None for k in keys}
        try:
            async with self._redis.pipeline() as pipe:
                for k in keys:
                    pipe.get(f"{self.meta_prefix}{k}")
                values = await asyncio.wait_for(pipe.execute(), timeout=timeout)
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
                await asyncio.wait_for(pipe.execute(), timeout=timeout)
            return True
        except Exception as e:
            logger.error(f"batch_set_metadata failed: {e}")
            return False

    async def check_recent_alert(self, pair: str, alert_key: str, ts: int, window_sec: Optional[int] = None) -> bool:
        if self.degraded:
            return True
        if not self._redis:
            logger.critical(
                f"check_recent_alert: degraded=False but _redis is None (state desync) — "
                f"failing closed for {pair}:{alert_key}, this alert will be blocked"
            )
            return False  # fail-closed, consistent with the except-branch policy below
        recent_key = f"{RedisKeyPrefix.RECENT_ALERT}{pair}:{alert_key}"
        effective_window = window_sec if window_sec is not None else cfg.ALERT_DEDUP_WINDOW_SEC
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
            logger.error(f"Dedup check FAILED for {pair}:{alert_key}: {e}")
            return False   # fail-closed, not fail-open

    async def release_recent_alert(self, pair: str, alert_key: str) -> None:
        """Undo a dedup claim if the message didn't actually get delivered."""
        if self.degraded:
            return
        recent_key = f"{RedisKeyPrefix.RECENT_ALERT}{pair}:{alert_key}"
        try:
            await asyncio.wait_for(self._redis.delete(recent_key), timeout=1.0)
        except Exception as e:
            logger.warning(f"Failed to release dedup claim for {pair}:{alert_key}: {e}")

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
    ) -> None:

        if self.degraded or not cfg.ENABLE_WIN_RATE_FILTER:
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
            })
        except Exception as e:
            logger.warning(
                f"Failed to serialize pending outcome for {pair}:{alert_key}: {e}"
            )
            return

        ttl = (cfg.OUTCOME_LOOKAHEAD_CANDLES + 4) * 15 * 60  # lookahead + buffer, in seconds

        try:
            await asyncio.wait_for(
                self._redis.set(key, payload, ex=ttl),
                timeout=2.0,
            )
        except Exception as e:
            logger.warning(
                f"Failed to record pending outcome for {pair}:{alert_key}: {e}"
            )

        if confluence_votes is not None:
            await self.record_vote_count(alert_key, confluence_votes)

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

        ttl = (cfg.OUTCOME_LOOKAHEAD_CANDLES + 4) * 15 * 60

        try:
            await asyncio.wait_for(
                self._redis.set(key, payload, ex=ttl),
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

                await self._safe_redis_op(
                    lambda: pipe.execute(),
                    2.0,
                    f"vote_count_save:{alert_key}",
                )
        except Exception:
            pass

    async def get_vote_count_history(self, alert_key: str) -> List[int]:
        if self.degraded or not self._redis:
            return []

        key = f"{RedisKeyPrefix.VOTE_COUNT_HISTORY}{alert_key}"

        try:
            raw_list = await self._safe_redis_op(
                lambda: self._redis.lrange(
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
            return [k async for k in self._redis.scan_iter(match=pattern, count=100)]
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

        target_idx = entry_idx + cfg.OUTCOME_LOOKAHEAD_CANDLES
        if target_idx > i15:
            return None, "not_ready"

        future_price = float(data_15m.close[target_idx])
        pct_move = (future_price - entry_price) / entry_price * 100.0
        win = (
            pct_move >= cfg.OUTCOME_FAVORABLE_MOVE_PCT
            if is_buy
            else pct_move <= -cfg.OUTCOME_FAVORABLE_MOVE_PCT
        )

        # ── MAE / MFE (path-aware, not just the binary close-vs-target outcome) ──
        path_low = data_15m.low[entry_idx:target_idx]
        path_high = data_15m.high[entry_idx:target_idx]
        if len(path_low) and len(path_high):
            if is_buy:
                mae = max(0.0, (entry_price - float(np.min(path_low))) / entry_price)
                mfe = max(0.0, (float(np.max(path_high)) - entry_price) / entry_price)
            else:
                mae = max(0.0, (float(np.max(path_high)) - entry_price) / entry_price)
                mfe = max(0.0, (entry_price - float(np.min(path_low))) / entry_price)
        else:
            mae = mfe = None

        return {
            "alert_key": key.split(":")[-2],
            "direction": direction,
            "entry_ts": entry_ts,
            "is_buy": is_buy,
            "pct_move": pct_move,
            "win": win,
            "mae": mae,
            "mfe": mfe,
            "conf_score": conf_score,
            "conf_total": conf_total,
            "conf_votes": conf_votes,
            "adx_val": adx_val,
            "context": data.get("context"),
        }, ""

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
                raw_values = await asyncio.wait_for(read_pipe.execute(), timeout=2.0)
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
                        stats_key = f"{RedisKeyPrefix.ALERT_STATS}{pair}:{alert_key}"
                        write_pipe.hincrby(stats_key, "wins" if win else "losses", 1)
                        write_pipe.expire(stats_key, stats_ttl)
                        session = _get_session_from_ts(entry_ts) if entry_ts else "dead"
                        session_stats_key = f"{stats_key}:{session}"
                        write_pipe.hincrby(session_stats_key, "wins" if win else "losses", 1)
                        write_pipe.expire(session_stats_key, stats_ttl)
                        stream_fields = None
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
                                "session": session,
                                "votes": conf_votes,
                                "adx_val": adx_val,
                                "context": row_context,
                            })
                    except Exception as e:
                        logger_pair.debug(f"Failed to resolve pending outcome {key}: {e}")
                        bad_payload_count += 1
                        continue

                if pending_writes:
                    await asyncio.wait_for(write_pipe.execute(), timeout=2.0)

        except Exception as e:
            logger_pair.debug(f"Failed to persist resolved outcomes for {pair}: {e}")
            return

        if resolved_for_file and getattr(cfg, "BRAIN_USE_FILE_STORAGE", False):
            try:
                from outcome_storage import append_outcome_batch
                append_outcome_batch(resolved_for_file, shadow=False)
            except Exception as e:
                logger_pair.warning(f"[{pair}] File archive write failed: {e}")

        logger_pair.debug(
            f"[{pair}] Outcome resolution | "
            f"pending={len(keys)} | "
            f"resolved={resolved_count} | "
            f"not_ready={not_ready_count} | "
            f"ts_mismatch={ts_mismatch_count} | "
            f"missing_score={missing_score_count} | "
            f"bad_payload={bad_payload_count}"
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
            "shadow pending",
        )
        if not keys:
            return

        try:
            async with self._redis.pipeline() as read_pipe:
                for key in keys:
                    read_pipe.get(key)
                raw_values = await asyncio.wait_for(read_pipe.execute(), timeout=2.0)
        except Exception as e:
            logger_pair.warning(
                f"Failed to batch-fetch shadow pending outcomes for {pair}: {e}"
            )
            return

        resolved_count = 0
        hiconf_pct = getattr(cfg, "BRAIN_REWARDABLE_MIN_CONFLUENCE_PCT", 80.0)
        stats_ttl = max(cfg.STATE_EXPIRY_DAYS * 86400, 7 * 86400)
        resolved_for_file: List[Dict[str, Any]] = []
        
        try:
            async with self._redis.pipeline() as write_pipe:
                pending_writes = 0

                for key, raw in zip(keys, raw_values):
                    try:
                        result, skip_reason = self._parse_pending_outcome_row(
                            key, raw, data_15m, i15
                        )
                        if skip_reason:
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
                                write_pipe.xadd(
                                    RedisKeyPrefix.SHADOW_LOG_STREAM,
                                    {
                                        "pair": str(pair),
                                        "alert_key": str(alert_key),
                                        "direction": str(direction),
                                        "score": str(conf_score),
                                        "total": str(conf_total),
                                        "pct_move": f"{pct_move:.4f}",
                                        "win": "1" if win else "0",
                                        "entry_ts": str(entry_ts),
                                        "session": _get_session_from_ts(entry_ts)
                                        if entry_ts
                                        else "dead",
                                        "mae": f"{mae:.5f}" if mae is not None else "",
                                        "mfe": f"{mfe:.5f}" if mfe is not None else "",
                                        "votes": json_dumps(conf_votes)
                                        if conf_votes is not None
                                        else "",
                                    },
                                    maxlen=2000,
                                    approximate=True,
                                )
                            if conf_pct >= hiconf_pct:
                                hiconf_key = (
                                    f"{RedisKeyPrefix.SHADOW_HICONF_STATS}{alert_key}"
                                )
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
                            })
                
                    except Exception as e:
                        logger_pair.debug(
                            f"Failed to resolve shadow pending outcome {key}: {e}"
                        )
                        continue

                if pending_writes:
                    await asyncio.wait_for(write_pipe.execute(), timeout=2.0)

        except Exception as e:
            logger_pair.debug(
                f"Failed to persist resolved shadow outcomes for {pair}: {e}"
            )
            return
    
        if resolved_for_file and getattr(cfg, "BRAIN_USE_FILE_STORAGE", False):
            try:
                from outcome_storage import append_outcome_batch
                append_outcome_batch(resolved_for_file, shadow=True)
            except Exception as e:
                logger_pair.warning(f"[{pair}] Shadow file archive write failed: {e}")

        if resolved_count:
            logger_pair.debug(
                f"[{pair}] Shadow outcome resolution | resolved={resolved_count}"
            )

    async def get_alert_win_rate(self, pair: str, alert_key: str) -> Tuple[Optional[float], int]:
        """Returns (win_rate, sample_size). win_rate is None until MIN_WIN_RATE_SAMPLE is reached."""
        if self.degraded or not cfg.ENABLE_WIN_RATE_FILTER:
            return None, 0
        stats_key = f"{RedisKeyPrefix.ALERT_STATS}{pair}:{alert_key}"
        try:
            data = await asyncio.wait_for(self._redis.hgetall(stats_key), timeout=2.0)
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
            data = await asyncio.wait_for(self._redis.hgetall(stats_key), timeout=2.0)
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
            async with self._redis.pipeline() as pipe:
                for ak in alert_keys:
                    pipe.hgetall(f"{RedisKeyPrefix.ALERT_STATS}{pair}:{ak}")
                raw_results = await asyncio.wait_for(pipe.execute(), timeout=timeout)
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
                self._redis.hgetall(hash_key),
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

    async def atomic_batch_update(self, updates: List[Tuple[str, Any, Optional[int]]], deletes: Optional[List[str]] = None, timeout: float = 4.0) -> bool:
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
            await self._record_redis_failure("atomic_batch_update", e)
            return False
        except Exception as e:
            await self._record_redis_failure("atomic_batch_update", e)
            return False

    # ── CUSUM state persistence ──────────────────────────────────────────
    async def load_cusum_state(self, alert_key: str) -> Optional[Dict[str, Any]]:
        """Load persisted CUSUM accumulator for one alert_key."""
        if self.degraded or not self._redis:
            return None
        key = f"{RedisKeyPrefix.CUSUM_STATE}{alert_key}"
        raw = await self._safe_redis_op(
            lambda: self._redis.get(key), 2.0, f"cusum_load:{alert_key}",
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
                lambda: self._redis.set(key, json_dumps(state), ex=30 * 86400),
                2.0, f"cusum_save:{alert_key}",
            )
        except Exception:
            pass

    async def load_cusum_watermark(self, alert_key: str) -> int:
        """Last entry_ts already fed into this alert_key's CUSUM detector.
        0 means nothing has been fed yet."""
        if self.degraded or not self._redis:
            return 0
        key = f"{RedisKeyPrefix.CUSUM_WATERMARK}{alert_key}"
        try:
            raw = await self._safe_redis_op(
                lambda: self._redis.get(key), 2.0, f"cusum_watermark_load:{alert_key}",
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
                lambda: self._redis.set(key, str(entry_ts), ex=30 * 86400),
                2.0, f"cusum_watermark_save:{alert_key}",
            )
        except Exception:
            pass

    async def load_threshold_history(self, key_suffix: str = "") -> List[float]:
        """key_suffix="" (default) is the existing global CONFLUENCE_MIN_ABS_SCORE
        history, unchanged. Pass a pair name to track that pair's own history
        under a separate list (used by per-pair threshold stability checks)."""
        if self.degraded or not self._redis:
            return []
        key = f"{RedisKeyPrefix.THRESHOLD_HISTORY}{key_suffix}"
        try:
            raw_list = await self._safe_redis_op(
                lambda: self._redis.lrange(key, 0, 9),
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
                await self._safe_redis_op(
                    lambda: pipe.execute(), 2.0, f"threshold_history_save:{key_suffix or 'global'}",
                )
        except Exception:
            pass       

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
                self.redis.eval(
                    self.EXTEND_LUA,
                    1,
                    self.lock_key,
                    self.token,
                    self.expire,
                ),
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
                self.redis.eval(self.RELEASE_LUA, 1, self.lock_key, self.token),
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

