# ============================================================================
# RedisStateStore - GitHub JSON backend (lock-serialized, 409-free)
# ============================================================================

import json
import asyncio
import logging
import time
import base64
import random
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
import aiohttp
from pathlib import Path

logger = logging.getLogger("macd_bot.redis_state_store")


class GitHubStateStore:
    """
    Drop-in replacement for RedisStateStore using GitHub repo storage.
    All writes are serialized through an async lock to prevent 409 conflicts.
    """

    def __init__(self, github_token: str, repo: str, branch: str = "main"):
        self.github_token = github_token
        self.repo = repo
        self.branch = branch
        self.degraded = False
        self.degraded_alerted = False

        # Local cache (fast reads)
        self._local_cache: Dict[str, Dict[str, Any]] = {}
        self._metadata_cache: Dict[str, str] = {}
        self._session: Optional[aiohttp.ClientSession] = None

        # State files in repo
        self.state_file = "bot_state/alert_state.json"
        self.metadata_file = "bot_state/metadata.json"

        # Expiry
        self.expiry_seconds = 30 * 86400  # 30 days
        self.metadata_expiry_seconds = 7 * 86400  # 7 days

        # Write serialization lock (prevents 409s)
        self._write_lock = asyncio.Lock()
        self._pending_writes = 0  # Track queued writes

        self._api_calls = 0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _read_file_from_github(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Read JSON file from GitHub repo"""
        try:
            session = await self._get_session()
            url = f"https://api.github.com/repos/{self.repo}/contents/{file_path}?ref={self.branch}"
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3.raw",
                "User-Agent": "macd-bot"
            }

            async with session.get(url, headers=headers, timeout=10) as resp:
                self._api_calls += 1
                if resp.status == 200:
                    text = await resp.text()
                    return json.loads(text)
                elif resp.status == 404:
                    logger.debug(f"File not found: {file_path}")
                    return None
                else:
                    logger.warning(f"GitHub read error ({file_path}): {resp.status}")
                    return None

        except asyncio.TimeoutError:
            logger.warning(f"Timeout reading {file_path} from GitHub")
            return None
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
            return None

    async def _write_file_to_github_locked(self, file_path: str, data: Dict[str, Any], message: str) -> bool:
        """
        Internal write method - MUST be called while holding _write_lock.
        No 409 handling needed because lock guarantees single-writer.
        """
        try:
            session = await self._get_session()
            url = f"https://api.github.com/repos/{self.repo}/contents/{file_path}"
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "macd-bot"
            }

            # Get current SHA
            sha = None
            async with session.get(url, headers=headers, timeout=5) as resp:
                self._api_calls += 1
                if resp.status == 200:
                    sha = (await resp.json()).get("sha")

            # Encode and write
            content = json.dumps(data, indent=2, default=str)
            content_b64 = base64.b64encode(content.encode()).decode()
            payload = {
                "message": message,
                "content": content_b64,
                "branch": self.branch
            }
            if sha:
                payload["sha"] = sha

            async with session.put(url, json=payload, headers=headers, timeout=10) as resp:
                self._api_calls += 1
                if resp.status in (200, 201):
                    logger.debug(f"✅ Synced {file_path} to GitHub")
                    return True
                logger.warning(f"GitHub write error ({file_path}): {resp.status}")
                return False

        except asyncio.TimeoutError:
            logger.warning(f"Timeout writing {file_path} to GitHub")
            return False
        except Exception as e:
            logger.warning(f"Error writing {file_path}: {e}")
            return False

    async def connect(self, timeout: float = 5.0) -> None:
        """Initialize - test GitHub API access and load state"""
        try:
            session = await self._get_session()
            headers = {
                "Authorization": f"token {self.github_token}",
                "User-Agent": "macd-bot"
            }

            async with session.get(
                f"https://api.github.com/repos/{self.repo}",
                headers=headers,
                timeout=timeout
            ) as resp:
                self._api_calls += 1
                if resp.status == 200:
                    repo_data = await resp.json()
                    logger.info(f"✅ GitHub connected: {repo_data.get('full_name')}")
                    self.degraded = False

                    # Load existing state
                    state_data = await self._read_file_from_github(self.state_file)
                    if state_data:
                        self._local_cache = state_data
                        logger.info(f"Loaded {len(self._local_cache)} alert states from GitHub")

                    # Load metadata
                    meta_data = await self._read_file_from_github(self.metadata_file)
                    if meta_data:
                        self._metadata_cache = meta_data

                    return
                else:
                    logger.warning(f"GitHub connection failed: {resp.status}")
                    self.degraded = True

        except Exception as e:
            logger.error(f"GitHub connection error: {e}")
            self.degraded = True

    async def close(self) -> None:
        """Close session and sync final state"""
        # One final sync under lock
        async with self._write_lock:
            await self._sync_to_github_locked()
        if self._session and not self._session.closed:
            await self._session.close()

    async def get(self, key: str, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
        """Get state from local cache (fast, no lock needed)"""
        if key in self._local_cache:
            data = self._local_cache[key]
            age = time.time() - data.get("ts", 0)
            if age > self.expiry_seconds:
                del self._local_cache[key]
                return None
            return data
        return None

    async def set(self, key: str, state: Optional[Any], ts: Optional[int] = None, timeout: float = 2.0) -> None:
        """Set state in local cache, lazy sync"""
        ts = int(ts or time.time())
        self._local_cache[key] = {"state": state, "ts": ts}
        # Sync happens in batch operations, not here

    async def get_metadata(self, key: str, timeout: float = 2.0) -> Optional[str]:
        """Get metadata value"""
        return self._metadata_cache.get(key)

    async def set_metadata(self, key: str, value: str, timeout: float = 2.0) -> None:
        """Set metadata in cache, lazy sync"""
        self._metadata_cache[key] = value

    async def _sync_to_github_locked(self) -> bool:
        """
        Sync both state files to GitHub.
        MUST be called while holding _write_lock.
        """
        if self.degraded:
            return False

        # Clean expired entries
        now = time.time()
        expired = [k for k, v in self._local_cache.items()
                   if (now - v.get("ts", 0)) > self.expiry_seconds]
        for k in expired:
            del self._local_cache[k]

        # Write state file
        state_ok = await self._write_file_to_github_locked(
            self.state_file,
            self._local_cache,
            f"🤖 Alert state sync | {len(self._local_cache)} active | {datetime.now(timezone.utc).isoformat()}"
        )

        # Write metadata file
        meta_ok = await self._write_file_to_github_locked(
            self.metadata_file,
            self._metadata_cache,
            f"🤖 Metadata sync | {datetime.now(timezone.utc).isoformat()}"
        )

        return state_ok and meta_ok

    async def check_recent_alert(self, pair: str, alert_key: str, ts: int) -> bool:
        """Check if alert was recently sent (dedup)"""
        dedup_key = f"{pair}:{alert_key}"
        if dedup_key in self._local_cache:
            last_ts = self._local_cache[dedup_key].get("ts", 0)
            if (ts - last_ts) < 300:  # 5 minutes
                return False
        return True

    async def batch_check_recent_alerts(self, checks: List[Tuple[str, str, int]]) -> Dict[str, bool]:
        """Batch check recent alerts (pure cache, no lock)"""
        result = {}
        for pair, alert_key, ts in checks:
            dedup_key = f"{pair}:{alert_key}"
            result[dedup_key] = await self.check_recent_alert(pair, alert_key, ts)
        return result

    async def batch_get_and_set_alerts(self, pair: str, alert_keys: List[str],
                                       updates: List[Tuple[str, Any, Optional[int]]]) -> Dict[str, Optional[Dict[str, Any]]]:
        """Batch get and set - writes are queued, sync happens separately"""
        # Get current states (no lock needed)
        result = {}
        for key in alert_keys:
            state_key = f"{pair}:{key}"
            result[key] = await self.get(state_key)

        # Apply updates to cache only
        now = int(time.time())
        for full_key, state_value, custom_ts in updates:
            ts = custom_ts or now
            await self.set(full_key, state_value, ts)

        return result

    async def atomic_batch_update(self, updates: List[Tuple[str, Any, Optional[int]]],
                                  deletes: Optional[List[str]] = None, timeout: float = 4.0) -> bool:
        """
        Atomic batch update with serialized GitHub write.
        All workers queue here; only one writes at a time.
        """
        if self.degraded:
            return False

        # Apply changes to local cache first (fast, no lock)
        now = int(time.time())
        for key, state, custom_ts in (updates or []):
            ts = custom_ts or now
            self._local_cache[key] = {"state": state, "ts": ts}

        if deletes:
            for key in deletes:
                if key in self._local_cache:
                    del self._local_cache[key]

        # Serialize the actual GitHub write
        async with self._write_lock:
            self._pending_writes += 1
            try:
                # Small jitter to prevent thundering herd on first acquisition
                if self._pending_writes > 1:
                    await asyncio.sleep(random.uniform(0.01, 0.05))

                return await self._sync_to_github_locked()
            finally:
                self._pending_writes -= 1

    @staticmethod
    async def shutdown_global_pool():
        """Placeholder for compatibility with old Redis code"""
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics"""
        return {
            "degraded": self.degraded,
            "alert_states_cached": len(self._local_cache),
            "metadata_cached": len(self._metadata_cache),
            "repo": self.repo,
            "api_calls": self._api_calls,
            "pending_writes": self._pending_writes,
        }


# Backwards-compatible alias so old imports keep working
RedisStateStore = GitHubStateStore
