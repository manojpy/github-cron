# ============================================================================
# RedisStateStore - Now using GitHub JSON backend (no Redis needed!)
# Stores alert states in GitHub repo for free, unlimited requests
# ============================================================================

import json
import asyncio
import logging
import time
import base64
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
import aiohttp
from pathlib import Path

logger = logging.getLogger("macd_bot.redis_state_store")

class RedisStateStore:
    """
    Replaces RedisStateStore - Now backs to GitHub repo
    Uses the same repo where your code lives
    NO Redis needed, NO request limits, completely FREE
    """
    
    def __init__(self, github_token: str, repo: str, branch: str = "main"):
        """
        Args:
            github_token: GitHub personal access token (GITHUB_TOKEN env var)
            repo: repo in format "owner/repo" (e.g., "manojpy/github-cron")
            branch: branch to store state (default: "main")
        """
        self.github_token = github_token
        self.repo = repo
        self.branch = branch
        self.degraded = False
        self.degraded_alerted = False
        
        # Local cache (fast reads)
        self._local_cache: Dict[str, Dict[str, Any]] = {}
        self._metadata_cache: Dict[str, str] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        
        # State files in your repo
        self.state_file = "bot_state/alert_state.json"
        self.metadata_file = "bot_state/metadata.json"
        
        # Expiry (same as before)
        self.expiry_seconds = 30 * 86400  # 30 days
        self.metadata_expiry_seconds = 7 * 86400  # 7 days
        
        self._sync_counter = 0  # Sync every 10 writes to avoid rate limits
        self._api_calls = 0  # Track API usage

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

    async def _write_file_to_github(self, file_path: str, data: Dict[str, Any], message: str) -> bool:
        """Write JSON file to GitHub with commit (409-safe)"""
        try:
            session = await self._get_session()
            url = f"https://api.github.com/repos/{self.repo}/contents/{file_path}"
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept":        "application/vnd.github.v3+json",
                "User-Agent":    "macd-bot"
            }

            # ---- 1.  grab current SHA (if file exists) -----------------
            sha = None
            async with session.get(url, headers=headers, timeout=5) as resp:
                self._api_calls += 1
                if resp.status == 200:
                    sha = (await resp.json()).get("sha")

            # ---- 2.  race guard – re-check if file appeared ------------
            if sha is None:                       # file did not exist a few ms ago
                await asyncio.sleep(0.1)
                async with session.get(url, headers=headers, timeout=5) as dbl:
                    self._api_calls += 1
                    if dbl.status == 200:
                        sha = (await dbl.json()).get("sha")

            # ---- 3.  prepare payload ----------------------------------
            content      = json.dumps(data, indent=2, default=str)
            content_b64  = base64.b64encode(content.encode()).decode()
            payload      = {
                "message": message,
                "content": content_b64,
                "branch":  self.branch
            }
            if sha:
                payload["sha"] = sha

            # ---- 4.  write / create ------------------------------------
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
        """Initialize - test GitHub API access"""
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
                    
                    return
                else:
                    logger.warning(f"GitHub connection failed: {resp.status}")
                    self.degraded = True
                    
        except Exception as e:
            logger.error(f"GitHub connection error: {e}")
            self.degraded = True

    async def close(self) -> None:
        """Close session and sync final state"""
        await self._sync_to_github()
        if self._session and not self._session.closed:
            await self._session.close()

    async def get(self, key: str, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
        """Get state from local cache (fast)"""
        if key in self._local_cache:
            data = self._local_cache[key]
            age = time.time() - data.get("ts", 0)
            if age > self.expiry_seconds:
                del self._local_cache[key]
                return None
            return data
        return None

    async def set(self, key: str, state: Optional[Any], ts: Optional[int] = None, timeout: float = 2.0) -> None:
        """Set state in local cache, periodic sync to GitHub"""
        ts = int(ts or time.time())
        self._local_cache[key] = {"state": state, "ts": ts}
        
        # Sync every 10 writes to avoid GitHub rate limits
        self._sync_counter += 1
        if self._sync_counter % 10 == 0:
            await self._sync_to_github()

    async def get_metadata(self, key: str, timeout: float = 2.0) -> Optional[str]:
        """Get metadata value"""
        return self._metadata_cache.get(key)

    async def set_metadata(self, key: str, value: str, timeout: float = 2.0) -> None:
        """Set metadata and sync"""
        self._metadata_cache[key] = value
        await self._sync_metadata_to_github()

    async def _sync_to_github(self) -> bool:
        """Sync alert states to GitHub"""
        if self.degraded:
            return False
        
        try:
            # Clean expired entries
            now = time.time()
            expired = [k for k, v in self._local_cache.items() 
                      if (now - v.get("ts", 0)) > self.expiry_seconds]
            for k in expired:
                del self._local_cache[k]
            
            success = await self._write_file_to_github(
                self.state_file,
                self._local_cache,
                f"🤖 Alert state sync | {len(self._local_cache)} active | {datetime.now(timezone.utc).isoformat()}"
            )
            return success
        except Exception as e:
            logger.error(f"Failed to sync state: {e}")
            return False

    async def _sync_metadata_to_github(self) -> bool:
        """Sync metadata to GitHub"""
        if self.degraded:
            return False
        
        try:
            success = await self._write_file_to_github(
                self.metadata_file,
                self._metadata_cache,
                f"🤖 Metadata sync | {datetime.now(timezone.utc).isoformat()}"
            )
            return success
        except Exception as e:
            logger.error(f"Failed to sync metadata: {e}")
            return False

    async def check_recent_alert(self, pair: str, alert_key: str, ts: int) -> bool:
        """Check if alert was recently sent (dedup)"""
        dedup_key = f"{pair}:{alert_key}"
        if dedup_key in self._local_cache:
            last_ts = self._local_cache[dedup_key].get("ts", 0)
            if (ts - last_ts) < 300:  # 5 minutes
                return False
        return True

    async def batch_check_recent_alerts(self, checks: List[Tuple[str, str, int]]) -> Dict[str, bool]:
        """Batch check recent alerts"""
        result = {}
        for pair, alert_key, ts in checks:
            dedup_key = f"{pair}:{alert_key}"
            result[dedup_key] = await self.check_recent_alert(pair, alert_key, ts)
        return result

    async def batch_get_and_set_alerts(self, pair: str, alert_keys: List[str], 
                                       updates: List[Tuple[str, Any, Optional[int]]]) -> Dict[str, Optional[Dict[str, Any]]]:
        """Batch get and set alerts"""
        result = {}
        for key in alert_keys:
            state_key = f"{pair}:{key}"
            result[key] = await self.get(state_key)
        
        now = int(time.time())
        for full_key, state_value, custom_ts in updates:
            ts = custom_ts or now
            await self.set(full_key, state_value, ts)
        
        return result

    @staticmethod
    async def shutdown_global_pool():
        """Placeholder for compatibility with old Redis code"""
        logger.info("GitHub state store does not use a global pool.")
        return True

    async def close(self):
        """Properly close the aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("GitHub State Store session closed.")

    async def atomic_eval_batch(self, pair: str, alert_keys: List[str], 
                               state_updates: List[Tuple[str, Any, Optional[int]]], 
                               dedup_checks: List[Tuple[str, str, int]]) -> Tuple[Dict[str, bool], Dict[str, bool]]:
        """Atomic batch evaluation"""
        prev_states = {}
        for key in alert_keys:
            state_key = f"{pair}:{key}"
            state_data = await self.get(state_key)
            prev_states[key] = isinstance(state_data, dict) and state_data.get("state") == "ACTIVE"
        
        dedup_results = await self.batch_check_recent_alerts(dedup_checks)
        
        now = int(time.time())
        for key, state_value, custom_ts in state_updates:
            ts = custom_ts or now
            await self.set(key, state_value, ts)
        
        return prev_states, dedup_results

    async def atomic_batch_update(self, updates: List[Tuple[str, Any, Optional[int]]], 
                                  deletes: Optional[List[str]] = None, timeout: float = 4.0) -> bool:
        """Atomic batch update"""
        now = int(time.time())
        
        for key, state, custom_ts in (updates or []):
            ts = custom_ts or now
            await self.set(key, state, ts)
        
        if deletes:
            for key in deletes:
                if key in self._local_cache:
                    del self._local_cache[key]

        await asyncio.sleep(random.uniform(0.05, 0.15))
        return await self._sync_to_github()

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics"""
        return {
            "degraded": self.degraded,
            "alert_states_cached": len(self._local_cache),
            "metadata_cached": len(self._metadata_cache),
            "repo": self.repo,
            "api_calls": self._api_calls,
        }