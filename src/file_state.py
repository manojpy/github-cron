#!/usr/bin/env python3
"""file_state.py — Redis-shaped adapter persisted to JSON/JSONL files.

Substituted for the real redis-py client when STATE_BACKEND=file. Every
method mirrors redis-py's async interface, so RedisStateStore's higher-
level methods run unchanged.

Design assumptions:
  • Single runner. GitHub Actions concurrency groups serialize runs.
  • Crash-during-run loses that run's writes. Matches Redis-flush failure.
  • State fits in memory. Empirically <2 MB for a 30-pair universe.
  • Writes flush on close(). run_once() calls close() in its finally.
"""
from __future__ import annotations
import asyncio
import fnmatch
import json
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple


class _FilePipeline:
    def __init__(self, adapter: "_FileRedisAdapter") -> None:
        self._a = adapter
        self._ops: List[Tuple[str, tuple, dict]] = []

    def get(self, key: str) -> "_FilePipeline":
        self._ops.append(("get", (key,), {})); return self
    def set(self, key: str, value: str, **kw) -> "_FilePipeline":
        self._ops.append(("set", (key, value), kw)); return self
    def delete(self, *keys: str) -> "_FilePipeline":
        self._ops.append(("delete", keys, {})); return self
    def expire(self, key: str, seconds: int) -> "_FilePipeline":
        self._ops.append(("expire", (key, seconds), {})); return self
    def hset(self, key: str, mapping: Optional[Dict[str, str]] = None, **kw) -> "_FilePipeline":
        self._ops.append(("hset", (key,), {"mapping": mapping, **kw})); return self
    def hgetall(self, key: str) -> "_FilePipeline":
        self._ops.append(("hgetall", (key,), {})); return self
    def hdel(self, key: str, *fields: str) -> "_FilePipeline":
        self._ops.append(("hdel", (key, *fields), {})); return self
    def hincrby(self, key: str, field: str, amount: int = 1) -> "_FilePipeline":
        self._ops.append(("hincrby", (key, field, amount), {})); return self
    def lpush(self, key: str, *values: str) -> "_FilePipeline":
        self._ops.append(("lpush", (key, *values), {})); return self
    def ltrim(self, key: str, start: int, end: int) -> "_FilePipeline":
        self._ops.append(("ltrim", (key, start, end), {})); return self
    def xadd(self, key: str, fields: Dict[str, Any], **kw) -> "_FilePipeline":
        self._ops.append(("xadd", (key, fields), kw)); return self

    async def execute(self) -> List[Any]:
        results = []
        for name, args, kw in self._ops:
            results.append(await getattr(self._a, name)(*args, **kw))
        return results

    async def __aenter__(self) -> "_FilePipeline":
        return self
    async def __aexit__(self, *exc: Any) -> None:
        pass


class _FileRedisAdapter:
    """Redis-py-shaped client backed by JSON + JSONL files.

    Storage layout under <data_dir>/state/:
        kv.json       {key: value}
        hashes.json   {key: {field: value}}
        lists.json    {key: [values]}
        ttls.json     {key: expiry_unix_ts}
        streams/<name>.jsonl
    """

    def __init__(self, data_dir: str) -> None:
        self._dir = os.path.join(data_dir, "state")
        self._streams_dir = os.path.join(self._dir, "streams")
        self._kv: Dict[str, str] = {}
        self._hashes: Dict[str, Dict[str, str]] = {}
        self._lists: Dict[str, List[str]] = {}
        self._ttls: Dict[str, float] = {}
        self._streams: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        self._stream_maxlen: Dict[str, int] = {}
        self._lock = asyncio.Lock()

        class _Pool:
            max_connections = 32
        self.connection_pool = _Pool()

    async def connect(self) -> None:
        os.makedirs(self._dir, exist_ok=True)
        os.makedirs(self._streams_dir, exist_ok=True)
        for name, target in (
            ("kv", self._kv), ("hashes", self._hashes),
            ("lists", self._lists), ("ttls", self._ttls),
        ):
            path = os.path.join(self._dir, f"{name}.json")
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        target.update(data)
                except Exception:
                    pass
        for fname in os.listdir(self._streams_dir):
            if not fname.endswith(".jsonl"):
                continue
            key = fname[:-6]
            entries: List[Tuple[str, Dict[str, Any]]] = []
            path = os.path.join(self._streams_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            eid, fields = json.loads(line)
                            entries.append((eid, fields))
                        except Exception:
                            continue
            except Exception:
                continue
            self._streams[key] = entries

    async def close(self) -> None:
        os.makedirs(self._dir, exist_ok=True)
        for name, data in (
            ("kv", self._kv), ("hashes", self._hashes),
            ("lists", self._lists), ("ttls", self._ttls),
        ):
            path = os.path.join(self._dir, f"{name}.json")
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"), sort_keys=True)
            os.replace(tmp, path)
        for key, entries in self._streams.items():
            maxlen = self._stream_maxlen.get(key)
            if maxlen:
                entries = entries[-maxlen:]
            path = os.path.join(self._streams_dir, f"{key}.jsonl")
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for eid, fields in entries:
                    f.write(json.dumps([eid, fields], separators=(",", ":")) + "\n")
            os.replace(tmp, path)

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        await self.close()

    def _expired(self, key: str) -> bool:
        exp = self._ttls.get(key)
        if exp is None:
            return False
        if time.time() <= exp:
            return False
        self._kv.pop(key, None)
        self._hashes.pop(key, None)
        self._lists.pop(key, None)
        self._ttls.pop(key, None)
        return True

    async def get(self, key: str) -> Optional[str]:
        if self._expired(key):
            return None
        return self._kv.get(key)

    async def set(
        self, key: str, value: str,
        ex: Optional[int] = None, nx: bool = False,
    ) -> Optional[bool]:
        self._expired(key)
        if nx and key in self._kv:
            return None
        self._kv[key] = str(value)
        if ex:
            self._ttls[key] = time.time() + ex
        else:
            self._ttls.pop(key, None)
        return True

    async def delete(self, *keys: str) -> int:
        n = 0
        for k in keys:
            hit = False
            if k in self._kv:
                del self._kv[k]; hit = True
            if k in self._hashes:
                del self._hashes[k]; hit = True
            if k in self._lists:
                del self._lists[k]; hit = True
            if k in self._streams:
                del self._streams[k]; hit = True
            self._ttls.pop(k, None)
            if hit:
                n += 1
        return n

    async def unlink(self, *keys: str) -> int:
        """Alias for delete — _clear_all_redis_states and CLEAR_REDIS call
        unlink(), which in real Redis frees memory in a background thread.
        The file adapter just calls delete()."""
        return await self.delete(*keys)

    async def exists(self, *keys: str) -> int:
        n = 0
        for k in keys:
            if self._expired(k):
                continue
            if k in self._kv or k in self._hashes or k in self._lists:
                n += 1
        return n

    async def incr(self, key: str) -> int:
        self._expired(key)
        cur = int(self._kv.get(key, "0") or "0")
        new = cur + 1
        self._kv[key] = str(new)
        return new

    async def decrby(self, key: str, amount: int) -> int:
        self._expired(key)
        cur = int(self._kv.get(key, "0") or "0")
        new = cur - amount
        self._kv[key] = str(new)
        return new

    async def expire(self, key: str, seconds: int) -> bool:
        self._ttls[key] = time.time() + seconds
        return True

    async def hgetall(self, key: str) -> Dict[str, str]:
        if self._expired(key):
            return {}
        return dict(self._hashes.get(key, {}))

    async def hset(
        self, key: str,
        mapping: Optional[Dict[str, str]] = None,
        **kwargs: str,
    ) -> int:
        self._expired(key)
        h = self._hashes.setdefault(key, {})
        added = 0
        if mapping:
            for k, v in mapping.items():
                if k not in h:
                    added += 1
                h[k] = str(v)
        for k, v in kwargs.items():
            if k not in h:
                added += 1
            h[k] = str(v)
        return added

    async def hdel(self, key: str, *fields: str) -> int:
        h = self._hashes.get(key)
        if not h:
            return 0
        n = 0
        for f in fields:
            if f in h:
                del h[f]; n += 1
        return n

    async def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        self._expired(key)
        h = self._hashes.setdefault(key, {})
        cur = int(h.get(field, "0") or "0")
        new = cur + amount
        h[field] = str(new)
        return new

    async def lrange(self, key: str, start: int, end: int) -> List[str]:
        if self._expired(key):
            return []
        lst = self._lists.get(key, [])
        return lst[start:] if end == -1 else lst[start:end + 1]

    async def lpush(self, key: str, *values: str) -> int:
        self._expired(key)
        lst = self._lists.setdefault(key, [])
        for v in values:
            lst.insert(0, str(v))
        return len(lst)

    async def ltrim(self, key: str, start: int, end: int) -> bool:
        if self._expired(key):
            return True
        lst = self._lists.get(key, [])
        self._lists[key] = lst[start:] if end == -1 else lst[start:end + 1]
        return True

    async def llen(self, key: str) -> int:
        if self._expired(key):
            return 0
        return len(self._lists.get(key, []))

    async def xadd(
        self, key: str, fields: Dict[str, Any],
        maxlen: Optional[int] = None, approximate: bool = True,
    ) -> str:
        stream = self._streams.setdefault(key, [])
        if maxlen:
            self._stream_maxlen[key] = int(maxlen)
        eid = f"{int(time.time() * 1000)}-{len(stream)}"
        clean: Dict[str, Any] = {}
        for k, v in fields.items():
            clean[str(k)] = v if isinstance(v, (int, float)) else str(v)
        stream.append((eid, clean))
        return eid

    async def xrevrange(
        self, key: str, count: Optional[int] = None,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        stream = list(self._streams.get(key, []))
        stream.reverse()
        if count:
            stream = stream[:count]
        return stream

    async def scan_iter(
        self, match: str = "*", count: int = 100,
    ) -> AsyncIterator[str]:
        seen = set(self._kv) | set(self._hashes) | set(self._lists) | set(self._streams)
        for k in sorted(seen):
            if fnmatch.fnmatchcase(k, match):
                yield k

    def pipeline(self) -> _FilePipeline:
        return _FilePipeline(self)

    async def eval(
        self, script: str, numkeys: int, *keys_and_args: str,
    ) -> Optional[int]:
        if numkeys != 1 or len(keys_and_args) < 2:
            return None
        key, token = keys_and_args[0], keys_and_args[1]
        current = self._kv.get(key)
        if "DEL" in script and "EXPIRE" not in script:
            if current == token:
                del self._kv[key]
                self._ttls.pop(key, None)
                return 1
            return 0
        if "EXPIRE" in script:
            if current == token:
                self._ttls[key] = time.time() + int(keys_and_args[2])
                return 1
            return 0
        return None

    async def dbsize(self) -> int:
        return len(self._kv) + len(self._hashes) + len(self._lists)