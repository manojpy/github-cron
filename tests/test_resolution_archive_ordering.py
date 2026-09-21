"""Resolved outcomes must reach the file archive BEFORE the Redis pending key
is deleted. If the archive write fails, the pending key must survive."""
import asyncio
import logging


import outcome_storage
from bot_config import cfg
from state import RedisStateStore


class FakePipe:
    def __init__(self, store, log):
        self.store, self.log, self.ops = store, log, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def get(self, key):
        self.ops.append(("get", key))

    def __getattr__(self, name):          # hincrby / expire / xadd / delete ...
        def _rec(*args, **kw):
            self.ops.append((name, args))
        return _rec

    async def execute(self):
        self.log.append("redis_execute")
        if any(op[0] == "get" for op in self.ops):
            return ["raw"] * sum(1 for op in self.ops if op[0] == "get")
        if any(op[0] == "delete" for op in self.ops):
            self.store["deleted"] = True
        return []


class FakeRedis:
    def __init__(self, log):
        self.log, self.store = log, {"deleted": False}

    def pipeline(self):
        return FakePipe(self.store, self.log)


def _make_store(monkeypatch, log):
    sdb = object.__new__(RedisStateStore)
    sdb.degraded = False
    sdb._redis = FakeRedis(log)
    sdb._run_resolved_total = 0
    sdb._run_archived_total = 0

    async def fake_keys(*a, **k):
        return ["pending:ETHUSD:ppo_cross_up:1700000000"]

    result = {
        "alert_key": "ppo_cross_up", "direction": "buy", "entry_ts": 1_700_000_000,
        "pct_move": 1.5, "win": True, "mae": 0.1, "mfe": 2.0,
        "conf_score": 5.0, "conf_total": 10.0, "conf_votes": {}, "adx_val": 20.0,
    }
    monkeypatch.setattr(sdb, "_fetch_pending_keys", fake_keys)
    monkeypatch.setattr(sdb, "_parse_pending_outcome_row", lambda *a, **k: (result, ""))
    monkeypatch.setattr(cfg, "ENABLE_WIN_RATE_FILTER", True, raising=False)
    monkeypatch.setattr(cfg, "BRAIN_USE_FILE_STORAGE", True, raising=False)
    return sdb


def _resolve(sdb):
    asyncio.run(sdb.resolve_pending_outcomes("ETHUSD", None, 0, logging.getLogger("t")))


def test_archive_written_before_redis_delete(monkeypatch):
    log = []
    sdb = _make_store(monkeypatch, log)
    written = []

    def fake_append(rows, shadow=False):
        log.append("archive_write")
        written.extend(rows)

    monkeypatch.setattr(outcome_storage, "append_outcome_batch", fake_append)
    _resolve(sdb)
    assert log.index("archive_write") < len(log) - 1          # before the write pipeline
    assert log[-1] == "redis_execute" and sdb._redis.store["deleted"] is True
    assert written[0]["_stream_id"] == "ETHUSD:ppo_cross_up:1700000000"


def test_archive_failure_keeps_pending_key(monkeypatch):
    log = []
    sdb = _make_store(monkeypatch, log)

    def boom(rows, shadow=False):
        raise OSError("disk full")

    monkeypatch.setattr(outcome_storage, "append_outcome_batch", boom)
    _resolve(sdb)
    assert sdb._redis.store["deleted"] is False               # pending outcome NOT deleted
    assert sdb._run_resolved_total == 0