"""Live CONFLUENCE_WEIGHTS may change only via promotion or rollback."""
import asyncio
from types import SimpleNamespace

import state as state_mod


class _FakeRedisStateStore:
    _WEIGHT_WRITE_SOURCES = state_mod.RedisStateStore._WEIGHT_WRITE_SOURCES
    degraded = False
    _redis = object()

    def __init__(self):
        self.written = {}

    async def set_metadata(self, key, value, ttl=None):
        self.written[key] = value


def _call(**kw):
    db = _FakeRedisStateStore()
    ok = asyncio.run(state_mod.RedisStateStore.set_dynamic_weights(db, {"x": 1.0}, **kw))
    return ok, db.written


def test_direct_write_refused_when_enforced(monkeypatch):
    monkeypatch.setattr(state_mod, "cfg", SimpleNamespace(ENFORCE_SINGLE_WEIGHT_PATH=True))
    ok, written = _call()
    assert ok is False and not written


def test_promotion_and_rollback_allowed(monkeypatch):
    monkeypatch.setattr(state_mod, "cfg", SimpleNamespace(ENFORCE_SINGLE_WEIGHT_PATH=True))
    for src in ("promotion", "rollback"):
        ok, written = _call(source=src)
        assert ok is True and "dynamic_weights" in written


def test_direct_write_allowed_when_not_enforced(monkeypatch):
    monkeypatch.setattr(state_mod, "cfg", SimpleNamespace(ENFORCE_SINGLE_WEIGHT_PATH=False))
    ok, _ = _call()
    assert ok is True
