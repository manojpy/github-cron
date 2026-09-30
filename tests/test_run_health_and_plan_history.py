"""Run-health counters, Brain plan audit trail, and the JSONL round trip."""
import asyncio
import json

import alerts
import outcome_storage
from brain_enhanced import BrainEngineV2, PLAN_HISTORY_KEY


class FakeSdb:
    def __init__(self):
        self.meta = {}

    async def get_metadata(self, key):
        return self.meta.get(key)

    async def set_metadata(self, key, value, ttl=None):
        self.meta[key] = value
        return True


def _engine():
    eng = object.__new__(BrainEngineV2)
    eng.sdb = FakeSdb()
    return eng


def test_plan_history_appends_and_caps():
    eng = _engine()
    asyncio.run(eng._record_plan_event("PLAN-001", "pending", "gate passed"))
    asyncio.run(eng._record_plan_event("PLAN-001", "applied"))
    hist = json.loads(eng.sdb.meta[PLAN_HISTORY_KEY])
    assert [h["status"] for h in hist] == ["pending", "applied"]
    for i in range(150):
        asyncio.run(eng._record_plan_event(f"PLAN-{i}", "blocked"))
    assert len(json.loads(eng.sdb.meta[PLAN_HISTORY_KEY])) == 100


def test_plan_history_ignores_missing_id():
    eng = _engine()
    asyncio.run(eng._record_plan_event(None, "applied"))
    assert PLAN_HISTORY_KEY not in eng.sdb.meta


def test_dedup_stats_reset():
    alerts.DEDUP_STATS["released"] = 3
    alerts.DEDUP_STATS["kept_repaint"] = 2
    alerts.reset_dedup_stats()
    assert all(v == 0 for v in alerts.DEDUP_STATS.values())


def test_telegram_counters(monkeypatch):
    q = object.__new__(alerts.TelegramQueue)
    q.sent_ok = 0
    q.sent_failed = 0
    results = iter([True, False])

    async def fake_impl(self, message):
        return next(results)

    monkeypatch.setattr(alerts.TelegramQueue, "_send_impl", fake_impl)
    assert asyncio.run(q.send("a")) is True
    assert asyncio.run(q.send("b")) is False
    assert (q.sent_ok, q.sent_failed) == (1, 1)


def test_jsonl_round_trip_filters_signal_only_rows(tmp_path, monkeypatch):
    for sub in ("outcomes", "shadow"):
        (tmp_path / sub).mkdir()
    monkeypatch.setattr(outcome_storage, "_OUTCOME_DIR", str(tmp_path))
    now = 1_900_000_000
    monkeypatch.setattr(outcome_storage.time, "time", lambda: now)
    # pre-resolution shadow row (no `win`) + resolved outcome row
    outcome_storage.append_outcome({"pair": "BTCUSD", "alert_key": "vwap_buy", "entry_ts": now}, shadow=True)
    outcome_storage.append_outcome_batch(
        [{"pair": "BTCUSD", "alert_key": "vwap_buy", "entry_ts": now, "win": True, "pct_move": 1.2}],
        shadow=True,
    )
    rows = outcome_storage.load_recent_outcomes(days=1, shadow=True)
    assert len(rows) == 1 and rows[0]["win"] is True
    assert rows[0]["schema_version"] == outcome_storage.OUTCOME_SCHEMA_VERSION


def test_redis_key_inventory_counts_prefixes_and_no_ttl():
    import macd_unified

    class Pipe:
        def __init__(self):
            self.keys = []

        def ttl(self, key):
            self.keys.append(key)

        async def execute(self):
            return [-1 if k.startswith("metadata:") else 300 for k in self.keys]

    class R:
        def __init__(self, keys):
            self._keys = keys

        async def scan_iter(self, match="*", count=500):
            for k in self._keys:
                yield k

        def pipeline(self):
            return Pipe()

    class S:
        pass

    s = S()
    s._redis = R(["pending:a:1", "pending:b:2", "metadata:x", "dedup:c"])
    inv = asyncio.run(macd_unified._redis_key_inventory(s))
    assert inv["pending"] == {"keys": 2, "no_ttl": 0}
    assert inv["metadata"] == {"keys": 1, "no_ttl": 1}
    assert inv["dedup"]["keys"] == 1
