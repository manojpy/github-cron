"""#6: snapshot at apply + objective post-apply harm monitor + safe revert."""
from __future__ import annotations

import asyncio
import logging
import random
import time

import brain_enhanced as be
from bot_config import cfg, json_dumps, json_loads

HOUR = 3600
LOG = logging.getLogger("t")


class FakeSDB:
    degraded = False

    def __init__(self):
        self.meta = {}
        self.override = {}
        self.disabled = set()
        self.weights = None

    async def get_metadata(self, k):
        return self.meta.get(k)

    async def set_metadata(self, k, v, ttl=None):
        self.meta[k] = v
        return True

    async def get_config_override(self):
        return dict(self.override)

    async def write_config_override(self, f, v):
        self.override[f] = v
        return True

    async def remove_config_override_field(self, f):
        self.override.pop(f, None)
        return True

    async def get_disabled_alert_keys(self):
        return set(self.disabled)

    async def set_alert_key_disabled(self, ak, flag):
        (self.disabled.add if flag else self.disabled.discard)(ak)
        return True

    async def get_dynamic_weights(self):
        return None if self.weights is None else dict(self.weights)

    async def set_dynamic_weights(self, w, ttl=None):
        self.weights = dict(w)
        return True

    async def clear_dynamic_weights(self):
        self.weights = None
        return True


class FakeQ:
    def __init__(self):
        self.sent = []

    async def send(self, msg, priority="normal"):
        self.sent.append(msg)
        return True


def _engine(sdb):
    cls = be.BrainEngineV2
    e = cls.__new__(cls)
    e.sdb = sdb
    return e


def _rows(n, wr, t0, t1, seed, adx=True):
    rnd = random.Random(seed)
    return [{
        "win": rnd.random() < wr, "entry_ts": int(rnd.uniform(t0, t1)),
        "adx_val": rnd.uniform(10, 45) if adx else None,
        "net_pnl_pct": 0.5, "pct_move": 0.5,
    } for _ in range(n)]


def _seed_snapshot(sdb, applied_ago_h=72, **kw):
    now = int(time.time())
    snap = {
        "plan_id": "plan-1", "applied_at": now - int(applied_ago_h * HOUR), "status": "active",
        "overrides": kw.get("overrides", {}), "disabled": kw.get("disabled", {}),
        "weights": kw.get("weights"),
    }
    sdb.meta[be.APPLY_SNAPSHOT_KEY] = json_dumps([snap])
    return snap, now


def _status(sdb):
    return json_loads(sdb.meta[be.APPLY_SNAPSHOT_KEY])[0]["status"]


def test_apply_captures_prior_state():
    sdb = FakeSDB()
    sdb.override = {"CONFLUENCE_MIN_PCT": 70.0}
    sdb.weights = {"a": 1.0}
    sdb.disabled = {"old_key"}
    plan = {
        "plan_id": "p9", "generated_at": int(time.time()), "_action_gate_passed": True,
        "config_patch": [
            {"path": "CONFLUENCE_MIN_PCT", "suggested": 75.0},
            {"path": "CONFLUENCE_MIN_ABS_SCORE", "suggested": 20.0},
        ],
        "disable_alerts": ["bad_key"], "weight_adjustments": [],
    }
    sdb.meta["brain_pending_plan"] = json_dumps(plan)
    eng = _engine(sdb)
    ok = asyncio.run(eng.apply_pending_plan(FakeQ(), LOG))
    assert ok
    snap = json_loads(sdb.meta[be.APPLY_SNAPSHOT_KEY])[0]
    assert snap["overrides"]["CONFLUENCE_MIN_PCT"] == {"prev": 70.0, "new": 75.0}
    assert snap["overrides"]["CONFLUENCE_MIN_ABS_SCORE"] == {"prev": None, "new": 20.0}
    assert snap["disabled"]["bad_key"] == {"prev": False, "new": True}
    assert snap["status"] == "active"


def test_harm_reverts_overrides_to_previous_values_and_absence():
    sdb = FakeSDB()
    sdb.override = {"CONFLUENCE_MIN_PCT": 75.0, "CONFLUENCE_MIN_ABS_SCORE": 20.0}
    sdb.disabled = {"bad_key"}
    snap, now = _seed_snapshot(
        sdb,
        overrides={"CONFLUENCE_MIN_PCT": {"prev": 70.0, "new": 75.0},
                   "CONFLUENCE_MIN_ABS_SCORE": {"prev": None, "new": 20.0}},
        disabled={"bad_key": {"prev": False, "new": True}},
    )
    at = snap["applied_at"]
    rows = _rows(80, 0.62, at - 72 * HOUR, at - 1, 1) + _rows(80, 0.35, at + 1, now, 2)
    ev = asyncio.run(_engine(sdb).monitor_applied_plans(rows, LOG))
    assert [e["status"] for e in ev] == ["rolled_back"], ev
    assert sdb.override == {"CONFLUENCE_MIN_PCT": 70.0}       # restored / removed
    assert "bad_key" not in sdb.disabled                      # re-enabled
    assert _status(sdb) == "rolled_back"
    assert ev[0]["needs_restart"] is True
    hist = json_loads(sdb.meta[be.PLAN_HISTORY_KEY])
    assert hist[-1]["status"] == "rolled_back"


def test_harm_reverts_weights_to_none_or_prev():
    for prev in (None, {"a": 1.0, "b": 2.0}):
        sdb = FakeSDB()
        sdb.weights = {"a": 0.5, "b": 2.0}
        snap, now = _seed_snapshot(sdb, weights={"prev": prev, "new": {"a": 0.5, "b": 2.0}})
        at = snap["applied_at"]
        rows = _rows(80, 0.62, at - 72 * HOUR, at - 1, 1) + _rows(80, 0.35, at + 1, now, 2)
        asyncio.run(_engine(sdb).monitor_applied_plans(rows, LOG))
        assert sdb.weights == prev


def test_no_revert_when_value_changed_since():
    sdb = FakeSDB()
    sdb.override = {"CONFLUENCE_MIN_PCT": 80.0}               # a LATER plan set 80, ours wrote 75
    snap, now = _seed_snapshot(
        sdb, overrides={"CONFLUENCE_MIN_PCT": {"prev": 70.0, "new": 75.0}})
    at = snap["applied_at"]
    rows = _rows(80, 0.62, at - 72 * HOUR, at - 1, 1) + _rows(80, 0.35, at + 1, now, 2)
    ev = asyncio.run(_engine(sdb).monitor_applied_plans(rows, LOG))
    assert sdb.override == {"CONFLUENCE_MIN_PCT": 80.0}       # untouched
    assert "CONFLUENCE_MIN_PCT (changed since)" in ev[0]["result"]["skipped"]


def test_no_revert_on_noise_or_small_sample():
    sdb = FakeSDB()
    sdb.override = {"CONFLUENCE_MIN_PCT": 75.0}
    snap, now = _seed_snapshot(
        sdb, overrides={"CONFLUENCE_MIN_PCT": {"prev": 70.0, "new": 75.0}})
    at = snap["applied_at"]
    # same WR before/after
    rows = _rows(80, 0.55, at - 72 * HOUR, at - 1, 1) + _rows(80, 0.55, at + 1, now, 2)
    assert asyncio.run(_engine(sdb).monitor_applied_plans(rows, LOG)) == []
    # big drop but only 12 post outcomes
    rows = _rows(80, 0.62, at - 72 * HOUR, at - 1, 1) + _rows(12, 0.20, at + 1, now, 2)
    assert asyncio.run(_engine(sdb).monitor_applied_plans(rows, LOG)) == []
    assert sdb.override == {"CONFLUENCE_MIN_PCT": 75.0} and _status(sdb) == "active"


def test_too_early_is_not_judged():
    sdb = FakeSDB()
    sdb.override = {"CONFLUENCE_MIN_PCT": 75.0}
    snap, now = _seed_snapshot(
        sdb, applied_ago_h=3, overrides={"CONFLUENCE_MIN_PCT": {"prev": 70.0, "new": 75.0}})
    at = snap["applied_at"]
    rows = _rows(80, 0.62, at - 3 * HOUR, at - 1, 1) + _rows(80, 0.30, at + 1, now, 2)
    assert asyncio.run(_engine(sdb).monitor_applied_plans(rows, LOG)) == []


def test_regime_attributed_drop_is_withheld():
    sdb = FakeSDB()
    sdb.override = {"CONFLUENCE_MIN_PCT": 75.0}
    snap, now = _seed_snapshot(
        sdb, overrides={"CONFLUENCE_MIN_PCT": {"prev": 70.0, "new": 75.0}})
    at = snap["applied_at"]
    rows = _rows(80, 0.62, at - 72 * HOUR, at - 1, 1, adx=False) + _rows(80, 0.35, at + 1, now, 2, adx=False)
    ev = asyncio.run(_engine(sdb).monitor_applied_plans(rows, LOG))
    assert ev and ev[0]["status"] == "rollback_withheld"      # no ADX → cannot rule out regime
    assert sdb.override == {"CONFLUENCE_MIN_PCT": 75.0}


def test_daily_cap_and_clear_after_monitor_window():
    sdb = FakeSDB()
    now = int(time.time())
    old = {"plan_id": "old", "applied_at": now - 20 * 86400, "status": "active",
           "overrides": {}, "disabled": {}, "weights": None}
    done = {"plan_id": "done", "applied_at": now - 5 * 86400, "status": "rolled_back",
            "rolled_back_at": now - HOUR, "overrides": {}, "disabled": {}, "weights": None}
    sdb.meta[be.APPLY_SNAPSHOT_KEY] = json_dumps([done, old])
    rows = _rows(80, 0.55, now - 40 * 86400, now - 20 * 86400 - 1, 1) + _rows(80, 0.55, now - 20 * 86400 + 1, now, 2)
    ev = asyncio.run(_engine(sdb).monitor_applied_plans(rows, LOG))
    assert [e["status"] for e in ev] == ["cleared"]


def test_disabled_by_flag(monkeypatch):
    monkeypatch.setattr(cfg, "BRAIN_AUTO_ROLLBACK_HURT", False, raising=False)
    sdb = FakeSDB()
    snap, now = _seed_snapshot(sdb, overrides={"X": {"prev": 1, "new": 2}})
    assert asyncio.run(_engine(sdb).monitor_applied_plans([], LOG)) == []


def test_regime_shift_explains_drop_is_withheld():
    # Per-regime WR unchanged (trending 72%, ranging 38%), but post-apply trades are
    # almost all ranging -> overall WR drops purely from the mix.
    sdb = FakeSDB()
    sdb.override = {"CONFLUENCE_MIN_PCT": 75.0}
    snap, now = _seed_snapshot(
        sdb, overrides={"CONFLUENCE_MIN_PCT": {"prev": 70.0, "new": 75.0}})
    at = snap["applied_at"]

    def mk(n, wr, lo, hi, adx_lo, adx_hi, seed):
        rnd = random.Random(seed)
        return [{"win": rnd.random() < wr, "entry_ts": int(rnd.uniform(lo, hi)),
                 "adx_val": rnd.uniform(adx_lo, adx_hi), "net_pnl_pct": 0.5} for _ in range(n)]

    pre = mk(70, 0.72, at - 72 * HOUR, at - 1, 30, 45, 1) + mk(70, 0.38, at - 72 * HOUR, at - 1, 10, 22, 2)
    post = mk(10, 0.72, at + 1, now, 30, 45, 3) + mk(110, 0.38, at + 1, now, 10, 22, 4)
    ev = asyncio.run(_engine(sdb).monitor_applied_plans(pre + post, LOG))
    assert ev and ev[0]["status"] == "rollback_withheld", ev
    assert ev[0]["reason"].startswith("drop attributed to regime")
    assert sdb.override == {"CONFLUENCE_MIN_PCT": 75.0}
