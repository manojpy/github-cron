"""Regression tests: online calibration keeps history + is cursor-safe,
ablation separates informative votes from noise, reasoning chain reads real data."""
from __future__ import annotations

import asyncio
import random
import time

import threshold_engine as engine


def _rows(n, seed, p_win=0.55, ak="ppo_cross_buy"):
    rnd = random.Random(seed)
    return [
        {"alert_key": ak, "conf_pct": rnd.uniform(50, 90), "win": rnd.random() < p_win}
        for _ in range(n)
    ]


# ── online calibration (pure function) ────────────────────────────────
def test_fold_keeps_history_and_adds_rows():
    base = engine.build_calibration_curves(_rows(300, 1), bucket_pct=20, min_sample=15)
    n_before = base["curves"]["ppo_cross_buy"]["n"]
    n_buckets = len(base["curves"]["ppo_cross_buy"]["buckets"])
    built_at = base["built_at"]

    folded = engine.fold_outcomes_into_calibration(base, _rows(20, 2), min_sample=15)

    c = base["curves"]["ppo_cross_buy"]
    assert folded == 20
    assert c["n"] == n_before + 20                    # history retained
    assert len(c["buckets"]) == n_buckets             # boundaries untouched
    assert sum(b["n"] for b in c["buckets"]) == n_before + 20
    assert base["built_at"] == built_at               # age-based rebuild still fires


def test_fold_equals_full_rebuild_on_observed_rate():
    old, new = _rows(300, 1), _rows(40, 2)
    base = engine.build_calibration_curves(old, bucket_pct=20, min_sample=15)
    engine.fold_outcomes_into_calibration(base, new, min_sample=15)
    folded_wins = sum(round(b["observed"] * b["n"]) for b in base["curves"]["ppo_cross_buy"]["buckets"])
    assert abs(folded_wins - sum(r["win"] for r in old + new)) <= len(base["curves"]["ppo_cross_buy"]["buckets"])


def test_fold_ignores_unknown_alert_key_and_out_of_range():
    base = engine.build_calibration_curves(_rows(200, 1), bucket_pct=20, min_sample=15)
    assert engine.fold_outcomes_into_calibration(base, _rows(5, 3, ak="other_key"), 15) == 0


# ── online calibration (cursor flow against a fake Redis) ─────────────
class _FakeRedis:
    def __init__(self):
        self.streams = {"outcome_log_stream": [], "shadow_log_stream": []}
        self.kv = {}
        self._seq = 0

    def add(self, stream, fields):
        self._seq += 1
        self.streams[stream].append((f"{1000 + self._seq}-0", fields))

    async def xrevrange(self, key, count=1):
        return list(reversed(self.streams[key]))[:count]

    async def xrange(self, key, min="-", max="+", count=None):
        def key_of(i):
            ms, seq = i.split("-")
            return (int(ms), int(seq))
        out = [e for e in self.streams[key] if key_of(e[0]) >= key_of(min)]
        return out[:count] if count else out

    async def get(self, k):
        return self.kv.get(k)

    async def set(self, k, v, ex=None):
        self.kv[k] = v
        return True


class _FakeSDB:
    degraded = False

    def __init__(self, r):
        self._redis = r

    async def _safe_redis_op(self, fn, timeout, label):
        return await fn()


def _outcome(conf, win, ts):
    return {"pair": "BTCUSD", "alert_key": "ppo_cross_buy", "direction": "buy",
            "score": str(conf), "total": "100", "win": "1" if win else "0",
            "pct_move": "0.5", "entry_ts": str(ts)}


def test_cursor_flow_no_double_count(monkeypatch):
    import brain
    from bot_config import cfg, json_loads
    monkeypatch.setattr(brain, "_rc", lambda r: r)
    monkeypatch.setattr(cfg, "ENABLE_CALIBRATION_GATE", True, raising=False)

    r = _FakeRedis()
    now = int(time.time())
    rnd = random.Random(5)
    for i in range(120):
        r.add("outcome_log_stream", _outcome(rnd.uniform(50, 90), rnd.random() < 0.55, now - 100000 + i))

    eng = brain.BaseBrainEngine.__new__(brain.BaseBrainEngine) if hasattr(brain, "BaseBrainEngine") else None
    cls = [c for n, c in vars(brain).items() if isinstance(c, type) and hasattr(c, "maybe_refresh_calibration")][0]
    eng = cls.__new__(cls)
    eng.sdb = _FakeSDB(r)
    eng._calib_cache = None
    eng._calib_cache_ts = 0.0
    import logging
    log = logging.getLogger("t")

    async def scenario():
        await eng.maybe_refresh_calibration(log)                       # full rebuild
        first = json_loads(r.kv[brain.CALIBRATION_CURVES_KEY])
        n0 = first["curves"]["ppo_cross_buy"]["n"]
        assert "stream_cursors" in first

        for i in range(15):                                            # 15 new outcomes
            r.add("outcome_log_stream", _outcome(rnd.uniform(50, 90), rnd.random() < 0.55, now + i))
        await eng.maybe_refresh_calibration(log)                       # fold
        second = json_loads(r.kv[brain.CALIBRATION_CURVES_KEY])
        assert second["curves"]["ppo_cross_buy"]["n"] == n0 + 15
        assert second["built_at"] == first["built_at"]

        await eng.maybe_refresh_calibration(log)                       # nothing new
        third = json_loads(r.kv[brain.CALIBRATION_CURVES_KEY])
        assert third["curves"]["ppo_cross_buy"]["n"] == n0 + 15        # idempotent

    asyncio.run(scenario())


# ── ablation ───────────────────────────────────────────────────────────
def _vote_rows(seed):
    rnd = random.Random(seed)
    rows = []
    for _ in range(400):
        a, b = rnd.random() < 0.5, rnd.random() < 0.5
        rows.append({"win": rnd.random() < (0.75 if a else 0.30),
                     "votes": {"informative": a, "noise": b}})
    return rows


def test_ablation_never_cuts_informative_vote():
    for seed in range(15):
        res = {a["vote"]: a for a in engine.actionable_condition_ablation(
            _vote_rows(seed), min_sample=30, n_permutations=15)}
        assert res["informative"]["action"] == "keep"


def test_ablation_flags_noise_vote_often():
    cuts = sum(
        {a["vote"]: a for a in engine.actionable_condition_ablation(
            _vote_rows(s), min_sample=30, n_permutations=15)}["noise"]["action"] == "reduce_weight"
        for s in range(30)
    )
    assert cuts >= 10

def test_ablation_skips_unmeasurable_votes():
    rows = _vote_rows(1)
    for r in rows:
        r["votes"]["always_on"] = True
    names = {a["vote"] for a in engine.actionable_condition_ablation(rows, min_sample=30)}
    assert "always_on" not in names


# ── #15: Regime-transition fields don't break calibration ──────────
def test_calibration_ignores_regime_transition_fields():
    """Rows carrying regime_transition or learned_tp_sl enrichment
    fields should not break build_calibration_curves."""
    rows = _rows(300, 1)
    for r in rows[:10]:
        r["regime_transition"] = {"from": "a", "to": "b"}
        r["learned_tp_sl"] = {"sl_suggested_pct": 0.5}
        r["lifecycle_state"] = "APPLIED"
    base = engine.build_calibration_curves(rows, bucket_pct=20, min_sample=15)
    assert "ppo_cross_buy" in base["curves"]
    assert base["curves"]["ppo_cross_buy"]["n"] == 300


# ── #14: Learned TP/SL zone function ───────────────────────────────
def test_learned_tp_sl_zone_returns_valid():
    """learned_tp_sl_zone should return a valid result with enough
    data and OOS validation."""
    import random
    rnd = random.Random(7)
    rows = []
    for i in range(80):
        mae = rnd.uniform(0.001, 0.02)
        mfe = rnd.uniform(0.001, 0.03)
        rows.append({
            "entry_ts": 1700000000 + i * 900,
            "mae": mae,
            "mfe": mfe,
            "win": mfe > mae,
        })
    result = engine.learned_tp_sl_zone(rows, min_sample=20)
    assert result["valid"] is True
    assert "sl_suggested_pct" in result
    assert "tp1_suggested_pct" in result
    assert "tp2_suggested_pct" in result
    assert "oos_passed" in result
    assert result["n_train"] >= 20
    assert result["n_holdout"] >= 10


def test_learned_tp_sl_zone_insufficient_data():
    """With too few rows, learned_tp_sl_zone should return invalid."""
    rows = [{"entry_ts": i, "mae": 0.01, "mfe": 0.02, "win": True} for i in range(10)]
    result = engine.learned_tp_sl_zone(rows, min_sample=20)
    assert result["valid"] is False
    assert result.get("error") == "insufficient_data"