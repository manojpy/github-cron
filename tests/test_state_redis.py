from __future__ import annotations
import json
import redis_audit as ra
import asyncio
import redis_audit
import state
from alerts import ALERT_KEYS
from threshold_engine import alert_family_of
import time
from archive_reader import load_archived_outcomes
from outcome_storage import OUTCOME_SCHEMA_VERSION

# ======================================================================
# from test_redis_audit.py
# ======================================================================
"""Redis key inventory / TTL audit: classification, findings, read-only scan."""



DAY = 86400


class _Pipe:
    def __init__(self, c):
        self.c, self.ops = c, []

    def type(self, k):
        self.ops.append(("type", k))

    def ttl(self, k):
        self.ops.append(("ttl", k))

    def execute(self):
        return [self.c.type(k) if op == "type" else self.c.ttl(k) for op, k in self.ops]


class _Redis:
    """Minimal read-only fake; records every command so tests can prove no writes."""

    def __init__(self, data, no_memory=False):
        self.data, self.calls, self.no_memory = data, [], no_memory

    def scan_iter(self, match="*", count=500):
        self.calls.append("scan")
        yield from list(self.data)

    def pipeline(self, transaction=False):
        return _Pipe(self)

    def type(self, k):
        self.calls.append("type")
        return b"string"

    def ttl(self, k):
        self.calls.append("ttl")
        return self.data[k]

    def memory_usage(self, k):
        self.calls.append("memory")
        if self.no_memory:
            raise RuntimeError("unsupported")
        return 100

    def xlen(self, s):
        self.calls.append("xlen")
        return 42


def test_classification_longest_prefix_and_orphans():
    assert ra.classify_key("brain_cusum_watermark:ppo")[0] == "brain_cusum_watermark"
    assert ra.classify_key("brain_cusum:ppo")[0] == "brain_cusum"
    assert ra.classify_key("alert_stats:x")[0] == "alert_stats"
    assert ra.classify_key("alert:BTC:k")[0] == "alert_dedup"
    assert ra.classify_key("outcome_log_stream") == ("outcome_log_stream", "no_ttl_ok")
    assert ra.classify_key("some_old_thing:1") == ("unknown", "any")
    assert ra.metadata_name("metadata:daily_cache:BTCUSD") == "daily_cache"


def _rec(key, ttl, t="string"):
    return {"key": key, "type": t, "ttl": ttl, "bytes": 100}


def test_findings_orphan_leak_and_durable():
    recs = [
        _rec("pair_state:BTCUSD", 5 * DAY),
        _rec("pair_state:ETHUSD", -1),                     # leak
        _rec("outcome_log_stream", -1, "stream"),          # fine: streams persist
        _rec("legacy_blob", 100),                          # orphan
        _rec("metadata:config_override", 3 * DAY),         # durable, finite TTL
        _rec("metadata:daily_cache:BTCUSD", 2 * DAY),
        _rec("gone", -2),                                  # expired mid-scan: ignored
    ]
    rep = ra.summarize(recs, {"outcome_log_stream": 10})
    assert rep["scanned"] == 6
    kinds = {(f["kind"], f["family"]) for f in rep["findings"]}
    assert ("NO_TTL_LEAK", "pair_state") in kinds
    assert ("ORPHAN_KEYS", "unknown") in kinds
    assert ("DURABLE_DECISION_FINITE_TTL", "metadata:config_override") in kinds
    assert not any(f["family"] == "outcome_log_stream" for f in rep["findings"])
    assert rep["families"]["pair_state"]["no_ttl"] == 1
    assert rep["families"]["metadata"]["keys"] == 2
    assert rep["metadata"]["daily_cache"]["keys"] == 1
    assert rep["streams"] == {"outcome_log_stream": 10}


def test_clean_database_has_no_findings():
    rep = ra.summarize([_rec("pair_state:BTCUSD", DAY), _rec("metadata:dynamic_weights", 30 * DAY)])
    assert rep["findings"] == []


def test_collect_is_read_only_and_samples_bytes():
    r = _Redis({"pair_state:A": DAY, "pair_state:B": DAY, "metadata:x": -1})
    recs, streams, truncated = ra.collect(r, sample_bytes=1)
    assert not truncated and len(recs) == 3
    assert set(r.calls) <= {"scan", "type", "ttl", "memory", "xlen"}   # no writes, ever
    rep = ra.summarize(recs, streams)
    assert rep["families"]["pair_state"]["bytes_est"] == 200            # 1 sampled x 2 keys
    assert streams == {"outcome_log_stream": 42, "shadow_log_stream": 42}


def test_collect_survives_missing_memory_usage_and_caps_keys():
    r = _Redis({f"pair_state:{i}": DAY for i in range(10)}, no_memory=True)
    recs, _, truncated = ra.collect(r, max_keys=4)
    assert truncated and len(recs) == 4
    assert ra.summarize(recs)["families"]["pair_state"]["bytes_est"] is None


def test_render_and_json_roundtrip():
    rep = ra.summarize([_rec("pair_state:X", -1)])
    text = ra.render(rep)
    assert "NO_TTL_LEAK" in text and "pair_state" in text
    assert json.loads(json.dumps(rep))["findings"][0]["kind"] == "NO_TTL_LEAK"


def test_cli_needs_url(monkeypatch, capsys):
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert ra.main([]) == 2
    assert "REDIS_URL" in capsys.readouterr().err


# ── one-off durable-TTL heal ────────────────────────────────────────────────

class _HealRedis:
    """Fake with writable TTLs. gt=True mimics EXPIRE ... GT (never shortens)."""

    def __init__(self, ttls, supports_gt=True):
        self.ttls, self.supports_gt, self.expired = dict(ttls), supports_gt, []

    def pipeline(self, transaction=False):
        return _HealPipe(self)

    def ttl(self, k):
        return self.ttls.get(k, -2)

    def expire(self, k, seconds, gt=False, nx=False):
        if gt and not self.supports_gt:
            raise RuntimeError("ERR syntax error")
        cur = self.ttls.get(k, -2)
        if cur == -2:
            return False
        if nx and cur != -1:
            return False
        if gt and cur != -1 and seconds <= cur:
            return False
        self.ttls[k] = seconds
        self.expired.append(k)
        return True

class _HealPipe:
    def __init__(self, c):
        self.c, self.ops = c, []

    def type(self, k):
        self.ops.append(("type", k))

    def ttl(self, k):
        self.ops.append(("ttl", k))

    def execute(self):
        return [b"string" if op == "type" else self.c.ttl(k) for op, k in self.ops]


TARGET = ra.HEAL_TARGET_TTL_SEC


def test_heal_set_excludes_long_lived_keys_and_matches_durable():
    assert ra.HEAL_METADATA <= ra.DURABLE_METADATA
    assert "dynamic_weights" not in ra.HEAL_METADATA
    assert "brain_apply_snapshots" not in ra.HEAL_METADATA
    assert "config_override" in ra.HEAL_METADATA


def test_plan_only_raises_short_finite_ttls_on_exact_keys():
    recs = [
        _rec("metadata:config_override", 3 * DAY),                 # planned
        _rec("metadata:brain_disabled_alert_keys", TARGET),        # already at target
        _rec("metadata:brain_alert_key_history", -1),              # no expiry: leave
        _rec("metadata:pair_confluence_thresholds", -2),           # absent
        _rec("metadata:dynamic_weights", 2 * DAY),                 # exempt
        _rec("metadata:brain_apply_snapshots", 2 * DAY),           # exempt
        _rec("metadata:config_override:extra", 2 * DAY),           # not the exact key
        _rec("pair_state:BTCUSD", 2 * DAY),                        # other family
    ]
    plan = ra.plan_durable_ttl_heal(recs)
    assert [p["key"] for p in plan] == ["metadata:config_override"]
    assert plan[0]["ttl_before"] == 3 * DAY and plan[0]["target"] == TARGET


def test_apply_raises_and_verifies_after():
    r = _HealRedis({"metadata:config_override": 3 * DAY})
    plan = ra.plan_durable_ttl_heal(ra.collect_heal_candidates(r))
    out = ra.apply_durable_ttl_heal(r, plan)
    row = out["keys"][0]
    assert row["status"] == "raised" and row["ttl_after"] == TARGET
    assert r.ttls["metadata:config_override"] == TARGET


def test_dry_run_writes_nothing():
    r = _HealRedis({"metadata:config_override": 3 * DAY})
    plan = ra.plan_durable_ttl_heal(ra.collect_heal_candidates(r))
    out = ra.apply_durable_ttl_heal(r, plan, dry_run=True)
    assert out["keys"][0]["status"] == "would_raise" and "ttl_after" not in out["keys"][0]
    assert r.expired == [] and r.ttls["metadata:config_override"] == 3 * DAY


def test_heal_never_shortens_a_concurrent_longer_write():
    # the bot rewrote the key with a 365d TTL between our read and our EXPIRE
    r = _HealRedis({"metadata:config_override": 365 * DAY})
    out = ra.apply_durable_ttl_heal(r, [{"key": "metadata:config_override", "ttl_before": 3 * DAY, "target": TARGET}])
    assert out["keys"][0]["status"] == "skipped"
    assert r.ttls["metadata:config_override"] == 365 * DAY


def test_heal_fallback_without_gt_still_never_shortens():
    r = _HealRedis({"metadata:config_override": 365 * DAY, "metadata:brain_alert_key_history": 3 * DAY}, supports_gt=False)
    plan = [
        {"key": "metadata:config_override", "ttl_before": 3 * DAY, "target": TARGET},
        {"key": "metadata:brain_alert_key_history", "ttl_before": 3 * DAY, "target": TARGET},
    ]
    out = ra.apply_durable_ttl_heal(r, plan)
    status = {x["key"]: x["status"] for x in out["keys"]}
    assert status == {"metadata:config_override": "skipped", "metadata:brain_alert_key_history": "raised"}
    assert r.ttls["metadata:config_override"] == 365 * DAY


def test_render_includes_heal_section_and_empty_case():
    rep = ra.summarize([_rec("pair_state:X", DAY)])
    rep["heal"] = {"target_days": 90.0, "dry_run": False, "keys": [
        {"key": "metadata:config_override", "ttl_before": 3 * DAY, "target": TARGET, "ttl_after": TARGET, "status": "raised"}]}
    text = ra.render(rep)
    assert "Durable-TTL heal" in text and "3.0d → 90.0d [raised]" in text
    rep["heal"] = {"target_days": 90.0, "dry_run": True, "keys": []}
    assert "nothing to heal" in ra.render(rep) and "DRY RUN" in ra.render(rep)


def test_cli_heal_end_to_end(monkeypatch, capsys):
    import sys, types
    fake = _HealRedis({"metadata:config_override": 3 * DAY, "metadata:dynamic_weights": 2 * DAY})
    fake.ping = lambda: True
    fake.scan_iter = lambda match="*", count=500: iter(list(fake.ttls))
    fake.type = lambda k: b"string"
    fake.memory_usage = lambda k: 10
    fake.xlen = lambda s: 0
    mod = types.SimpleNamespace(from_url=lambda *a, **k: fake)
    monkeypatch.setitem(sys.modules, "redis", mod)
    assert ra.main(["--url", "redis://x", "--heal-durable-ttl"]) == 0
    out = capsys.readouterr().out
    assert "metadata:config_override" in out and "[raised]" in out
    assert fake.ttls["metadata:config_override"] == TARGET
    assert fake.ttls["metadata:dynamic_weights"] == 2 * DAY      # exempt: untouched


def test_cli_without_heal_flag_writes_nothing(monkeypatch):
    import sys, types
    fake = _HealRedis({"metadata:config_override": 3 * DAY})
    fake.ping = lambda: True
    fake.scan_iter = lambda match="*", count=500: iter(list(fake.ttls))
    fake.type = lambda k: b"string"
    fake.memory_usage = lambda k: 10
    fake.xlen = lambda s: 0
    monkeypatch.setitem(sys.modules, "redis", types.SimpleNamespace(from_url=lambda *a, **k: fake))
    assert ra.main(["--url", "redis://x"]) == 0
    assert fake.expired == [] and fake.ttls["metadata:config_override"] == 3 * DAY

# ── one-off TTL-leak heal ───────────────────────────────────────────────────

def test_plan_ttl_leak_heal_only_allowlisted_families():
    recs = [
        _rec("brain_threshold_history:BCHUSD", -1),   # allowed → planned
        _rec("brain_threshold_history:LABUSD", -1),   # allowed → planned
        _rec("pair_state:BTCUSD", -1),                # ttl_required but not allow-listed
        _rec("lock:foo", -1),                         # must never be healed
        _rec("outcome_pending:x", -1),                # must never be healed
        _rec("recent_alert:y", -1),                   # must never be healed
        _rec("outcome_log_stream", -1, "stream"),     # no_ttl_ok
        _rec("legacy_blob", -1),                      # orphan
        _rec("brain_threshold_history:XAUTUSD", 5 * DAY),  # already has TTL
    ]
    plan = ra.plan_ttl_leak_heal(recs)
    assert [p["key"] for p in plan] == [
        "brain_threshold_history:BCHUSD",
        "brain_threshold_history:LABUSD",
    ]
    assert all(p["target"] == ra.LEAK_HEAL_TTL_SEC for p in plan)
    assert all(p["family"] == "brain_threshold_history" for p in plan)


def test_set_ttl_only_if_missing_respects_nx():
    r = _HealRedis({"brain_threshold_history:BCHUSD": -1})
    assert ra._set_ttl_only_if_missing(r, "brain_threshold_history:BCHUSD", ra.LEAK_HEAL_TTL_SEC)
    assert r.ttls["brain_threshold_history:BCHUSD"] == ra.LEAK_HEAL_TTL_SEC

    # Already has a TTL → must not change it
    r2 = _HealRedis({"brain_threshold_history:BCHUSD": 5 * DAY})
    assert not ra._set_ttl_only_if_missing(r2, "brain_threshold_history:BCHUSD", ra.LEAK_HEAL_TTL_SEC)
    assert r2.ttls["brain_threshold_history:BCHUSD"] == 5 * DAY


def test_apply_ttl_leak_heal_sets_and_dry_run():
    r = _HealRedis({
        "brain_threshold_history:BCHUSD": -1,
        "brain_threshold_history:LABUSD": -1,
    })
    plan = ra.plan_ttl_leak_heal([
        _rec("brain_threshold_history:BCHUSD", -1),
        _rec("brain_threshold_history:LABUSD", -1),
    ])
    out = ra.apply_ttl_leak_heal(r, plan)
    assert all(row["status"] == "set" for row in out["keys"])
    assert r.ttls["brain_threshold_history:BCHUSD"] == ra.LEAK_HEAL_TTL_SEC
    assert r.ttls["brain_threshold_history:LABUSD"] == ra.LEAK_HEAL_TTL_SEC

    r2 = _HealRedis({"brain_threshold_history:BCHUSD": -1})
    out2 = ra.apply_ttl_leak_heal(r2, plan[:1], dry_run=True)
    assert out2["keys"][0]["status"] == "would_set"
    assert r2.expired == [] and r2.ttls["brain_threshold_history:BCHUSD"] == -1


def test_render_includes_leak_heal_section():
    rep = ra.summarize([_rec("brain_threshold_history:BCHUSD", -1)])
    rep["leak_heal"] = {
        "target_days": 30.0,
        "dry_run": False,
        "keys": [{
            "key": "brain_threshold_history:BCHUSD",
            "family": "brain_threshold_history",
            "ttl_before": -1,
            "target": ra.LEAK_HEAL_TTL_SEC,
            "ttl_after": ra.LEAK_HEAL_TTL_SEC,
            "status": "set",
        }],
    }
    text = ra.render(rep)
    assert "TTL-leak heal → 30d" in text
    assert "brain_threshold_history:BCHUSD" in text
    assert "[set]" in text

    rep["leak_heal"] = {"target_days": 30.0, "dry_run": True, "keys": []}
    assert "nothing to heal" in ra.render(rep) and "DRY RUN" in ra.render(rep)

def test_cli_heal_ttl_leaks_end_to_end(monkeypatch, capsys):
    import sys, types
    fake = _HealRedis({
        "brain_threshold_history:BCHUSD": -1,
        "pair_state:BTCUSD": -1,          # not allow-listed → must stay -1
    })
    fake.ping = lambda: True
    fake.scan_iter = lambda match="*", count=500: iter(list(fake.ttls))
    fake.type = lambda k: b"string"
    fake.memory_usage = lambda k: 10
    fake.xlen = lambda s: 0
    mod = types.SimpleNamespace(from_url=lambda *a, **k: fake)
    monkeypatch.setitem(sys.modules, "redis", mod)
    assert ra.main(["--url", "redis://x", "--heal-ttl-leaks"]) == 0
    out = capsys.readouterr().out
    assert "brain_threshold_history:BCHUSD" in out and "[set]" in out
    assert fake.ttls["brain_threshold_history:BCHUSD"] == ra.LEAK_HEAL_TTL_SEC
    assert fake.ttls["pair_state:BTCUSD"] == -1          # untouched


# ======================================================================
# from test_durable_metadata_and_registry.py
# ======================================================================
"""Guards for standing-decision TTLs and the alert-key registry.

1. Every durable metadata key gets the long TTL (not the 7-day default) and
   the set stays identical to redis_audit.DURABLE_METADATA.
2. Every alert key maps to a real family (none fall into "Other"), so a newly
   added alert cannot silently skip family analysis.
"""



class _Store:
    """Just enough of RedisStateStore for set_metadata()."""

    meta_prefix = "metadata:"
    metadata_expiry_seconds = 7 * 86400
    set_metadata = state.RedisStateStore.set_metadata

    def __init__(self):
        self.calls = []

    async def _safe_redis_op(self, op, timeout, name, **_kw):
        class _R:
            async def set(_s, key, value, ex=None):
                self.calls.append((key, ex))

        import state as _state
        orig = _state._rc
        _state._rc = lambda _r: _R()
        try:
            return await op()
        finally:
            _state._rc = orig

    _redis = object()


def _ttl_for(key, **kw):
    store = _Store()
    asyncio.run(store.set_metadata(key, "{}", **kw))
    return store.calls[0][1]


def test_durable_keys_get_long_ttl():
    for key in state.DURABLE_METADATA_KEYS:
        assert _ttl_for(key) == state.DURABLE_METADATA_TTL_SEC


def test_non_durable_key_keeps_default_ttl():
    assert _ttl_for("daily_cache:BTCUSD:2026-10-01") == 7 * 86400


def test_explicit_ttl_wins():
    assert _ttl_for("dynamic_weights", ttl=123) == 123


def test_durable_sets_match_audit():
    assert set(state.DURABLE_METADATA_KEYS) == set(redis_audit.DURABLE_METADATA)


def test_alert_registry_is_self_consistent():
    from alert_registry import (
        REGISTRY,
        ALERT_KEYS,
        BUY_ALERT_KEYS,
        SELL_ALERT_KEYS,
        alert_family_of,
        registry_problems,
    )

    assert REGISTRY
    assert not registry_problems()

    assert set(ALERT_KEYS) == set(REGISTRY)

    assert not (BUY_ALERT_KEYS & SELL_ALERT_KEYS)

    for key, spec in REGISTRY.items():
        assert spec.key == key
        assert spec.family
        assert spec.family != "Other"
        assert spec.direction in {"buy", "sell"}
        assert spec.config_flag
        assert alert_family_of(key) == spec.family

def test_unknown_alert_key_does_not_inherit_family():
    from alert_registry import alert_family_of

    assert alert_family_of("vwap_future") == "Other"
    assert alert_family_of("ppo_unknown") == "Other"
    assert alert_family_of("rsi_unknown") == "Other"
    assert alert_family_of("not_registered") == "Other"

def test_all_alert_keys_have_a_family():
    other = [k for k in ALERT_KEYS if alert_family_of(k) == "Other"]
    assert not other, f"alert keys with no family: {other}"


# ======================================================================
# from test_archive_integrity.py
# ======================================================================
"""Archive read-path integrity: idempotent rows and reconcilable counters."""



def _row(pair="ETHUSD", key="ppo_cross_up", ts=None, **over):
    ts = ts or int(time.time()) - 3600
    r = {
        "pair": pair, "alert_key": key, "direction": "buy", "entry_ts": ts,
        "score": 5.0, "total": 10.0, "win": True, "pct_move": 1.2,
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "_stream_id": f"{pair}:{key}:{ts}",
    }
    r.update(over)
    return r


def _write(tmp_path, rows):
    d = tmp_path / "outcomes"
    d.mkdir()
    (d / "2026-09-21.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_retried_archive_write_is_deduplicated(tmp_path):
    r = _row()
    _write(tmp_path, [r, dict(r)])          # same outcome appended twice (retry)
    rows, stats = load_archived_outcomes(str(tmp_path), 30, return_stats=True)
    assert len(rows) == 1
    assert stats["dropped_duplicate_sid"] == 1


def test_counters_reconcile_including_unparseable(tmp_path):
    good = _row(ts=int(time.time()) - 3600)
    bad_total = _row(key="rsi_up", ts=int(time.time()) - 7200, total=0, _stream_id="x:1")
    _write(tmp_path, [good, bad_total])
    rows, stats = load_archived_outcomes(str(tmp_path), 30, return_stats=True)
    assert len(rows) == 1 and stats["kept"] == 1
    assert stats["dropped_unparseable"] == 1
    dropped = sum(v for k, v in stats.items() if k.startswith("dropped_")) + stats["lines_malformed"]
    assert stats["lines_total"] == stats["kept"] + dropped
