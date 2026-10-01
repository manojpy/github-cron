"""Redis key inventory / TTL audit: classification, findings, read-only scan."""
from __future__ import annotations

import json

import redis_audit as ra

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
