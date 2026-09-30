"""Archive read-path integrity: idempotent rows and reconcilable counters."""
import json
import time

from archive_reader import load_archived_outcomes
from outcome_storage import OUTCOME_SCHEMA_VERSION


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
    dropped = sum(v for k, v in stats.items() if k.startswith("dropped")) + stats["lines_malformed"]
    assert stats["lines_total"] == stats["kept"] + dropped


# ── New enrichment fields (#15, #14, #18) ──────────────────────────
def test_new_enrichment_fields_parsed(tmp_path):
    """Rows carrying regime_transition, learned_tp_sl, and
    lifecycle_state should parse them into the output dict."""
    r = _row(
        regime_transition={"from": "ranging", "to": "trending", "ts": 123},
        learned_tp_sl={"sl_suggested_pct": 0.5, "tp1_suggested_pct": 0.9, "oos_passed": True},
        lifecycle_state="APPLIED",
    )
    _write(tmp_path, [r])
    rows, stats = load_archived_outcomes(str(tmp_path), 30, return_stats=True)
    assert len(rows) == 1
    assert rows[0]["regime_transition"]["from"] == "ranging"
    assert rows[0]["learned_tp_sl"]["oos_passed"] is True
    assert rows[0]["lifecycle_state"] == "APPLIED"


def test_missing_enrichment_fields_default_to_none(tmp_path):
    """Old archive rows without the new enrichment fields should still
    parse correctly with None defaults — no crash, no dropped rows."""
    r = _row()  # no regime_transition, learned_tp_sl, lifecycle_state
    _write(tmp_path, [r])
    rows, stats = load_archived_outcomes(str(tmp_path), 30, return_stats=True)
    assert len(rows) == 1
    assert rows[0].get("regime_transition") is None
    assert rows[0].get("learned_tp_sl") is None
    assert rows[0].get("lifecycle_state") is None
    assert stats["kept"] == 1
    assert stats["dropped_unmigratable"] == 0


def test_enrichment_fields_do_not_affect_dedup(tmp_path):
    """Two rows with the same _stream_id but different enrichment
    fields should still be deduplicated."""
    r1 = _row(regime_transition={"from": "a", "to": "b"})
    r2 = dict(r1)  # same _stream_id
    r2["lifecycle_state"] = "ROLLBACK"
    _write(tmp_path, [r1, r2])
    rows, stats = load_archived_outcomes(str(tmp_path), 30, return_stats=True)
    assert len(rows) == 1
    assert stats["dropped_duplicate_sid"] == 1