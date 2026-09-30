"""Smoke: resolved outcome → JSONL → trade_quality / calibration / ablation."""
from __future__ import annotations

import json
import time
from pathlib import Path

def test_alert_pending_resolve_jsonl_brain_quality(tmp_path, monkeypatch):
    """Smoke: resolved outcome → JSONL → trade_quality / calibration / ablation."""
    from outcome_storage import append_outcome_batch, OUTCOME_SCHEMA_VERSION
    import threshold_engine as engine
    import outcome_storage as osmod

    # append_outcome_batch writes under the private module global _OUTCOME_DIR
    # (set at import from cfg). Point it at tmp_path and create subdirs.
    root = str(tmp_path)
    monkeypatch.setattr(osmod, "_OUTCOME_DIR", root)
    for sub in ("outcomes", "shadow", "reports"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)

    row = {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "pair": "BTCUSD",
        "alert_key": "ppo_cross_buy",
        "direction": "buy",
        "entry_ts": int(time.time()) - 3600,
        "entry_price": 50000.0,
        "win": True,
        "conf_pct": 72.0,
        "confluence_score": 22.0,
        "confluence_total": 30.0,
        "mfe": 0.018,
        "mae": 0.004,
        "net_pnl_pct": 0.012,
        "outcome_reason": "target_hit",
        "votes": {"base_trend": True, "ppo_cross": True, "adx": True},
        "adx_val": 28.0,
        "session": "london",
    }
    append_outcome_batch([row], shadow=False)

    files = list(Path(root).rglob("*.jsonl"))
    assert files, "resolved outcome was not written to JSONL"
    written = json.loads(files[0].read_text(encoding="utf-8").strip().splitlines()[0])
    assert written["alert_key"] == "ppo_cross_buy"
    assert written["win"] is True

    tq = engine.trade_quality_score(
        {
            "pair": "BTCUSD",
            "alert_key": "ppo_cross_buy",
            "direction": "buy",
            "conf_pct": 72.0,
        },
        {
            "p_profit": 0.62,
            "expected_return": 0.008,
            "net_ev": 0.006,
        },
        None,
        None,
    )
    assert "verdict" in tq

    calib = engine.build_calibration_curves([row], bucket_pct=20.0, min_sample=1)
    assert "curves" in calib

    # Only if actionable_condition_ablation was added to threshold_engine
    if hasattr(engine, "actionable_condition_ablation"):
        ablation = engine.actionable_condition_ablation(
            [row], min_sample=1, n_permutations=3,
        )
        assert isinstance(ablation, list)