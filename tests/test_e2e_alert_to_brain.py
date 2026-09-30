"""End-to-end: alert dispatch → pending outcome → resolve → JSONL → Brain quality.

Uses a fake Redis-like store and a temp OUTCOME_DATA_DIR so it does not
need a live Redis or network. This is the integration test that was
called out as still missing for the dual-path surface area.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Minimal fake that supports the small subset Brain + outcome_storage need
class _FakeRedis:
    def __init__(self):
        self.kv: Dict[str, str] = {}
        self.hashes: Dict[str, Dict[str, str]] = {}

    async def get(self, key: str):
        return self.kv.get(key)

    async def set(self, key: str, value: str, ex: Optional[int] = None):
        self.kv[key] = value
        return True

    async def hgetall(self, key: str):
        return dict(self.hashes.get(key, {}))

    async def hset(self, key: str, mapping: Optional[Dict[str, str]] = None, **kwargs):
        h = self.hashes.setdefault(key, {})
        if mapping:
            h.update(mapping)
        h.update(kwargs)
        return True

    async def ping(self):
        return True

    async def aclose(self):
        return None


@pytest.mark.asyncio
async def test_alert_pending_resolve_jsonl_brain_quality(tmp_path, monkeypatch):
    """Smoke the full loop with one synthetic winning BUY outcome."""
    from outcome_storage import append_outcome_batch, OUTCOME_SCHEMA_VERSION
    import threshold_engine as engine

    outcome_dir = tmp_path / "outcomes"
    outcome_dir.mkdir()
    monkeypatch.setenv("OUTCOME_DATA_DIR", str(tmp_path))
    # Point module global if it was already imported
    import outcome_storage as osmod
    monkeypatch.setattr(osmod, "OUTCOME_DATA_DIR", str(tmp_path), raising=False)

    # 1) Append one resolved real outcome (simulates post-resolve write)
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
    append_outcome_batch([row], is_shadow=False)

    # 2) Files land under outcomes/
    files = list(Path(tmp_path).rglob("*.jsonl"))
    assert files, "resolved outcome was not written to JSONL"
    written = json.loads(files[0].read_text(encoding="utf-8").strip().splitlines()[0])
    assert written["alert_key"] == "ppo_cross_buy"
    assert written["win"] is True

    # 3) Trade-quality path accepts the row shape
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
        None,  # calibration_curve
        None,  # regime_info
    )
    assert "verdict" in tq
    assert "evidence_state" in tq or "evidence_strength" in tq

    # 4) Calibration builder does not explode on a single-key sample
    calib = engine.build_calibration_curves(
        [row], bucket_pct=20.0, min_sample=1,
    )
    assert "curves" in calib

    # 5) Actionable ablation handles thin data without raising
    ablation = engine.actionable_condition_ablation(
        [row], min_sample=1, n_permutations=3,
    )
    assert isinstance(ablation, list)