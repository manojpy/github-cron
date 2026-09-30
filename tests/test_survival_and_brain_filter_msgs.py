"""#25 survival checklist + #26 BRAIN FILTER Telegram notice (build_* / _notify_brain_filter)."""
from __future__ import annotations

import asyncio

import alerts
from bot_config import cfg


class _Q:
    def __init__(self, ok=True):
        self.sent, self.ok = [], ok

    async def send(self, msg, priority="normal"):
        self.sent.append(msg)
        return self.ok


class _Redis:
    def __init__(self):
        self.keys = set()

    def set(self, key, val, nx=False, ex=None):
        if nx and key in self.keys:
            return None
        self.keys.add(key)
        return True


class _SDB:
    degraded = False

    def __init__(self):
        self._redis = _Redis()

    async def _safe_redis_op(self, fn, timeout, label):
        return fn()


def _notify(q, sdb, key="ppo_cross_buy", gate="calibration gate", tq=None):
    return asyncio.run(alerts._notify_brain_filter(
        q, sdb, pair_name="BTCUSD", alert_key=key, gate=gate,
        lines=["calibrated WR 38% below 55% floor"], tq=tq, score=22.0, total=30.0,
    ))


def _on(monkeypatch, cap=3):
    monkeypatch.setattr(cfg, "ENABLE_BRAIN_FILTER_TELEGRAM", True, raising=False)
    monkeypatch.setattr(cfg, "BRAIN_FILTER_TELEGRAM_MAX_PER_RUN", cap, raising=False)
    monkeypatch.setattr(alerts, "_rc", lambda r: r)


TQ = {"verdict": "BLOCKED", "p_ev_positive": 0.41, "net_ev": -0.12,
      "evidence_state": "SHADOW", "n_oos": 38, "drift_warning": True,
      "regime_warning": "ranging regime WR=35%"}


def test_survival_checklist_passes_and_warnings():
    out = alerts.build_survival_checklist(
        score=22.0, total=30.0, required=18.0, win_rate=0.61, win_sample=45,
        cal_wr=0.6, cal_reason="ok",
        tq={"verdict": "MEDIUM", "p_ev_positive": 0.66, "net_ev": 0.21, "drift_warning": True},
    )
    assert "✓ gates 22/30 (need 18)" in out
    assert "✓ WR 61% n=45" in out and "✓ calib 60%" in out
    assert "✓ brain MEDIUM P=66% EV=+0.21%" in out and "⚠ drift" in out


def test_survival_checklist_marks_unevaluated_not_passed():
    out = alerts.build_survival_checklist(
        score=None, total=None, required=None, win_rate=None, win_sample=0,
        cal_wr=None, cal_reason="thin_bucket_fail_open", tq=None,
    )
    assert "○ WR no history" in out and "○ calib thin" in out
    assert "✓" not in out        # nothing was actually verified


def test_message_has_fields_and_is_markdown_safe():
    msg = alerts.build_brain_filter_message(
        pair="BTC_USD", alert_key="ppo_cross_buy", gate="calibration gate",
        lines=["x (y)"], tq=TQ, score=22.0, total=30.0,
    )
    for needle in ("BRAIN FILTER", "Signal gates: PASS", "BLOCKED", "evidence SHADOW",
                   "recent WR drift", "ranging regime"):
        assert needle in msg
    assert "P\\(profit\\) 41%" in msg and "netEV \\-0\\.12%" in msg
    assert "BTC\\_USD" in msg and "x \\(y\\)" in msg


def test_off_by_default(monkeypatch):
    monkeypatch.setattr(cfg, "ENABLE_BRAIN_FILTER_TELEGRAM", False, raising=False)
    q = _Q()
    assert _notify(q, _SDB()) is False and q.sent == []


def test_cooldown_dedupes_same_pair_alert_gate(monkeypatch):
    _on(monkeypatch)
    q, sdb = _Q(), _SDB()
    assert _notify(q, sdb) is True
    assert _notify(q, sdb) is False                      # same key inside cooldown
    assert _notify(q, sdb, gate="quality hard block") is True   # different gate
    assert len(q.sent) == 2


def test_per_run_cap(monkeypatch):
    _on(monkeypatch, cap=2)
    q, sdb = _Q(), _SDB()
    results = [_notify(q, sdb, key=f"k{i}") for i in range(4)]
    assert results == [True, True, False, False]


def test_degraded_redis_sends_nothing(monkeypatch):
    _on(monkeypatch)
    q, sdb = _Q(), _SDB()
    sdb.degraded = True
    assert _notify(q, sdb) is False and q.sent == []


def test_never_raises_on_send_failure(monkeypatch):
    _on(monkeypatch)

    class _Boom(_Q):
        async def send(self, msg, priority="normal"):
            raise RuntimeError("telegram down")

    assert _notify(_Boom(), _SDB()) is False
