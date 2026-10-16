#!/usr/bin/env python3
"""repair_ledger.py — closed-loop tracking of every repair the Brain issues.

Without this, repair_shop_diagnosis() is open-loop: it fires recommendations,
the plan is applied, and nothing ever checks whether the repair worked. Every
trigger in the shop is a hand-tuned constant that has never been asked to
learn. This module gives each repair an identity, a "before" snapshot, an
"applied" event, and a post-hoc verdict — the labeled data every ML upgrade
in the shop depends on.
"""
from __future__ import annotations
import asyncio
import hashlib
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from bot_config import cfg, json_dumps, json_loads
from threshold_engine import _flatten_row_features, wilson_ci, ev_and_kelly_for, row_net_pnl_pct
from plan_replay import cluster_se

LEDGER_KEY = "brain_repair_ledger"
LEDGER_MAX = 500
LEDGER_TTL_DAYS = 90
# A plan is considered to have "applied" every repair issued within this
# window of the plan's own timestamp. The report generator and _store_
# pending_plan run back-to-back in the same brain cycle, so ±60s is safe.
APPLY_WINDOW_SEC = 60

def _matches_scope(row: dict, scope: Optional[dict]) -> bool:
    """True if `row` falls inside the repair's affected subset.

    A repair that disables three alerts, or tightens one threshold band,
    or blocks one segment rule, only touches a slice of trades. Measuring
    the verdict against the whole book drowns that signal in the other
    97% of rows — the Wilson band on global WR will cover any local
    effect, and 90%+ of verdicts collapse to 'neutral' regardless of
    whether the repair actually worked. Scope narrows the measurement
    back to where the repair can matter.

    None / unknown / {'kind':'global'} → no filtering (legacy behavior).
    """
    if not scope:
        return True
    kind = scope.get("kind")
    if kind in (None, "global", "none"):
        return True
    val = scope.get("value")

    if kind == "direction":
        return row.get("direction") == val
    if kind == "alert_keys":
        return row.get("alert_key") in (val or [])
    if kind == "pair":
        return row.get("pair") == val
    if kind == "score_band":
        lo, hi = val if isinstance(val, (list, tuple)) and len(val) == 2 else (0.0, 0.0)
        return lo <= row.get("score", 0.0) < hi
    if kind == "config_version":
        return (row.get("context") or {}).get("config_version") == val
    if kind == "segment":
        if not isinstance(val, dict):
            return False
        feats = _flatten_row_features(row)
        feature = val.get("feature")
        if not isinstance(feature, str):
            return False
        v = feats.get(feature)
        if v is None:
            return False
        op = val.get("op")
        thr = float(val.get("threshold", 0.0))
        if op == ">":
            return v > thr
        if op == "<=":
            return v <= thr
        return False
    return True

def _scope_metrics(rows: List[dict], scope: Optional[dict]):
    """(wr, n, net_ev) on the subset matching scope, or (None, 0, None) if empty."""
    subset = [r for r in rows if _matches_scope(r, scope)]
    if not subset:
        return None, 0, None
    wins = sum(1 for r in subset if r["win"])
    wr = wins / len(subset)
    net_ev, _hk, _wr = ev_and_kelly_for(subset)
    return wr, len(subset), net_ev

def _repair_id(rec: Dict[str, Any], ts: int) -> str:
    """Stable ID — retries within the same 15m bucket produce the same ID,
    so a crash-and-resume does not fork the ledger."""
    payload = json_dumps({
        "type": rec.get("type"),
        "category": rec.get("category"),
        "alert": rec.get("alert"),
        "param": rec.get("param"),
        "ts_bucket": ts // 900,
    })
    return hashlib.md5(payload.encode()).hexdigest()[:12]

async def _load_ledger(sdb) -> List[dict]:
    raw = await sdb.get_metadata(LEDGER_KEY)
    if not raw:
        return []
    try:
        data = json_loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        return []

async def _load_ledger_strict(sdb) -> List[dict]:
    """Like _load_ledger, but raises when Redis can't be read or the payload
    is corrupt, so a writer never overwrites real history with an empty list."""
    rc = sdb._redis
    if rc is None or sdb.degraded:
        raise RuntimeError("repair ledger: Redis unavailable")
    raw = await asyncio.wait_for(rc.get(f"{sdb.meta_prefix}{LEDGER_KEY}"), timeout=5.0)
    if not raw:
        return []
    data = json_loads(raw)
    if not isinstance(data, list):
        raise ValueError("repair ledger: payload is not a list")
    return data

async def _save_ledger(sdb, entries: List[dict]) -> None:
    entries = entries[:LEDGER_MAX]
    await sdb.set_metadata(LEDGER_KEY, json_dumps(entries),
                            ttl=LEDGER_TTL_DAYS * 86400)

async def record_repair_issued(sdb, rec: Dict[str, Any], snapshot: Dict[str, Any], real_rows: Optional[List[dict]] = None) -> Optional[str]:
    """Call once per emitted repair. snapshot carries the pre-repair metrics
    the verdict will be measured against. When `real_rows` is supplied and
    the repair carries a `scope`, also stores scope_wr/scope_n — the WR of
    the affected subset right now, which the verdict will be compared to."""
    ts = int(time.time())
    rid = _repair_id(rec, ts)
    scope = rec.get("scope")
    snapshot = dict(snapshot)
    if scope and real_rows:
        scope_wr, scope_n, scope_ev = _scope_metrics(real_rows, scope)
        if scope_n > 0:
            snapshot["scope_wr"] = scope_wr
            snapshot["scope_n"] = scope_n
            snapshot["scope_ev"] = scope_ev
    entry = {
        "id": rid,
        "issued_at": ts,
        "type": rec.get("type"),
        "category": rec.get("category"),
        "severity": rec.get("severity"),
        "alert": rec.get("alert"),
        "param": rec.get("param"),
        "message": rec.get("message"),
        "scope": scope,
        "snapshot_before": snapshot,
        "applied_at": None,
        "verdict": None,
        "delta_observed": None,
    }
    if sdb.degraded or not sdb._redis:
        return None
    try:
        entries = await _load_ledger_strict(sdb)
    except Exception:
        return None
    # If this repair was already issued in the same 15m bucket, refresh it
    # rather than duplicating.
    for i, e in enumerate(entries):
        if e["id"] == rid:
            entries[i] = entry
            await _save_ledger(sdb, entries)
            return rid
    entries.insert(0, entry)
    await _save_ledger(sdb, entries)
    return rid

async def mark_plan_applied(sdb, plan_ts: int, repair_ids: Optional[List[str]] = None) -> int:
    """Called by apply_pending_plan. With `repair_ids`, marks exactly those
    repairs (the ones whose change was actually applied). Without it (plans
    stored before ids were recorded), falls back to every repair whose issued_at
    is within APPLY_WINDOW_SEC of the plan's own timestamp. Returns count."""
    try:
        entries = await _load_ledger_strict(sdb)
    except Exception:
        return 0
    n = 0
    now = int(time.time())
    wanted = set(repair_ids) if repair_ids is not None else None
    for e in entries:
        if e["applied_at"] is not None:
            continue
        if wanted is not None:
            hit = e["id"] in wanted
        else:
            hit = abs(e["issued_at"] - plan_ts) <= APPLY_WINDOW_SEC
        if hit:
            e["applied_at"] = now
            n += 1
    if n:
        await _save_ledger(sdb, entries)
    return n

async def record_applied_alert_toggles(sdb, toggles: List[Tuple[str, str]], real_rows: Optional[List[dict]] = None) -> int:
    """Put alert disable / reinstate actions into the ledger as already-applied
    repairs, so they get a verdict and count toward family accountability like
    every other change. `toggles` is a list of (category, alert_key) where
    category is 'disable_alert' or 'reinstate_alert'. Returns how many entries
    were added."""
    if not toggles or sdb.degraded or not sdb._redis:
        return 0
    try:
        entries = await _load_ledger_strict(sdb)
    except Exception:
        return 0
    now = int(time.time())
    known = {e.get("id") for e in entries}
    added = 0
    for category, ak in toggles:
        rec = {"type": category, "category": category, "alert": ak, "param": None}
        rid = _repair_id(rec, now)
        if rid in known:
            continue
        scope = {"kind": "alert_keys", "value": [ak]}
        snapshot: Dict[str, Any] = {"overall_wr": None, "n": len(real_rows or []), "net_ev": None}
        if real_rows:
            s_wr, s_n, s_ev = _scope_metrics(real_rows, scope)
            if s_n > 0:
                snapshot["scope_wr"] = s_wr
                snapshot["scope_n"] = s_n
                snapshot["scope_ev"] = s_ev
        entries.insert(0, {
            "id": rid, "issued_at": now, "type": category, "category": category,
            "severity": "medium", "alert": ak, "param": None,
            "message": f"{category}: {ak}", "scope": scope,
            "snapshot_before": snapshot, "applied_at": now,
            "verdict": None, "delta_observed": None,
        })
        known.add(rid)
        added += 1
    if added:
        await _save_ledger(sdb, entries)
    return added

def _row_rejection(r: dict) -> Optional[str]:
    return r.get("rejection_reason") or (r.get("context") or {}).get("rejection_reason")

def _disable_verdict(e: dict, shadow_rows: List[dict], min_n: int) -> Optional[Dict[str, Any]]:
    """A disabled alert produces no real trades, so judge the disable on the
    trades it blocked (shadow rows tagged brain_disabled) since it was applied.
    lift = minus the blocked trades' net EV: positive when the blocked trades
    were losers (the disable helped). Same +/-0.02 band as other verdicts."""
    keys = set((e.get("scope") or {}).get("value") or [])
    applied_at = int(e["applied_at"])
    blocked = [
        r for r in shadow_rows
        if r.get("alert_key") in keys and int(r.get("entry_ts") or 0) >= applied_at
        and _row_rejection(r) == "brain_disabled"
    ]
    if len(blocked) < min_n:
        return None
    ev, _hk, _wr = ev_and_kelly_for(blocked)
    if ev is None:
        return None
    lift = -float(ev)
    verdict = "helped" if lift > 0.02 else "hurt" if lift < -0.02 else "neutral"
    wins = sum(1 for r in blocked if r["win"])
    lo, hi, _ = wilson_ci(wins, len(blocked))
    return {
        "verdict": verdict,
        "delta": {
            "wr": None, "ev": round(lift, 4), "blocked_net_ev": round(float(ev), 4),
            "n_post": len(blocked), "n_post_total": len(blocked), "scoped": True,
            "scope_kind": "alert_keys", "basis": "blocked_shadow_trades",
            "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4),
        },
    }

def control_comparison(rows: List[dict], scope: Optional[dict], applied_at: int, now: int, min_scope: int = 8, min_control: int = 15, z: float = 1.645) -> Dict[str, Any]:
    """Did the repaired slice move differently from trades the repair did NOT
    touch, over the same two periods? (difference-in-differences, net % per trade)

    pre  = the period of equal length just BEFORE the repair was applied
    post = applied_at .. now
    effect = (scope_post - scope_pre) - (control_post - control_pre)
    so a market-wide move that lifts or sinks every alert cancels out.
    The standard error uses 3-hour time blocks (alerts on one move are not
    independent). Verdicts: helped / hurt (the bound clears zero),
    indistinguishable, insufficient (too few trades), no_control (global scope
    or nothing outside the scope). Descriptive: a repair that also changes the
    control group (a global threshold) is not separable this way."""
    kind = (scope or {}).get("kind")
    if kind in (None, "global", "none"):
        return {"verdict": "no_control"}
    span = int(now) - int(applied_at)
    if span <= 0:
        return {"verdict": "insufficient"}
    lo_t = int(applied_at) - span
    cost = (float(getattr(cfg, "BRAIN_FEE_PCT", 0.0006)) + float(getattr(cfg, "BRAIN_SLIPPAGE_PCT", 0.0003))) * 2 * 100
    groups: Dict[str, List[Any]] = {"s0": [], "s1": [], "c0": [], "c1": []}
    for r in rows:
        t = r.get("entry_ts")
        if t is None or t < lo_t or t > now:
            continue
        side = "s" if _matches_scope(r, scope) else "c"
        groups[side + ("1" if t >= applied_at else "0")].append((float(t), row_net_pnl_pct(r, cost)))
    n = {k: len(v) for k, v in groups.items()}
    if min(n["s0"], n["s1"]) < min_scope or min(n["c0"], n["c1"]) < min_control:
        return {"verdict": "insufficient" if (n["c0"] + n["c1"]) else "no_control", "n": n}
    means: Dict[str, float] = {}
    var = 0.0
    for k, v in groups.items():
        vals = [x[1] for x in v]
        means[k] = sum(vals) / len(vals)
        se, _nb = cluster_se(vals, [x[0] for x in v])
        if se == float("inf"):
            return {"verdict": "insufficient", "n": n}
        var += se * se
    s_chg = means["s1"] - means["s0"]
    c_chg = means["c1"] - means["c0"]
    did = s_chg - c_chg
    se_d = var ** 0.5
    lo, hi = did - z * se_d, did + z * se_d
    verdict = "helped" if lo > 0 else "hurt" if hi < 0 else "indistinguishable"
    return {
        "verdict": verdict, "effect": round(did, 4), "lo": round(lo, 4), "hi": round(hi, 4),
        "scope_change": round(s_chg, 4), "control_change": round(c_chg, 4), "n": n,
    }

def control_summary(entries: List[dict], now: int, window_days: int = 60) -> Dict[str, Any]:
    """Of the repairs measured in the window, how many verdicts survive the
    same-period comparison? Counts by (before/after verdict, control verdict)."""
    cutoff = now - window_days * 86400
    out: Dict[str, Any] = {"n": 0, "checked": 0, "confirmed": 0, "drift": 0, "contradicted": 0}
    for e in entries:
        if e.get("verdict") not in ("helped", "hurt"):
            continue
        if int(e.get("verdict_at") or e.get("applied_at") or 0) < cutoff:
            continue
        out["n"] += 1
        cv = (e.get("control") or {}).get("verdict")
        if cv in ("helped", "hurt", "indistinguishable"):
            out["checked"] += 1
            if cv == "indistinguishable":
                out["drift"] += 1
            elif cv == e["verdict"]:
                out["confirmed"] += 1
            else:
                out["contradicted"] += 1
    return out

async def evaluate_pending_repairs(sdb, current_rows: List[dict], horizon_hours: int = 48, min_outcomes: int = 30, shadow_rows: Optional[List[dict]] = None) -> List[dict]:
    """Called once per brain report. For each applied repair whose horizon
    has passed, compare post-application metrics against the pre-application
    snapshot using a Wilson band so a repair is only marked helped/hurt when
    the move clears noise.

    Scope-aware: measures the AFFECTED SUBSET, not the whole book. A repair
    that touches 3% of trades cannot move global WR by more than a Wilson
    band's width, so a global-only verdict is noise by construction. When
    the ledger entry carries a `scope`, verdicts are measured on rows
    matching that scope; when it doesn't (legacy entries), falls back to
    global — identical behavior to before this change.
    """
    now = int(time.time())
    cutoff = now - horizon_hours * 3600
    entries = await _load_ledger(sdb)
    fresh_verdicts: List[dict] = []
    changed = False

    # Scoped verdicts need fewer outcomes than global ones — the subset is
    # a fraction of the book, so 30 scoped outcomes means 150+ total rows.
    # Floor at 10 so narrow scopes (segment rules, single-alert) can still
    # get a verdict within one report cycle instead of starving.
    min_outcomes_scoped = max(10, min_outcomes // 3)

    for e in entries:
        if e["applied_at"] is None or e["verdict"] is not None:
            continue
        if e["applied_at"] > cutoff:
            continue

        scope = e.get("scope")
        scope_kind = (scope or {}).get("kind")
        is_scoped = scope_kind not in (None, "global", "none")

        if e.get("category") == "disable_alert":
            _dv = _disable_verdict(e, shadow_rows or [], min_outcomes_scoped)
            if _dv is None:
                continue
            e["verdict"] = _dv["verdict"]
            e["verdict_at"] = now
            e["control"] = {"verdict": "no_control"}
            e["delta_observed"] = _dv["delta"]
            fresh_verdicts.append(e)
            changed = True
            continue

        post = [r for r in current_rows if r["entry_ts"] >= e["applied_at"]]
        post_scope = [r for r in post if _matches_scope(r, scope)]
        effective_min = min_outcomes_scoped if is_scoped else min_outcomes
        if len(post_scope) < effective_min:
            continue

        post_wins = sum(1 for r in post_scope if r["win"])
        post_wr = post_wins / len(post_scope)
        post_ev, _hk, _wr = ev_and_kelly_for(post_scope)

        # Pre-repair baseline: prefer the scoped snapshot; fall back to
        # overall only if the scope was empty at issue time.
        if is_scoped:
            pre_wr = e["snapshot_before"].get("scope_wr")
            pre_ev = e["snapshot_before"].get("scope_ev")
        else:
            pre_wr = None
            pre_ev = None
        if pre_wr is None:
            pre_wr = e["snapshot_before"].get("overall_wr")
        if pre_wr is None:
            pre_wr = post_wr
        if pre_ev is None:
            pre_ev = e["snapshot_before"].get("net_ev")
        if pre_ev is None:
            pre_ev = post_ev

        delta_wr = post_wr - pre_wr
        delta_ev = post_ev - pre_ev

        lo, hi, _ = wilson_ci(post_wins, len(post_scope))

        # ── EV-first verdict ──      
        if delta_ev > 0.02 and delta_wr > -0.03:
            verdict = "helped"
        elif delta_ev < -0.02 and delta_wr < 0.03:
            verdict = "hurt"
        elif delta_wr > 0.03 and lo > pre_wr:
            verdict = "helped"
        elif delta_wr < -0.03 and hi < pre_wr:
            verdict = "hurt"
        else:
            verdict = "neutral"
        e["verdict"] = verdict
        e["verdict_at"] = now
        if str(getattr(cfg, "ACCOUNTABILITY_CONTROL_MODE", "shadow")) != "off":
            try:
                e["control"] = control_comparison(current_rows, scope, int(e["applied_at"]), now)
            except Exception:
                e["control"] = {"verdict": "insufficient"}
        e["delta_observed"] = {
            "wr": round(delta_wr, 4),
            "ev": round(delta_ev, 4),
            "n_post": len(post_scope),
            "n_post_total": len(post),
            "scoped": is_scoped,
            "scope_kind": scope_kind,
            "wilson_lo": round(lo, 4),
            "wilson_hi": round(hi, 4),
        }
        fresh_verdicts.append(e)
        changed = True
    if changed:
        await _save_ledger(sdb, entries)
    return fresh_verdicts

async def repair_success_rates(sdb) -> Dict[str, Dict[str, float]]:
    """Aggregate verdicts → per-category help/hurt rates.
    Neutral counts half-credit: the repair didn't hurt, but it didn't help."""
    entries = await _load_ledger(sdb)
    stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"helped": 0, "hurt": 0, "neutral": 0, "n": 0}
    )
    for e in entries:
        if e["verdict"] is None:
            continue
        key = e.get("category") or e.get("type") or "unknown"
        stats[key][e["verdict"]] += 1
        stats[key]["n"] += 1
    return {
        k: {
            "help_rate": (v["helped"] + 0.5 * v["neutral"]) / max(1, v["n"]),
            "hurt_rate": v["hurt"] / max(1, v["n"]),
            "n": v["n"],
        }
        for k, v in stats.items()
    }

def family_accountability(entries: List[dict], now: int, window_days: int = 60, min_verdicts: int = 6, hurt_streak: int = 3, suspend_days: int = 14, use_control: bool = False) -> Dict[str, Dict[str, Any]]:
    """Per recommendation family (ledger category), has acting on it helped?
    Pure. Only resolved verdicts inside the window count. A family is
    'suspended' when its last `hurt_streak` verdicts were all 'hurt', or when
    at least 5 decisive verdicts have a 'hurt' share whose Wilson lower bound
    exceeds 50%. A suspended family returns to 'probation' `suspend_days` after
    its last 'hurt' verdict (it has to be allowed to try again, or it could
    never show recovery). 'insufficient' = fewer than `min_verdicts` verdicts.
    eligible is False only for 'suspended'."""
    cutoff = now - window_days * 86400
    by_cat: Dict[str, List[dict]] = defaultdict(list)
    for e in entries:
        if e.get("verdict") is None:
            continue
        ts = int(e.get("verdict_at") or e.get("applied_at") or e.get("issued_at") or 0)
        if ts < cutoff:
            continue
        item = {**e, "_ts": ts}
        if use_control and e["verdict"] in ("helped", "hurt") and (e.get("control") or {}).get("verdict") == "indistinguishable":
            item["verdict"] = "neutral"     # the untouched trades moved the same way
            item["_drift"] = True
        by_cat[str(e.get("category") or e.get("type") or "unknown")].append(item)
    out: Dict[str, Dict[str, Any]] = {}
    for cat, items in by_cat.items():
        items.sort(key=lambda x: x["_ts"])
        helped = sum(1 for x in items if x["verdict"] == "helped")
        hurt = sum(1 for x in items if x["verdict"] == "hurt")
        neutral = len(items) - helped - hurt
        decisive = helped + hurt
        streak = len(items) >= hurt_streak and all(x["verdict"] == "hurt" for x in items[-hurt_streak:])
        majority = False
        if decisive >= 5:
            lo, _hi, _ = wilson_ci(hurt, decisive)
            majority = lo > 0.5
        last_hurt = max((x["_ts"] for x in items if x["verdict"] == "hurt"), default=0)
        lifts = [float(x["delta_observed"]["ev"]) for x in items
                 if isinstance(x.get("delta_observed"), dict) and x["delta_observed"].get("ev") is not None]
        ctrl = [float(x["control"]["effect"]) for x in items
                if isinstance(x.get("control"), dict) and x["control"].get("effect") is not None]
        ev_lift = (sum(lifts) / len(lifts)) if lifts else None
        if ev_lift is None or len(lifts) < min_verdicts:
            lift_tag = "insufficient"
        elif ev_lift > 0.02:
            lift_tag = "positive_lift"
        elif ev_lift < -0.02:
            lift_tag = "negative_lift"
        else:
            lift_tag = "flat"
        reason = ""
        if streak or majority:
            reason = f"last {hurt_streak} verdicts all hurt" if streak else "hurt verdicts outnumber helped (95% bound)"
            status = "suspended" if now < last_hurt + suspend_days * 86400 else "probation"
        elif len(items) < min_verdicts:
            status = "insufficient"
        else:
            status = "eligible"
        out[cat] = {
            "status": status, "eligible": status != "suspended", "reason": reason,
            "n": len(items), "helped": helped, "hurt": hurt, "neutral": neutral,
            "hurt_rate": round(hurt / len(items), 3),
            "drift_discounted": sum(1 for x in items if x.get("_drift")),
            "ev_lift_mean": round(ev_lift, 4) if ev_lift is not None else None,
            "ev_lift_n": len(lifts),
            "control_effect_mean": round(sum(ctrl) / len(ctrl), 4) if ctrl else None,
            "control_effect_n": len(ctrl),
            "lift_tag": lift_tag,
            "suspended_until": (last_hurt + suspend_days * 86400) if status == "suspended" else None,
        }
    return out

async def ledger_stats(sdb) -> Dict[str, Any]:
    """For the report header: how many repairs issued/applied/verdict'd."""
    entries = await _load_ledger(sdb)
    return {
        "issued": len(entries),
        "applied": sum(1 for e in entries if e["applied_at"] is not None),
        "verdicts": sum(1 for e in entries if e["verdict"] is not None),
        "helped": sum(1 for e in entries if e["verdict"] == "helped"),
        "hurt": sum(1 for e in entries if e["verdict"] == "hurt"),
    }

async def load_ledger_entries(sdb) -> List[dict]:
    """Public accessor for the raw ledger — used by the contextual
    repair-effectiveness model (threshold_engine.learn_repair_effectiveness).
    Kept separate from repair_success_rates() because the model needs the
    per-entry snapshot_before + verdict, not just the aggregated rates."""
    return await _load_ledger(sdb)