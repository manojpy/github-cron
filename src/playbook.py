"""playbook — the slow learner's output: a small, versioned, evidence-gated
trade-plan playbook that the fast 15-minute advisor only READS.

Pure functions (numpy only).  No Redis, no cfg, no network: everything the
learner needs is passed in, so the logic is deterministic and replayable.

Governing rules
  * The fixed bracket (OUTCOME_MAE_LOSS_PCT x OUTCOME_RR_TARGET) is a permanent
    CONTROL arm.  It is evaluated beside every entry and never replaced.
  * A plan is only ever *proposed* by the learner.  It becomes the served
    CHAMPION after: out-of-sample validation with a cluster-robust lower bound
    (stricter the more groups, plans and rules were tested), positive "alpha"
    against a MATCHED no-alert control built from raw candles (same pair, side,
    session, volatility and momentum bucket), a drawdown cap and a stability
    check, N consecutive re-validations of the FROZEN plan on newer data, and
    positive forward-only evidence.
  * Champions decay: demoted on bad recent results, flagged RECONFIRM_DUE when
    no fresh confirmation arrives, EXPIRED when the window passes.
  * The learner can only RESTRICT.  Each entry carries restrict = None | WATCH |
    AVOID (an AVOID needs proof on two consecutive cycles).  Shallow "do not
    take when ..." rules are restrict-only too.
  * Every cycle is versioned; the previous playbook is kept for rollback.

Entry status values
  VALIDATED       served champion plan (confirmed recently)
  RECONFIRM_DUE   served champion plan, confirmation is overdue
  PROMISING       a frozen challenger plan is being re-validated (streak k/N)
  COLLECTING      not enough independent path data yet
  NO_EDGE         no plan in the grid is profitable -> the SIGNAL has no edge
  FAILED_OOS      a plan looked good on train but failed on unseen data
  DEMOTED         a champion was just demoted (cooldown running)
  EXPIRED         a champion went unconfirmed past the expiry window
"""
from __future__ import annotations

import statistics
from statistics import NormalDist
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from plan_replay import (
    EXIT_VARIANTS, Row, cluster_se, diagnose_row, evaluate_plan, has_path, plan_exit_suffix, replay_plan, select_plan,
)
from rulemine import mine_rules, rule_matches

PLAYBOOK_SCHEMA = 2
# sdb metadata keys (written by learner.py, read by the advisor)
KEY_CURRENT = "playbook_current"
KEY_PREV = "playbook_prev"
KEY_STATE = "playbook_state"
KEY_STATE_PREV = "playbook_state_prev"
KEY_LEDGER = "playbook_ledger"
KEY_SCORE = "playbook_scoreboard"

SERVED_STATUSES = ("VALIDATED", "RECONFIRM_DUE")
_GRID_SIZE = 8 * 9 * 4 + EXIT_VARIANTS      # bracket grid + breakeven/trail variants
_DAY = 86400.0
_NEGATIVE = ("NO_EDGE", "FAILED_OOS", "DEMOTED", "EXPIRED")


def default_params() -> Dict[str, Any]:
    return {
        "min_n": 40,                 # path rows a group needs before anything is tested
        "min_holdout": 15,
        "min_blocks": 8,             # independent 3h blocks the alpha test needs
        "promote_consecutive": 3,    # cycles the frozen plan must keep passing
        "min_forward": 15,           # rows strictly AFTER the plan was frozen
        "demote_window": 40,         # most recent rows used to judge a champion
        "demote_min": 20,
        "reconfirm_days": 7.0,
        "expire_days": 14.0,
        "cooldown_days": 2.0,
        "family_alpha": 0.20,        # family-wise error budget across every hypothesis tried
        "min_ev_lift": 0.02,         # a challenger must beat the champion by this (pct points)
        "require_control": True,
        "family_pool": True,
        "max_dd_r": 15.0,            # holdout drawdown cap, in multiples of the plan's stop
        "stab_floor_r": 0.25,        # neither half of the holdout may lose more than this many stops on average
        "avoid_consecutive": 2,      # cycles of proof before an AVOID restriction is issued
        "max_rules": 120,            # rule candidates scored per group per cycle (hard cap)
        "size_full_lcb": 0.30,       # EV lower bound (pct) that earns full size
        "dd_haircut_r": 8.0,         # holdout drawdown (in stops) above which size is halved
    }


# ── No-alert control (built from raw candles) ─────────────────────────────
def _exit_vec(fav: np.ndarray, adv: np.ndarray, cls_h: np.ndarray, sl: float, tp: float,
              h: int, be: float, trail: float) -> np.ndarray:
    """Vectorised twin of plan_replay.replay_exit: gross P&L % per path (rows x candles)."""
    n = fav.shape[0]
    stop = np.full(n, -sl)
    done = np.zeros(n, dtype=bool)
    gross = np.zeros(n)
    best = np.full(n, -1e9)
    be_on = np.zeros(n, dtype=bool)
    for i in range(h + 1):
        hit_stop = ~done & (-adv[:, i] <= stop)
        gross[hit_stop] = stop[hit_stop]
        done |= hit_stop
        hit_tp = ~done & (fav[:, i] >= tp)
        gross[hit_tp] = tp
        done |= hit_tp
        alive = ~done
        best = np.where(alive, np.maximum(best, fav[:, i]), best)
        if be > 0:
            be_on |= alive & (fav[:, i] >= be)
        lvl = np.where(be_on, 0.0, -sl) if be > 0 else np.full(n, -sl)
        if trail > 0:
            lvl = np.maximum(lvl, best - trail)
        stop = np.where(alive, np.maximum(stop, lvl), stop)
    gross[~done] = cls_h[~done]
    return gross


def _terciles(x: np.ndarray) -> Tuple[float, float]:
    v = x[np.isfinite(x)]
    if len(v) < 30:
        return float("nan"), float("nan")
    return float(np.quantile(v, 1 / 3)), float(np.quantile(v, 2 / 3))


def _bucket(x: np.ndarray, edges: Tuple[float, float]) -> np.ndarray:
    out = np.full(len(x), -1, dtype=np.int8)
    if not np.isfinite(edges[0]):
        return out
    ok = np.isfinite(x)
    out[ok] = np.digitize(x[ok], edges).astype(np.int8)
    return out


class ControlModel:
    """What a plan would have earned on EVERY candle of a pair, alert or not.

    The alert's edge is its P&L minus this baseline.  The baseline is MATCHED:
    same pair, same direction, same session, and the same volatility tercile
    (mean candle range of the prior 24 candles) and momentum tercile (directional
    move of the prior 8 candles) as the alert's signal candle.  If a cell is too
    thin the match relaxes step by step: session+vol+momentum -> session+vol ->
    session -> pair/direction."""

    LOOK_VOL = 24
    LOOK_MOM = 8

    def __init__(self, candles: Mapping[str, Tuple[Any, Any, Any, Any, Any]], *,
                 max_h: int = 12, session_fn: Callable[[Any], str], min_rows: int = 30) -> None:
        from numpy.lib.stride_tricks import sliding_window_view as swv
        self.max_h = int(max_h)
        self._min_rows = int(min_rows)
        self._d: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
        self._cache: Dict[Tuple[Any, ...], Optional[Tuple[float, int, int]]] = {}
        self.span: Dict[str, Tuple[int, int]] = {}
        H = self.max_h + 1
        for pair, (ts, o, h, l, c) in candles.items():
            ts = np.asarray(ts, dtype=np.int64)
            o, h, l, c = (np.asarray(x, dtype=np.float64) for x in (o, h, l, c))
            n = len(ts)
            if n < H + min_rows + self.LOOK_VOL + 1:
                continue
            if np.any(c <= 0):
                continue
            hi_w, lo_w, cl_w = swv(h, H)[1:], swv(l, H)[1:], swv(c, H)[1:]
            anchor = o[1:n - H + 1][:, None]
            m = n - H
            sig_ts = ts[0:m]
            sess = np.array([session_fn(int(t)) for t in sig_ts], dtype=object)
            rng = (h - l) / c * 100.0
            csum = np.concatenate([[0.0], np.cumsum(rng)])
            vol = np.full(n, np.nan)
            k = self.LOOK_VOL
            vol[k - 1:] = (csum[k:] - csum[:-k]) / k
            mom = np.full(n, np.nan)
            j = self.LOOK_MOM
            mom[j:] = (c[j:] - c[:-j]) / c[:-j] * 100.0
            vol_b = _bucket(vol[:m], _terciles(vol[:m]))
            for side in ("buy", "sell"):
                if side == "buy":
                    fav = (hi_w - anchor) / anchor * 100.0
                    adv = (anchor - lo_w) / anchor * 100.0
                    cls = (cl_w - anchor) / anchor * 100.0
                    dmom = mom[:m]
                else:
                    fav = (anchor - lo_w) / anchor * 100.0
                    adv = (hi_w - anchor) / anchor * 100.0
                    cls = (anchor - cl_w) / anchor * 100.0
                    dmom = -mom[:m]
                self._d[(pair, side)] = {"ts": sig_ts, "fav": fav, "adv": adv, "cls": cls, "sess": sess,
                                         "vb": vol_b, "mb": _bucket(dmom, _terciles(dmom))}
            self.span[pair] = (int(sig_ts[0]), int(sig_ts[-1]))

    @property
    def n_pairs(self) -> int:
        return len(self.span)

    def covers(self, pair: str, ts: Any) -> bool:
        sp = self.span.get(pair)
        return bool(sp and ts is not None and sp[0] <= int(ts) <= sp[1])

    def buckets_of(self, pair: str, side: str, ts: Any) -> Tuple[Optional[int], Optional[int]]:
        """(volatility, momentum) bucket of the candle an alert fired on."""
        d = self._d.get((pair, side))
        if d is None or ts is None:
            return None, None
        i = int(np.searchsorted(d["ts"], int(ts), side="right")) - 1
        if i < 0 or i >= len(d["ts"]):
            return None, None
        vb, mb = int(d["vb"][i]), int(d["mb"][i])
        return (vb if vb >= 0 else None), (mb if mb >= 0 else None)

    def expected_gross(self, pair: str, side: str, session: Optional[str], sl: float, tp: float, h: int,
                       be: float = 0.0, trail: float = 0.0,
                       vb: Optional[int] = None, mb: Optional[int] = None) -> Optional[Tuple[float, int, int]]:
        """(mean gross P&L %, n, match_level) of the plan over comparable no-alert
        entries.  match_level 3 = session+vol+momentum, 2 = session+vol, 1 = session,
        0 = pair/direction only."""
        key = (pair, side, session, vb, mb, round(sl, 4), round(tp, 4), int(h), round(be, 4), round(trail, 4))
        if key in self._cache:
            return self._cache[key]
        d = self._d.get((pair, side))
        out: Optional[Tuple[float, int, int]] = None
        if d is not None and 0 <= h <= self.max_h:
            total = len(d["ts"])
            has_s = session not in (None, "", "unknown")
            sm = (d["sess"] == session) if has_s else np.ones(total, dtype=bool)
            levels = []
            if has_s and vb is not None and mb is not None:
                levels.append((3, sm & (d["vb"] == vb) & (d["mb"] == mb)))
            if has_s and vb is not None:
                levels.append((2, sm & (d["vb"] == vb)))
            if has_s:
                levels.append((1, sm))
            levels.append((0, np.ones(total, dtype=bool)))
            for lv, mask in levels:
                if int(mask.sum()) >= self._min_rows:
                    gross = _exit_vec(d["fav"][mask][:, :h + 1], d["adv"][mask][:, :h + 1],
                                      d["cls"][mask][:, h], sl, tp, h, be, trail)
                    out = (float(gross.mean()), int(mask.sum()), lv)
                    break
        self._cache[key] = out
        return out


def _pt(plan: Mapping[str, Any]) -> Tuple[float, float, int, float, float]:
    return (float(plan["sl"]), float(plan["tp"]), int(plan["h"]),
            float(plan.get("be") or 0.0), float(plan.get("trail") or 0.0))


def _ev(rows: Sequence[Row], plan: Mapping[str, Any], cost: float) -> Optional[Dict[str, Any]]:
    sl, tp, h, be, tr = _pt(plan)
    return evaluate_plan(rows, sl, tp, h, cost, be, tr)


def alpha_vs_control(
    rows: Sequence[Row], plan: Mapping[str, Any], control: Optional[ControlModel],
    default_cost_pct: float, z: float = 1.645,
) -> Optional[Dict[str, Any]]:
    """Paired difference  (alert P&L) - (matched no-alert control P&L) under the
    SAME plan and cost, with time-clustered standard errors."""
    if control is None:
        return None
    sl, tp, h, be, tr = _pt(plan)
    diffs: List[float] = []
    stamps: List[Optional[float]] = []
    ctl_net: List[float] = []
    levels: Dict[str, int] = {}
    for r in rows:
        if not has_path(r) or not control.covers(str(r.get("pair")), r.get("entry_ts")):
            continue
        res = replay_plan(r, sl, tp, h, default_cost_pct, be, tr)
        if res is None:
            continue
        side = str(r.get("direction")).lower()
        pair = str(r.get("pair"))
        vb, mb = control.buckets_of(pair, side, r.get("entry_ts"))
        cg = control.expected_gross(pair, side, r.get("session"), sl, tp, h, be, tr, vb, mb)
        if cg is None:
            continue
        cost = r.get("realized_cost_pct")
        cost = float(cost) if cost is not None else float(default_cost_pct)
        ctl = cg[0] - cost
        diffs.append(res[1] - ctl)
        ctl_net.append(ctl)
        stamps.append(r.get("entry_ts"))
        lk = str(cg[2])
        levels[lk] = levels.get(lk, 0) + 1
    n = len(diffs)
    if n < 2:
        return {"n": n, "n_blocks": 0, "alpha": None, "alpha_lcb": None, "control_ev": None, "match_levels": levels}
    mean = statistics.fmean(diffs)
    se, nb = cluster_se(diffs, stamps)
    return {
        "n": n, "n_blocks": nb, "alpha": mean,
        "alpha_lcb": mean - z * se if se != float("inf") else float("-inf"),
        "control_ev": statistics.fmean(ctl_net), "match_levels": levels,
    }


# ── Playbook construction ─────────────────────────────────────────────────
def _plan(p: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"sl": float(p["sl"]), "tp": float(p["tp"]), "h": int(p["h"])}
    if float(p.get("be") or 0) > 0:
        out["be"] = float(p["be"])
    if float(p.get("trail") or 0) > 0:
        out["trail"] = float(p["trail"])
    return out


def _stats(e: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not e:
        return {}

    def _v(k: str) -> Any:
        v = e.get(k)
        if k in ("n", "n_blocks") and isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, (int, float)) and v not in (float("inf"), float("-inf")):
            return round(float(v), 4)
        return v
    return {k: _v(k)
            for k in ("n", "n_blocks", "ev", "ev_lcb", "p_profit", "tp_rate", "sl_rate", "timeout_rate", "worst10",
                      "max_dd", "half_min")
            if k in e}


def _z_for(n_tests: int, alpha: float) -> float:
    n = max(1, n_tests)
    return max(1.645, NormalDist().inv_cdf(1.0 - alpha / n))


def _criteria(he: Optional[Mapping[str, Any]], plan: Mapping[str, Any], z: float, P: Mapping[str, Any]) -> Tuple[bool, str]:
    """Out-of-sample criteria a plan must keep meeting: positive EV with a lower
    bound above zero, a drawdown cap and a stability check across the holdout."""
    if not he or he["n"] < P["min_holdout"]:
        return False, "holdout too small"
    if he["ev"] <= 0 or he["se"] == float("inf") or he["ev"] - z * he["se"] <= 0:
        return False, "EV lower bound not above zero"
    sl = float(plan["sl"])
    if he["max_dd"] > P["max_dd_r"] * sl:
        return False, f"drawdown {he['max_dd']:.1f}% exceeds {P['max_dd_r']:g} stops"
    if he["half_min"] < -P["stab_floor_r"] * sl:
        return False, "one half of the holdout lost money"
    return True, ""


def group_rows(rows: Sequence[Row], family_fn: Optional[Callable[[str], str]], family_pool: bool = True) -> Dict[str, List[Row]]:
    groups: Dict[str, List[Row]] = {}
    for r in rows:
        if not has_path(r) or r.get("entry_ts") is None:
            continue
        ak, d = str(r.get("alert_key", "?")), str(r.get("direction", "?"))
        groups.setdefault(f"{ak}|{d}", []).append(r)
        if family_pool and family_fn is not None:
            try:
                groups.setdefault(f"fam:{family_fn(ak)}|{d}", []).append(r)
            except Exception:
                pass
    return groups


def _with_costs(rows: Sequence[Row], cost_pct: float, cost_by_pair: Optional[Mapping[str, float]]) -> List[Row]:
    """Rows without a realized cost get their PAIR's typical cost (spreads and
    slippage differ per pair), else the global default."""
    out = []
    for r in rows:
        if r.get("realized_cost_pct") is None:
            r = dict(r)
            r["realized_cost_pct"] = float((cost_by_pair or {}).get(str(r.get("pair")), cost_pct))
        out.append(r)
    return out


def _size_mult(he: Mapping[str, Any], plan: Mapping[str, Any], P: Mapping[str, Any]) -> float:
    """Advisory size multiple from the EV LOWER bound (never the point estimate)."""
    if he["se"] == float("inf"):
        return 0.0
    lcb = he["ev"] - 1.645 * he["se"]
    if lcb <= 0:
        return 0.0
    m = min(1.0, max(0.1, lcb / P["size_full_lcb"]))
    if he["max_dd"] > P["dd_haircut_r"] * float(plan["sl"]):
        m *= 0.5
    return round(m, 2)


def build_playbook(
    rows: Sequence[Row], *, now_ts: float, prev_state: Optional[Mapping[str, Any]],
    cost_pct: float, fixed: Mapping[str, float],
    params: Optional[Mapping[str, Any]] = None, control: Optional[ControlModel] = None,
    control_note: str = "", family_fn: Optional[Callable[[str], str]] = None,
    cost_by_pair: Optional[Mapping[str, float]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Returns (playbook_blob, new_state, changes, ledger).

    rows   outcome rows (with stored paths) tagged row['source'] in LIVE / SHADOW.
    fixed  {'sl','tp','h'}  the permanent control bracket.
    """
    P = dict(default_params())
    P.update(params or {})
    prev_state = dict(prev_state or {})
    prev_entries: Dict[str, Any] = dict(prev_state.get("entries") or {})
    fixed_sl, fixed_tp, fixed_h = float(fixed["sl"]), float(fixed["tp"]), int(fixed["h"])
    fixed_plan = {"sl": fixed_sl, "tp": fixed_tp, "h": fixed_h}

    rows = _with_costs(rows, cost_pct, cost_by_pair)
    groups = group_rows(rows, family_fn, bool(P["family_pool"]))

    eligible = [k for k, g in groups.items() if len(g) >= P["min_n"]]
    z_adj = _z_for(len(eligible), float(P["family_alpha"]))
    z_rule = _z_for(len(eligible) * int(P["max_rules"]), float(P["family_alpha"]))
    control_ok = control is not None and control.n_pairs > 0
    entries: Dict[str, Any] = {}
    new_entries_state: Dict[str, Any] = {}
    changes: List[Dict[str, Any]] = []
    ledger: List[Dict[str, Any]] = []

    for key in sorted(groups):
        grp = sorted(groups[key], key=lambda r: r["entry_ts"])
        st = dict(prev_entries.get(key) or {})
        champ = dict(st["champion"]) if st.get("champion") else None
        chall = dict(st["challenger"]) if st.get("challenger") else None
        cooldown_until = float(st.get("cooldown_until") or 0.0)
        was_expired = bool(st.get("expired"))
        old_status = st.get("status", "COLLECTING")
        old_restrict = st.get("restrict")
        avoid_streak = int(st.get("avoid_streak") or 0)
        avoid_since = st.get("avoid_since_ts")
        n_live = sum(1 for r in grp if r.get("source") != "SHADOW")
        entry: Dict[str, Any] = {
            "status": "COLLECTING", "n_path": len(grp), "n_live": n_live, "n_shadow": len(grp) - n_live,
            "last_row_ts": int(grp[-1]["entry_ts"]), "restrict": None,
        }
        event: Optional[str] = None
        reason = ""

        if len(grp) < P["min_n"]:
            entry["reason"] = f"need ≥{P['min_n']} path rows (have {len(grp)})"
            entries[key] = entry
            new_entries_state[key] = {"status": "COLLECTING", "champion": champ, "challenger": None,
                                      "cooldown_until": cooldown_until, "expired": was_expired,
                                      "last_version": int(st.get("last_version", 0))}
            continue

        sel = select_plan(grp, cost_pct, fixed_sl, fixed_tp, fixed_h,
                          min_n=P["min_n"], min_holdout=P["min_holdout"], z=z_adj)
        hold_rows: List[Row] = sel.get("_hold_rows") or []
        ctl_eval = evaluate_plan(grp, fixed_sl, fixed_tp, fixed_h, cost_pct)
        entry["control_plan"] = {**_stats(ctl_eval), **fixed_plan}

        diag: Dict[str, int] = {}
        for r in grp:
            dg = diagnose_row(r, fixed_sl, fixed_tp)
            if dg != "WIN":
                diag[dg] = diag.get(dg, 0) + 1
        if diag:
            entry["dominant_failure"] = max(diag.items(), key=lambda kv: kv[1])[0]

        # ── 1. maintain the champion (demote / expire / confirm) ──
        if champ:
            recent = grp[-int(P["demote_window"]):]
            ev_recent = _ev(recent, champ["plan"], cost_pct)
            stale = (now_ts - entry["last_row_ts"]) > P["expire_days"] * _DAY or \
                    (now_ts - float(champ.get("last_confirmed_ts", now_ts))) > P["expire_days"] * _DAY
            if stale:
                champ, event, reason = None, "EXPIRED", "no fresh confirmation within the expiry window"
                was_expired = True
            elif ev_recent and ev_recent["n"] >= P["demote_min"] and ev_recent["ev"] < 0 \
                    and ev_recent["ev"] + 1.28 * ev_recent["se"] < 0:
                champ, event = None, "DEMOTED"
                reason = f"recent EV {ev_recent['ev']:+.2f}% over last {ev_recent['n']} trades"
                cooldown_until = now_ts + P["cooldown_days"] * _DAY
            elif ev_recent and ev_recent["n"] >= P["demote_min"] and entry["last_row_ts"] > int(champ.get("last_data_ts", 0)):
                champ["last_confirmed_ts"] = now_ts
                champ["last_data_ts"] = entry["last_row_ts"]
            if champ and ev_recent:
                champ["recent"] = _stats(ev_recent)

        # ── 2. frozen challenger: re-validate on newer data ──
        cand_plan: Optional[Dict[str, Any]] = None
        if sel["status"] == "VALIDATED" and sel.get("best"):
            cand_plan = _plan(sel["best"])
        if chall:
            cp = chall["plan"]
            he = _ev(hold_rows, cp, cost_pct)
            al = alpha_vs_control(hold_rows, cp, control, cost_pct, z_adj) if control_ok else None
            pass_hold, why_hold = _criteria(he, cp, z_adj, P)
            pass_alpha = (not P["require_control"]) or bool(
                al and al["n_blocks"] >= P["min_blocks"] and al["alpha_lcb"] is not None and al["alpha_lcb"] > 0)
            if pass_hold and pass_alpha:
                chall["streak"] = int(chall.get("streak", 1)) + 1
                chall["last"] = {**_stats(he), "alpha": al}
            else:
                chall, event = None, "REJECTED"
                reason = "frozen plan failed re-validation" + ("" if pass_hold else f" ({why_hold})") + \
                         ("" if pass_alpha else " (alpha vs matched control)")
            if chall and chall["streak"] >= P["promote_consecutive"]:
                fwd = [r for r in grp if r["entry_ts"] > chall["found_ts"]]
                fe = _ev(fwd, cp, cost_pct)
                fwd_ok = bool(fe and fe["n"] >= P["min_forward"] and fe["ev"] > 0)
                beats = True
                if champ:
                    ce = _ev(hold_rows, champ["plan"], cost_pct)
                    beats = bool(he and ce and he["ev"] - ce["ev"] >= P["min_ev_lift"])
                if fe is not None and fwd_ok and beats:
                    prev_ver = int(champ.get("version", 0)) if champ else int(st.get("last_version", 0))
                    champ = {"plan": _plan(cp), "since_ts": now_ts, "last_confirmed_ts": now_ts,
                             "last_data_ts": entry["last_row_ts"], "version": prev_ver + 1,
                             "previous_plan": champ["plan"] if champ else None,
                             "evidence": {**_stats(he), "alpha": al, "forward": _stats(fe), "z": round(z_adj, 3)}}
                    chall = None
                    was_expired = False
                    event = "PROMOTED"
                    reason = f"{P['promote_consecutive']} consecutive passes + {fe['n']} forward trades EV {fe['ev']:+.2f}%"
                elif not beats:
                    chall, event, reason = None, "REJECTED", "does not beat the current champion by the required margin"
                else:
                    chall["waiting"] = "forward evidence"
        elif cand_plan and now_ts >= cooldown_until:
            he = sel.get("holdout")
            ok_c, why_c = _criteria(he, cand_plan, z_adj, P)
            al = alpha_vs_control(hold_rows, cand_plan, control, cost_pct, z_adj) if control_ok else None
            pass_alpha = (not P["require_control"]) or bool(
                al and al["n_blocks"] >= P["min_blocks"] and al["alpha_lcb"] is not None and al["alpha_lcb"] > 0)
            if ok_c and pass_alpha:
                chall = {"plan": cand_plan, "found_ts": now_ts, "streak": 1, "last": {**_stats(he), "alpha": al}}
                event, reason = "CHALLENGER", "validated out of sample; plan frozen for re-validation on newer data"
            else:
                if not ok_c:
                    entry["reason"] = f"candidate rejected: {why_c}"
                else:
                    entry["reason"] = ("no positive alpha vs the matched no-alert control" if control_ok
                                       else f"control baseline unavailable{(' (' + control_note + ')') if control_note else ''}")
                entry["candidate"] = {"plan": cand_plan, **_stats(he), "alpha": al}

        # ── 3. assemble the entry ──
        if champ:
            cp = champ["plan"]
            he = _ev(hold_rows, cp, cost_pct)
            all_e = _ev(grp, cp, cost_pct)
            due = (now_ts - float(champ.get("last_confirmed_ts", now_ts))) > P["reconfirm_days"] * _DAY
            entry.update({
                "status": "RECONFIRM_DUE" if due else "VALIDATED", "plan": cp,
                "version": champ.get("version", 1), "since_ts": champ.get("since_ts"),
                "last_confirmed_ts": champ.get("last_confirmed_ts"),
                "holdout": _stats(he), "all": _stats(all_e), "recent": champ.get("recent"),
                "evidence": champ.get("evidence"), "previous_plan": champ.get("previous_plan"),
            })
            if he:
                entry["size_mult"] = _size_mult(he, cp, P)
            if chall:
                entry["challenger"] = {"plan": chall["plan"], "streak": chall["streak"], "needs": P["promote_consecutive"]}
        elif chall:
            entry.update({"status": "PROMISING", "challenger": {"plan": chall["plan"], "streak": chall["streak"],
                          "needs": P["promote_consecutive"], "last": chall.get("last"), "waiting": chall.get("waiting")},
                          "reason": f"frozen plan re-validating {chall['streak']}/{P['promote_consecutive']}"})
        elif now_ts < cooldown_until:
            entry.update({"status": "DEMOTED", "reason": "champion demoted; cooling down"})
        elif was_expired:
            entry.update({"status": "EXPIRED", "reason": "validated plan expired without fresh confirmation"})
        else:
            s = sel["status"]
            entry["status"] = "COLLECTING" if s == "COLLECTING" else ("PROMISING" if s in ("VALIDATED", "PROMISING") else s)
            if s in ("NO_EDGE", "FAILED_OOS", "PROMISING"):
                entry.setdefault("reason", {"NO_EDGE": "no stop/target/horizon combination is profitable",
                                            "FAILED_OOS": "best train plan failed on unseen data",
                                            "PROMISING": "positive out of sample but lower bound not yet above zero"}[s])
            if s in ("PROMISING", "VALIDATED", "FAILED_OOS") and sel.get("best"):
                entry.setdefault("candidate", {"plan": _plan(sel["best"]), **_stats(sel.get("holdout"))})

        # ── 4. restriction tier (restrict-only: WATCH caps a TAKE, AVOID is proven loss) ──
        recent_ctl = _ev(grp[-int(P["demote_window"]):], fixed_plan, cost_pct)
        proof = bool(
            sel["status"] == "NO_EDGE" and ctl_eval and ctl_eval["n_blocks"] >= P["min_blocks"]
            and ctl_eval["se"] != float("inf") and ctl_eval["ev"] + z_adj * ctl_eval["se"] < 0
            and recent_ctl and recent_ctl["ev"] < 0
        )
        avoid_streak = avoid_streak + 1 if proof else 0
        restrict: Optional[str] = None
        if entry["status"] in _NEGATIVE:
            restrict = "WATCH"
        if avoid_streak >= int(P["avoid_consecutive"]) and ctl_eval is not None and entry["status"] not in SERVED_STATUSES:
            restrict = "AVOID"
            avoid_since = avoid_since or int(now_ts)
            entry["avoid_proof"] = {**_stats(ctl_eval), "ucb": round(ctl_eval["ev"] + z_adj * ctl_eval["se"], 4)}
        else:
            avoid_since = None if restrict != "AVOID" else avoid_since
        entry["restrict"] = restrict
        entry["avoid_streak"] = avoid_streak
        if avoid_since:
            entry["avoid_since_ts"] = avoid_since

        # ── 5. shallow "do not take when ..." rules (restrict-only) ──
        n_rules_tested = 0
        if entry["status"] not in ("COLLECTING",) and entry["restrict"] != "AVOID":
            plan_for_rules = champ["plan"] if champ else fixed_plan
            r_sl, r_tp, r_h, r_be, r_tr = _pt(plan_for_rules)
            items = []
            for r in grp:
                res = replay_plan(r, r_sl, r_tp, r_h, cost_pct, r_be, r_tr)
                if res is not None:
                    items.append((r, res[1]))
            rules, n_rules_tested = mine_rules(items, z=z_rule, max_rules=int(P["max_rules"]))
            if rules:
                entry["avoid_rules"] = rules

        entry["z_used"] = round(z_adj, 3)
        entries[key] = entry
        new_entries_state[key] = {
            "status": entry["status"], "champion": champ, "challenger": chall, "cooldown_until": cooldown_until,
            "last_version": int(champ["version"]) if champ else int(st.get("last_version", 0)),
            "expired": was_expired and champ is None, "restrict": restrict,
            "avoid_streak": avoid_streak, "avoid_since_ts": avoid_since,
        }

        ledger.append({
            "ts": int(now_ts), "key": key, "n_path": len(grp), "plans_tested": _GRID_SIZE,
            "rules_tested": n_rules_tested,
            "groups_tested": len(eligible), "z": round(z_adj, 3), "select_status": sel["status"],
            "decision": event or entry["status"], "restrict": restrict, "reason": reason or entry.get("reason", ""),
        })
        if event or old_status != entry["status"]:
            changes.append({"key": key, "event": event or "STATUS", "from": old_status, "to": entry["status"],
                            "detail": reason, "plan": entry.get("plan") or (chall or {}).get("plan")})
        if old_restrict != restrict:
            changes.append({"key": key, "event": "RESTRICT", "from": old_restrict or "none", "to": restrict or "none",
                            "detail": ("proven losing on two consecutive cycles" if restrict == "AVOID" else ""),
                            "plan": None})

    ver = int(prev_state.get("version", 0))
    if changes:
        ver += 1
    served = {k: e for k, e in entries.items() if e["status"] in SERVED_STATUSES}
    summary: Dict[str, int] = {}
    for e in entries.values():
        summary[e["status"]] = summary.get(e["status"], 0) + 1
    restricted = {k: e["restrict"] for k, e in entries.items() if e.get("restrict")}
    blob = {
        "schema": PLAYBOOK_SCHEMA, "version": ver,
        "label": f"{__import__('time').strftime('%Y-%m-%d', __import__('time').gmtime(now_ts))}.{ver}",
        "generated_ts": int(now_ts),
        "control": {"available": bool(control_ok), "pairs": control.n_pairs if control else 0,
                    "note": control_note, "fixed_plan": fixed_plan},
        "z": round(z_adj, 3), "z_rule": round(z_rule, 3), "groups_tested": len(eligible),
        "entries": entries, "served": sorted(served), "restricted": restricted, "summary": summary,
    }
    new_state = {
        "version": ver, "entries": new_entries_state,
        "cum_hypotheses": int(prev_state.get("cum_hypotheses", 0)) + len(eligible) * (_GRID_SIZE + int(P["max_rules"])),
    }
    return blob, new_state, changes, ledger


# ── Advisor-side lookup (cheap, pure) ─────────────────────────────────────
def playbook_lookup(
    blob: Optional[Mapping[str, Any]], alert_key: str, direction: str, *,
    family_fn: Optional[Callable[[str], str]] = None,
    now_ts: Optional[float] = None, max_age_sec: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Most specific usable playbook entry for an alert, or None.
    A stale playbook (learner outage) is ignored entirely: fail safe = no change."""
    if not blob or not isinstance(blob, Mapping) or blob.get("schema") != PLAYBOOK_SCHEMA:
        return None
    if now_ts is not None and max_age_sec is not None:
        if now_ts - float(blob.get("generated_ts", 0)) > max_age_sec:
            return None
    entries = blob.get("entries") or {}
    d = str(direction).lower()
    keys = [f"{alert_key}|{d}"]
    if family_fn is not None:
        try:
            keys.append(f"fam:{family_fn(alert_key)}|{d}")
        except Exception:
            pass
    for i, k in enumerate(keys):
        e = entries.get(k)
        if e and (e.get("status") in SERVED_STATUSES + _NEGATIVE + ("PROMISING",) or e.get("restrict") or e.get("avoid_rules")):
            return {**e, "key": k, "playbook_version": blob.get("label"), "via_family": i > 0}
    return None


def matching_rule(entry: Optional[Mapping[str, Any]], ctx: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """First 'do not take when ...' rule of an entry that matches this alert's context."""
    for r in (entry or {}).get("avoid_rules") or []:
        if rule_matches(r["rule"], ctx):
            return r
    return None


# ── Human-readable summary (plain text; the caller escapes for Telegram) ──
def format_summary(blob: Mapping[str, Any], changes: Sequence[Mapping[str, Any]],
                   name_fn: Callable[[str], str] = str, max_lines: int = 14) -> str:
    def nm(key: str) -> str:
        fam = key.startswith("fam:")
        body = key[4:] if fam else key
        a, _, d = body.partition("|")
        return f"{'[family] ' if fam else ''}{name_fn(a)} {d.upper()}"

    def pl(p: Optional[Mapping[str, Any]]) -> str:
        return (f"SL {p['sl']:g}% / TP {p['tp']:g}% / {p['h']}c{plan_exit_suffix(p)}") if p else "-"

    ctl = blob.get("control") or {}
    lines = [f"🧠 PLAYBOOK {blob.get('label')} · z={blob.get('z')} · {blob.get('groups_tested')} group(s) tested"]
    lines.append("Control baseline: " + (f"matched, on ({ctl.get('pairs')} pairs)" if ctl.get("available")
                                         else f"UNAVAILABLE {ctl.get('note', '')}".strip()))
    sm = blob.get("summary") or {}
    if sm:
        lines.append(" · ".join(f"{k} {v}" for k, v in sorted(sm.items())))
    if changes:
        lines.append("")
        lines.append("CHANGES")
        for c in list(changes)[:max_lines]:
            lines.append(f"• {nm(c['key'])}: {c['from']} → {c['to']}"
                         + (f" [{c['event']}]" if c.get("event") not in (None, "STATUS") else "")
                         + (f" {pl(c.get('plan'))}" if c.get("plan") else "")
                         + (f" — {c['detail']}" if c.get("detail") else ""))
    served = [(k, blob["entries"][k]) for k in blob.get("served", [])]
    if served:
        lines.append("")
        lines.append("ACTIVE PLANS")
        for k, e in served[:max_lines]:
            h = e.get("holdout") or {}
            sz = f" · size {e['size_mult']:.2f}×" if e.get("size_mult") is not None else ""
            lines.append(f"• {nm(k)}: {pl(e.get('plan'))} · holdout EV {h.get('ev', float('nan')):+.2f}% "
                         f"(n={h.get('n', '?')}){sz} · {e['status']}")
    avoid = [k for k, v in (blob.get("restricted") or {}).items() if v == "AVOID"]
    if avoid:
        lines.append("")
        lines.append("PROVEN LOSERS (AVOID)")
        for k in avoid[:max_lines]:
            lines.append(f"• {nm(k)}: fixed plan EV {(blob['entries'][k].get('control_plan') or {}).get('ev', float('nan')):+.2f}%")
    rules = [(k, e) for k, e in blob.get("entries", {}).items() if e.get("avoid_rules")]
    if rules:
        lines.append("")
        lines.append("DO NOT TAKE WHEN")
        for k, e in rules[:max_lines]:
            r = e["avoid_rules"][0]
            lines.append(f"• {nm(k)}: {r['text']} (EV {r['ev_hold']:+.2f}%, n={r['n_hold']})")
    prom = [(k, e) for k, e in blob.get("entries", {}).items() if e["status"] == "PROMISING" and e.get("challenger")]
    if prom:
        lines.append("")
        lines.append("RE-VALIDATING")
        for k, e in prom[:max_lines]:
            c = e["challenger"]
            lines.append(f"• {nm(k)}: {pl(c['plan'])} · pass {c['streak']}/{c['needs']}")
    lines.append("")
    lines.append("Nothing here changes alert rules. Plans are advisory and restrict-only.")
    return "\n".join(lines)
