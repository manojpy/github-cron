"""Per-pair trade advice for Telegram alerts.

Pure functions, no I/O and no config access: every threshold is a parameter,
so the logic is unit-testable and cannot disturb alert dispatch.

advise_pair() turns the brain's trade-quality dicts (``tq``) for the alert
families that fired on one pair/direction into:

  * a conviction % (a transparent blend, NOT the raw P(profit)),
  * a TAKE / WATCH / AVOID verdict,
  * the Brain / Edge / Why / Risk / SL-TP lines shown in the message.

Conviction blend (0-100, only when at least one family has brain data):
    40% expected value   (net EV +0.30% or better = full marks)
    20% regime win rate  (neutral 50 when unknown)
    20% TP1-first rate   (neutral 50 when unknown)
    10% gate margin      (confluence score above the required minimum)
    10% bias alignment   (with = 100, neutral = 50, against = 0)
Evidence limits: with only SHADOW / INSUFFICIENT evidence conviction is capped
(default 55) so nothing can reach TAKE before it has real results.

Hard rules (applied after the score):
    net EV < 0                                -> AVOID
    quality gate verdict BLOCKED              -> AVOID
    against the market bias AND OI/funding failed -> AVOID
    no brain data at all (no conviction % is ever invented):
        blocked, technicals < 75%, or counter-trend with OI/funding failed -> AVOID
        counter-trend (OI ok) -> WATCH only if technicals >= the unproven bar, else AVOID
        technicals >= the unproven bar, not counter-trend, OI/funding ok,
            no opposing signal                 -> TAKE (small size, unproven)
        anything else                          -> WATCH
    opposing signals on the same pair         -> TAKE is downgraded to WATCH
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

TAKE, WATCH, AVOID = "TAKE", "WATCH", "AVOID"
VERDICT_LABEL = {TAKE: "✅ TAKE", WATCH: "🟡 WATCH", AVOID: "⛔ AVOID"}

VOTE_LABELS = {
    "base_trend": "trend",
    "ichimoku_cloud": "Ichimoku cloud",
    "rma_cloud": "RMA cloud",
    "dynamic_flow_ribbon": "market flow",
    "ppo_cross": "PPO cross",
    "rsi_guard": "RSI",
    "tk_guard": "Tenkan/Kijun",
    "adx": "ADX",
    "rvol": "relative volume",
    "cpr": "CPR",
    "oi_funding": "OI/funding",
    "order_block": "order block",
    "adx_strength": "trend strength",
    "atr_percentile": "volatility",
    "volume_percentile": "volume",
    "ppo_gate_momentum": "PPO momentum",
    "rsi_guard_momentum": "RSI momentum",
    "rma_cloud_momentum": "cloud momentum",
    "vwap_momentum": "VWAP momentum",
}

_CONFIDENCE = {"ACTIONABLE": "HIGH", "ELIGIBLE": "MEDIUM"}
_LIMITED_STATES = ("INSUFFICIENT", "SHADOW", "")


@dataclass
class PairAdvice:
    verdict: str
    conviction: Optional[int]          # None = no brain data (never faked)
    confidence: str                    # HIGH / MEDIUM / LOW
    brain_line: str
    edge_line: str
    why_line: str
    risk_label: str                    # "Risk" or "Missing"
    risk_line: str
    plan_line: str
    size_line: str = ""
    notes: List[str] = field(default_factory=list)
    qualifier: str = ""                # e.g. "small size" for an unproven TAKE

    @property
    def label(self) -> str:
        base = VERDICT_LABEL[self.verdict]
        return f"{base} ({self.qualifier})" if self.qualifier else base

def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None          # drop NaN

def _mean(vals: Sequence[Optional[float]]) -> Optional[float]:
    clean = [v for v in vals if v is not None]
    return sum(clean) / len(clean) if clean else None

def _best_evidence(tqs: Sequence[Dict[str, Any]]) -> str:
    rank = {"": 0, "INSUFFICIENT": 0, "SHADOW": 1, "ELIGIBLE": 2, "ACTIONABLE": 3}
    best = ""
    for t in tqs:
        s = str(t.get("evidence_state") or "").upper()
        if rank.get(s, 0) >= rank.get(best, 0):
            best = s
    return best

def _plan(tqs: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Trade plan from the family with the most outcome history behind it."""
    best, best_n, best_validated = None, -1, False
    for t in tqs:
        zone = t.get("trade_zone")
        validated = isinstance(zone, dict)
        p = zone if validated else t.get("trade_plan")
        if not isinstance(p, dict):
            continue
        if validated:
            sl, tp1, tp2 = (_num(p.get(k)) for k in ("sl_pct", "tp1_pct", "tp2_pct"))
        else:
            sl, tp1, tp2 = (_num(p.get(k)) for k in
                            ("sl_suggested_pct", "tp1_suggested_pct", "tp2_suggested_pct"))
        if sl is None or tp1 is None or tp2 is None or sl <= 0:
            continue

        n = int(_num(p.get("n")) or 0)
        _ntf = _num(p.get("n_tp_first"))
        tp_n = int(_ntf) if _ntf is not None else n
        # A validated zone always beats an unvalidated plan; within a class,
        # the one with the most history behind it wins.
        if (validated, n) > (best_validated, best_n):
            best, best_n, best_validated = {
                "sl": sl, "tp1": tp1, "tp2": tp2,
                "tp_first": _num(p.get("tp_first_rate")), "n": n,
                "tp_n": tp_n,
                "validated": validated,
            }, n, validated
    return best

def _px(v: float) -> str:
    """Price with enough decimals for low-priced pairs (levels sit well under 1% apart)."""
    v = float(v)
    if v >= 1000:
        return f"${v:,.2f}"
    if v >= 1:
        return f"${v:,.4f}"
    if v >= 0.01:
        return f"${v:,.5f}"
    return f"${v:,.7f}"

def _level(price: float, direction: Optional[str], pct: float, favourable: bool) -> float:
    """Absolute price of a level `pct` percent from entry. favourable=True is the
    profit side (TP): above entry for a buy, below for a sell; False is the stop."""
    sign = 1.0 if (str(direction).lower() == "buy") == favourable else -1.0
    return price * (1.0 + sign * float(pct) / 100.0)

def _plan_line(plan: Optional[Dict[str, float]], default_sl: float, rr: float,
               direction: Optional[str] = None, price: Optional[float] = None) -> str:
    use_px = isinstance(price, (int, float)) and price > 0 and direction in ("buy", "sell")
    if plan:
        r1, r2 = plan["tp1"] / plan["sl"], plan["tp2"] / plan["sl"]
        tag = " · validated zone" if plan.get("validated") else ""
        if use_px:
            return (f"🛡 SL {_px(_level(price, direction, plan['sl'], False))}"
                    f" | TP1 {_px(_level(price, direction, plan['tp1'], True))} ({r1:.1f}R)"
                    f" | TP2 {_px(_level(price, direction, plan['tp2'], True))} ({r2:.1f}R){tag}")
        return (f"🛡 SL -{plan['sl']:.2f}% | TP1 +{plan['tp1']:.2f}% ({r1:.1f}R)"
                f" | TP2 +{plan['tp2']:.2f}% ({r2:.1f}R){tag}")
    if use_px:
        return (f"🛡 SL {_px(_level(price, direction, default_sl, False))}"
                f" | TP {_px(_level(price, direction, default_sl * rr, True))} "
                f"({rr:.1f}R) · default plan, no history")
    return (f"🛡 SL -{default_sl:.2f}% | TP +{default_sl * rr:.2f}% "
            f"({rr:.1f}R) · default plan, no history")

def _support_labels(votes: Optional[Dict[str, bool]], weights: Optional[Dict[str, float]],
                    passing: bool, limit: int) -> List[str]:
    if not votes:
        return []
    w = weights or {}
    names = [n for n, ok in votes.items() if bool(ok) == passing]
    names.sort(key=lambda n: (-float(w.get(n, 0.0)), n))
    return [VOTE_LABELS.get(n, n.replace("_", " ")) for n in names[:limit]]

def _support_text(votes: Optional[Dict[str, bool]], weights: Optional[Dict[str, float]],
                  support: List[str]) -> str:
    """Informative support wording. When every high-weight check passes say so
    (the top-3 list was identical for nearly every alert and told nothing)."""
    if not votes:
        return ""
    w = weights or {}
    core_fail = [n for n, ok in votes.items() if not ok and float(w.get(n, 0.0)) >= 2.0]
    if w and not core_fail:
        return " All high-weight checks pass."
    return f" Supported by {', '.join(support)}." if support else ""

def _advise_pair_core(
    *,
    direction: str,
    score: Optional[float],
    total: Optional[float],
    required: Optional[float],
    votes: Optional[Dict[str, bool]],
    tqs: Sequence[Optional[Dict[str, Any]]],
    bias: str,                         # "with" | "against" | "neutral"
    conflicting: bool = False,
    take_min: float = 70.0,
    watch_min: float = 50.0,
    shadow_cap: float = 55.0,
    default_sl_pct: float = 0.5,
    default_rr: float = 2.0,
    vote_weights: Optional[Dict[str, float]] = None,
    unproven_take_min_pct: Optional[float] = None,   # 0-100; None = feature off
    unproven_size: float = 0.25,
    price: Optional[float] = None,
) -> PairAdvice:
    data = [t for t in tqs if isinstance(t, dict) and _num(t.get("net_ev")) is not None]
    blocked = any(isinstance(t, dict) and str(t.get("verdict")) == "BLOCKED" for t in tqs)
    pct = (score / total) if (score is not None and total) else None
    oi_failed = votes is not None and votes.get("oi_funding") is False
    against = bias == "against"

    # ── technical strength wording ──
    if pct is None:
        tech = "Technical setup"
    elif pct >= 0.75:
        tech = "Strong technical setup"
    elif pct >= 0.60:
        tech = "Solid technical setup"
    else:
        tech = "Marginal technical setup"

    plan = _plan(data)  
    plan_line = _plan_line(plan, default_sl_pct, default_rr, direction, price)
    support = _support_labels(votes, vote_weights, True, 3)
    weak = _support_labels(votes, vote_weights, False, 2)
    support_txt = _support_text(votes, vote_weights, support)

    # ═════════ no brain data at all ═════════
    if not data:
        strong_bar = (unproven_take_min_pct if unproven_take_min_pct is not None else 90.0) / 100.0
        qualifier = ""
        if blocked or pct is None or pct < 0.75:
            verdict = AVOID
        elif against and oi_failed:
            verdict = AVOID
        elif against:
            # counter-trend: only a very strong setup earns a WATCH
            verdict = WATCH if pct >= strong_bar else AVOID
        elif (unproven_take_min_pct is not None and pct >= strong_bar
              and not oi_failed and not conflicting):
            verdict, qualifier = TAKE, "small size"
        else:
            verdict = WATCH
        if against and oi_failed:
            reason = "it fights the market bias, OI/funding disagrees and it has no track record"
        elif against:
            reason = "it is counter-trend and has no track record yet"
        elif verdict == AVOID:
            reason = "technicals are not strong enough to act on without a track record"
        elif verdict == TAKE:
            reason = ""
        elif oi_failed:
            reason = "OI/funding disagrees and there is no track record yet"
        else:
            reason = "there is no track record for this setup yet"
        no_data_risks = []
        if against:
            no_data_risks.append("against the market bias")
        if oi_failed:
            no_data_risks.append("OI/funding check failed")
        if conflicting:
            no_data_risks.append("opposing signal on the same pair")
        if weak and not no_data_risks:
            no_data_risks.append(f"weak: {', '.join(weak)}")
        if no_data_risks:
            label = "Risk"
            risk_line = "; ".join(no_data_risks) + "; no reliable profitability evidence."
        elif verdict == WATCH:
            label, risk_line = "Missing", "Reliable profitability evidence."
        elif verdict == TAKE:
            label, risk_line = "Risk", "unproven setup, no reliable profitability evidence."
        else:
            label = "Risk"
            risk_line = "weak technicals and no reliable profitability evidence."
        if verdict == TAKE:
            why = (f"{tech} with no track record yet, so the technical confluence "
                   f"is the only evidence; size small.{support_txt}")
            size_line = f"📐 Size {unproven_size:.2f}× (unproven setup, advisory)"
        else:
            why = f"{tech}, but {reason}.{support_txt}"
            size_line = ""
        return PairAdvice(
            verdict=verdict, conviction=None, confidence="LOW",
            brain_line="🧠 Brain: no track record for this setup yet",
            edge_line="📊 Edge: Insufficient history",
            why_line=f"💡 Why: {why}",
            risk_label=label, risk_line=risk_line, plan_line=plan_line,
            size_line=size_line, qualifier=qualifier,
        )

    # ═════════ conviction blend ═════════
    ev = _mean([_num(t["net_ev"]) for t in data])
    p_vals = [_num(t.get("p_ev_positive_ensemble", t.get("p_ev_positive"))) for t in data]
    p_vals = [p for p in p_vals if p is not None]
    p_mean = _mean(p_vals)
    reg_vals = [_num(t.get("regime_wr")) for t in data]
    reg_vals = [r for r in reg_vals if r is not None]
    reg_wr = _mean(reg_vals)
    reg_name = next((str(t.get("regime_name")) for t in data if t.get("regime_name")), "")
    tp_first = plan["tp_first"] if plan else None
    tp_n = plan.get("tp_n", plan["n"]) if plan else 0
    s_ev = _clamp(50.0 + (ev / 0.30) * 50.0) if ev is not None else 50.0
    s_reg = _clamp(reg_wr * 100.0) if reg_wr is not None else 50.0
    s_tp1 = _clamp(tp_first * 100.0) if tp_first is not None else 50.0
    if pct is not None and score is not None and total:
        if required is not None and total > required:
            s_gate = _clamp((score - required) / (total - required) * 100.0)
        else:
            s_gate = _clamp((pct - 0.60) / 0.40 * 100.0)
    else:
        s_gate = 50.0
    s_bias = {"with": 100.0, "against": 0.0}.get(bias, 50.0)
    conv = 0.40 * s_ev + 0.20 * s_reg + 0.20 * s_tp1 + 0.10 * s_gate + 0.10 * s_bias

    state = _best_evidence(data)
    limited = state in _LIMITED_STATES
    if limited:
        conv = min(conv, shadow_cap)
    conviction = int(round(_clamp(conv)))

    # ── verdict ──
    if blocked:
        verdict, why_neg = AVOID, "the quality gate vetoed it"
    elif ev is not None and ev < 0:
        verdict, why_neg = AVOID, "its historical expected value is negative"
    elif against and oi_failed:
        verdict, why_neg = AVOID, "it fights the market bias and OI/funding disagrees"
    elif conviction >= take_min:
        verdict, why_neg = TAKE, ""
    elif conviction >= watch_min:
        verdict, why_neg = WATCH, ""
    else:
        verdict, why_neg = AVOID, "historical evidence does not justify the risk"
    if conflicting and verdict == TAKE:
        verdict = WATCH

    confidence = _CONFIDENCE.get(state, "LOW")

    if p_mean is None or (state in ("SHADOW", "") and abs(p_mean - 0.5) < 0.05):
        p_txt = "n/a (thin data)"
    else:
        p_txt = f"{round(p_mean * 100)}%"
    brain_line = f"🧠 Brain: P(EV>0) {p_txt}, EV {ev:+.2f}%, {confidence} confidence"
    ev_label = {"ACTIONABLE": "Strong evidence", "ELIGIBLE": "Moderate evidence",
                "SHADOW": "Limited evidence (shadow)"}.get(state, "Insufficient history")
    edge_bits = [ev_label]
    if reg_wr is not None:
        edge_bits.append(f"{reg_name + ' ' if reg_name else ''}regime WR {round(reg_wr * 100)}%")
    if tp_first is not None:
        edge_bits.append(f"target-before-stop {round(tp_first * 100)}% (n={tp_n})")
    edge_line = "📊 Edge: " + " · ".join(edge_bits)

    if verdict == TAKE:
        why = f"{tech} backed by a positive historical edge.{support_txt}"
    elif verdict == WATCH:
        if conflicting:
            reason = "opposing signals exist on this pair"
        elif limited:
            reason = "historical evidence is still limited"
        elif against:
            reason = "it runs against the market bias"
        else:
            reason = "conviction is below the take threshold"
        why = f"{tech}, but {reason}.{support_txt}"
    else:
        why = f"{tech}, but {why_neg}.{support_txt}"

    risks: List[str] = []
    if against:
        risks.append("against the market bias")
    if conflicting:
        risks.append("opposing signal on the same pair")
    if oi_failed:
        risks.append("OI/funding check failed")
    if reg_wr is not None and reg_wr < 0.45:
        risks.append(f"{reg_name + ' ' if reg_name else ''}regime WR only {round(reg_wr * 100)}%")
    if tp_first is not None and tp_n >= 20 and tp_first < 0.40:
        risks.append(f"target reached before stop only {round(tp_first * 100)}%")
    if any(t.get("drift_warning") for t in data):
        risks.append("recent win-rate drift")
    if ev is not None and ev < 0:
        risks.append(f"expected value {ev:+.2f}%")
    if not risks:
        risks.append("setup invalidated by a close beyond the stop")
    risk_line = "; ".join(risks[:3]) + "."

    size_line = ""
    sizes: List[float] = [
        s for t in data
        if (s := _num(t.get("size_hint"))) is not None
    ]
    if sizes and verdict != AVOID:
        size_line = f"📐 Size {min(sizes):.2f}× (advisory)"

    return PairAdvice(
        verdict=verdict, conviction=conviction, confidence=confidence,
        brain_line=brain_line, edge_line=edge_line, why_line=f"💡 Why: {why}",
        risk_label="Risk", risk_line=risk_line, plan_line=plan_line, size_line=size_line,
    )

# ── Playbook overlay (learner output; restrict-only by default) ───────────
_PB_NEGATIVE = ("NO_EDGE", "FAILED_OOS", "DEMOTED", "EXPIRED")

def _pb_plan_text(p: Dict[str, Any], direction: Optional[str] = None,
                  price: Optional[float] = None) -> str:
    if isinstance(price, (int, float)) and price > 0 and direction in ("buy", "sell"):
        txt = (f"SL {_px(_level(price, direction, float(p['sl']), False))} | "
               f"TP {_px(_level(price, direction, float(p['tp']), True))} | "
               f"exit by {int(p['h'])} candles")
        if float(p.get("be") or 0) > 0:
            txt += f" | stop to breakeven at {_px(_level(price, direction, float(p['be']), True))}"
    else:
        txt = f"SL -{float(p['sl']):g}% | TP +{float(p['tp']):g}% | exit by {int(p['h'])} candles"
        if float(p.get("be") or 0) > 0:
            txt += f" | stop to breakeven at +{float(p['be']):g}%"
    if float(p.get("trail") or 0) > 0:
        txt += f" | trail {float(p['trail']):g}%"
    return txt

def _pb_rule_hit(playbook: Dict[str, Any], ctx: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not ctx:
        return None
    from rulemine import rule_matches
    for r in playbook.get("avoid_rules") or []:
        if rule_matches(r["rule"], ctx):
            return r
    return None

def advise_pair(
    *,
    playbook: Optional[Dict[str, Any]] = None,
    playbook_mode: str = "off",          # off | shadow | live
    playbook_allow_upgrade: bool = False,
    playbook_ctx: Optional[Dict[str, Any]] = None,   # {"session", "conf", "adx"} of THIS alert
    **kwargs: Any,
) -> PairAdvice:
    """Core advice, optionally overlaid with the learner's validated playbook entry.

    off / shadow : the advice is byte-for-byte the core advice (shadow only records).
    live         : RESTRICT-ONLY.
        * a served champion plan replaces the displayed SL/TP (and size, from the EV lower bound);
        * restrict=AVOID (proven loser, two cycles)     -> TAKE/WATCH read AVOID;
        * restrict=WATCH, or a status with no edge / demoted / expired -> TAKE reads WATCH;
        * a matching "do not take when ..." rule       -> TAKE reads WATCH;
        * a champion whose EV lower bound is not above zero gets no size -> TAKE reads WATCH.
      A WATCH->TAKE upgrade needs playbook_allow_upgrade AND a VALIDATED champion
      AND no restriction or rule hit AND no conflicts.
    Any problem in the overlay falls back to the core advice."""
    advice = _advise_pair_core(**kwargs)
    if playbook_mode != "live" or not isinstance(playbook, dict):
        return advice
    try:
        status = str(playbook.get("status", ""))
        restrict = playbook.get("restrict")
        ver = playbook.get("playbook_version") or ""
        notes: List[str] = []
        capped = False
        if status in ("VALIDATED", "RECONFIRM_DUE") and isinstance(playbook.get("plan"), dict):
            h = playbook.get("holdout") or {}
            tag = "validated" if status == "VALIDATED" else "validation overdue"
            ev = h.get("ev")
            ev_txt = f", out-of-sample EV {float(ev):+.2f}% (n={int(h.get('n', 0))})" if isinstance(ev, (int, float)) else ""
            advice.plan_line = f"🛡 {_pb_plan_text(playbook['plan'], kwargs.get('direction'), kwargs.get('price'))} · {tag} plan{ev_txt} [{ver}]"
            sm = playbook.get("size_mult")
            if isinstance(sm, (int, float)):
                if sm <= 0:
                    capped = True
                    notes.append("the plan's EV lower bound is no longer above zero")
                elif advice.verdict != AVOID:
                    advice.size_line = f"📐 Size {float(sm):.2f}× (from EV lower bound, advisory)"
        elif status in _PB_NEGATIVE:
            capped = True
            notes.append({"NO_EDGE": "no stop/target combination has been profitable for this setup",
                          "FAILED_OOS": "its best plan failed on unseen data",
                          "DEMOTED": "its validated plan was just demoted",
                          "EXPIRED": "its validated plan expired unconfirmed"}[status])
        if restrict == "WATCH":
            capped = True
        rule = _pb_rule_hit(playbook, playbook_ctx)
        if rule is not None:
            capped = True
            notes.append(f"it lost money on unseen trades when {rule['text']}")
        if restrict == "AVOID" and advice.verdict in (TAKE, WATCH):
            advice.verdict = AVOID
            advice.qualifier = ""
            advice.size_line = ""
            advice.risk_label = "Risk"
            advice.risk_line = "playbook: proven losing setup (two consecutive learner cycles); " + advice.risk_line
        elif capped:
            if advice.verdict == TAKE:
                advice.verdict = WATCH
                advice.qualifier = ""
                advice.size_line = ""
            if notes:
                advice.risk_label = "Risk"
                advice.risk_line = "playbook: " + "; ".join(notes) + "; " + advice.risk_line
        elif (playbook_allow_upgrade and status == "VALIDATED" and advice.verdict == WATCH
              and not kwargs.get("conflicting") and kwargs.get("bias") != "against"
              and (advice.conviction is None or advice.conviction >= float(kwargs.get("watch_min", 50.0)))):
            advice.verdict = TAKE
            advice.qualifier = "validated plan"
    except Exception:
        return _advise_pair_core(**kwargs)
    return advice


