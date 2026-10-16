"""brain_helpers — storage-key constants and pure helper functions shared by the Brain modules.

Split out of the original module without logic changes; the original
module remains as a facade re-exporting every name."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from alert_registry import resolve_alert_config_path
from bot_config import cfg
import threshold_engine as engine

_OVERRIDE_COOLDOWN_PREFIX = "brain_override_cooldown:"

CALIBRATION_CURVES_KEY = "brain:calibration_curves"

QUALITY_INPUTS_KEY = "brain:quality_inputs"

KILL_SWITCH_KEY = "brain:kill_switch_active"

MARKET_STATE_MODEL_KEY = "brain:market_state_model"

ML_CALIBRATION_KEY = "brain:ml_calibration_curve"

EXPERIMENT_LEDGER_KEY = "brain_experiment_ledger"

HYPOTHESIS_LEDGER_KEY = "brain_hypothesis_ledger"

ACTION_PLAN_KEY = "brain_action_plan"

def count_brain_trials(ai_metrics: Dict[str, Any], n_recommendations: int) -> Dict[str, int]:
    """How many candidate hypotheses this Brain cycle examined (not just the
    ones that survived): TP/SL zone candidates, alert+direction+regime segments
    and recommendations issued."""
    zp = ai_metrics.get("zone_profiles") or {}
    rg = ai_metrics.get("regime_gate") or {}
    return {
        "zone_candidates": int(zp.get("n_candidates") or 0),
        "regime_segments": int(rg.get("n_evaluated") or 0),
        "recommendations": int(n_recommendations),
        "plan_variants": int((ai_metrics.get("vol_horizon_study") or {}).get("n_candidates") or 0),
    }

def update_experiment_ledger(prev: Optional[Dict[str, Any]], trials: Dict[str, int], now: int) -> Dict[str, Any]:
    """Fold one cycle's trial counts into the persisted ledger. `last` is the
    family size of the latest cycle (what a chance-hit estimate should use);
    `cum` counts repeated looks at the same data since `since_ts` and is only
    a measure of how much has been tried, not of independent hypotheses."""
    prev = prev if isinstance(prev, dict) else {}
    cum_prev_raw = prev.get("cum")
    cum_prev: Dict[str, Any] = cum_prev_raw if isinstance(cum_prev_raw, dict) else {}
    cum = {k: int(cum_prev.get(k, 0)) + int(v) for k, v in trials.items()}
    return {
        "since_ts": int(prev.get("since_ts") or now),
        "cycles": int(prev.get("cycles") or 0) + 1,
        "cum": cum,
        "last": {"ts": int(now), **{k: int(v) for k, v in trials.items()}},
    }

def collect_conclusions(ai_metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The standing conclusions this report rests on, each with a polarity:
    +1 supports using it, -1 argues against, 0 undecided. Only conclusions the
    report actually computed this cycle are returned."""
    out: Dict[str, Dict[str, Any]] = {}
    nev = ai_metrics.get("net_ev_audit") or {}
    if nev.get("status"):
        out["net_ev_model"] = {
            "claim": "the market-state model picks better trades than simple rules",
            "status": nev["status"],
            "polarity": {"validated": 1, "not_better": -1}.get(nev["status"], 0),
            "evidence": {k: nev.get(k) for k in ("delta_vs_best", "delta_lo", "n_test", "best_baseline")},
        }
    ou = ((ai_metrics.get("net_ev_audit") or {}).get("online") or {}).get("online_vs_static") or {}
    if ou.get("status") and ou["status"] != "insufficient":
        out["online_updates"] = {
            "claim": "updating the model after every resolved trade beats leaving it fixed",
            "status": ou["status"],
            "polarity": {"better": 1, "worse": -1}.get(ou["status"], 0),
            "evidence": {"delta": ou.get("delta"), "lo": ou.get("lo")},
        }
    vs = ai_metrics.get("vol_horizon_study") or {}
    if vs.get("status") and vs["status"] != "insufficient":
        out["volatility_plans"] = {
            "claim": "a plan per volatility regime beats one tuned plan",
            "status": vs["status"],
            "polarity": {"supported": 1, "worse": -1}.get(vs["status"], 0),
            "evidence": {k: vs.get(k) for k in ("overall_delta", "overall_lo", "n_hold_pairs")},
        }
    fe = ai_metrics.get("future_edge_record") or {}
    if fe.get("status"):
        out["future_edge"] = {
            "claim": "alerts with a positive Future Edge Score earn more than the rest",
            "status": fe["status"],
            "polarity": {"predictive": 1, "not_predictive": -1}.get(fe["status"], 0),
            "evidence": {k: fe.get(k) for k in ("positive_minus_rest", "lo", "n")},
        }
    for name, p in ((ai_metrics.get("prediction_track_record") or {}).get("predictors") or {}).items():
        out[f"prediction:{name}"] = {
            "claim": f"the stated odds from {name} beat the base rate",
            "status": p.get("status"),
            "polarity": {"skilful": 1, "no_skill": -1}.get(p.get("status"), 0),
            "evidence": {k: p.get(k) for k in ("n", "brier_skill", "ece")},
        }
    for cat, f in (ai_metrics.get("family_accountability") or {}).items():
        if f.get("status") == "suspended":
            pol = -1
        elif f.get("status") == "eligible" and f.get("helped", 0) > f.get("hurt", 0):
            pol = 1
        else:
            pol = 0
        out[f"family:{cat}"] = {
            "claim": f"acting on '{cat}' recommendations helps",
            "status": f.get("status"), "polarity": pol,
            "evidence": {k: f.get(k) for k in ("n", "helped", "hurt", "neutral")},
        }
    return out

def update_hypothesis_ledger(prev: Optional[Dict[str, Any]], conclusions: Dict[str, Dict[str, Any]], now: int, max_events: int = 100) -> Dict[str, Any]:
    """Remember what the Brain concluded and say so when a later cycle reverses it.

    A subject is OVERTURNED when its polarity flips between +1 and -1, and
    LOST SUPPORT when it falls from +/-1 to undecided. Each event keeps the
    evidence before and after and how long the old conclusion stood, and each
    subject counts its flips so a conclusion that keeps reversing reads as
    unstable rather than as a series of discoveries."""
    prev = prev if isinstance(prev, dict) else {}
    subjects: Dict[str, Any] = {k: dict(v) for k, v in (prev.get("subjects") or {}).items()}
    events: List[Dict[str, Any]] = list(prev.get("events") or [])
    new_overturned = 0
    for sid, c in conclusions.items():
        pol = int(c.get("polarity") or 0)
        old = subjects.get(sid)
        if old is None:
            subjects[sid] = {"claim": c.get("claim"), "status": c.get("status"), "polarity": pol,
                             "since": int(now), "flips": 0, "evidence": c.get("evidence")}
            continue
        old_pol = int(old.get("polarity") or 0)
        if old_pol != 0 and pol != old_pol:
            events.insert(0, {
                "ts": int(now), "subject": sid, "claim": c.get("claim"),
                "kind": "overturned" if pol != 0 else "lost_support",
                "from": old.get("status"), "to": c.get("status"),
                "held_sec": int(now) - int(old.get("since") or now),
                "evidence_before": old.get("evidence"), "evidence_after": c.get("evidence"),
            })
            if pol != 0:
                old["flips"] = int(old.get("flips") or 0) + 1
                new_overturned += 1
        if pol != old_pol:
            old["since"] = int(now)
        old.update({"claim": c.get("claim"), "status": c.get("status"), "polarity": pol, "evidence": c.get("evidence")})
        subjects[sid] = old
    return {
        "subjects": subjects, "events": events[:max_events],
        "n_overturned": int(prev.get("n_overturned") or 0) + new_overturned,
        "updated_at": int(now),
    }

def multiple_testing_note(n_tests: int, alpha: float = 0.05) -> Dict[str, Any]:
    """Rough chance-hit estimate for a family of n_tests tests at level alpha:
    about n_tests * alpha would look significant with no real effect at all,
    and a Bonferroni-corrected level would be alpha / n_tests."""
    n = max(0, int(n_tests))
    return {
        "tests_this_cycle": n,
        "alpha": alpha,
        "expected_chance_hits": round(n * alpha, 2),
        "bonferroni_alpha": (alpha / n) if n else alpha,
    }

def _resolve_config_path(alert_key: str) -> Optional[str]:
    """Single registry: delegates to alerts.ALERT_CONFIG_MAP."""
    return resolve_alert_config_path(alert_key)

def _hget_int(data: dict, key: str, default: int = 0) -> int:
    value = data.get(key)
    if value is None:
        value = data.get(key.encode())
    if value is None:
        return default
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    try:
        return int(value)
    except Exception:
        return default

def _to_opt_float(f: Dict[str, str], key: str) -> Optional[float]:
    raw = f.get(key)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None

def _extract_p_value_for_fdr(rec: Dict[str, Any]) -> Optional[float]:
    """Extract a single-hypothesis p-value from a recommendation for the
    Benjamini-Hochberg FDR pass.

    FIX (Issue 5 cleanup): dropped `real_rows` and `min_sample` —
    neither was read anywhere in this function's body. All p-values
    are reconstructed from fields already stamped on the rec itself.
    """
    rtype = rec.get("type")

    # ── Interactions: p_value already stamped by the miner ──
    if rtype == "vote_interaction":
        p = rec.get("p_value")
        return float(p) if isinstance(p, (int, float)) else None

    # ── Repair shop: p_value stamped by the ML diagnostics ──
    if rtype == "repair_shop":
        p = rec.get("p_value")
        return float(p) if isinstance(p, (int, float)) else None

    # ── Calibration: one-sample against the train-split prediction ─
    if rtype == "calibration_divergence":
        n = rec.get("n")
        pred = rec.get("predicted")
        obs = rec.get("observed")
        if (isinstance(n, (int, float)) and isinstance(pred, (int, float)) and isinstance(obs, (int, float))
                and n > 0):
            wins_obs = int(round(obs * n))
            return engine.one_proportion_p_value(wins_obs, int(n), float(pred))
        return None

    # ── Parameter autopsy: one-sample against MIN_WIN_RATE ──
    # Claim: the worst bucket's WR is inconsistent with the target.
    if rtype == "parameter_autopsy":
        n = rec.get("n")
        wr = rec.get("wr")
        if isinstance(n, int) and isinstance(wr, (int, float)) and n > 0:
            wins = int(round(wr * n))
            return engine.one_proportion_p_value(wins, n, cfg.MIN_WIN_RATE)
        return None

    # ── Conditional gating: two-proportion, above-threshold vs below ──
    # Claim: WR differs materially depending on which side of the
    # condition the row falls on.
    if rtype == "conditional_gating":
        a_n, a_wr = rec.get("above_n"), rec.get("above_wr")
        b_n, b_wr = rec.get("below_n"), rec.get("below_wr")
        if (isinstance(a_n, int) and isinstance(b_n, int)
                and isinstance(a_wr, (int, float)) and isinstance(b_wr, (int, float))
                and a_n > 0 and b_n > 0):
            wins_a = int(round(a_wr * a_n))
            wins_b = int(round(b_wr * b_n))
            return engine.two_proportion_p_value(wins_a, a_n, wins_b, b_n)
        return None

    # ── Disable alert: one-sample against BRAIN_ALERT_DISABLE_THRESHOLD_WR ──
    if rtype == "disable_alert":
        n = rec.get("sample_size")
        wr = rec.get("win_rate")
        if isinstance(n, int) and isinstance(wr, (int, float)) and n > 0:
            wins = int(round(wr * n))
            p0 = getattr(cfg, "BRAIN_ALERT_DISABLE_THRESHOLD_WR", 0.40)
            return engine.one_proportion_p_value(wins, n, p0)
        return None

    # ── Recovered alert: one-sample against MIN_WIN_RATE ──
    if rtype == "recovered_alert":
        n = rec.get("sample_size")
        wr = rec.get("win_rate")
        if isinstance(n, int) and isinstance(wr, (int, float)) and n > 0:
            wins = int(round(wr * n))
            return engine.one_proportion_p_value(wins, n, cfg.MIN_WIN_RATE)
        return None

    # ── Config regression: two-proportion, prev vs current ──
    if rtype in ("config_regression", "config_improvement"):
        prev_n, cur_n = rec.get("prev_n"), rec.get("cur_n")
        prev_wr, cur_wr = rec.get("prev_wr"), rec.get("cur_wr")
        if (isinstance(prev_n, int) and isinstance(cur_n, int)
                and isinstance(prev_wr, (int, float)) and isinstance(cur_wr, (int, float))
                and prev_n > 0 and cur_n > 0):
            wins_prev = int(round(prev_wr * prev_n))
            wins_cur = int(round(cur_wr * cur_n))
            return engine.two_proportion_p_value(wins_cur, cur_n, wins_prev, prev_n)
        return None

    # ── Three-metric close-vs-MFE gap: exact McNemar (paired) ──
    if rtype == "three_metric_evaluation":
        mfe_only = rec.get("mfe_only")
        close_only = rec.get("close_only")
        if isinstance(mfe_only, int) and isinstance(close_only, int):
            return engine.mcnemar_exact_p(mfe_only, close_only)
        return None

    # ── Permutation importance: top-signal t-test against 0 ──
    # Approximate: the shuffle distribution gives a standard error for the
    if rtype == "permutation_importance":
        # A correct permutation-based p-value would require the raw
        # shuffle distribution (fraction of |shuffled_importance| >=
        # |observed_importance|), which is not carried on the rec —
        # only mean and std are. Reconstructing a z-score from those
        # and treating it as a p-value over-rejects under BH, so we
        # exclude permutation_importance from the FDR pass entirely.
        return None

    # ── Fallthrough: no clean single-hypothesis claim ──
    return None
