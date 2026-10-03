"""brain_helpers — storage-key constants and pure helper functions shared by the Brain modules.

Split out of the original module without logic changes; the original
module remains as a facade re-exporting every name."""
from __future__ import annotations
from typing import Any, Dict, Optional
from alert_registry import resolve_alert_config_path
from bot_config import cfg
import threshold_engine as engine

_OVERRIDE_COOLDOWN_PREFIX = "brain_override_cooldown:"

CALIBRATION_CURVES_KEY = "brain:calibration_curves"

QUALITY_INPUTS_KEY = "brain:quality_inputs"

KILL_SWITCH_KEY = "brain:kill_switch_active"

MARKET_STATE_MODEL_KEY = "brain:market_state_model"

ML_CALIBRATION_KEY = "brain:ml_calibration_curve"

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
