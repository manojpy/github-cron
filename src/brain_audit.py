#!/usr/bin/env python3
"""
brain_audit.py — Brain Audit Layer: data-integrity, history-coverage,
analysis-health, and outcome-reconciliation control plane.

Sits ABOVE all Brain analysis components. No analysis runs without
the audit layer first validating whether the data population is
sufficient for that analysis class. No silent failure goes unreported.

Architecture position:

    ┌──────────────────────────────────────────┐
    │            BRAIN AUDIT LAYER             │
    │                                          │
    │  • Data integrity & reconciliation       │
    │  • History coverage validation           │
    │  • Analysis health tracking              │
    │  • Statistical sufficiency gates         │
    │  • Schema migration advisory             │
    └──────────────────┬───────────────────────┘
                       │
    ┌──────────────────┼──────────────────────┐
    ↓                  ↓                      ↓
 Outcome Data     Brain Analysis       Recommendations
    ↓                  ↓                      ↓
 Resolved         Statistics           Action Gate
    ↓                  ↓                      ↓
 Archive          Diagnostics          Apply / Reject
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from bot_config import cfg, format_ist_time

_log = logging.getLogger("macd_bot")


# ══════════════════════════════════════════════════════════════════════
#  ENUMS & CONSTANTS
# ══════════════════════════════════════════════════════════════════════

class HealthStatus(str, Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"
    FAILED = "FAILED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    SKIPPED = "SKIPPED"

    @property
    def icon(self) -> str:
        return {
            HealthStatus.OK: "🟢",
            HealthStatus.DEGRADED: "🟡",
            HealthStatus.UNAVAILABLE: "⚪",
            HealthStatus.FAILED: "🔴",
            HealthStatus.INSUFFICIENT_DATA: "🟠",
            HealthStatus.SKIPPED: "⚪",
        }[self]

    @property
    def allows_analysis(self) -> bool:
        """Whether this status permits running the associated analysis."""
        return self in (HealthStatus.OK, HealthStatus.DEGRADED)

    @property
    def allows_recommendation(self) -> bool:
        """Whether this status permits emitting actionable recommendations."""
        return self == HealthStatus.OK


class DataCoverage(str, Enum):
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    SEVERELY_LIMITED = "SEVERELY_LIMITED"
    CRITICAL = "CRITICAL"

    @property
    def icon(self) -> str:
        return {
            DataCoverage.FULL: "🟢",
            DataCoverage.PARTIAL: "🟡",
            DataCoverage.SEVERELY_LIMITED: "🟠",
            DataCoverage.CRITICAL: "🔴",
        }[self]


class RecommendationTier(int, Enum):
    """Evidence tiers for recommendations — prevents mixing descriptive
    observations with actionable changes."""
    DESCRIPTIVE = 1        # "Here's what we observed"
    STATISTICAL = 2        # "This is statistically notable"
    CANDIDATE = 3          # "This might be worth changing"
    ACTIONABLE = 4         # "Apply this change"

    @property
    def label(self) -> str:
        return {
            RecommendationTier.DESCRIPTIVE: "📊 Observation",
            RecommendationTier.STATISTICAL: "📈 Statistical Signal",
            RecommendationTier.CANDIDATE: "🧪 Simulation — NOT Validated",
            RecommendationTier.ACTIONABLE: "✅ Validated Recommendation",
        }[self]


# Minimum sample requirements per analysis class
_ANALYSIS_MIN_SAMPLES: Dict[str, int] = {
    "walk_forward": 60,
    "monte_carlo": 100,
    "regime_comparison": 40,
    "temporal_drift": 40,
    "parameter_optimization": 30,
    "layered_window": 60,
    "hierarchical": 50,
    "weight_optimizer": 100,
    "calibration": 30,
    "cusum": 20,
    "threshold_recommendation": 20,
    "ev_assessment": 20,
    "vote_interactions": 20,
    "counterfactual": 10,
    "repair_shop": 20,
    "config_regression": 40,
}

# History coverage requirements (days) per analysis class
_ANALYSIS_MIN_HISTORY_DAYS: Dict[str, int] = {
    "walk_forward": 14,
    "monte_carlo": 21,
    "regime_comparison": 7,
    "temporal_drift": 28,
    "layered_window": 30,
    "hierarchical": 14,
    "weight_optimizer": 21,
    "config_regression": 14,
}


# ══════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class OutcomeReconciliation:
    """Tracks the full lifecycle of outcomes from alert to Brain input."""
    pending_count: Optional[int] = None
    resolved_this_run: Optional[int] = None
    archived_this_run: Optional[int] = None
    total_archived: Optional[int] = None
    loaded_by_brain: int = 0
    archive_rejects_stale_schema: int = 0
    archive_rejects_signal_only: int = 0
    archive_rejects_malformed: int = 0
    archive_rejects_duplicate: int = 0
    archive_rejects_out_of_window: int = 0
    shadow_loaded: int = 0
    status: HealthStatus = HealthStatus.OK
    notes: List[str] = field(default_factory=list)

    @property
    def is_consistent(self) -> bool:
        return self.status in (HealthStatus.OK, HealthStatus.DEGRADED)

    def to_report_lines(self) -> List[str]:
        def _n(v: Optional[int]) -> str:
            return "n/a" if v is None else str(v)

        lines = [
            f"   Pending outcomes: {_n(self.pending_count)}",
            f"   Resolved this run: {_n(self.resolved_this_run)}",
            f"   Archived this run: {_n(self.archived_this_run)}",
            f"   Archive lines read: {_n(self.total_archived)}",
            f"   Loaded by Brain: {self.loaded_by_brain}",
            f"   Shadow loaded: {self.shadow_loaded}",
        ]
        if self.archive_rejects_stale_schema > 0:
            lines.append(
                f"   ⚠️ Stale-schema rows excluded: {self.archive_rejects_stale_schema}"
            )
        if self.archive_rejects_signal_only > 0:
            lines.append(
                f"   Signal-only rows excluded: {self.archive_rejects_signal_only}"
            )
        if self.archive_rejects_malformed > 0:
            lines.append(
                f"   Malformed rows excluded: {self.archive_rejects_malformed}"
            )
        lines.append(f"   Status: {self.status.icon} {self.status.value}")
        return lines


@dataclass
class HistoryCoverage:
    """Validates whether the loaded data covers the requested time window."""
    requested_days: int = 0
    actual_days: float = 0.0
    oldest_ts: Optional[float] = None
    newest_ts: Optional[float] = None
    n_rows: int = 0
    coverage: DataCoverage = DataCoverage.CRITICAL
    warnings: List[str] = field(default_factory=list)

    @property
    def coverage_ratio(self) -> float:
        if self.requested_days <= 0:
            return 0.0
        return min(1.0, self.actual_days / self.requested_days)

    def to_report_lines(self) -> List[str]:
        lines = [
            f"   History requested: {self.requested_days} days",
            f"   History available: {self.actual_days:.1f} days",
            f"   Coverage: {self.coverage_ratio:.0%}",
        ]
        if self.oldest_ts:
            lines.append(f"   Oldest trade: {format_ist_time(self.oldest_ts)}")
        if self.newest_ts:
            lines.append(f"   Newest trade: {format_ist_time(self.newest_ts)}")
        lines.append(f"   Resolved trades: {self.n_rows}")
        if self.warnings:
            for w in self.warnings:
                lines.append(f"   ⚠️ {w}")
        return lines


@dataclass
class AnalysisHealthEntry:
    """Health status for one analytical component."""
    name: str
    status: HealthStatus
    detail: str = ""
    error: Optional[str] = None
    duration_ms: float = 0.0

    @property
    def report_line(self) -> str:
        base = f"   {self.status.icon} {self.name}: {self.status.value}"
        if self.detail:
            base += f" — {self.detail}"
        return base


# ══════════════════════════════════════════════════════════════════════
#  MAIN AUDIT LAYER
# ══════════════════════════════════════════════════════════════════════

class BrainAuditLayer:
    """Central audit layer for the Brain analysis pipeline.

    Usage:
        audit = BrainAuditLayer()
        audit.begin_cycle()
        audit.set_history_coverage(rows, requested_days=180)
        audit.set_reconciliation(...)

        # Before each analysis:
        if audit.can_run("walk_forward"):
            result = engine.validate_threshold_walk_forward(...)
            audit.record_analysis("walk_forward", HealthStatus.OK)
        else:
            audit.record_analysis(
                "walk_forward", HealthStatus.INSUFFICIENT_DATA,
                detail=f"Need {_ANALYSIS_MIN_SAMPLES['walk_forward']} rows"
            )

        # At report time:
        header = audit.build_data_quality_header()
    """

    def __init__(self) -> None:
        self._cycle_start: float = 0.0
        self._history: Optional[HistoryCoverage] = None
        self._reconciliation: Optional[OutcomeReconciliation] = None
        self._analysis_health: Dict[str, AnalysisHealthEntry] = {}
        self._suppressed_analyses: List[str] = []
        self._n_rows: int = 0
        self._n_shadow_rows: int = 0
        self._schema_version: Optional[int] = None
        self._archive_stats: Optional[Dict[str, int]] = None

    # ── Lifecycle ─────────────────────────────────────────────────────

    def begin_cycle(self) -> None:
        """Reset all state for a new Brain report cycle."""
        self._cycle_start = time.time()
        self._history = None
        self._reconciliation = None
        self._analysis_health = {}
        self._suppressed_analyses = []
        self._n_rows = 0
        self._n_shadow_rows = 0
        self._archive_stats = None

    # ── Data Population ───────────────────────────────────────────────

    def set_history_coverage(
        self,
        rows: List[Dict[str, Any]],
        requested_days: Optional[int] = None,
        analysis_rows: Optional[List[Dict[str, Any]]] = None,
    ) -> HistoryCoverage:
        """Compute and store history coverage from loaded rows.

        `rows` = long-window rows (used for the history SPAN).
        `analysis_rows` = rows the analyses actually run on (used for the
        sample-size count). Falls back to `rows` when not given."""
        if requested_days is None:
            requested_days = getattr(cfg, "BRAIN_LONG_WINDOW_DAYS", 180)
        n = len(analysis_rows) if analysis_rows is not None else len(rows)
        self._n_rows = n

        if not rows:
            self._history = HistoryCoverage(
                requested_days=requested_days,
                actual_days=0.0,
                n_rows=0,
                coverage=DataCoverage.CRITICAL,
                warnings=["No outcome rows loaded — Brain cannot analyze anything."],
            )
            return self._history

        timestamps = [r.get("entry_ts", 0) for r in rows if r.get("entry_ts", 0) > 0]
        if not timestamps:
            self._history = HistoryCoverage(
                requested_days=requested_days,
                actual_days=0.0,
                n_rows=n,
                coverage=DataCoverage.CRITICAL,
                warnings=["All rows have entry_ts=0 — cannot determine history span."],
            )
            return self._history

        oldest = min(timestamps)
        newest = max(timestamps)
        actual_days = (newest - oldest) / 86400.0

        # Determine coverage level
        ratio = actual_days / max(requested_days, 1)
        if ratio >= 0.8:
            coverage = DataCoverage.FULL
            warnings: List[str] = []
        elif ratio >= 0.4:
            coverage = DataCoverage.PARTIAL
            warnings = [
                f"History covers only {actual_days:.0f} of {requested_days} "
                f"requested days. Long-window analyses will be degraded."
            ]
        elif ratio >= 0.1:
            coverage = DataCoverage.SEVERELY_LIMITED
            warnings = [
                f"SEVERELY LIMITED: only {actual_days:.1f} days of "
                f"{requested_days} requested. Most multi-window analyses "
                f"are unreliable."
            ]
        else:
            coverage = DataCoverage.CRITICAL
            warnings = [
                f"CRITICAL: only {actual_days:.1f} days available vs "
                f"{requested_days} requested. Statistical conclusions "
                f"are not defensible."
            ]

        if n < 30:
            warnings.append(
                f"Only {n} resolved trades — below the minimum for "
                f"any confident statistical claim."
            )

        self._history = HistoryCoverage(
            requested_days=requested_days,
            actual_days=actual_days,
            oldest_ts=oldest,
            newest_ts=newest,
            n_rows=n,
            coverage=coverage,
            warnings=warnings,
        )
        return self._history

    def set_reconciliation(
        self,
        pending_count: Optional[int] = None,
        resolved_this_run: Optional[int] = None,
        archived_this_run: Optional[int] = None,
        total_archived: Optional[int] = None,
        loaded_by_brain: int = 0,
        shadow_loaded: int = 0,
        archive_stats: Optional[Dict[str, int]] = None,
    ) -> OutcomeReconciliation:
        """Record outcome reconciliation data."""
        self._n_shadow_rows = shadow_loaded
        self._archive_stats = archive_stats

        recon = OutcomeReconciliation(
            pending_count=pending_count,
            resolved_this_run=resolved_this_run,
            archived_this_run=archived_this_run,
            total_archived=total_archived,
            loaded_by_brain=loaded_by_brain,
            shadow_loaded=shadow_loaded,
        )

        if archive_stats:
            if recon.total_archived is None:
                recon.total_archived = archive_stats.get("lines_total")
            recon.archive_rejects_stale_schema = archive_stats.get(
                "dropped_unmigratable", 0
            )
            recon.archive_rejects_signal_only = archive_stats.get(
                "dropped_missing_win", 0
            )
            recon.archive_rejects_malformed = archive_stats.get(
                "lines_malformed", 0
            )
            recon.archive_rejects_duplicate = archive_stats.get(
                "dropped_duplicate_sid", 0
            )
            recon.archive_rejects_out_of_window = archive_stats.get(
                "dropped_before_window", 0
            )

        # Consistency check
        total_rejects = (
            recon.archive_rejects_stale_schema
            + recon.archive_rejects_signal_only
            + recon.archive_rejects_malformed
            + recon.archive_rejects_duplicate
        )
        if total_rejects > 0 and loaded_by_brain == 0:
            recon.status = HealthStatus.FAILED
            recon.notes.append(
                "Archive has rows but Brain loaded zero — "
                "possible schema mismatch or window misconfiguration."
            )
        elif total_rejects > loaded_by_brain:
            recon.status = HealthStatus.DEGRADED
            recon.notes.append(
                f"More rows rejected ({total_rejects}) than loaded "
                f"({loaded_by_brain}) — significant data loss."
            )
        else:
            recon.status = HealthStatus.OK

        self._reconciliation = recon
        return recon

    def set_shadow_count(self, n_shadow: int) -> None:
        self._n_shadow_rows = n_shadow

    # ── Analysis Gating ───────────────────────────────────────────────

    def can_run(self, analysis_name: str) -> Tuple[bool, str]:
        """Check whether an analysis has sufficient data to run.

        Returns (allowed, reason). If not allowed, the caller should
        record the analysis as INSUFFICIENT_DATA and skip it.
        """
        # Sample size check
        min_sample = _ANALYSIS_MIN_SAMPLES.get(analysis_name, 20)
        if self._n_rows < min_sample:
            return False, (
                f"Need ≥{min_sample} rows, have {self._n_rows}"
            )

        # History coverage check
        min_days = _ANALYSIS_MIN_HISTORY_DAYS.get(analysis_name)
        if min_days is not None and self._history is not None:
            if self._history.actual_days < min_days:
                return False, (
                    f"Need ≥{min_days} days history, "
                    f"have {self._history.actual_days:.1f}"
                )

        # Coverage-level gate for specific analysis classes
        if self._history is not None:
            if analysis_name in (
                "layered_window", "temporal_drift", "config_regression"
            ) and self._history.coverage in (
                DataCoverage.SEVERELY_LIMITED, DataCoverage.CRITICAL
            ):
                return False, (
                    f"History coverage is {self._history.coverage.value} — "
                    f"this analysis requires at least PARTIAL coverage"
                )

        return True, "ok"

    def can_recommend(self, analysis_name: str) -> bool:
        """Whether the analysis result can produce an ACTIONABLE
        recommendation (Tier 4) vs being capped at CANDIDATE (Tier 3)."""
        if self._history is None:
            return False
        if self._history.actual_days < 21:
            return False
        if self._n_rows < 100:
            return False
        entry = self._analysis_health.get(analysis_name)
        if entry and not entry.status.allows_recommendation:
            return False
        return True

    def max_recommendation_tier(self, analysis_name: str) -> RecommendationTier:
        """The highest tier a recommendation from this analysis can reach."""
        if self.can_recommend(analysis_name):
            return RecommendationTier.ACTIONABLE
        if self.can_run(analysis_name)[0]:
            return RecommendationTier.CANDIDATE
        return RecommendationTier.DESCRIPTIVE

    # ── Analysis Health Recording ─────────────────────────────────────

    def record_analysis(
        self,
        name: str,
        status: HealthStatus,
        detail: str = "",
        error: Optional[str] = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Record the health outcome of an analytical component."""
        self._analysis_health[name] = AnalysisHealthEntry(
            name=name,
            status=status,
            detail=detail,
            error=error,
            duration_ms=duration_ms,
        )
        if status in (
            HealthStatus.FAILED,
            HealthStatus.INSUFFICIENT_DATA,
            HealthStatus.UNAVAILABLE,
        ):
            self._suppressed_analyses.append(name)
            if status == HealthStatus.FAILED:
                _log.warning(
                    f"Brain audit: analysis '{name}' FAILED: {error or detail}"
                )

    def record_analysis_exception(
        self, name: str, exc: Exception
    ) -> None:
        """Record a silent-failure path that was caught by try/except."""
        self.record_analysis(
            name,
            HealthStatus.FAILED,
            detail=f"Exception caught: {type(exc).__name__}",
            error=str(exc)[:200],
        )

    # ── CUSUM Watermark Diagnostics ───────────────────────────────────

    def validate_cusum_watermark(
        self,
        alert_key: str,
        old_watermark: int,
        new_watermark: int,
        rows_consumed: int,
    ) -> Optional[str]:
        """Verify monotonic watermark advancement. Returns warning or None."""
        if new_watermark < old_watermark:
            msg = (
                f"CUSUM watermark REGRESSION for {alert_key}: "
                f"{old_watermark} → {new_watermark}. "
                f"History replay risk."
            )
            _log.warning(f"Brain audit: {msg}")
            self.record_analysis(
                "cusum", HealthStatus.DEGRADED, detail=msg
            )
            return msg
        if rows_consumed > 0 and new_watermark == old_watermark:
            msg = (
                f"CUSUM watermark STALLED for {alert_key}: "
                f"consumed {rows_consumed} rows but watermark unchanged."
            )
            _log.warning(f"Brain audit: {msg}")
            return msg
        return None

    # ── Report Generation ─────────────────────────────────────────────

    def build_data_quality_header(self) -> List[str]:
        """Build the 📊 BRAIN DATA QUALITY section for the report."""
        lines: List[str] = []
        lines.append("📊 BRAIN DATA QUALITY")

        # History coverage
        if self._history:
            cov = self._history
            lines.append(
                f"   {cov.coverage.icon} DATA COVERAGE: {cov.coverage.value}"
            )
            for line in cov.to_report_lines():
                lines.append(line)
        else:
            lines.append("   ⚪ No history coverage data available")

        lines.append("")

        # Outcome reconciliation
        if self._reconciliation:
            recon = self._reconciliation
            lines.append("   Outcome reconciliation:")
            for line in recon.to_report_lines():
                lines.append(line)
            if recon.notes:
                for note in recon.notes:
                    lines.append(f"   ⚠️ {note}")
        else:
            lines.append("   ⚪ No reconciliation data available")

        lines.append("")

        # Analysis health summary
        if self._analysis_health:
            lines.append("   Analysis health:")
            for entry in self._analysis_health.values():
                lines.append(entry.report_line)

        if self._suppressed_analyses:
            lines.append("")
            lines.append(
                f"   ⚠️ {len(self._suppressed_analyses)} analysis(es) "
                f"suppressed due to insufficient data: "
                f"{', '.join(self._suppressed_analyses[:6])}"
            )

        lines.append("")
        return lines

    def build_action_gate_summary(
        self, action_gate: Dict[str, Any]
    ) -> List[str]:
        """Build explicit action gate status for the report."""
        lines: List[str] = []
        actionable = action_gate.get("actionable", False)

        checks = {
            "data_quality": "≥100 trades logged",
            "oos_prediction": "out-of-sample EV positive (P>0.70)",
            "profitability": "net EV confident (P>0.85, P5 > -0.10)",
            "stability": "no active CUSUM drift alarms",
            "risk": "drawdown within kill-switch budget",
            "execution": "fee/slippage assumptions configured",
        }

        passed = sum(1 for k in checks if action_gate.get(k, False))
        total = len(checks)

        icon = "🟢" if actionable else "🔴"
        lines.append(
            f"{icon} ACTION GATE: "
            f"{'PASSED' if actionable else 'BLOCKED'} "
            f"({passed}/{total} conditions met)"
        )

        if not actionable:
            for key, label in checks.items():
                status = action_gate.get(key, False)
                check_icon = "✅" if status else "❌"
                lines.append(f"   {check_icon} {label}")

        return lines

    # ── Statistical Language Helpers ──────────────────────────────────

    def statistical_confidence_label(self) -> str:
        """Plain-language confidence qualifier for the current sample."""
        n = self._n_rows
        if n >= 300:
            return "HIGH"
        if n >= 100:
            return "MODERATE"
        if n >= 30:
            return "LOW"
        return "VERY LOW"

    def qualify_projection(self, metric_name: str, value_str: str) -> str:
        """Replace 'projected' language with statistically honest phrasing."""
        confidence = self.statistical_confidence_label()
        if confidence in ("LOW", "VERY LOW"):
            return (
                f"{metric_name}: {value_str} "
                f"(historical simulation estimate, {confidence} confidence "
                f"— NOT a forecast)"
            )
        return f"{metric_name}: {value_str} (historical estimate)"

    def qualify_verdict(
        self,
        net_ev: float,
        wr: float,
        n: int,
        p_ev_positive: float = 0.0,
    ) -> str:
        """Statistically disciplined verdict instead of absolute claims."""
        confidence = self.statistical_confidence_label()

        if confidence == "VERY LOW":
            if net_ev > 0:
                return (
                    f"⚠️ OBSERVED SAMPLE: Net EV {net_ev:+.2f}%/trade, "
                    f"WR={wr:.0%} (n={n}).\n"
                    f"   Statistical confidence: {confidence}.\n"
                    f"   Conclusion: Sample is too small for a definitive "
                    f"profitability claim. Accumulate more outcomes."
                )
            else:
                return (
                    f"⚠️ OBSERVED SAMPLE: Net EV {net_ev:+.2f}%/trade, "
                    f"WR={wr:.0%} (n={n}).\n"
                    f"   Statistical confidence: {confidence}.\n"
                    f"   Conclusion: Current sample does not demonstrate "
                    f"positive expectancy, but the sample is too small "
                    f"to rule it out. Accumulate more outcomes."
                )

        if confidence == "LOW":
            if net_ev > 0 and p_ev_positive >= 0.85:
                return (
                    f"⚠️ MARGINALLY POSITIVE: Net EV {net_ev:+.2f}%/trade, "
                    f"WR={wr:.0%} (n={n}), P(EV>0)={p_ev_positive:.0%}.\n"
                    f"   Statistical confidence: {confidence}.\n"
                    f"   Conclusion: Suggestive of positive expectancy but "
                    f"not yet conclusive."
                )
            elif net_ev > 0:
                return (
                    f"⚠️ THIN EVIDENCE: Net EV {net_ev:+.2f}%/trade, "
                    f"WR={wr:.0%} (n={n}), P(EV>0)={p_ev_positive:.0%}.\n"
                    f"   Statistical confidence: {confidence}.\n"
                    f"   Conclusion: Positive point estimate but insufficient "
                    f"statistical power to confirm."
                )
            else:
                return (
                    f"🟠 NEGATIVE SAMPLE: Net EV {net_ev:+.2f}%/trade, "
                    f"WR={wr:.0%} (n={n}).\n"
                    f"   Statistical confidence: {confidence}.\n"
                    f"   Conclusion: Current sample does not demonstrate "
                    f"positive expectancy."
                )

        # MODERATE or HIGH confidence
        if net_ev > 0 and p_ev_positive >= 0.85:
            return (
                f"✅ PROFITABLE: Net EV {net_ev:+.2f}%/trade, "
                f"WR={wr:.0%} (n={n}), P(EV>0)={p_ev_positive:.0%}.\n"
                f"   Statistical confidence: {confidence}."
            )
        elif net_ev > 0:
            return (
                f"⚠️ MARGINALLY POSITIVE: Net EV {net_ev:+.2f}%/trade, "
                f"WR={wr:.0%} (n={n}), P(EV>0)={p_ev_positive:.0%}.\n"
                f"   Statistical confidence: {confidence}."
            )
        else:
            return (
                f"🔴 UNPROFITABLE: Net EV {net_ev:+.2f}%/trade, "
                f"WR={wr:.0%} (n={n}).\n"
                f"   Statistical confidence: {confidence}.\n"
                f"   Conclusion: Costs exceed gains in the observed sample."
            )

    # ── Schema Migration Advisory ─────────────────────────────────────

    def schema_migration_advisory(
        self,
        current_version: int,
        stale_count: int,
        migrated_count: int,
        unmigratable_count: int,
        total_archive_rows: int,
    ) -> Optional[str]:
        """Advise on schema migration health. With the migration layer active,
        stale rows are mapped forward rather than dropped. This advisory now
        reports migration success and flags genuinely lost rows."""
        if stale_count == 0 and migrated_count == 0:
            return None

        if unmigratable_count > 0:
            pct_lost = unmigratable_count / max(total_archive_rows, 1) * 100
            return (
                f"⚠️ SCHEMA MIGRATION: {migrated_count} rows successfully "
                f"migrated from v<{current_version}. "
                f"{unmigratable_count} rows ({pct_lost:.0f}%) were genuinely "
                f"unusable (missing win/pct_move/score) and discarded."
            )

        if migrated_count > 0:
            return (
                f"✅ SCHEMA MIGRATION: {migrated_count} historical rows "
                f"migrated forward from v<{current_version}. "
                f"Three-metric and R:R analyses will show reduced coverage "
                f"for these rows (fields set to None)."
            )

        return None

    # ── Full Audit Summary ────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize audit state for persistence / ai_metrics."""
        return {
            "cycle_start": self._cycle_start,
            "n_rows": self._n_rows,
            "n_shadow_rows": self._n_shadow_rows,
            "history_coverage": {
                "requested_days": self._history.requested_days if self._history else 0,
                "actual_days": self._history.actual_days if self._history else 0.0,
                "coverage": self._history.coverage.value if self._history else "UNKNOWN",
                "n_rows": self._history.n_rows if self._history else 0,
            },
            "reconciliation": {
                "status": self._reconciliation.status.value if self._reconciliation else "UNKNOWN",
                "loaded": self._reconciliation.loaded_by_brain if self._reconciliation else 0,
                "stale_schema_dropped": (
                    self._reconciliation.archive_rejects_stale_schema
                    if self._reconciliation else 0
                ),
            },
            "analysis_health": {
                name: entry.status.value
                for name, entry in self._analysis_health.items()
            },
            "suppressed_analyses": self._suppressed_analyses,
            "statistical_confidence": self.statistical_confidence_label(),
        }


# ══════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL SINGLETON (one per Brain cycle)
# ══════════════════════════════════════════════════════════════════════

_audit_instance: Optional[BrainAuditLayer] = None


def get_audit() -> BrainAuditLayer:
    """Get the current cycle's audit layer (creates if needed)."""
    global _audit_instance
    if _audit_instance is None:
        _audit_instance = BrainAuditLayer()
    return _audit_instance


def reset_audit() -> BrainAuditLayer:
    """Reset for a new cycle and return the fresh instance."""
    global _audit_instance
    _audit_instance = BrainAuditLayer()
    _audit_instance.begin_cycle()
    return _audit_instance