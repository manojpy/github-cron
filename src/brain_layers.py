"""brain_layers — the Research / Validation / Policy boundary, written down and checkable.

Three layers, one direction of travel:

  research    finds patterns and proposes things. May read everything, may keep its
             own records (ledgers, caches) but may NOT change live behaviour.
  validation  tests research output on data the research did not choose. Same
             write limits as research.
  policy      the only layer that may change what the bot does (weights, config
             overrides, alert switches, per-pair thresholds) and only after
             validation. Policy modules may import research; research and
             validation modules may NOT import policy modules.
  infra       plumbing (storage, config, indicators, data fetch). Unrestricted
             here because it holds no decisions of its own.

`audit_layer_boundary()` reads the source with the AST (nothing is imported or
run) and reports:
  * violations    a research/validation module calls a live-behaviour writer or
                  imports a policy module; a "pure" module touches Redis, files
                  or the network layer
  * exceptions    places where a recommendation builder DOES call a live writer
                  today, each with its documented reason; a new one is a violation
  * unclassified  modules not listed here (new files must be classified)
  * stale         documented exceptions that no longer exist in the code

Run `python3 brain_layers.py` to print the audit; it exits 1 on violations."""
from __future__ import annotations

import ast
import functools
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RESEARCH, VALIDATION, POLICY, INFRA = "research", "validation", "policy", "infra"

LAYER_OF: Dict[str, str] = {
    # ── research ──
    "brain_report": RESEARCH, "brain_helpers": RESEARCH, "feature_store": RESEARCH,
    "hypothesis_gen": RESEARCH, "plan_replay": RESEARCH, "rulemine": RESEARCH,
    "playbook": RESEARCH, "threshold_stats": RESEARCH, "threshold_analysis": RESEARCH,
    "threshold_models": RESEARCH, "threshold_quality": RESEARCH, "threshold_repair": RESEARCH,
    "threshold_engine": RESEARCH, "archive_reader": RESEARCH, "scoreboard": RESEARCH,
    "confluence_tier_report": RESEARCH, "learner": RESEARCH, "autopsy": RESEARCH,
    "brain_recommend_baseline": RESEARCH, "brain_recommend_full": RESEARCH,
    # ── validation ──
    "threshold_validation": VALIDATION, "repair_ledger": VALIDATION, "brain_audit": VALIDATION,
    # ── policy ──
    "brain_enhanced": POLICY, "brain": POLICY, "alerts": POLICY, "gates": POLICY,
    "macd_unified": POLICY, "apply_config_override": POLICY, "alert_advisor": POLICY,
    "brain_engine": POLICY,
    # ── infra ──
    "state": INFRA, "bot_config": INFRA, "config_base": INFRA, "fetcher": INFRA,
    "outcome_storage": INFRA, "cleanup_outcomes": INFRA, "health_server": INFRA,
    "redis_audit": INFRA, "aot_bridge": INFRA, "aot_meta": INFRA,
    "numba_functions_shared": INFRA, "indicators": INFRA, "numeric_selftest": INFRA,
    "pathrecon": INFRA, "feedback": INFRA, "brain_layers": INFRA, "alert_registry": INFRA,
}

# Modules that must be pure computation: no Redis, no state store, no files.
PURE = frozenset({
    "feature_store", "hypothesis_gen", "plan_replay", "rulemine", "autopsy", "threshold_stats",
    "threshold_analysis", "threshold_models", "threshold_quality", "threshold_repair",
    "threshold_validation",
})

# State-store methods that change what the bot does.
POLICY_WRITERS = frozenset({
    "write_config_override", "set_alert_key_disabled", "set_pair_threshold",
    "set_dynamic_weights", "clear_dynamic_weights", "promote_challenger_to_champion",
})

POLICY_MODULES = frozenset(m for m, l in LAYER_OF.items() if l == POLICY)
_PURE_FORBIDDEN_IMPORTS = frozenset({"redis", "state", "asyncio", "aiohttp", "requests", "fetcher", "outcome_storage"})
_PURE_FORBIDDEN_CALLS = frozenset({"set_metadata", "get_metadata", "open"})

# Today a recommendation builder applies these itself. Documented so they are
# visible; any NEW call of this kind is a violation.
KNOWN_EXCEPTIONS: Dict[Tuple[str, str], str] = {
    ("brain_recommend_baseline", "set_pair_threshold"):
        "per-pair confluence threshold, applied only after walk-forward and the stability gate",
    ("brain_recommend_full", "set_alert_key_disabled"):
        "protective auto-disable, and re-enable only after the action gate passes",
    ("brain_recommend_full", "clear_dynamic_weights"):
        "auto-rollback of a weight repair that was measured as harmful",
}

# Research modules that import a policy module today, only for text / delivery
# helpers (no state changes). Documented so they are visible; a new one is a violation.
ALLOWED_IMPORTS: Dict[Tuple[str, str], str] = {
    ("brain_report", "alerts"): "escape_markdown_v2 text helper",
    ("learner", "alerts"): "TelegramQueue and escape_markdown_v2 for report delivery",
}


def _scan(path: Path) -> Dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: List[Tuple[str, int]] = []
    calls: List[Tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports += [(a.name.split(".")[0], node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.module.split(".")[0], node.lineno))
        elif isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
            if name:
                calls.append((name, node.lineno))
    return {"imports": imports, "calls": calls}


def audit_layer_boundary(src_dir: Optional[str] = None) -> Dict[str, Any]:
    root = Path(src_dir) if src_dir else Path(__file__).resolve().parent
    violations: List[Dict[str, Any]] = []
    exceptions: Dict[Tuple[str, str], int] = {}
    imports_seen: Dict[Tuple[str, str], int] = {}
    unclassified: List[str] = []
    counts: Dict[str, int] = {}
    for p in sorted(root.glob("*.py")):
        mod = p.stem
        if mod.startswith("test_") or mod in ("conftest", "setup", "__init__"):
            continue
        layer = LAYER_OF.get(mod)
        if layer is None:
            unclassified.append(mod)
            continue
        counts[layer] = counts.get(layer, 0) + 1
        if layer not in (RESEARCH, VALIDATION):
            continue
        try:
            s = _scan(p)
        except SyntaxError as e:
            violations.append({"module": mod, "layer": layer, "line": e.lineno or 0, "kind": "unreadable", "detail": "syntax error"})
            continue
        for name, line in s["calls"]:
            if name in POLICY_WRITERS:
                if (mod, name) in KNOWN_EXCEPTIONS:
                    exceptions[(mod, name)] = exceptions.get((mod, name), 0) + 1
                else:
                    violations.append({"module": mod, "layer": layer, "line": line, "kind": "policy_write", "detail": name})
        for name, line in s["imports"]:
            if name in POLICY_MODULES:
                if (mod, name) in ALLOWED_IMPORTS:
                    imports_seen[(mod, name)] = imports_seen.get((mod, name), 0) + 1
                else:
                    violations.append({"module": mod, "layer": layer, "line": line, "kind": "imports_policy", "detail": name})
        if mod in PURE:
            for name, line in s["imports"]:
                if name in _PURE_FORBIDDEN_IMPORTS:
                    violations.append({"module": mod, "layer": layer, "line": line, "kind": "pure_module_io", "detail": f"imports {name}"})
            for name, line in s["calls"]:
                if name in _PURE_FORBIDDEN_CALLS:
                    violations.append({"module": mod, "layer": layer, "line": line, "kind": "pure_module_io", "detail": f"calls {name}"})
    stale = [f"{m}.{w}" for (m, w) in KNOWN_EXCEPTIONS if (m, w) not in exceptions]
    stale += [f"{m} imports {w}" for (m, w) in ALLOWED_IMPORTS if (m, w) not in imports_seen]
    return {
        "layers": counts, "unclassified": unclassified, "violations": violations,
        "exceptions": [{"module": m, "writer": w, "calls": n, "reason": KNOWN_EXCEPTIONS[(m, w)]}
                       for (m, w), n in sorted(exceptions.items())],
        "allowed_imports": [{"module": m, "imports": w, "reason": ALLOWED_IMPORTS[(m, w)]} for (m, w) in sorted(imports_seen)],
        "stale_exceptions": stale,
        "ok": not violations,
    }


@functools.lru_cache(maxsize=1)
def cached_audit() -> Dict[str, Any]:
    """The source does not change while the bot runs, so one scan is enough."""
    return audit_layer_boundary()


def boundary_allows_apply(mode: str) -> Tuple[bool, str]:
    """May the policy layer apply a stored plan? Only 'block' mode ever says no,
    and only while the audit has violations (restrict-only)."""
    if mode != "block":
        return True, ""
    a = cached_audit()
    if a["ok"]:
        return True, ""
    first = a["violations"][0]
    return False, f"{len(a['violations'])} layer violation(s), first: {first['module']}:{first['line']} {first['kind']} {first['detail']}"


if __name__ == "__main__":
    res = audit_layer_boundary()
    print(f"layers: {res['layers']}")
    for e in res["exceptions"]:
        print(f"  known exception: {e['module']} calls {e['writer']} x{e['calls']} — {e['reason']}")
    for a in res["allowed_imports"]:
        print(f"  allowed import: {a['module']} imports {a['imports']} — {a['reason']}")
    for s in res["stale_exceptions"]:
        print(f"  stale exception (no longer in code): {s}")
    for m in res["unclassified"]:
        print(f"  UNCLASSIFIED module: {m}")
    for v in res["violations"]:
        print(f"  VIOLATION {v['module']}:{v['line']} [{v['layer']}] {v['kind']} {v['detail']}")
    print("OK" if res["ok"] else f"{len(res['violations'])} violation(s)")
    sys.exit(0 if res["ok"] else 1)