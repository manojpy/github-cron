"""feature_store — point-in-time feature rows + multi-horizon labels.

Read-side and pure by default: no Redis, no files. The Brain already
snapshots every feature at alert time (`votes` + `context`, written BEFORE the
outcome is known) and every resolved row carries a 12-candle price path, so a
versioned, leakage-checked dataset can be MATERIALISED from the archive instead
of being re-derived ad hoc by each analysis.

Optional on-disk cache: set BRAIN_DATASET_CACHE to a directory to reuse the
materialised dataset across runs. The cache key covers the feature-set version,
the feature-builder source, the horizon set, and the identity of every input
row — any change invalidates it. A cache miss/hit failure is a pure
optimisation loss, never a run failure.

Guarantees:
  * features come only from `votes` / `context` / session / direction, i.e. what
    was knowable at the signal candle; keys that only exist after resolution are
    dropped and counted (`leaked_keys`);
  * every row carries the feature-set version it was recorded under
    (`"legacy"` when it predates the stamp), so analyses can refuse to mix
    incompatible feature sets;
  * labels exist per horizon (candles after the fill) and are `None`, never
    guessed, when the stored path is too short.
"""
from __future__ import annotations

import hashlib
import inspect
import os
import pickle
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from plan_replay import has_path, replay_plan

# ── Optional dataset cache (Phase B: stop re-parsing the archive every run) ──
# Empty string = disabled (default, pure in-memory behaviour unchanged).
DATASET_CACHE_DIR: str = os.environ.get("BRAIN_DATASET_CACHE", "")
# 0 = entries never expire on their own (row-identity key already invalidates
# when new outcomes arrive); >0 = max cache age in seconds.
DATASET_CACHE_TTL_SEC: int = int(os.environ.get("BRAIN_DATASET_CACHE_TTL", "0"))

Row = Dict[str, Any]

# Bump when the set or meaning of fields written into `context` / `votes`
# changes. Stamped into every new pending outcome (alerts.py).
FEATURE_SET_VERSION = "fv1"
LEGACY_VERSION = "legacy"

# 15m candles after the fill. The stored path is 12 candles long, so longer
# horizons cannot be labelled without extending the outcome path itself.
LABEL_HORIZONS: Tuple[int, ...] = (4, 6, 8, 12)

# Fields that only exist once the trade has resolved. They must never be
# features; if one ever shows up inside `context` it is dropped and counted.
_POST_HOC_KEYS = frozenset({
    "win", "pct_move", "mae", "mfe", "close_win", "mfe_win", "mae_loss",
    "tp_first", "outcome_reason", "bonus_win", "rr_achieved", "win_weight",
    "net_pnl_pct", "net_pnl_hold_pct", "outcome_class", "realized_cost_pct",
    "tp_candle", "sl_candle", "mfe_candle", "mae_candle",
})


def row_feature_version(row: Row) -> str:
    v = (row.get("context") or {}).get("feature_version")
    return str(v) if v else LEGACY_VERSION


def _clean_row(row: Row) -> Tuple[Row, int]:
    """Copy of `row` whose context has no post-resolution keys."""
    ctx = row.get("context") or {}
    bad = [k for k in ctx if k in _POST_HOC_KEYS]
    if not bad:
        return row, 0
    out = dict(row)
    out["context"] = {k: v for k, v in ctx.items() if k not in _POST_HOC_KEYS}
    return out, len(bad)

# ── Dataset cache helpers ────────────────────────────────────────────────

def _row_identity(r: Row) -> str:
    rid = r.get("_stream_id") or f'{r.get("pair")}:{r.get("alert_key")}:{r.get("entry_ts")}'
    return f"{rid},{r.get('entry_ts')}"


def _dataset_cache_key(rows: Sequence[Row], horizons: Sequence[int],
                       feature_names: Optional[List[str]],
                       default_cost_pct: float) -> str:
    """Anything that could change the materialised dataset feeds the key:
    row identities, horizons, requested features, cost, feature-set version,
    and the source of the feature builder itself (so a code change to
    build_market_state_features invalidates old caches automatically)."""
    from threshold_models import build_market_state_features   # lazy, same as below
    try:
        code_hash = hashlib.sha1(
            inspect.getsource(build_market_state_features).encode()).hexdigest()[:10]
    except Exception:
        code_hash = "unknown"
    h = hashlib.sha1()
    for r in rows:
        if r.get("entry_ts") is not None:
            h.update(_row_identity(r).encode())
    h.update(str(tuple(int(x) for x in horizons)).encode())
    h.update(repr(feature_names).encode())
    h.update(str(default_cost_pct).encode())
    h.update(FEATURE_SET_VERSION.encode())
    h.update(code_hash.encode())
    return h.hexdigest()[:24]

def _dataset_cache_path(key: str) -> str:
    return os.path.join(DATASET_CACHE_DIR, f"pit_dataset_{FEATURE_SET_VERSION}_{key}.pkl")


def load_cached_dataset(key: str) -> Optional[Dict[str, Any]]:
    if not DATASET_CACHE_DIR:
        return None
    path = _dataset_cache_path(key)
    try:
        if not os.path.exists(path):
            return None
        if DATASET_CACHE_TTL_SEC and time.time() - os.path.getmtime(path) > DATASET_CACHE_TTL_SEC:
            return None
        with open(path, "rb") as fh:
            ds = pickle.load(fh)
        if not isinstance(ds, dict) or "rows" not in ds or "feature_hash" not in ds:
            return None
        return ds
    except Exception:
        return None    # cache is a pure optimisation; never a failure source


def save_cached_dataset(ds: Dict[str, Any], key: str) -> None:
    if not DATASET_CACHE_DIR:
        return
    try:
        os.makedirs(DATASET_CACHE_DIR, exist_ok=True)
        tmp = _dataset_cache_path(key) + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(ds, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, _dataset_cache_path(key))   # atomic-ish: no partial reads
    except Exception:
        pass           # shadow: a failed cache write must not break the run

def row_labels(row: Row, horizons: Sequence[int] = LABEL_HORIZONS,
               sl_pct: Optional[float] = None, tp_pct: Optional[float] = None,
               default_cost_pct: float = 0.0) -> Dict[int, Optional[Dict[str, Any]]]:
    """Per-horizon outcome of this trade, in percent of fill price, net of cost.

    net      stop/target bracket replayed to that horizon (same pessimistic
             tie rule as plan_replay), exit at the horizon close otherwise
    hold     simply the horizon close, net of cost
    mfe/mae  best / worst excursion inside the horizon
    t_mfe    candle index of the best excursion (time to MFE)
    None     the row has no path, or the path is shorter than the horizon."""
    out: Dict[int, Optional[Dict[str, Any]]] = {}
    if not has_path(row):
        return {int(h): None for h in horizons}
    sl = sl_pct if sl_pct is not None else row.get("plan_sl_pct")
    tp = tp_pct if tp_pct is not None else row.get("plan_tp_pct")
    fav, adv, cls = row["path_fav"], row["path_adv"], row["path_close"]
    cost = row.get("realized_cost_pct")
    cost = float(cost) if cost is not None else float(default_cost_pct)
    for h in horizons:
        h = int(h)
        if len(fav) <= h:
            out[h] = None
            continue
        rep = replay_plan(row, float(sl), float(tp), h, default_cost_pct) if (sl and tp) else None
        window = fav[: h + 1]
        out[h] = {
            "net": None if rep is None else round(rep[1], 6),
            "class": None if rep is None else rep[0],
            "hold": round(float(cls[h]) - cost, 6),
            "mfe": round(float(max(window)), 6),
            "mae": round(float(max(adv[: h + 1])), 6),
            "t_mfe": int(max(range(len(window)), key=window.__getitem__)),
        }
    return out

def build_point_in_time_dataset(rows: Sequence[Row], horizons: Sequence[int] = LABEL_HORIZONS,
                                feature_names: Optional[List[str]] = None,
                                default_cost_pct: float = 0.0,
                                use_cache: bool = True) -> Dict[str, Any]:
    """Materialise the dataset. Rows are ordered by entry_ts and need an
    entry_ts; everything else degrades to None rather than being invented.

    Returns {"feature_names", "feature_hash", "rows", "stats"}; each dataset row
    is {id, ts, pair, alert_key, direction, version, x, labels}.

    With BRAIN_DATASET_CACHE set, an unchanged input set returns the cached
    materialisation instead of re-running the feature builder."""
    if use_cache:
        key = _dataset_cache_key(rows, horizons, feature_names, default_cost_pct)
        cached = load_cached_dataset(key)
        if cached is not None:
            stats = dict(cached.get("stats") or {})
            stats["cache_hit"] = True
            cached["stats"] = stats
            return cached
        ds = _build_point_in_time_dataset_impl(rows, horizons, feature_names, default_cost_pct)
        ds.setdefault("stats", {})["cache_hit"] = False
        save_cached_dataset(ds, key)
        return ds
    return _build_point_in_time_dataset_impl(rows, horizons, feature_names, default_cost_pct)


def _build_point_in_time_dataset_impl(rows: Sequence[Row], horizons: Sequence[int] = LABEL_HORIZONS,
                                      feature_names: Optional[List[str]] = None,
                                      default_cost_pct: float = 0.0) -> Dict[str, Any]:
    """Materialise the dataset. Rows are ordered by entry_ts and need an
    entry_ts; everything else degrades to None rather than being invented.

    Returns {"feature_names", "feature_hash", "rows", "stats"}; each dataset row
    is {id, ts, pair, alert_key, direction, version, x, labels}."""
    from threshold_models import build_market_state_features   # lazy: keeps this module light
    usable: List[Row] = []
    leaked = 0
    for r in rows:
        if r.get("entry_ts") is None:
            continue
        c, n = _clean_row(r)
        leaked += n
        usable.append(c)
    usable.sort(key=lambda r: r["entry_ts"])

    X, _y, names = build_market_state_features(
        [dict(r, win=bool(r.get("win"))) for r in usable], feature_names)
    fhash = hashlib.sha1("|".join(names).encode()).hexdigest()[:10]

    out_rows: List[Dict[str, Any]] = []
    for r, x in zip(usable, X):
        out_rows.append({
            "id": r.get("_stream_id") or f'{r.get("pair")}:{r.get("alert_key")}:{r["entry_ts"]}',
            "ts": int(r["entry_ts"]),
            "pair": r.get("pair"),
            "alert_key": r.get("alert_key"),
            "direction": r.get("direction"),
            "version": row_feature_version(r),
            "x": x,
            "labels": row_labels(r, horizons, default_cost_pct=default_cost_pct),
        })
    return {
        "feature_names": names, "feature_hash": fhash, "rows": out_rows,
        "stats": {"n_input": len(rows), "n_rows": len(out_rows), "leaked_keys": leaked},
    }


def chronological_split(ds_rows: Sequence[Dict[str, Any]], test_frac: float = 0.3,
                        max_horizon: int = max(LABEL_HORIZONS), candle_sec: int = 900,
                        embargo_candles: int = 2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Oldest rows train, newest test. A training row is purged when its label
    window (entry + horizon candles + embargo) reaches the first test timestamp,
    so no training label overlaps the test period."""
    rows = sorted(ds_rows, key=lambda r: r["ts"])
    if len(rows) < 4:
        return list(rows), []
    cut = max(1, min(len(rows) - 1, int(round(len(rows) * (1.0 - test_frac)))))
    test = rows[cut:]
    first_test = test[0]["ts"]
    reach = (max_horizon + embargo_candles) * candle_sec
    train = [r for r in rows[:cut] if r["ts"] + reach <= first_test]
    return train, test


def dataset_coverage(ds: Dict[str, Any], horizons: Sequence[int] = LABEL_HORIZONS) -> Dict[str, Any]:
    """Is the dataset good enough to trust? Counts only — no model, no verdict."""
    rows = ds.get("rows") or []
    n = len(rows)
    versions = Counter(r["version"] for r in rows)
    per_h: Dict[int, Dict[str, Any]] = {}
    for h in horizons:
        done = [r["labels"].get(int(h)) for r in rows]
        done = [d for d in done if d and d.get("net") is not None]
        per_h[int(h)] = {"n": len(done), "frac": round(len(done) / n, 3) if n else 0.0,
                         "mean_net": round(sum(d["net"] for d in done) / len(done), 4) if done else None}
    zero_share = None
    if n and rows[0].get("x"):
        k = len(rows[0]["x"])
        zeros = [sum(1 for r in rows if not r["x"][j]) / n for j in range(1, k)]
        names = ds.get("feature_names") or []
        worst = sorted(zip(names, zeros), key=lambda t: -t[1])[:5]
        zero_share = [(nm, round(z, 3)) for nm, z in worst if z >= 0.5]
    mixed = len([v for v in versions if v != LEGACY_VERSION]) > 1
    return {
        "n": n, "versions": dict(versions), "mixed_versions": mixed,
        "feature_hash": ds.get("feature_hash"), "n_features": len(ds.get("feature_names") or []),
        "leaked_keys": (ds.get("stats") or {}).get("leaked_keys", 0),
        "horizons": per_h, "mostly_empty_features": zero_share or [],
        "current_version": FEATURE_SET_VERSION,
        "current_version_share": round(versions.get(FEATURE_SET_VERSION, 0) / n, 3) if n else 0.0,
    }