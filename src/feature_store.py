"""feature_store — point-in-time feature rows + multi-horizon labels.

Read-side and pure: no Redis, no files, no new storage. The Brain already
snapshots every feature at alert time (`votes` + `context`, written BEFORE the
outcome is known) and every resolved row carries a 12-candle price path, so a
versioned, leakage-checked dataset can be MATERIALISED from the archive instead
of being re-derived ad hoc by each analysis.

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

import base64
import gzip
import hashlib
import json
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from plan_replay import has_path, replay_plan

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

def _row_id(r: Row) -> str:
    return r.get("_stream_id") or f'{r.get("pair")}:{r.get("alert_key")}:{r["entry_ts"]}'

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
            "id": _row_id(r),
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

DATASET_CACHE_VERSION = 1


def pack_dataset(ds: Dict[str, Any], horizons: Sequence[int], default_cost_pct: float) -> str:
    """ASCII blob (gzip + base64 of JSON) of a materialised dataset, stamped with
    everything a later run must match before it may reuse the rows."""
    payload = {
        "cv": DATASET_CACHE_VERSION, "fv": FEATURE_SET_VERSION,
        "horizons": [int(h) for h in horizons], "cost": round(float(default_cost_pct), 6),
        "feature_hash": ds.get("feature_hash"), "feature_names": ds.get("feature_names"),
        "rows": ds.get("rows") or [],
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(gzip.compress(raw, 6)).decode("ascii")


def unpack_dataset(blob: str) -> Optional[Dict[str, Any]]:
    """Inverse of pack_dataset. None (never raises) if the blob is unreadable."""
    try:
        data = json.loads(gzip.decompress(base64.b64decode(blob.encode("ascii"))).decode("utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
            return None
        for r in data["rows"]:     # JSON turns the integer horizon keys into strings
            r["labels"] = {int(k): v for k, v in (r.get("labels") or {}).items()}
        return data
    except Exception:
        return None


def _num_eq(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return abs(float(a) - float(b)) <= 1e-9
    return a == b


def _same_ds_row(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> bool:
    if not a or not b:
        return False
    for k in ("id", "ts", "pair", "alert_key", "direction", "version"):
        if a.get(k) != b.get(k):
            return False
    xa, xb = a.get("x") or [], b.get("x") or []
    if len(xa) != len(xb) or not all(_num_eq(u, v) for u, v in zip(xa, xb)):
        return False
    la, lb = a.get("labels") or {}, b.get("labels") or {}
    if set(la) != set(lb):
        return False
    for h in la:
        ha, hb = la[h], lb[h]
        if (ha is None) != (hb is None):
            return False
        if ha is not None and (set(ha) != set(hb) or not all(_num_eq(ha[f], hb[f]) for f in ha)):
            return False
    return True


def build_point_in_time_dataset_cached(rows: Sequence[Row], cached: Optional[Dict[str, Any]], horizons: Sequence[int] = LABEL_HORIZONS,
                                       default_cost_pct: float = 0.0, verify_n: int = 5) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Same dataset as build_point_in_time_dataset, but rows already present in
    `cached` are reused and only new rows are built. Returns (dataset, info).

    A cached row is reused only when the cache was written under the same
    feature-set version, feature names/hash, horizons and cost, and ids are
    unique. A resolved row's features and labels never change, so reuse is exact;
    to prove it, up to `verify_n` reused rows are rebuilt every call and compared
    -- any difference discards the cache and rebuilds everything.
    info: {hit, built, verified, reason, feature_hash}."""
    from threshold_models import derive_market_state_feature_names

    hz = tuple(int(h) for h in horizons)
    usable: List[Row] = []
    leaked = 0
    for r in rows:
        if r.get("entry_ts") is None:
            continue
        c, n = _clean_row(r)
        leaked += n
        usable.append(c)
    usable.sort(key=lambda r: r["entry_ts"])
    names = derive_market_state_feature_names(usable)
    fhash = hashlib.sha1("|".join(names).encode()).hexdigest()[:10]
    info: Dict[str, Any] = {"hit": 0, "built": 0, "verified": None, "reason": None, "feature_hash": fhash}

    def _full(reason: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ds = build_point_in_time_dataset(rows, hz, None, default_cost_pct)
        info.update({"hit": 0, "built": len(ds["rows"]), "reason": reason})
        return ds, info

    if not cached:
        return _full("no_cache")
    if not (cached.get("cv") == DATASET_CACHE_VERSION and cached.get("fv") == FEATURE_SET_VERSION
            and cached.get("feature_hash") == fhash and cached.get("feature_names") == names
            and tuple(cached.get("horizons") or ()) == hz
            and _num_eq(float(cached.get("cost", 0.0)), round(float(default_cost_pct), 6))):
        return _full("signature_changed")
    ids = [_row_id(r) for r in usable]
    by_id = {r["id"]: r for r in cached["rows"]}
    if len(set(ids)) != len(ids) or len(by_id) != len(cached["rows"]):
        return _full("duplicate_ids")
    have = [i for i in ids if i in by_id]
    if have:
        k = max(1, int(verify_n))
        picks = sorted({0, len(have) - 1, *(int(j * (len(have) - 1) / max(1, k - 1)) for j in range(k))})
        sample = [have[j] for j in picks]
        row_of = dict(zip(ids, usable))
        fresh = {r["id"]: r for r in build_point_in_time_dataset([row_of[i] for i in sample], hz, names, default_cost_pct)["rows"]}
        if not all(_same_ds_row(fresh.get(i), by_id[i]) for i in sample):
            info["verified"] = False
            return _full("verification_failed")
        info["verified"] = True
    new_rows = [r for r, i in zip(usable, ids) if i not in by_id]
    built = {r["id"]: r for r in build_point_in_time_dataset(new_rows, hz, names, default_cost_pct)["rows"]} if new_rows else {}
    out_rows = [by_id[i] if i in by_id else built[i] for i in ids]
    info.update({"hit": len(have), "built": len(new_rows), "reason": "reused"})
    return {
        "feature_names": names, "feature_hash": fhash, "rows": out_rows,
        "stats": {"n_input": len(rows), "n_rows": len(out_rows), "leaked_keys": leaked},
    }, info

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