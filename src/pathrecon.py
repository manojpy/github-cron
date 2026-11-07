"""pathrecon — recover the candle path of REAL alerts that were archived without one,
and COMPLETE the path of trades that closed early.

Early-closed trades (stop or bonus target touched before OUTCOME_LOOKAHEAD_CANDLES) are
archived with a path that ends at the exit candle. Every plan replay then sees them as
"too short" and drops them, so plan comparisons only ever saw trades that ran the full
distance (survivor bias). needs_recon(row, full_len) flags those rows too; recon_path then
rebuilds the full forward path and, for such rows, REQUIRES the rebuilt prefix to equal
the stored one (expect_prefix), anchored on the stored fill_price, so a misaligned
candle history can never silently replace a real path.

Alerts that fired before path capture existed (or whose path failed to store) have
no stored candle path, so they are invisible to the plan lab and the learner.  The
alerts themselves really fired, so this is NOT a hypothetical backtest: only the
price path after the simulated fill is rebuilt from the exchange's candle history,
with the SAME arithmetic the live resolver uses (state._parse_pending_outcome_row).

Safety checks
  * the candle open at the fill index must match the archived fill_price
    (tolerance 0.05%); otherwise the row is rejected (wrong candle / bad data);
  * every candle of the path must be closed history (not the forming candle);
  * rebuilt rows are tagged path_source = "RECON".

Pure functions plus a small cache codec.  Fetching lives in learner.py.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

KEY_RECON = "path_recon"     # Redis metadata key of the rebuilt-path cache (written by learner.py)
CANDLE_SEC = 900
FILL_TOL = 0.0005
_ROUND = 4

def cache_key(pair: Any, direction: Any, entry_ts: Any) -> str:
    return f"{pair}|{str(direction).lower()}|{int(entry_ts)}"

PREFIX_TOL = 0.001   # %-points; stored paths are rounded to 4 decimals


def recon_path(ts: np.ndarray, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, *, entry_ts: int, direction: str, fill_price: Optional[float], fill_delay: int, lookahead: int, tol: float = FILL_TOL, expect_prefix: Optional[Mapping[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """{'f','a','c', fav/adv/close lists + mfe_candle/mae_candle} or None.

    expect_prefix: the row's stored (short) path_fav/path_adv/path_close. When given, the
    path is anchored on the stored fill_price (the live resolver's anchor, so the prefix
    reproduces exactly) and the rebuilt path must start with the stored values within
    PREFIX_TOL, else None."""
    d = str(direction).lower()
    if d not in ("buy", "sell", "long", "short") or not fill_price or fill_price <= 0:
        return None
    is_buy = d in ("buy", "long")
    idx = np.flatnonzero(ts == entry_ts)
    if idx.size == 0:
        return None
    entry_idx = int(idx[-1])
    fill_idx = entry_idx + max(0, int(fill_delay))
    target_idx = fill_idx + int(lookahead)
    if target_idx >= len(ts):                       # path not fully closed yet
        return None
    if expect_prefix is not None:
        anchor = float(fill_price)                   # live anchor; the prefix check below is the validation
    else:
        anchor = float(o[fill_idx]) if fill_delay > 0 else None
        if anchor is None:
            return None
        if abs(anchor - fill_price) / fill_price > tol:
            return None                              # candle history does not match what was archived
    path_start = fill_idx if fill_delay > 0 else entry_idx + 1
    hi, lo, cl = h[path_start:target_idx + 1], l[path_start:target_idx + 1], c[path_start:target_idx + 1]
    if not len(hi):
        return None
    if is_buy:
        fav = (hi - anchor) / anchor * 100.0
        adv = (anchor - lo) / anchor * 100.0
        cls = (cl - anchor) / anchor * 100.0
    else:
        fav = (anchor - lo) / anchor * 100.0
        adv = (hi - anchor) / anchor * 100.0
        cls = (anchor - cl) / anchor * 100.0
    out = {
        "path_fav": [round(float(x), _ROUND) for x in fav],
        "path_adv": [round(float(x), _ROUND) for x in adv],
        "path_close": [round(float(x), _ROUND) for x in cls],
        "mfe_candle": int(np.argmax(fav)), "mae_candle": int(np.argmax(adv)),
    }
    if expect_prefix is not None:
        for key in ("path_fav", "path_adv", "path_close"):
            stored = expect_prefix.get(key) or []
            if not stored or len(stored) > len(out[key]):
                return None
            for s, v in zip(stored, out[key]):
                if abs(float(s) - float(v)) > PREFIX_TOL:
                    return None                      # candle history disagrees with the live path
    return out


def needs_recon(row: Mapping[str, Any], full_len: Optional[int] = None) -> bool:
    """True for a real row with no stored path; and, when full_len is given (the full
    path length = OUTCOME_LOOKAHEAD_CANDLES + 1), also for a row whose stored path is
    shorter than that, i.e. a trade that closed early."""
    if not (row.get("entry_ts") and row.get("fill_price")
            and str(row.get("direction", "")).lower() in ("buy", "sell") and row.get("pair")):
        return False
    fav = row.get("path_fav")
    if not fav:
        return True
    return bool(full_len and len(fav) < int(full_len))


def windows_for(entry_tss: Sequence[int], span_candles: int = 1880, tail: int = 16) -> List[Tuple[int, int]]:
    """Greedy (start_ts, end_ts) candle windows covering every entry_ts plus its path."""
    out: List[Tuple[int, int]] = []
    pending = sorted(set(int(t) for t in entry_tss))
    i = 0
    while i < len(pending):
        start = pending[i] - 2 * CANDLE_SEC
        end = start + span_candles * CANDLE_SEC
        j = i
        while j < len(pending) and pending[j] + tail * CANDLE_SEC <= end:
            j += 1
        j = max(j, i + 1)
        out.append((start, end))
        i = j
    return out


def apply_recon(row: Mapping[str, Any], rec: Mapping[str, Any]) -> Dict[str, Any]:
    r = dict(row)
    extended = bool(row.get("path_fav"))             # row had a (short) live path -> completing it
    r.update({k: rec[k] for k in ("path_fav", "path_adv", "path_close", "mfe_candle", "mae_candle") if k in rec})
    r["path_source"] = "RECON_EXT" if extended else "RECON"
    return r

def apply_cached(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, Any], full_len: Optional[int] = None) -> Tuple[List[Dict[str, Any]], int]:
    """Cache-only: give rows that have no stored path (or, with full_len, a path cut short
    by an early exit) the full one the learner already rebuilt.
    No network. Rows with no usable cache entry are returned unchanged."""
    out: List[Dict[str, Any]] = []
    applied = 0
    for r in rows:
        if needs_recon(r, full_len):
            p = cache.get(cache_key(r["pair"], r["direction"], r["entry_ts"]))
            if isinstance(p, dict) and p.get("f"):
                r = apply_recon(r, cache_unpack(p))
                applied += 1
        out.append(r if isinstance(r, dict) else dict(r))
    return out, applied

def cache_pack(rec: Mapping[str, Any]) -> Dict[str, Any]:
    return {"f": rec["path_fav"], "a": rec["path_adv"], "c": rec["path_close"],
            "mf": rec.get("mfe_candle"), "ma": rec.get("mae_candle")}


def cache_unpack(p: Mapping[str, Any]) -> Dict[str, Any]:
    return {"path_fav": p["f"], "path_adv": p["a"], "path_close": p["c"],
            "mfe_candle": p.get("mf"), "mae_candle": p.get("ma")}
