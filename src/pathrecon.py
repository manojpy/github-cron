"""pathrecon — recover the candle path of REAL alerts that were archived without one.

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

CANDLE_SEC = 900
FILL_TOL = 0.0005
_ROUND = 4


def cache_key(pair: Any, direction: Any, entry_ts: Any) -> str:
    return f"{pair}|{str(direction).lower()}|{int(entry_ts)}"


def recon_path(
    ts: np.ndarray, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, *,
    entry_ts: int, direction: str, fill_price: Optional[float],
    fill_delay: int, lookahead: int, tol: float = FILL_TOL,
) -> Optional[Dict[str, Any]]:
    """{'f','a','c', fav/adv/close lists + mfe_candle/mae_candle} or None."""
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
    anchor = float(o[fill_idx]) if fill_delay > 0 else None
    if anchor is None:
        return None
    if abs(anchor - fill_price) / fill_price > tol:
        return None                                  # candle history does not match what was archived
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
    return {
        "path_fav": [round(float(x), _ROUND) for x in fav],
        "path_adv": [round(float(x), _ROUND) for x in adv],
        "path_close": [round(float(x), _ROUND) for x in cls],
        "mfe_candle": int(np.argmax(fav)), "mae_candle": int(np.argmax(adv)),
    }


def needs_recon(row: Mapping[str, Any]) -> bool:
    return bool(
        not row.get("path_fav") and row.get("entry_ts") and row.get("fill_price")
        and str(row.get("direction", "")).lower() in ("buy", "sell") and row.get("pair")
    )


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
    r.update({k: rec[k] for k in ("path_fav", "path_adv", "path_close", "mfe_candle", "mae_candle") if k in rec})
    r["path_source"] = "RECON"
    return r


def cache_pack(rec: Mapping[str, Any]) -> Dict[str, Any]:
    return {"f": rec["path_fav"], "a": rec["path_adv"], "c": rec["path_close"],
            "mf": rec.get("mfe_candle"), "ma": rec.get("mae_candle")}


def cache_unpack(p: Mapping[str, Any]) -> Dict[str, Any]:
    return {"path_fav": p["f"], "path_adv": p["a"], "path_close": p["c"],
            "mfe_candle": p.get("mf"), "mae_candle": p.get("ma")}
