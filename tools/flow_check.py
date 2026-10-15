import os, sys, calendar, time
import numpy as np
import requests

BASE = os.environ["DELTA_API_BASE"].rstrip("/")
SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "RIVERUSD"
TARGET_IST = sys.argv[2] if len(sys.argv) > 2 else "2026-10-14 19:30"   # candle OPEN time, IST
FACTOR, EMA_LEN, SMA_LEN = 3.0, 15, 200

target_ts = calendar.timegm(time.strptime(TARGET_IST, "%Y-%m-%d %H:%M")) - 19800   # IST -> UTC

def fetch(n):
    r = requests.get(f"{BASE}/v2/chart/history", params={
        "resolution": "15", "symbol": SYMBOL,
        "from": target_ts - n * 900, "to": target_ts + 900}, timeout=30).json()["result"]
    t = np.array(r["t"], dtype=np.int64)
    o = {k: np.array(r[k], dtype=float) for k in ("h", "l", "c")}
    m = t <= target_ts
    return t[m], o["h"][m], o["l"][m], o["c"][m]

def flow(h, l, c):
    src = (h + l + c) / 3.0
    n = len(src)
    basis = np.empty(n); basis[0] = src[0]; a = 2.0 / (EMA_LEN + 1)
    for i in range(1, n):
        basis[i] = a * src[i] + (1 - a) * basis[i - 1]
    dist = np.full(n, np.nan)
    cs = np.cumsum(np.insert(h - l, 0, 0.0))
    dist[SMA_LEN - 1:] = (cs[SMA_LEN:] - cs[:-SMA_LEN]) / SMA_LEN
    lo_p = up_p = 0.0
    upper_prev = True
    out = []
    for i in range(n):
        d = dist[i]
        if np.isnan(d):
            lo, up = lo_p, up_p
        else:
            ru, rl = basis[i] + FACTOR * d, basis[i] - FACTOR * d
            sp = src[i - 1] if i else np.nan
            lo = rl if (rl > lo_p or sp < lo_p) else lo_p
            up = ru if (ru < up_p or sp > up_p) else up_p
        if i == 0 or np.isnan(dist[i - 1]):
            dr = 1.0
        elif upper_prev:
            dr = -1.0 if src[i] > up else 1.0
        else:
            dr = 1.0 if src[i] < lo else -1.0
        upper_prev = (dr == 1.0)
        out.append((dr, lo, up, (lo + up) / 2.0, src[i], d))
        lo_p, up_p = lo, up
    return out

print(f"{SYMBOL} candle open {TARGET_IST} IST")
for W in (300, 600, 1500):
    t, h, l, c = fetch(W)
    if len(t) == 0 or t[-1] != target_ts:
        print(f"window {W}: target candle not returned"); continue
    dr, lo, up, mid, s, d = flow(h, l, c)[-1]
    print(f"window {len(t):5d} candles | dir={dr:+.0f} ({'BEARISH' if dr == 1 else 'bullish'}) "
          f"hlc3={s:.5f} close={c[-1]:.5f} lowerBand={lo:.5f} upperBand={up:.5f} line={mid:.5f} dist={d:.5f}")