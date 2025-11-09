import requests
import pandas as pd
import numpy as np
import time
import os
import json
import pytz
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============ CONFIGURATION ============
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '8462496498:AAHYZ4xDIHvrVRjmCmZyoPhupCjRaRgiITc')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '203813932')

DEBUG_MODE = os.environ.get('DEBUG_MODE', 'True').lower() == 'true'
SEND_TEST_MESSAGE = os.environ.get('SEND_TEST_MESSAGE', 'True').lower() == 'true'

DELTA_API_BASE = "https://api.delta.exchange"

PAIRS = {
    "BTCUSD": None, "ETHUSD": None, "SOLUSD": None, "AVAXUSD": None, "BCHUSD": None,
    "XRPUSD": None, "BNBUSD": None, "LTCUSD": None, "DOTUSD": None, "ADAUSD": None,
    "SUIUSD": None, "AAVEUSD": None
}

SPECIAL_PAIRS = {
    "SOLUSD": {"limit_15m": 150, "min_required": 74, "limit_5m": 250, "min_required_5m": 183}
}

PPO_FAST, PPO_SLOW, PPO_SIGNAL, PPO_USE_SMA = 7, 16, 5, False
RMA_50_PERIOD, RMA_200_PERIOD = 50, 200
X1, X2, X3, X4 = 22, 9, 15, 5
SRSI_RSI_LEN, SRSI_KALMAN_LEN = 21, 5
STATE_FILE = 'alert_state.json'

# ============ UTILITIES ============

def debug_log(msg): 
    if DEBUG_MODE: 
        print(f"[DEBUG] {msg}")

def load_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                s = json.load(f)
                debug_log(f"Loaded state: {s}")
                return s
    except Exception as e:
        print(f"Error loading state: {e}")
    return {}

def save_state(s):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(s, f)
    except Exception as e:
        print(f"Error saving state: {e}")

def send_telegram_alert(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
        if r.json().get("ok"):
            print("✓ Alert sent")
            return True
        print("❌ Telegram error:", r.text)
    except Exception as e:
        print("❌ Telegram send failed:", e)
        if DEBUG_MODE: traceback.print_exc()
    return False

def send_test_message():
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist).strftime("%d-%m-%Y @ %H:%M IST")
    msg = f"🔔 Bot Started\nTest message\nTime: {now}\nDebug: {'ON' if DEBUG_MODE else 'OFF'}"
    send_telegram_alert(msg)

def get_product_ids():
    try:
        r = requests.get(f"{DELTA_API_BASE}/v2/products", timeout=10).json()
        if not r.get("success"): return False
        for p in r["result"]:
            symbol = p["symbol"].replace("_USDT","USD").replace("USDT","USD")
            if p.get("contract_type") != "perpetual_futures": continue
            for k in PAIRS:
                if symbol == k or symbol.replace("_","") == k:
                    PAIRS[k] = {"id": p["id"], "symbol": p["symbol"]}
        return True
    except Exception as e:
        print("Product fetch error:", e)
        if DEBUG_MODE: traceback.print_exc()
        return False

def get_candles(symbol, res="15", limit=150, retries=2):
    for i in range(retries):
        try:
            now, frm = int(time.time()), int(time.time()) - limit * int(res) * 60
            r = requests.get(f"{DELTA_API_BASE}/v2/chart/history",
                             params={"symbol": symbol, "resolution": res, "from": frm, "to": now},
                             timeout=15).json()
            if r.get("success"):
                d = r["result"]
                return pd.DataFrame({"timestamp": d["t"], "open": d["o"], "high": d["h"],
                                     "low": d["l"], "close": d["c"], "volume": d["v"]})
        except Exception as e:
            debug_log(f"Retry {i+1}/{retries} error: {e}")
            time.sleep(1)
    print(f"❌ Candle fetch failed for {symbol} {res}m")
    return None

# ============ INDICATORS ============

def ema(x,n): return x.ewm(span=n,adjust=False).mean()
def sma(x,n): return x.rolling(n).mean()
def rma(x,n): return x.ewm(alpha=1/n,adjust=False).mean()

def ppo(df,f=7,s=16,signal=5,use_sma=False):
    c=df.close
    fma,sma_ = (sma(c,f),sma(c,s)) if use_sma else (ema(c,f),ema(c,s))
    p=(fma-sma_)/sma_*100
    sig=sma(p,signal) if use_sma else ema(p,signal)
    return p,sig

def smoothrng(x,t,m):
    w=t*2-1
    return ema(ema(abs(x.diff()),t),w)*m

def rngfilt(x,r):
    out=[x.iloc[0]]
    for i in range(1,len(x)):
        pf,xx,rr=out[-1],x.iloc[i],r.iloc[i]
        out.append(pf if (xx>pf and xx-rr<pf) or (xx<=pf and xx+rr>pf)
                   else (xx-rr if xx>pf else xx+rr))
    return pd.Series(out,index=x.index)

def cirrus(df):
    c=df.close
    f1,f2=smoothrng(c,X1,X2),smoothrng(c,X3,X4)
    a,b=rngfilt(c,f1),rngfilt(c,f2)
    return a<b,a>b,a,b

def kalman(src,n,R=.01,Q=.1):
    res,est,err=[],np.nan,1.0
    em,qd=R*n,Q/n
    for v in src:
        if np.isnan(est):
            if len(res)>0: est=res[-1]
            else: res.append(np.nan); continue
        kg=err/(err+em)
        est=est+kg*(v-est)
        err=(1-kg)*err+qd
        res.append(est)
    return pd.Series(res,index=src.index)

def srsi(df):
    c=df.close
    d=c.diff()
    g,l=d.clip(lower=0),(-d).clip(lower=0)
    ag,al=rma(g,SRSI_RSI_LEN),rma(l,SRSI_RSI_LEN)
    rs=ag/al.replace(0,np.nan)
    rsi=100-100/(1+rs)
    return kalman(rsi,SRSI_KALMAN_LEN)

# ============ SIGNAL CHECK ============

def check_pair(name,info,last_alerts):
    try:
        if not info: return None
        sp=SPECIAL_PAIRS.get(name,{})
        lim15,req15=sp.get("limit_15m",210),sp.get("min_required",150)
        lim5,req5=max(sp.get("limit_5m",300),RMA_200_PERIOD+50),max(sp.get("min_required_5m",250),RMA_200_PERIOD)

        d15=get_candles(info["symbol"],"15",lim15)
        d5=get_candles(info["symbol"],"5",lim5)
        if d15 is None or len(d15)<req15: return None
        if d5 is None or len(d5)<req5: return None

        p,sig=ppo(d15,PPO_FAST,PPO_SLOW,PPO_SIGNAL,PPO_USE_SMA)
        r50,r200=rma(d15.close,RMA_50_PERIOD),rma(d5.close,RMA_200_PERIOD)
        upw,dnw,f1,f2=cirrus(d15)
        rs=srsi(d15)

        if any(pd.isna(v) for v in [p.iloc[-1],r50.iloc[-1],r200.iloc[-1],rs.iloc[-1]]):
            return None

        ppo_c,ppo_p=p.iloc[-1],p.iloc[-2]
        sig_c,sig_p=sig.iloc[-1],sig.iloc[-2]
        rsi_c,rsi_p=rs.iloc[-1],rs.iloc[-2]
        close_c=d15.close.iloc[-1]; open_c=d15.open.iloc[-1]
        high_c,low_c=d15.high.iloc[-1],d15.low.iloc[-1]
        upw_c,dnw_c=upw.iloc[-1],dnw.iloc[-1]
        close_5=r200.index and d5.close.iloc[-1]; r200_c=r200.iloc[-1]

        # Candle structure
        rng=high_c-low_c
        strong_bull=(close_c>open_c and rng>0 and (high_c-max(open_c,close_c))/rng<.2)
        strong_bear=(close_c<open_c and rng>0 and (min(open_c,close_c)-low_c)/rng<.2)

        # PPO crosses
        x_up=(ppo_p<=sig_p and ppo_c>sig_c)
        x_dn=(ppo_p>=sig_p and ppo_c<sig_c)
        above0=(ppo_p<=0 and ppo_c>0)
        below0=(ppo_p>=0 and ppo_c<0)
        above011=(ppo_p<=.11 and ppo_c>.11)
        below011=(ppo_p>=-.11 and ppo_c<-.11)

        cond=dict(
            ppo_up_sig=ppo_c>sig_c, ppo_dn_sig=ppo_c<sig_c,
            ppo_lt020=ppo_c<.2, ppo_gtm020=ppo_c>-.2,
            ppo_lt030=ppo_c<.3, ppo_gtm030=ppo_c>-.3,
            c_abv50=close_c>r50.iloc[-1], c_blw50=close_c<r50.iloc[-1],
            c_abv200=close_5>r200_c, c_blw200=close_5<r200_c,
            srsi_up=(rsi_p<=50 and rsi_c>50),
            srsi_dn=(rsi_p>=50 and rsi_c<50)
        )

        ist=pytz.timezone("Asia/Kolkata")
        t=datetime.now(ist).strftime("%d-%m-%Y @ %H:%M IST")
        price=close_c

        msg=None; state=None
        # --- signals ---
        if x_up and cond["ppo_lt020"] and cond["c_abv50"] and cond["c_abv200"] and upw_c and not dnw_c and strong_bull:
            state="buy"; msg=f"🟢 {name} BUY\nPPO CrossUp\nPrice ${price:.2f}\n{t}"
        elif x_dn and cond["ppo_gtm020"] and cond["c_blw50"] and cond["c_blw200"] and dnw_c and not upw_c and strong_bear:
            state="sell"; msg=f"🔴 {name} SELL\nPPO CrossDown\nPrice ${price:.2f}\n{t}"
        elif cond["srsi_up"] and cond["ppo_up_sig"] and cond["ppo_lt030"] and cond["c_abv50"] and cond["c_abv200"] and upw_c and strong_bull:
            state="buy_srsi"; msg=f"⬆️ {name} BUY (SRSI50)\nRSI {rsi_c:.2f}\n{t}"
        elif cond["srsi_dn"] and cond["ppo_dn_sig"] and cond["ppo_gtm030"] and cond["c_blw50"] and cond["c_blw200"] and dnw_c and strong_bear:
            state="sell_srsi"; msg=f"⬇️ {name} SELL (SRSI50)\nRSI {rsi_c:.2f}\n{t}"
        elif above0 and cond["ppo_up_sig"] and cond["c_abv50"] and cond["c_abv200"] and upw_c and strong_bull:
            state="long0"; msg=f"🟢 {name} LONG (0)\nPrice ${price:.2f}\n{t}"
        elif above011 and cond["ppo_up_sig"] and cond["c_abv50"] and cond["c_abv200"] and upw_c and strong_bull:
            state="long011"; msg=f"🟢 {name} LONG (0.11)\nPrice ${price:.2f}\n{t}"
        elif below0 and cond["ppo_dn_sig"] and cond["c_blw50"] and cond["c_blw200"] and dnw_c and strong_bear:
            state="short0"; msg=f"🔴 {name} SHORT (0)\nPrice ${price:.2f}\n{t}"
        elif below011 and cond["ppo_dn_sig"] and cond["c_blw50"] and cond["c_blw200"] and dnw_c and strong_bear:
            state="short011"; msg=f"🔴 {name} SHORT (-0.11)\nPrice ${price:.2f}\n{t}"

        if msg and last_alerts.get(name)!=state:
            send_telegram_alert(msg)
            return state
        return None
    except Exception as e:
        print("Error checking",name,":",e)
        if DEBUG_MODE: traceback.print_exc()
        return None

# ============ MAIN ============

def main():
    print("="*50)
    ist=pytz.timezone("Asia/Kolkata")
    start=datetime.now(ist)
    print(f"PPO/Cirrus Bot - {start.strftime('%d-%m-%Y @ %H:%M IST')}")
    print(f"Debug: {'ON' if DEBUG_MODE else 'OFF'}")
    print("="*50)
    if SEND_TEST_MESSAGE: send_test_message()
    state=load_state()
    if not get_product_ids():
        print("Product load failed"); return
    valid=sum(1 for v in PAIRS.values() if v)
    print(f"✓ Monitoring {valid} pairs")
    if valid==0: return
    alerts=0
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs={ex.submit(check_pair,k,v,state.copy()):k for k,v in PAIRS.items() if v}
        for f in as_completed(futs):
            k=futs[f]
            try:
                new=f.result()
                if new: state[k]=new; alerts+=1
            except Exception as e:
                print("Thread error",k,":",e)
                if DEBUG_MODE: traceback.print_exc()
    save_state(state)
    dur=(datetime.now(ist)-start).total_seconds()
    print(f"✓ Done {alerts} updates in {dur:.1f}s")
    print("="*50)

if __name__=="__main__":
    main()
