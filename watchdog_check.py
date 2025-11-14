#!/usr/bin/env python3
import os, time, json, requests

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

STATE_FILE = ".watchdog_state.json"

def send_alert(msg):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10
        )
    except Exception as e:
        print("Telegram send failed:", e)

def file_age_hours(path):
    return (time.time() - os.path.getmtime(path)) / 3600

ok = True

# MACD bot check
if not os.path.exists("macd_state.sqlite"):
    ok = False
    send_alert("⚠️ MACD Bot state file missing!")
elif file_age_hours("macd_state.sqlite") > 2:
    ok = False
    hrs = file_age_hours("macd_state.sqlite")
    send_alert(f"⚠️ MACD Bot has not updated for {hrs:.1f} hours.")

# Fibonacci bot check
if not os.path.exists("fib_state.sqlite"):
    ok = False
    send_alert("⚠️ Fibonacci Bot state file missing!")
elif file_age_hours("fib_state.sqlite") > 2:
    ok = False
    hrs = file_age_hours("fib_state.sqlite")
    send_alert(f"⚠️ Fibonacci Bot has not updated for {hrs:.1f} hours.")

# Daily heartbeat persistence
last_ping = 0
if os.path.exists(STATE_FILE):
    try:
        with open(STATE_FILE, "r") as f:
            last_ping = json.load(f).get("last_ping", 0)
    except Exception:
        last_ping = 0

hours_since = (time.time() - last_ping) / 3600

if ok and hours_since >= 24:
    send_alert("✅ Watchdog Daily Health Check:\nAll systems running normally.")
    with open(STATE_FILE, "w") as f:
        json.dump({"last_ping": time.time()}, f)
elif ok:
    print(f"Heartbeat OK. Last ping was {hours_since:.1f}h ago.")
else:
    print("Some bots unhealthy — alerts sent already.")
