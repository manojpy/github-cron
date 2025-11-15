#!/usr/bin/env python3
import os, time, json, requests
from datetime import datetime, timezone

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
# Set the max allowed idle time for a bot's state file to 1 hour
MAX_IDLE_HOURS = 1.0 
STATE_FILE = ".watchdog_state.json"

def send_alert(msg):
    # Add a timestamp to the alert for clarity
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    full_msg = f"*{now_utc}*\n{msg}"
    try:
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": full_msg, "parse_mode": "Markdown"}, # Changed to Markdown for better formatting
            timeout=10
        )
    except requests.exceptions.RequestException as e:
        # Use specific exception for network errors
        print("Telegram send failed (Network/API issue):", e)
    except Exception as e:
        print("Telegram send failed (Other issue):", e)

def file_age_hours(path):
    try:
        return (time.time() - os.path.getmtime(path)) / 3600
    except FileNotFoundError:
        # Should not happen due to 'exists' check, but safe guard is good.
        return float('inf') 

ok = True

# Check 1: MACD bot
state_file_macd = "macd_state.sqlite"
if not os.path.exists(state_file_macd):
    ok = False
    send_alert(f"⚠️ *MACD Bot* state file `{state_file_macd}` missing!")
elif file_age_hours(state_file_macd) > MAX_IDLE_HOURS:
    ok = False
    hrs = file_age_hours(state_file_macd)
    send_alert(f"⚠️ *MACD Bot* has not updated for *{hrs:.1f} hours* (Limit: {MAX_IDLE_HOURS}h).")

# Check 2: Fibonacci bot
state_file_fib = "fib_state.sqlite"
if not os.path.exists(state_file_fib):
    ok = False
    send_alert(f"⚠️ *Fibonacci Bot* state file `{state_file_fib}` missing!")
elif file_age_hours(state_file_fib) > MAX_IDLE_HOURS:
    ok = False
    hrs = file_age_hours(state_file_fib)
    send_alert(f"⚠️ *Fibonacci Bot* has not updated for *{hrs:.1f} hours* (Limit: {MAX_IDLE_HOURS}h).")

# Check 3: Daily heartbeat persistence
last_ping = 0
if os.path.exists(STATE_FILE):
    try:
        with open(STATE_FILE, "r") as f:
            last_ping = json.load(f).get("last_ping", 0)
    except Exception:
        print("Warning: Could not read or parse watchdog state file.")
        last_ping = 0

hours_since = (time.time() - last_ping) / 3600

if ok and hours_since >= 24:
    send_alert("✅ *Watchdog Daily Health Check:*\nAll systems running normally.")
    with open(STATE_FILE, "w") as f:
        json.dump({"last_ping": time.time()}, f)
elif ok:
    print(f"Heartbeat OK. Last ping was {hours_since:.1f}h ago.")
else:
    print("Some bots unhealthy — alerts sent already.")
