#!/usr/bin/env python3
import os, time, json, requests

# --- Configuration ---
# Environment variables set in GitHub Action
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

STATE_FILE = ".watchdog_state.json"
# The bot runs every 15 mins. A 1-hour failure window is 4 consecutive missed runs.
BOT_FAILURE_THRESHOLD_HOURS = 1.0 
HEARTBEAT_THRESHOLD_HOURS = 24.5 # Check once per day, 30 minutes grace

def send_alert(msg):
    """Sends a message to Telegram."""
    if not TOKEN or not CHAT_ID:
        print("Warning: Telegram credentials not set. Alert not sent.")
        return

    try:
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10
        )
    except Exception as e:
        print("Telegram send failed:", e)

def file_age_hours(path):
    """Calculates the age of a file in hours."""
    try:
        return (time.time() - os.path.getmtime(path)) / 3600
    except FileNotFoundError:
        return float('inf')

ok = True

# --- MACD bot check ---
MACD_STATE_FILE = "macd_state.sqlite"
if not os.path.exists(MACD_STATE_FILE):
    ok = False
    send_alert(f"⚠️ MACD Bot state file ({MACD_STATE_FILE}) missing! Check GitHub Actions run logs.")
elif file_age_hours(MACD_STATE_FILE) > BOT_FAILURE_THRESHOLD_HOURS:
    ok = False
    hrs = file_age_hours(MACD_STATE_FILE)
    send_alert(f"🚨 MACD Bot has not updated its state for {hrs:.1f} hours (> {BOT_FAILURE_THRESHOLD_HOURS:.1f} hr threshold). Immediate attention required!")

# --- Fibonacci bot check ---
FIB_STATE_FILE = "fib_state.sqlite"
if not os.path.exists(FIB_STATE_FILE):
    ok = False
    send_alert(f"⚠️ Fibonacci Bot state file ({FIB_STATE_FILE}) missing! Check GitHub Actions run logs.")
elif file_age_hours(FIB_STATE_FILE) > BOT_FAILURE_THRESHOLD_HOURS:
    ok = False
    hrs = file_age_hours(FIB_STATE_FILE)
    send_alert(f"🚨 Fibonacci Bot has not updated its state for {hrs:.1f} hours (> {BOT_FAILURE_THRESHOLD_HOURS:.1f} hr threshold). Immediate attention required!")

# --- Daily heartbeat persistence ---
last_ping = 0
current_time = time.time()
if os.path.exists(STATE_FILE):
    try:
        with open(STATE_FILE, "r") as f:
            last_ping = json.load(f).get("last_ping", 0)
    except Exception:
        last_ping = 0

hours_since = (current_time - last_ping) / 3600

if hours_since > HEARTBEAT_THRESHOLD_HOURS:
    # Send daily heartbeat (or failure if all is OK but heartbeat is too old)
    if ok:
        send_alert(f"💚 Watchdog Daily Heartbeat: Both bots are operating normally. Last state update was {BOT_FAILURE_THRESHOLD_HOURS:.1f} hours ago or less.")
    else:
        # Only alert if there was an underlying issue, otherwise the previous alerts suffice
        print(f"Watchdog failure detected, skipping heartbeat. Hours since last ping: {hours_since:.1f}")

    # Update state file regardless of success to reset the daily counter
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({"last_ping": current_time}, f)
    except Exception as e:
        print("Could not write watchdog state:", e)
 
