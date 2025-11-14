#!/usr/bin/env python3
import os, time, json, requests
from datetime import datetime

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

STATE_FILE = ".watchdog_state.json"
REPO = os.getenv("GITHUB_REPOSITORY", "your-username/your-repo")  # Fallback if not in Actions

def send_alert(msg):
    """Send formatted alert with timestamp"""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    full_msg = f"🤖 <b>Bot Watchdog Alert</b>\n⏰ {timestamp}\n\n{msg}"
    
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": full_msg, "parse_mode": "HTML"},
            timeout=10
        )
        if response.status_code == 200:
            print(f"Alert sent: {msg}")
        else:
            print(f"Telegram API error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Telegram send failed: {e}")

def send_critical_alert(msg):
    """Send critical alert - could extend with additional notification methods"""
    send_alert(f"🚨 CRITICAL: {msg}")
    # Future: Add Discord, email, or other notification methods here

def file_age_hours(path):
    """Calculate file age in hours, return 999 if file doesn't exist"""
    if not os.path.exists(path):
        return 999
    return (time.time() - os.path.getmtime(path)) / 3600

def check_recent_workflows():
    """Check if bots have run recently via GitHub Actions API"""
    if not GITHUB_TOKEN:
        print("No GITHUB_TOKEN available - skipping workflow check")
        return True, []
    
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    workflows_to_check = ["macd-bot", "fibonacci-bot"]  # Adjust to your workflow names
    problems = []
    
    for workflow in workflows_to_check:
        try:
            # Get workflow runs (most recent first)
            url = f"https://api.github.com/repos/{REPO}/actions/runs"
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                runs = response.json()["workflow_runs"]
                recent_runs = [r for r in runs if workflow in r["name"].lower()]
                
                if recent_runs:
                    latest_run = recent_runs[0]
                    run_time = datetime.fromisoformat(latest_run["created_at"].replace('Z', '+00:00'))
                    hours_ago = (datetime.utcnow() - run_time).total_seconds() / 3600
                    
                    if hours_ago > 6:  # Alert if no run in 6 hours
                        problems.append(f"{workflow}: last run {hours_ago:.1f}h ago")
                    elif latest_run["conclusion"] != "success":
                        problems.append(f"{workflow}: last run failed")
                else:
                    problems.append(f"{workflow}: no recent runs found")
            else:
                problems.append(f"{workflow}: API error {response.status_code}")
                
        except Exception as e:
            problems.append(f"{workflow}: check failed - {e}")
    
    return len(problems) == 0, problems

def main():
    print("🤖 Starting Bot Watchdog Check...")
    print("Current directory:", os.getcwd())
    print("Files in directory:", [f for f in os.listdir('.') if not f.startswith('.')])
    
    ok = True
    alerts_sent = False

    # Check state files with grace period for new deployments
    state_files = {
        "MACD Bot": "macd_state.sqlite",
        "Fibonacci Bot": "fib_state.sqlite"
    }
    
    for bot_name, state_file in state_files.items():
        age_hours = file_age_hours(state_file)
        
        if age_hours == 999:
            ok = False
            alerts_sent = True
            send_alert(f"⚠️ {bot_name} state file missing!\nFile: {state_file}")
        elif age_hours > 24:
            ok = False
            alerts_sent = True
            send_critical_alert(f"🚨 {bot_name} has not updated for {age_hours:.1f} hours!")
        elif age_hours > 2:
            # Only alert if file is between 2-24 hours old (grace period for new deployments)
            if age_hours < 24:
                ok = False
                alerts_sent = True
                send_alert(f"⚠️ {bot_name} has not updated for {age_hours:.1f} hours.")
        else:
            print(f"✅ {bot_name}: state file age {age_hours:.1f}h - OK")

    # Check recent workflow runs
    workflows_ok, workflow_problems = check_recent_workflows()
    if not workflows_ok:
        ok = False
        alerts_sent = True
        problems_text = "\n".join(f"• {p}" for p in workflow_problems)
        send_alert(f"⚠️ Workflow Issues:\n{problems_text}")

    # Daily heartbeat persistence
    last_ping = 0
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                state_data = json.load(f)
                last_ping = state_data.get("last_ping", 0)
                last_status = state_data.get("last_status", "unknown")
        except Exception as e:
            print(f"Error reading state file: {e}")
            last_ping = 0
            last_status = "error"

    hours_since = (time.time() - last_ping) / 3600

    if ok and hours_since >= 24:
        # Send daily all-clear message
        send_alert("✅ <b>Daily Health Check</b>\n\nAll systems running normally:\n• State files current\n• Workflows active\n• No critical issues")
        with open(STATE_FILE, "w") as f:
            json.dump({"last_ping": time.time(), "last_status": "healthy"}, f)
        print("Daily heartbeat sent - all systems OK")
    elif ok:
        print(f"Heartbeat OK. Last ping was {hours_since:.1f}h ago. Status: healthy")
        # Update state file even if we don't send a message
        with open(STATE_FILE, "w") as f:
            json.dump({"last_ping": time.time(), "last_status": "healthy"}, f)
    else:
        print("Some bots unhealthy — alerts sent already.")
        # Update state file with error status
        with open(STATE_FILE, "w") as f:
            json.dump({"last_ping": time.time(), "last_status": "errors"}, f)

    print("🤖 Watchdog check completed")

if __name__ == "__main__":
    main()
