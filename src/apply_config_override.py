#!/usr/bin/env python3
"""
apply_config_override.py — the manual, human-in-the-loop side of the
config hot-reload feature. The bot itself only ever READS the override
key from Redis (state.py:RedisStateStore.load_config_override) at the
start of each cron run; nothing in the bot ever writes to it. This
script is how a reviewed recommendation actually goes live, without a
git push + redeploy for a single number.

Deliberately NOT automatic. There is no "apply if EV improves" mode and
never will be in this script — that would just be the same overfitting
risk (a live-adapting system judging its own threshold off a small
sample) relocated from the bot into this script instead of solved.

Usage:
    # See what's currently overridden (if anything):
    python3 apply_config_override.py --show

    # Push a new override (only fields in the bot_config.py safelist
    # are ever honored — see CONFIG_OVERRIDE_ALLOWED_FIELDS):
    python3 apply_config_override.py --set CONFLUENCE_MIN_ABS_SCORE=24.5

    # Push several at once:
    python3 apply_config_override.py --set CONFLUENCE_MIN_ABS_SCORE=24.5 --set CONFLUENCE_MIN_PCT=78.0

    # Remove the override entirely (bot falls back to its static config):
    python3 apply_config_override.py --clear

Every write asks for confirmation unless --yes is passed. This talks
directly to Redis's "metadata:config_override" key — same key
state.py reads — so it works whether or not you have the rest of the
bot's dependencies (numba, aot_bridge, etc.) installed locally.
"""
import argparse
import json
import os
import sys

try:
    import redis
except ImportError:
    sys.exit("Missing dependency: pip install redis")

METADATA_PREFIX = "metadata:"
CONFIG_OVERRIDE_KEY = "config_override"
# Kept in sync with bot_config.py:CONFIG_OVERRIDE_ALLOWED_FIELDS by hand —
# duplicated here rather than imported so this script stays runnable
# without the rest of the bot's dependency stack. If you add a field to
# the safelist in bot_config.py, add it here too.
ALLOWED_FIELDS = {
    "CONFLUENCE_MIN_ABS_SCORE",
    "CONFLUENCE_MIN_PCT",
}


def parse_kv(s: str):
    if "=" not in s:
        sys.exit(f"--set expects FIELD=VALUE, got: {s!r}")
    field, value = s.split("=", 1)
    field = field.strip()
    if field not in ALLOWED_FIELDS:
        sys.exit(
            f"'{field}' is not in the allowed safelist ({sorted(ALLOWED_FIELDS)}). "
            f"The bot would ignore it anyway — add it to CONFIG_OVERRIDE_ALLOWED_FIELDS "
            f"in bot_config.py (and here) first if this is intentional."
        )
    try:
        parsed_value = float(value.strip())
    except ValueError:
        sys.exit(f"Value for {field} must be a number, got: {value!r}")
    return field, parsed_value


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                    help="Set one override field. Repeatable.")
    ap.add_argument("--clear", action="store_true", help="Remove the override entirely.")
    ap.add_argument("--show", action="store_true", help="Print the current override and exit.")
    ap.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    args = ap.parse_args()

    if not any([args.set, args.clear, args.show]):
        ap.error("Specify one of --show, --set FIELD=VALUE, or --clear")

    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        sys.exit("Set REDIS_URL in your environment first.")
    r = redis.from_url(redis_url, decode_responses=True)
    key = f"{METADATA_PREFIX}{CONFIG_OVERRIDE_KEY}"

    current_raw = r.get(key)
    current = json.loads(current_raw) if current_raw else {}

    if args.show:
        if not current:
            print("No config override currently set — bot is running its static config.")
        else:
            print("Current override (applied fresh at the start of every cron run):")
            for k, v in current.items():
                flag = "" if k in ALLOWED_FIELDS else "  ⚠️  NOT in allowed safelist — bot ignores this"
                print(f"  {k} = {v}{flag}")
        return

    if args.clear:
        if not current:
            print("Nothing to clear — no override is currently set.")
            return
        print(f"This will remove the override: {json.dumps(current, indent=2)}")
        if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
            print("Aborted.")
            return
        r.delete(key)
        print("✅ Override cleared. The bot falls back to its static config from the next run.")
        return

    updates = dict(parse_kv(s) for s in args.set)
    merged = {**current, **updates}

    print("Current override:", json.dumps(current, indent=2) if current else "(none)")
    print("Proposed override:", json.dumps(merged, indent=2))
    print("\nThis takes effect on the bot's NEXT cron run, and every run after until changed or cleared.")
    print("It is NOT validated against live data by this script — that's on you, from a")
    print("reviewed brain.py report or analyze_confluence_thresholds.py recommendation.")
    if not args.yes and input("\nPush this to Redis? [y/N] ").strip().lower() != "y":
        print("Aborted — nothing written.")
        return

    r.set(key, json.dumps(merged))
    print("✅ Pushed. Check the bot's next run log for a line starting with")
    print("   '⚙️ Config override active from Redis this run' to confirm it applied.")


if __name__ == "__main__":
    main()