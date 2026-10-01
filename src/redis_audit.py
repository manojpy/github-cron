#!/usr/bin/env python3

"""
redis_audit.py — Redis key inventory + TTL audit (read-only by default).

By default it only calls SCAN / TYPE / TTL / XLEN / MEMORY USAGE and never
writes. With --heal-leaks it additionally EXPIRE's keys in ttl_required
families that currently have no TTL (-1).

Usage:
    python3 redis_audit.py                    # uses $REDIS_URL, prints a table
    python3 redis_audit.py --json             # machine-readable report
    python3 redis_audit.py --fail-on-findings # exit 1 if any warning/error
    python3 redis_audit.py --heal-leaks       # EXPIRE no-TTL keys in ttl_required families
    python3 redis_audit.py --heal-leaks --dry-run
    python3 redis_audit.py --url redis://...  # explicit URL
    python3 redis_audit.py --max-keys 50000   # stop scanning after N keys

Exit codes: 0 ok, 1 findings (only with --fail-on-findings), 2 cannot connect.

Like apply_config_override.py it needs only the `redis` package, not the
rest of the bot's dependencies.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

# (name, key prefix, ttl policy). Longest prefix wins.
#   ttl_required : every key in the family is written with an expiry; a key
#                  with no TTL (-1) is a leak.
#   no_ttl_ok    : intentionally persistent (streams, counters).
#   any          : not asserted either way.
FAMILIES: List[Tuple[str, str, str]] = [
    ("pair_state", "pair_state:", "ttl_required"),
    ("alert_dedup", "alert:", "ttl_required"),
    ("recent_alert", "recent_alert:", "ttl_required"),
    ("lock", "lock:", "ttl_required"),
    ("outcome_pending", "outcome_pending:", "ttl_required"),
    ("alert_stats", "alert_stats:", "ttl_required"),
    ("shadow_pending", "shadow_pending:", "ttl_required"),
    ("shadow_stats", "shadow_stats:", "ttl_required"),
    ("shadow_hiconf", "shadow_hiconf:", "ttl_required"),
    ("brain_cusum_watermark", "brain_cusum_watermark:", "ttl_required"),
    ("brain_cusum", "brain_cusum:", "ttl_required"),
    ("brain_threshold_history", "brain_threshold_history:", "ttl_required"),
    ("brain_vote_counts", "brain_vote_counts:", "ttl_required"),
    ("last_processed_candle", "last_processed_candle:", "ttl_required"),
    ("brain_filter_tg", "brain_filter_tg:", "ttl_required"),
    ("metadata", "metadata:", "ttl_required"),
    ("brain_blob", "brain:", "any"),
    ("outcome_log_stream", "outcome_log_stream", "no_ttl_ok"),
    ("shadow_log_stream", "shadow_log_stream", "no_ttl_ok"),
    ("brain_run_counter", "brain_run_counter", "no_ttl_ok"),
]
_FAMILIES_BY_LEN = sorted(FAMILIES, key=lambda f: -len(f[1]))

KNOWN_STREAMS = ("outcome_log_stream", "shadow_log_stream")

# Metadata entries that record a standing decision. They are written with the
# default metadata TTL unless a caller passes one, so they silently lapse.
DURABLE_METADATA = {
    "config_override",
    "brain_disabled_alert_keys",
    "brain_alert_key_history",
    "pair_confluence_thresholds",
    "dynamic_weights",
    "brain_apply_snapshots",
}
DURABLE_WARN_BELOW_DAYS = 8.0
# Used only by --heal-leaks. Matches CUSUM / plan-history conventions.
HEAL_TTL_SECONDS = 30 * 86400
HEAL_DURABLE_TTL_SECONDS = 365 * 86400

def _text(key: Any) -> str:
    return key.decode("utf-8", "replace") if isinstance(key, (bytes, bytearray)) else str(key)


def classify_key(key: str) -> Tuple[str, str]:
    """Return (family, ttl_policy); ("unknown", "any") for an orphan key."""
    for name, prefix, policy in _FAMILIES_BY_LEN:
        if key.startswith(prefix):
            return name, policy
    return "unknown", "any"


def metadata_name(key: str) -> str:
    """'metadata:daily_cache:BTCUSD' -> 'daily_cache'; 'metadata:config_override' -> same."""
    rest = key[len("metadata:"):]
    return rest.split(":", 1)[0]


def _ttl_stats(ttls: List[int]) -> Dict[str, Optional[float]]:
    live = [t for t in ttls if t >= 0]
    if not live:
        return {"min": None, "median": None, "max": None}
    return {"min": min(live), "median": statistics.median(live), "max": max(live)}


def summarize(
    records: Iterable[Dict[str, Any]],
    stream_lengths: Optional[Dict[str, int]] = None,
    sample_examples: int = 5,
) -> Dict[str, Any]:
    """Pure function: turn per-key records into the audit report.

    Each record: {"key": str, "type": str, "ttl": int, "bytes": Optional[int]}
    where ttl follows Redis (-1 = no expiry, -2 = gone, otherwise seconds).
    """
    fam: Dict[str, Dict[str, Any]] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    orphans: List[str] = []
    leaks: Dict[str, List[str]] = {}
    total = 0

    for rec in records:
        key, ttl = rec["key"], int(rec["ttl"])
        if ttl == -2:  # expired between SCAN and TTL
            continue
        total += 1
        name, policy = classify_key(key)
        f = fam.setdefault(name, {
            "policy": policy, "keys": 0, "no_ttl": 0, "ttls": [],
            "bytes": 0, "bytes_n": 0, "types": {},
        })
        f["keys"] += 1
        f["types"][rec.get("type", "?")] = f["types"].get(rec.get("type", "?"), 0) + 1
        if ttl == -1:
            f["no_ttl"] += 1
        else:
            f["ttls"].append(ttl)
        if rec.get("bytes") is not None:
            f["bytes"] += int(rec["bytes"])
            f["bytes_n"] += 1

        if name == "unknown":
            orphans.append(key)
        elif policy == "ttl_required" and ttl == -1:
            leaks.setdefault(name, []).append(key)

        if name == "metadata":
            mn = metadata_name(key)
            m = meta.setdefault(mn, {"keys": 0, "no_ttl": 0, "ttls": []})
            m["keys"] += 1
            if ttl == -1:
                m["no_ttl"] += 1
            else:
                m["ttls"].append(ttl)

    families_out: Dict[str, Any] = {}
    for name, f in fam.items():
        est = None
        if f["bytes_n"]:
            est = int(f["bytes"] / f["bytes_n"] * f["keys"])
        families_out[name] = {
            "policy": f["policy"], "keys": f["keys"], "no_ttl": f["no_ttl"],
            "ttl": _ttl_stats(f["ttls"]), "bytes_est": est, "types": f["types"],
        }
    metadata_out = {
        n: {"keys": m["keys"], "no_ttl": m["no_ttl"], "ttl": _ttl_stats(m["ttls"])}
        for n, m in sorted(meta.items())
    }

    findings: List[Dict[str, Any]] = []
    if orphans:
        findings.append({
            "severity": "warning", "kind": "ORPHAN_KEYS", "family": "unknown",
            "count": len(orphans),
            "detail": "keys that match no known family (renamed prefix or leftover from an old version?)",
            "examples": sorted(orphans)[:sample_examples],
        })
    for name, keys in sorted(leaks.items()):
        findings.append({
            "severity": "error", "kind": "NO_TTL_LEAK", "family": name,
            "count": len(keys),
            "detail": "family is always written with an expiry, but these keys never expire",
            "examples": sorted(keys)[:sample_examples],
        })
    for mn in sorted(DURABLE_METADATA & set(meta)):
        stats = metadata_out[mn]["ttl"]
        if stats["max"] is not None and stats["max"] / 86400.0 < DURABLE_WARN_BELOW_DAYS:
            findings.append({
                "severity": "info", "kind": "DURABLE_DECISION_FINITE_TTL", "family": f"metadata:{mn}",
                "count": metadata_out[mn]["keys"],
                "detail": (
                    f"standing decision expires in at most {stats['max'] / 86400.0:.1f}d "
                    f"unless rewritten (default metadata TTL is 7d)"
                ),
                "examples": [],
            })

    return {
        "scanned": total,
        "families": dict(sorted(families_out.items())),
        "metadata": metadata_out,
        "streams": dict(stream_lengths or {}),
        "findings": findings,
    }


def collect(
    client: Any,
    max_keys: int = 200_000,
    scan_count: int = 500,
    sample_bytes: int = 25,
    match: str = "*",
) -> Tuple[List[Dict[str, Any]], Dict[str, int], bool]:
    """Read-only scan. Returns (records, stream_lengths, truncated)."""
    keys: List[str] = []
    truncated = False
    for raw in client.scan_iter(match=match, count=scan_count):
        keys.append(_text(raw))
        if len(keys) >= max_keys:
            truncated = True
            break

    records: List[Dict[str, Any]] = []
    for i in range(0, len(keys), scan_count):
        chunk = keys[i:i + scan_count]
        pipe = client.pipeline(transaction=False)
        for k in chunk:
            pipe.type(k)
            pipe.ttl(k)
        res = pipe.execute()
        for j, k in enumerate(chunk):
            ktype = _text(res[2 * j])
            ttl = int(res[2 * j + 1])
            records.append({"key": k, "type": ktype, "ttl": ttl, "bytes": None})

    # Byte sizes are sampled per family (MEMORY USAGE can be slow, and some
    # managed Redis offerings do not support it at all).
    seen: Dict[str, int] = {}
    for rec in records:
        name, _ = classify_key(rec["key"])
        if seen.get(name, 0) >= sample_bytes:
            continue
        try:
            rec["bytes"] = client.memory_usage(rec["key"])
        except Exception:
            rec["bytes"] = None
            seen[name] = sample_bytes  # unsupported: stop trying for this family
            continue
        seen[name] = seen.get(name, 0) + 1

    streams: Dict[str, int] = {}
    for s in KNOWN_STREAMS:
        try:
            streams[s] = int(client.xlen(s))
        except Exception:
            pass
    return records, streams, truncated

def heal_leaks(
    client: Any,
    records: List[Dict[str, Any]],
    dry_run: bool = False,
    ttl_seconds: int = HEAL_TTL_SECONDS,
    durable_ttl_seconds: int = HEAL_DURABLE_TTL_SECONDS,
) -> Dict[str, Any]:
    """EXPIRE keys in ttl_required families that currently have no TTL.

    Durable metadata names (DURABLE_METADATA) get durable_ttl_seconds so
    standing decisions are not shortened to the default 30d heal window.
    """
    healed: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for rec in records:
        key = rec["key"]
        ttl = int(rec["ttl"])
        if ttl != -1:
            continue
        name, policy = classify_key(key)
        if policy != "ttl_required":
            continue

        expire_for = ttl_seconds
        if name == "metadata":
            mn = metadata_name(key)
            if mn in DURABLE_METADATA:
                expire_for = durable_ttl_seconds

        entry = {"key": key, "family": name, "ttl_set": expire_for, "dry_run": dry_run}
        if dry_run:
            healed.append(entry)
            continue
        try:
            client.expire(key, expire_for)
            healed.append(entry)
        except Exception as e:
            skipped.append(f"{key}: {e}")

    return {
        "healed": len(healed),
        "skipped": len(skipped),
        "dry_run": dry_run,
        "details": healed,
        "errors": skipped,
    }

def _fmt_ttl(sec: Optional[float]) -> str:
    if sec is None:
        return "-"
    if sec >= 86400:
        return f"{sec / 86400:.1f}d"
    if sec >= 3600:
        return f"{sec / 3600:.1f}h"
    return f"{int(sec)}s"


def _fmt_bytes(n: Optional[int]) -> str:
    if n is None:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0  # type: ignore[assignment]
    return f"{n:.1f}TB"


def render(report: Dict[str, Any], truncated: bool = False) -> str:
    lines: List[str] = []
    lines.append(f"Redis key audit — {report['scanned']} keys scanned"
                 + (" (TRUNCATED at --max-keys)" if truncated else ""))
    lines.append("")
    lines.append(f"{'family':26s} {'keys':>7s} {'no-TTL':>7s} {'ttl min':>8s} {'median':>8s} {'max':>8s} {'~size':>9s}")
    for name, f in report["families"].items():
        t = f["ttl"]
        lines.append(
            f"{name:26s} {f['keys']:>7d} {f['no_ttl']:>7d} "
            f"{_fmt_ttl(t['min']):>8s} {_fmt_ttl(t['median']):>8s} {_fmt_ttl(t['max']):>8s} "
            f"{_fmt_bytes(f['bytes_est']):>9s}"
        )
    if report["streams"]:
        lines.append("")
        lines.append("streams: " + ", ".join(f"{k}={v}" for k, v in report["streams"].items()))
    if report["metadata"]:
        lines.append("")
        lines.append(f"{'metadata entry':34s} {'keys':>6s} {'no-TTL':>7s} {'ttl max':>8s}")
        for n, m in report["metadata"].items():
            lines.append(f"{n:34s} {m['keys']:>6d} {m['no_ttl']:>7d} {_fmt_ttl(m['ttl']['max']):>8s}")
    lines.append("")
    if not report["findings"]:
        lines.append("Findings: none")
    else:
        lines.append("Findings:")
        for fd in report["findings"]:
            lines.append(f"  [{fd['severity'].upper()}] {fd['kind']} {fd['family']} (n={fd['count']}): {fd['detail']}")
            for ex in fd.get("examples", []):
                lines.append(f"      e.g. {ex}")
    return "\n".join(lines)

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.getenv("REDIS_URL"), help="Redis URL (default: $REDIS_URL)")
    ap.add_argument("--json", action="store_true", help="Print the report as JSON")
    ap.add_argument("--fail-on-findings", action="store_true", help="Exit 1 on any warning/error finding")
    ap.add_argument("--heal-leaks", action="store_true", help="EXPIRE ttl_required keys that currently have no TTL(default off; audit stays read-only)")
    ap.add_argument("--dry-run", action="store_true", help="With --heal-leaks: print what would be expired without writing")
    ap.add_argument("--max-keys", type=int, default=200_000, help="Stop scanning after N keys")
    ap.add_argument("--scan-count", type=int, default=500)
    ap.add_argument("--sample-bytes", type=int, default=25, help="Keys per family sampled for MEMORY USAGE")
    ap.add_argument("--match", default="*", help="SCAN MATCH pattern (default: *)")
    args = ap.parse_args(argv)

    if not args.url:
        print("No Redis URL: set REDIS_URL or pass --url", file=sys.stderr)
        return 2
    try:
        import redis  # local import: only needed when actually running
    except ImportError:
        print("Missing dependency: pip install redis", file=sys.stderr)
        return 2
    try:
        client = redis.from_url(args.url, socket_timeout=10, socket_connect_timeout=10)
        client.ping()
    except Exception as e:
        print(f"Cannot connect to Redis: {e}", file=sys.stderr)
        return 2

    records, streams, truncated = collect(
        client, max_keys=args.max_keys, scan_count=args.scan_count,
        sample_bytes=args.sample_bytes, match=args.match,
    )
    report = summarize(records, streams)
    report["truncated"] = truncated

    if args.heal_leaks:
        heal_report = heal_leaks(client, records, dry_run=args.dry_run)
        report["heal"] = {
            "healed": heal_report["healed"],
            "skipped": heal_report["skipped"],
            "dry_run": heal_report["dry_run"],
            "errors": heal_report["errors"],
        }
        # Re-scan TTLs for a post-heal summary when we actually wrote.
        if not args.dry_run and heal_report["healed"]:
            records2, streams2, truncated2 = collect(
                client, max_keys=args.max_keys, scan_count=args.scan_count,
                sample_bytes=args.sample_bytes, match=args.match,
            )
            report = summarize(records2, streams2)
            report["truncated"] = truncated2
            report["heal"] = {
                "healed": heal_report["healed"],
                "skipped": heal_report["skipped"],
                "dry_run": False,
                "errors": heal_report["errors"],
            }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render(report, report.get("truncated", False)))
        if args.heal_leaks:
            h = report.get("heal", {})
            mode = "dry-run" if h.get("dry_run") else "applied"
            print("")
            print(f"Heal ({mode}): expired {h.get('healed', 0)} key(s), "
                  f"errors={h.get('skipped', 0)}")
            for err in h.get("errors") or []:
                print(f"  ! {err}")

    if args.fail_on_findings and any(f["severity"] in ("warning", "error") for f in report["findings"]):
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
