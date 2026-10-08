#!/usr/bin/env python3
"""
redis_audit.py — read-only Redis key inventory + TTL audit.

Answers, without touching anything: what is in Redis, how much of it, which
keys never expire, which keys belong to no known family, and which durable
Brain decisions only live as long as their TTL.

By default it only calls SCAN / TYPE / TTL / XLEN / MEMORY USAGE. It never
writes, deletes or expires a key, so it is safe to run against production.

The only write exceptions are the opt-in heal flags (see below). Both only
ever set or raise a TTL; nothing is deleted and no value is changed.

Usage:
    python3 redis_audit.py                    # uses $REDIS_URL, prints a table
    python3 redis_audit.py --json             # machine-readable report
    python3 redis_audit.py --fail-on-findings # exit 1 if any warning/error
    python3 redis_audit.py --url redis://...  # explicit URL
    python3 redis_audit.py --max-keys 50000   # stop scanning after N keys
    python3 redis_audit.py --heal-durable-ttl            # one-off: raise old 7d TTLs to 90d
    python3 redis_audit.py --heal-durable-ttl --dry-run  # same, but only show what would change
    python3 redis_audit.py --heal-ttl-leaks              # one-off: set 30d TTL on NO_TTL_LEAK keys
    python3 redis_audit.py --heal-ttl-leaks --dry-run    # same, but only show what would change

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
    ("trade_cooldown", "trade_cooldown:", "ttl_required"),
    ("brain_filter_tg", "brain_filter_tg:", "ttl_required"),
    ("telegram_dlq", "telegram_dlq:", "ttl_required"),
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

# One-off heal for standing decisions that were written under the old 7-day
# default TTL (state.py now writes them with DURABLE_METADATA_TTL_SEC, but a
# key only picks that up the next time it is rewritten).
HEAL_TARGET_TTL_SEC = 90 * 86400
# dynamic_weights is written with an explicit 30d TTL by design and
# brain_apply_snapshots with 365d; neither is ever touched by the heal.
HEAL_EXEMPT = {"dynamic_weights", "brain_apply_snapshots"}
HEAL_METADATA = DURABLE_METADATA - HEAL_EXEMPT
LEAK_HEAL_FAMILIES = {
    "brain_threshold_history",
    "brain_vote_counts",
}
LEAK_HEAL_TTL_SEC = 30 * 86400

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


def collect_heal_candidates(client: Any) -> List[Dict[str, Any]]:
    """Look up the TTL of each healable key directly by name (one pipelined
    round trip). Unlike the SCAN this cannot be cut short by --max-keys."""
    keys = [f"metadata:{n}" for n in sorted(HEAL_METADATA)]
    pipe = client.pipeline(transaction=False)
    for k in keys:
        pipe.ttl(k)
    res = pipe.execute()
    return [{"key": k, "ttl": int(t)} for k, t in zip(keys, res)]


def plan_durable_ttl_heal(
    records: Iterable[Dict[str, Any]],
    target_ttl: int = HEAL_TARGET_TTL_SEC,
) -> List[Dict[str, Any]]:
    """Pure: which keys would be raised. Only an exact 'metadata:<name>' key
    from HEAL_METADATA with a finite, positive TTL below target is planned.
    Absent keys (-2), no-expiry keys (-1) and keys already at/above target are
    left alone -- the heal can only ever lengthen a lifetime."""
    plan: List[Dict[str, Any]] = []
    for rec in records:
        key, ttl = str(rec["key"]), int(rec["ttl"])
        if not key.startswith("metadata:") or key[len("metadata:"):] not in HEAL_METADATA:
            continue
        if 0 < ttl < target_ttl:
            plan.append({"key": key, "ttl_before": ttl, "target": target_ttl})
    return sorted(plan, key=lambda p: p["key"])


def _raise_ttl(client: Any, key: str, target: int) -> bool:
    """EXPIRE ... GT: the server itself refuses to shorten a TTL, so a bot
    write that lands between our read and this call cannot be undone. Servers
    without GT (Redis < 7) fall back to re-reading the TTL first."""
    try:
        return bool(client.expire(key, target, gt=True))
    except Exception:
        ttl = int(client.ttl(key))
        if 0 < ttl < target:
            return bool(client.expire(key, target))
        return False


def apply_durable_ttl_heal(
    client: Any, plan: List[Dict[str, Any]], dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute (or, with dry_run, only describe) the plan, then re-read each
    TTL so the report shows the verified 'after' value."""
    rows: List[Dict[str, Any]] = []
    for p in plan:
        row = dict(p)
        if dry_run:
            row["status"] = "would_raise"
        else:
            changed = _raise_ttl(client, p["key"], p["target"])
            row["ttl_after"] = int(client.ttl(p["key"]))
            row["status"] = "raised" if changed else "skipped"
        rows.append(row)
    return {"target_days": HEAL_TARGET_TTL_SEC / 86400.0, "dry_run": dry_run, "keys": rows}

def plan_ttl_leak_heal(
    records: Iterable[Dict[str, Any]],
    target_ttl: int = LEAK_HEAL_TTL_SEC,
) -> List[Dict[str, Any]]:
    """Pure: keys that are NO_TTL_LEAK *and* belong to an allow-listed family.

    Only families in LEAK_HEAL_FAMILIES are healed automatically.
    """
    plan: List[Dict[str, Any]] = []
    for rec in records:
        key, ttl = str(rec["key"]), int(rec["ttl"])
        if ttl != -1:
            continue
        name, policy = classify_key(key)
        if policy != "ttl_required":
            continue
        if name not in LEAK_HEAL_FAMILIES:
            continue
        plan.append({
            "key": key,
            "family": name,
            "ttl_before": -1,
            "target": target_ttl,
        })
    return sorted(plan, key=lambda p: p["key"])


def _set_ttl_only_if_missing(client: Any, key: str, target_ttl: int) -> bool:
    """Set TTL only if the key currently has no TTL.

    Uses EXPIRE ... NX when available. Falls back to TTL check + EXPIRE
    for older Redis servers.
    """
    try:
        return bool(client.expire(key, target_ttl, nx=True))
    except Exception:
        try:
            if int(client.ttl(key)) == -1:
                return bool(client.expire(key, target_ttl))
        except Exception:
            return False
        return False


def apply_ttl_leak_heal(
    client: Any,
    plan: List[Dict[str, Any]],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute or describe the leak-heal plan, then re-read each TTL."""
    rows: List[Dict[str, Any]] = []
    for p in plan:
        row = dict(p)
        if dry_run:
            row["status"] = "would_set"
        else:
            changed = _set_ttl_only_if_missing(client, p["key"], p["target"])
            try:
                row["ttl_after"] = int(client.ttl(p["key"]))
            except Exception:
                row["ttl_after"] = None
            row["status"] = "set" if changed else "skipped"
        rows.append(row)
    return {
        "target_days": LEAK_HEAL_TTL_SEC / 86400.0,
        "dry_run": dry_run,
        "keys": rows,
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

    heal = report.get("heal")
    if heal is not None:
        lines.append("")
        mode = "DRY RUN — nothing written" if heal["dry_run"] else "applied"
        lines.append(f"Durable-TTL heal → {heal['target_days']:.0f}d ({mode})")
        if not heal["keys"]:
            lines.append("  nothing to heal: every durable decision key is absent, exempt, or already at the target")
        for row in heal["keys"]:
            after = f" → {_fmt_ttl(row['ttl_after'])}" if "ttl_after" in row else ""
            lines.append(f"  {row['key']}: {_fmt_ttl(row['ttl_before'])}{after} [{row['status']}]")

    leak_heal = report.get("leak_heal")
    if leak_heal is not None:
        lines.append("")
        mode = "DRY RUN — nothing written" if leak_heal["dry_run"] else "applied"
        lines.append(f"TTL-leak heal → {leak_heal['target_days']:.0f}d ({mode})")
        if not leak_heal["keys"]:
            lines.append("  nothing to heal: no allow-listed NO_TTL_LEAK keys found")
        for row in leak_heal["keys"]:
            after = f" → {_fmt_ttl(row['ttl_after'])}" if "ttl_after" in row else ""
            lines.append(
                f"  {row['key']} ({row['family']}): "
                f"{_fmt_ttl(row['ttl_before'])}{after} [{row['status']}]"
            )
    return "\n".join(lines)

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.getenv("REDIS_URL"), help="Redis URL (default: $REDIS_URL)")
    ap.add_argument("--json", action="store_true", help="Print the report as JSON")
    ap.add_argument("--fail-on-findings", action="store_true", help="Exit 1 on any warning/error finding")
    ap.add_argument("--max-keys", type=int, default=200_000, help="Stop scanning after N keys")
    ap.add_argument("--scan-count", type=int, default=500)
    ap.add_argument("--sample-bytes", type=int, default=25, help="Keys per family sampled for MEMORY USAGE")
    ap.add_argument("--match", default="*", help="SCAN MATCH pattern (default: *)")
    ap.add_argument("--heal-durable-ttl", action="store_true",
                    help="One-off, opt-in WRITE: raise standing-decision keys still on the old 7d TTL to 90d "
                         "(never lowers a TTL, never touches dynamic_weights / brain_apply_snapshots)")
    ap.add_argument("--heal-ttl-leaks", action="store_true",
                    help="One-off, opt-in WRITE: set a 30-day TTL on keys reported as NO_TTL_LEAK "
                         "that belong to LEAK_HEAL_FAMILIES. Never deletes.")
    ap.add_argument("--dry-run", action="store_true",
                    help="With --heal-durable-ttl or --heal-ttl-leaks: show what would change without writing")
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
    if args.heal_durable_ttl:
        plan = plan_durable_ttl_heal(collect_heal_candidates(client))
        report["heal"] = apply_durable_ttl_heal(client, plan, dry_run=args.dry_run)
    if args.heal_ttl_leaks:
        leak_plan = plan_ttl_leak_heal(records)
        report["leak_heal"] = apply_ttl_leak_heal(client, leak_plan, dry_run=args.dry_run)
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else render(report, truncated))

    if args.fail_on_findings and any(f["severity"] in ("warning", "error") for f in report["findings"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
