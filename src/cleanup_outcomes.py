#!/usr/bin/env python3
"""
cleanup_outcomes.py — Auto-clean old outcome archive files based on configurable limits
"""

import argparse
import gzip
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

def get_dir_size(path: Path) -> int:
    """Calculate total size of a directory in bytes."""
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except (OSError, PermissionError):
        pass
    return total

def format_size(bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"


def cleanup_by_age(data_dir: Path, max_age_days: int, dry_run: bool = False) -> int:
    """Remove files older than max_age_days."""
    removed = 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

    for label, pattern in (("outcomes", "*.jsonl*"), ("shadow", "*.jsonl*"), ("reports", "*.md")):
        label_dir = data_dir / label
        if not label_dir.exists():
            continue
        
        for file_path in label_dir.glob(pattern):
            try:
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
                if mtime < cutoff:
                    prefix = "🔎 [DRY RUN] Would remove" if dry_run else "🗑️ Removing"
                    print(f"{prefix} old file: {file_path}")
                    if not dry_run:
                        file_path.unlink()
                    removed += 1
            except OSError:
                continue
    
    return removed

def compress_large_files(data_dir: Path, max_size_mb: int, dry_run: bool = False,
                          no_compress_within_days: int = 0) -> int:
    """Compress files that exceed max_size_mb, but aren't old enough to delete —
    and aren't inside the brain's live analysis window (no_compress_within_days)."""
    compressed = 0
    max_size_bytes = max_size_mb * 1024 * 1024
    cutoff = datetime.now(timezone.utc) - timedelta(days=no_compress_within_days)

    for label in ["outcomes", "shadow"]:
        label_dir = data_dir / label
        if not label_dir.exists():
            continue
        
        for month_file in label_dir.glob("*.jsonl"):
            if no_compress_within_days > 0:
                mtime = datetime.fromtimestamp(month_file.stat().st_mtime, tz=timezone.utc)
                if mtime >= cutoff:
                    continue  # still inside the live analysis window — leave uncompressed
            if month_file.stat().st_size > max_size_bytes:
                gz_file = month_file.with_suffix(month_file.suffix + ".gz")
                if not gz_file.exists():
                    if dry_run:
                        print(f"🔎 [DRY RUN] Would compress: {month_file.name}")
                    else:
                        print(f"💾 Compressing large file: {month_file.name}")
                        with open(month_file, "rb") as f_in:
                            with gzip.open(gz_file, "wb", compresslevel=6) as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        month_file.unlink()
                    compressed += 1
    
    return compressed

def wipe_labels(data_dir: Path, labels: list, dry_run: bool = False) -> int:
    """DANGER: delete every file (any age, any size) under the given label dirs.
    
    SAFETY: Deletes FILES only. Parent folders are preserved, and a .gitkeep
    file is added so Git continues tracking the folder even when empty.
    """
    removed = 0
    for label in labels:
        label_dir = data_dir / label
        
        # 1. Ensure the folder exists (skipped in dry-run — nothing to preview creating)
        if not label_dir.exists():
            if dry_run:
                print(f"🔎 [DRY RUN] Would recreate missing folder: {label_dir}")
                continue
            label_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Recreated missing folder: {label_dir}")
        
        # 2. Delete all files inside (including nested ones)
        for file_path in label_dir.rglob('*'):
            if file_path.is_file():
                if dry_run:
                    print(f"🔎 [DRY RUN] Would WIPE: removing {file_path}")
                else:
                    print(f"🚨 WIPE: removing {file_path}")
                    file_path.unlink()
                removed += 1

        if dry_run:
            continue

        # 3. Clean up empty subdirectories (but NEVER the main label_dir)
        for dir_path in sorted(label_dir.rglob('*'), reverse=True):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                dir_path.rmdir()
        
        # 4. ADD .gitkeep so Git tracks the folder even when empty
        gitkeep = label_dir / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.write_text("# This file ensures Git tracks this empty directory.\n")
            print(f"📌 Added {gitkeep}")
                
    return removed

def main():
    ap = argparse.ArgumentParser(description="Cleanup old outcome archive files")
    ap.add_argument("--data-dir", default="src/data/outcomes",
                    help="Root directory for archived JSONL files")
    ap.add_argument("--max-age-days", type=int, default=90,
                    help="Remove files older than this many days (default: 90)")
    ap.add_argument("--max-size-mb", type=int, default=100,
                    help="Compress files larger than this size in MB (default: 100)")
    ap.add_argument("--max-total-mb", type=int, default=400,
                    help="Max total directory size in MB before aggressive cleanup (default: 400)") 
    ap.add_argument("--no-compress-within-days", type=int, default=0,
                    help="Never compress files newer than this many days, even if oversized "
                         "(protects the brain's live analysis window; 0 disables the guard)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be cleaned without actually deleting")
    ap.add_argument("--wipe-all", action="store_true",
                    help="DANGER: delete ALL files in outcomes/ and reports/ regardless "
                         "of age, instead of the normal age/size-based cleanup")

    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return

    if args.wipe_all:
        print("=" * 60)
        print("  🚨 OUTCOME DATA WIPE — outcomes/ + reports/" + ("  [DRY RUN]" if args.dry_run else ""))
        print("=" * 60)
        removed = wipe_labels(data_dir, ["outcomes", "reports"], dry_run=args.dry_run)
        verb = "Would remove" if args.dry_run else "removed"
        print(f"\n✅ Wipe {'preview' if args.dry_run else 'complete'} — {verb} {removed} file(s)")
        return

    print("=" * 60)
    print("  OUTCOME DATA CLEANUP" + ("  [DRY RUN]" if args.dry_run else ""))
    print("=" * 60)

    current_size = get_dir_size(data_dir)
    print(f"Current size: {format_size(current_size)}")
    print(f"Max total size: {args.max_total_mb} MB")
    
    print(f"\n📋 Checking for files older than {args.max_age_days} days...")
    removed = cleanup_by_age(data_dir, args.max_age_days, dry_run=args.dry_run)
    if removed:
        print(f"✅ {'Would remove' if args.dry_run else 'Removed'} {removed} old file(s)")
    else:
        print("No old files to remove")
    
    print(f"\n📋 Checking for files larger than {args.max_size_mb} MB...")
    compressed = compress_large_files(
        data_dir, args.max_size_mb, dry_run=args.dry_run,
        no_compress_within_days=args.no_compress_within_days,
    )
    if compressed:
        print(f"✅ {'Would compress' if args.dry_run else 'Compressed'} {compressed} large file(s)")
    else:
        print("No large files to compress")
    
    # In dry-run, nothing above actually changed the filesystem, so
    # get_dir_size() still reflects the original size — track a simulated
    # running total instead of re-stat'ing the (unchanged) directory.
    new_size = get_dir_size(data_dir)
    max_total_bytes = args.max_total_mb * 1024 * 1024
    
    if new_size > max_total_bytes:
        print(f"\n⚠️ Directory size ({format_size(new_size)}) exceeds limit ({args.max_total_mb} MB)")
        print("Performing aggressive cleanup..." if not args.dry_run else "Previewing aggressive cleanup...")
        
        simulated_size = new_size
        for label, pattern in (("outcomes", "*.jsonl*"), ("shadow", "*.jsonl*"), ("reports", "*.md")):
            label_dir = data_dir / label
            if not label_dir.exists():
                continue
            
            files = sorted(label_dir.glob(pattern))
            for month_file in files:
                current_check = simulated_size if args.dry_run else get_dir_size(data_dir)
                if current_check <= max_total_bytes:
                    break
                if args.dry_run:
                    print(f"🔎 [DRY RUN] Would remove {month_file.name} to stay under limit")
                    simulated_size -= month_file.stat().st_size
                else:
                    print(f"🗑️ Removing {month_file.name} to stay under limit")
                    month_file.unlink()
    
    final_size = get_dir_size(data_dir)
    print(f"\n{'=' * 60}")
    print(f"✅ Cleanup {'preview' if args.dry_run else 'complete'}")
    print(f"Final size: {format_size(final_size)}" + (" (unchanged — dry run)" if args.dry_run else ""))
    print(f"Size reduction: {format_size(current_size - final_size)}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
