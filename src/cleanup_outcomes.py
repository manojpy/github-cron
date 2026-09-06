#!/usr/bin/env python3
"""
cleanup_outcomes.py — Auto-clean old outcome archive files based on configurable limits

This script runs alongside export_outcomes.py to:
1. Remove old JSONL files that exceed a specified age
2. Ensure the repository stays within GitHub's size limits
3. Compress files that are getting large but not yet old enough to delete

Usage:
    python3 cleanup_outcomes.py --data-dir src/data/outcomes --max-age-days 90 --max-size-mb 500
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

def cleanup_by_age(data_dir: Path, max_age_days: int) -> int:
    """Remove files older than max_age_days."""
    removed = 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

    for label, pattern in (("outcomes", "*.jsonl*"), ("shadow", "*.jsonl*"), ("reports", "*.md")):
        label_dir = data_dir / label
        if not label_dir.exists():
            continue
        
        for file_path in label_dir.glob(pattern):  # includes .jsonl.gz for outcomes/shadow
            try:
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
                if mtime < cutoff:
                    print(f"🗑️ Removing old file: {file_path}")
                    file_path.unlink()
                    removed += 1
            except OSError:
                continue
    
    return removed

def compress_large_files(data_dir: Path, max_size_mb: int) -> int:
    """Compress files that exceed max_size_mb, but aren't old enough to delete."""
    compressed = 0
    max_size_bytes = max_size_mb * 1024 * 1024
    
    for label in ["outcomes", "shadow"]:
        label_dir = data_dir / label
        if not label_dir.exists():
            continue
        
        for month_file in label_dir.glob("*.jsonl"):
            if month_file.stat().st_size > max_size_bytes:
                gz_file = month_file.with_suffix(month_file.suffix + ".gz")
                if not gz_file.exists():
                    print(f"💾 Compressing large file: {month_file.name}")
                    with open(month_file, "rb") as f_in:
                        with gzip.open(gz_file, "wb", compresslevel=6) as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    month_file.unlink()  # Remove original after successful compression
                    compressed += 1
    
    return compressed

def wipe_labels(data_dir: Path, labels: list) -> int:
    """DANGER: delete every file (any age, any size) under the given label
    dirs. Used only when explicitly requested — bypasses age/size cleanup
    entirely rather than layering on top of it.
    
    SAFETY: This strictly deletes FILES only. The parent folders (outcomes/, reports/) 
    are preserved and will be automatically recreated if they somehow went missing."""
    removed = 0
    for label in labels:
        label_dir = data_dir / label
        
        # 1. Ensure the folder exists (recreate if missing to prevent silent failures)
        if not label_dir.exists():
            label_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Recreated missing folder: {label_dir}")
            continue
            
        # 2. Delete all files inside (using rglob to catch any nested subdirectories too)
        for file_path in label_dir.rglob('*'):
            if file_path.is_file():
                print(f"🚨 WIPE: removing {file_path}")
                file_path.unlink()
                removed += 1
                
        # 3. Clean up any empty subdirectories left behind (but NEVER the main label_dir)
        for dir_path in sorted(label_dir.rglob('*'), reverse=True):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                dir_path.rmdir()
                
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
        print("  🚨 OUTCOME DATA WIPE — outcomes/ + reports/")
        print("=" * 60)
        removed = wipe_labels(data_dir, ["outcomes", "reports"])
        print(f"\n✅ Wipe complete — removed {removed} file(s)")
        return

    print("=" * 60)
    print("  OUTCOME DATA CLEANUP")
    print("=" * 60)

    # Show current size
    current_size = get_dir_size(data_dir)
    print(f"Current size: {format_size(current_size)}")
    print(f"Max total size: {args.max_total_mb} MB")
    
    # 1. Remove old files
    print(f"\n📋 Checking for files older than {args.max_age_days} days...")
    removed = cleanup_by_age(data_dir, args.max_age_days)
    if removed:
        print(f"✅ Removed {removed} old file(s)")
    else:
        print("No old files to remove")
    
    # 2. Compress large files
    print(f"\n📋 Checking for files larger than {args.max_size_mb} MB...")
    compressed = compress_large_files(data_dir, args.max_size_mb)
    if compressed:
        print(f"✅ Compressed {compressed} large file(s)")
    else:
        print("No large files to compress")
    
    # 3. Aggressive cleanup if total is too large
    new_size = get_dir_size(data_dir)
    max_total_bytes = args.max_total_mb * 1024 * 1024
    
    if new_size > max_total_bytes:
        print(f"\n⚠️ Directory size ({format_size(new_size)}) exceeds limit ({args.max_total_mb} MB)")
        print("Performing aggressive cleanup...")
        
        for label, pattern in (("outcomes", "*.jsonl*"), ("shadow", "*.jsonl*"), ("reports", "*.md")):
            label_dir = data_dir / label
            if not label_dir.exists():
                continue
            
            # Get all files sorted by name (YYYY-MM-DD[...] format sorts chronologically)
            files = sorted(label_dir.glob(pattern))

            # Remove from oldest until under limit
            for month_file in files:
                if get_dir_size(data_dir) <= max_total_bytes:
                    break
                print(f"🗑️ Removing {month_file.name} to stay under limit")
                month_file.unlink()
    
    # Show final state
    final_size = get_dir_size(data_dir)
    print(f"\n{'=' * 60}")
    print(f"✅ Cleanup complete")
    print(f"Final size: {format_size(final_size)}")
    print(f"Size reduction: {format_size(current_size - final_size)}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()