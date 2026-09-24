"""Reports are kept 7 days (exact, from the file name); outcomes keep their own limit."""
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cleanup_outcomes as co


def _mk(dirpath: Path, name: str) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    f = dirpath / name
    f.write_text("x")
    return f


def _report_name(age: timedelta) -> str:
    return (datetime.now(timezone.utc) - age).strftime("%Y-%m-%d_%H-%M") + ".md"


def test_reports_older_than_7_days_removed_newer_kept(tmp_path):
    new = _mk(tmp_path / "reports", _report_name(timedelta(hours=1)))
    six_d = _mk(tmp_path / "reports", _report_name(timedelta(days=6, hours=23)))
    eight_d = _mk(tmp_path / "reports", _report_name(timedelta(days=7, hours=1)))
    keep = _mk(tmp_path / "reports", ".gitkeep")
    removed = co.cleanup_by_age(tmp_path, max_age_days=185, reports_max_age_days=7)
    assert removed == 1
    assert new.exists() and six_d.exists() and keep.exists()
    assert not eight_d.exists()


def test_outcomes_keep_their_own_185_day_limit(tmp_path):
    day = lambda n: (datetime.now(timezone.utc) - timedelta(days=n)).strftime("%Y-%m-%d")
    old_ok = _mk(tmp_path / "outcomes", f"{day(100)}.jsonl")
    too_old = _mk(tmp_path / "outcomes", f"{day(200)}.jsonl")
    co.cleanup_by_age(tmp_path, max_age_days=185, reports_max_age_days=7)
    assert old_ok.exists() and not too_old.exists()


def test_dry_run_removes_nothing(tmp_path):
    old = _mk(tmp_path / "reports", _report_name(timedelta(days=30)))
    assert co.cleanup_by_age(tmp_path, 185, dry_run=True, reports_max_age_days=7) == 1
    assert old.exists()


def test_report_age_uses_time_in_file_name():
    ref = co.file_age_reference(Path("2026-03-04_12-30.md"))
    assert ref == datetime(2026, 3, 4, 12, 30, tzinfo=timezone.utc)
    assert co.file_age_reference(Path("2026-03-04.jsonl")).hour == 23


def test_workflows_use_12_hour_cadence_and_stage_reports_and_state():
    root = Path(__file__).resolve().parent.parent / ".github" / "workflows"
    run = (root / "run-bot.yml").read_text(encoding="utf-8")

    assert "(NOW / 900) % 48 ))" in run
    assert "(NOW / 900) % 24 ))" not in run
    assert "% 16" not in run

    assert "git add --sparse -f reports" in run
    assert "git add --sparse -f state" in run

    assert "--reports-max-age-days 7" in (
        root / "cleanup-outcomes.yml"
    ).read_text(encoding="utf-8")

    build = (root / "build.yml").read_text(encoding="utf-8")
    assert re.search(r"Persist Brain report", build)
    assert "git add -f reports state" in build