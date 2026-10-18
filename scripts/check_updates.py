#!/usr/bin/env python3
"""
check_updates.py — read-only report of newer stable versions for everything the
bot's image is built from. It never edits a file; it only tells you what to change.

Checks
  1. requirements.txt   every pin vs the newest STABLE release on PyPI that allows the
                        image's Python, plus a pip dry-run that says whether the bump
                        resolves together with the other pins.
  2. Python             the FROM python:X.Y tag vs endoflife.date (EOL date, newer
                        stable minor) and whether wheels exist for that newer Python.
  3. uv                 the `uv==` pin in the Dockerfile vs PyPI.
  4. Debian base        the -slim-<codename> in the Dockerfile vs the newest stable Debian.
  5. GitHub Actions     every `uses: owner/repo@vN` in .github/workflows vs the latest release.

A passing dry-run only proves the pins can be installed together. It does NOT prove the
bot still runs: after changing anything, build the image and let build.yml's AOT smoke
test and the repo tests decide.

Usage:   python3 scripts/check_updates.py [--report update_report.md]
Needs:   Python 3.9+, `pip install packaging`, network access. Optional GITHUB_TOKEN raises
         the GitHub API rate limit.
Output:  Markdown on stdout and in the report file. With $GITHUB_OUTPUT set it also writes
         has_updates=true|false and count=<n>. Exit code is 0 unless the script itself breaks.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.version import InvalidVersion, Version
except ImportError:
    print("Missing dependency: pip install packaging", file=sys.stderr)
    sys.exit(2)

ROOT = Path(__file__).resolve().parent.parent
MIN_AGE_DAYS = 180          # a new Python / Debian must be this old before it is recommended
EOL_WARN_DAYS = 365         # warn when the current Python is closer than this to its EOL
USER_AGENT = "github-cron-update-check"


# ── helpers ──────────────────────────────────────────────────────────────────
def http_json(url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 20.0) -> Any:
    """GET + parse JSON. Two retries; returns None on a 404, raises on other failures."""
    req_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    req_headers.update(headers or {})
    last: Optional[Exception] = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=req_headers), timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last = e
        except Exception as e:  # network blips
            last = e
        time.sleep(1 + attempt)
    raise RuntimeError(f"{url}: {last}")


def parse_iso(s: Any) -> Optional[date]:
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def bump_kind(cur: Version, new: Version) -> str:
    if new.major != cur.major:
        return "MAJOR"
    if new.minor != cur.minor:
        return "minor"
    return "patch"


# ── 1. requirements.txt ──────────────────────────────────────────────────────
def read_requirements(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        try:
            req = Requirement(line)
        except Exception:
            continue
        specs = list(req.specifier)
        pinned = specs[0].version if len(specs) == 1 and specs[0].operator == "==" else None
        rows.append({"idx": idx, "raw": raw, "line": line, "req": req, "name": req.name, "pinned": pinned})
    return rows


def latest_stable(name: str, py_version: str) -> Optional[Version]:
    """Newest non-pre-release, non-yanked release that allows `py_version`."""
    data = http_json(f"https://pypi.org/pypi/{name}/json")
    if not data:
        return None
    best: Optional[Version] = None
    for ver_str, files in (data.get("releases") or {}).items():
        try:
            v = Version(ver_str)
        except InvalidVersion:
            continue
        if v.is_prerelease or v.is_devrelease:
            continue
        usable = [f for f in files if not f.get("yanked")]
        if not usable:
            continue
        if not any(
            not f.get("requires_python") or SpecifierSet(f["requires_python"]).contains(py_version)
            for f in usable
        ):
            continue
        if best is None or v > best:
            best = v
    return best


def dry_run(req_lines: List[str], py_version: str) -> Tuple[bool, str]:
    """Can these requirements be installed together for `py_version` (wheels only)?"""
    with tempfile.TemporaryDirectory() as tmp:
        req_file = Path(tmp) / "req.txt"
        req_file.write_text("\n".join(req_lines) + "\n", encoding="utf-8")
        cmd = [
            sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed",
            "--disable-pip-version-check", "--only-binary=:all:",
            "--python-version", py_version, "--target", str(Path(tmp) / "t"),
            "-r", str(req_file),
        ]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=240)
        except subprocess.TimeoutExpired:
            return False, "pip dry-run timed out"
        if p.returncode == 0:
            return True, ""
        tail = [ln.strip() for ln in (p.stdout + "\n" + p.stderr).splitlines() if ln.strip()]
        why = [ln for ln in tail if ln.startswith("The user requested") or " depends on " in ln]
        if not why:
            why = [ln for ln in tail if "no matching" in ln.lower() or "could not find" in ln.lower()]
        reason = "; ".join(why[:2]) if why else (tail[-1] if tail else "failed")
        reason = re.sub(r"\s*\(from versions:[^)]*\)", "", reason)   # pip lists every version it saw
        return False, reason[:260]


def check_requirements(py_version: str) -> Tuple[List[str], int]:
    path = ROOT / "requirements.txt"
    if not path.exists():
        return ["`requirements.txt` not found."], 0
    rows = read_requirements(path)
    base_lines = [r["raw"].split("#", 1)[0].strip() for r in rows]
    ok_base, why_base = dry_run(base_lines, py_version)
    out: List[str] = []
    if not ok_base:
        out.append(f"> ⚠️ The CURRENT requirements do not install for Python {py_version} "
                   f"(wheels only): {why_base}. Bump results below cannot be trusted.\n")

    outdated: List[Dict[str, Any]] = []
    floating: List[Dict[str, Any]] = []
    for r in rows:
        try:
            latest = latest_stable(r["name"], py_version)
        except Exception as e:
            out.append(f"- `{r['name']}`: PyPI lookup failed ({e})")
            continue
        if latest is None:
            continue
        r["latest"] = latest
        if r["pinned"]:
            cur = Version(r["pinned"])
            if latest > cur:
                r["kind"] = bump_kind(cur, latest)
                outdated.append(r)
        else:
            floating.append(r)

    for r in outdated:
        extras = f"[{','.join(sorted(r['req'].extras))}]" if r["req"].extras else ""
        new_line = f"{r['name']}{extras}=={r['latest']}"
        test_lines = [new_line if x is r else b for x, b in zip(rows, base_lines)]
        ok, why = dry_run(test_lines, py_version) if ok_base else (False, "baseline broken")
        r["new_line"], r["ok"], r["why"] = new_line, ok, why

    count = len(outdated)
    if outdated:
        out.append("| package | pinned | newest stable | change | resolves with the other pins? |")
        out.append("|---|---|---|---|---|")
        for r in sorted(outdated, key=lambda x: (x["kind"] != "MAJOR", x["name"].lower())):
            verdict = "✅ yes" if r["ok"] else f"⚠️ no — {r['why']}"
            out.append(f"| `{r['name']}` | {r['pinned']} | {r['latest']} | {r['kind']} | {verdict} |")
        out.append("")

        small = [r for r in outdated if r["kind"] != "MAJOR"]
        major = [r for r in outdated if r["kind"] == "MAJOR"]
        if small:
            combo = [next((x["new_line"] for x in small if x["idx"] == row["idx"]), b)
                     for row, b in zip(rows, base_lines)]
            ok_all, why_all = dry_run(combo, py_version) if ok_base else (False, "baseline broken")
            if ok_all:
                out.append(f"**Patch/minor bumps, all {len(small)} together: ✅ they install together.** "
                           "Edit `requirements.txt` (before → after):")
                out.append("```")
                for r in small:
                    out.append(f"- {r['line']}")
                    out.append(f"+ {r['new_line']}")
                out.append("```")
            else:
                out.append(f"**Patch/minor bumps, all {len(small)} together: ⚠️ they do NOT install together** — {why_all}")
                out.append("Safe one by one (each resolved on its own above):")
                out.append("```")
                for r in small:
                    if r["ok"]:
                        out.append(f"- {r['line']}")
                        out.append(f"+ {r['new_line']}")
                out.append("```")
        if major:
            out.append("")
            out.append("**MAJOR bumps** — installable does not mean compatible; read the release notes and "
                       "change one at a time: " + ", ".join(f"`{r['name']}` {r['pinned']} → {r['latest']}" for r in major) + ".")
        out.append("")
        out.append("After any change: rebuild the image and let `build.yml`'s AOT smoke test and the repo tests decide.")
    else:
        out.append("✅ Every pinned requirement is already the newest stable release.")

    if floating:
        out.append("")
        out.append("**Not pinned** (a rebuild silently installs whatever is newest):")
        for r in floating:
            out.append(f"- `{r['line']}` → would install **{r['latest']}** today. Pin it, or drop the line if an extra already pulls it in.")
        count += len(floating)
    return out, count


# ── 2–4. Dockerfile: Python, uv, Debian ──────────────────────────────────────
def dockerfile_text() -> str:
    p = ROOT / "Dockerfile"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def check_python(df: str, base_lines: List[str]) -> Tuple[List[str], int, str]:
    m = re.search(r"^FROM\s+python:(\d+)\.(\d+)", df, re.M)
    if not m:
        return ["No `FROM python:X.Y` line found in the Dockerfile."], 0, "3.11"
    cur_cycle = f"{m.group(1)}.{m.group(2)}"
    cycles = http_json("https://endoflife.date/api/python.json") or []
    today = date.today()
    cur = next((c for c in cycles if str(c.get("cycle")) == cur_cycle), None)
    out: List[str] = []
    count = 0
    if cur:
        eol = parse_iso(cur.get("eol"))
        days = (eol - today).days if eol else None
        line = f"Image uses Python **{cur_cycle}** (latest patch {cur.get('latest')}; the floating tag picks patches up on rebuild)."
        if days is not None:
            line += f" End of life: {eol} ({days} days)."
            if days < EOL_WARN_DAYS:
                line += " ⚠️ Plan an upgrade."
                count += 1
        out.append(line)
    candidates = []
    for c in cycles:
        rel, eol = parse_iso(c.get("releaseDate")), parse_iso(c.get("eol"))
        if rel and rel <= today - timedelta(days=MIN_AGE_DAYS) and (eol is None or eol > today):
            candidates.append(c)
    best = max(candidates, key=lambda c: tuple(int(x) for x in str(c["cycle"]).split("."))) if candidates else None
    if best and str(best["cycle"]) != cur_cycle and \
            tuple(int(x) for x in str(best["cycle"]).split(".")) > tuple(int(x) for x in cur_cycle.split(".")):
        ok, why = dry_run(base_lines, str(best["cycle"]))
        verdict = ("✅ every pinned requirement has wheels for it" if ok
                   else f"⚠️ not yet installable with the current pins: {why}")
        out.append(f"Newest Python at least {MIN_AGE_DAYS} days old: **{best['cycle']}** "
                   f"(released {best.get('releaseDate')}). {verdict}.")
        out.append(f"To move: change `FROM python:{cur_cycle}-…` to `python:{best['cycle']}-…` in all three stages, "
                   f"and `python3.11` in the `SP=` path of the runtime-venv stage.")
        count += 1
    else:
        out.append(f"No newer stable Python that is at least {MIN_AGE_DAYS} days old.")
    return out, count, cur_cycle


def check_uv(df: str) -> Tuple[List[str], int]:
    m = re.search(r"uv==(\d+(?:\.\d+)*)", df)
    if not m:
        return ["No `uv==` pin found in the Dockerfile."], 0
    cur = Version(m.group(1))
    latest = latest_stable("uv", "3.11")
    if latest and latest > cur:
        return [f"`uv` pinned at {cur}, newest stable **{latest}** ({bump_kind(cur, latest)}). "
                f"Change `uv=={cur}` to `uv=={latest}` in the `uv-installer` stage."], 1
    return [f"✅ `uv` {cur} is current."], 0


def check_debian(df: str) -> Tuple[List[str], int]:
    m = re.search(r"^FROM\s+python:[\w.]+-slim-(\w+)", df, re.M)
    if not m:
        return ["No `-slim-<codename>` base found in the Dockerfile."], 0
    cur_name = m.group(1).lower()
    cycles = http_json("https://endoflife.date/api/debian.json") or []
    today = date.today()
    cur = next((c for c in cycles if str(c.get("codename", "")).lower() == cur_name), None)
    out: List[str] = []
    count = 0
    if cur:
        eol, lts = parse_iso(cur.get("eol")), parse_iso(cur.get("extendedSupport"))
        regular = f"regular security support ended {eol}" if eol and eol < today else f"security support to {eol}"
        out.append(f"Base is Debian **{cur_name}** ({regular}"
                   + (f", long-term support to {lts}" if lts else "") + ").")
    stable = [c for c in cycles if c.get("codename") and parse_iso(c.get("releaseDate"))
              and parse_iso(c["releaseDate"]) <= today - timedelta(days=MIN_AGE_DAYS)]
    newest = max(stable, key=lambda c: parse_iso(c["releaseDate"])) if stable else None
    if newest and str(newest["codename"]).lower() != cur_name:
        out.append(f"Newer stable Debian at least {MIN_AGE_DAYS} days old: **{str(newest['codename']).lower()}** "
                   f"(released {newest['releaseDate']}). Not urgent while {cur_name} is supported. "
                   f"To move, change `-slim-{cur_name}` in the Dockerfile and check that `libtbb12` / `libgomp1` still install.")
        count += 1
    else:
        out.append("✅ Debian base is the newest stable.")
    return out, count


# ── 5. GitHub Actions ────────────────────────────────────────────────────────
def check_actions() -> Tuple[List[str], int]:
    wf_dir = ROOT / ".github" / "workflows"
    if not wf_dir.exists():
        return ["No workflows found."], 0
    uses: Dict[Tuple[str, str], List[str]] = {}
    pat = re.compile(r"uses:\s*([\w.-]+/[\w.-]+)(?:/[\w./-]+)?@([\w.-]+)")
    for f in sorted(wf_dir.glob("*.y*ml")):
        for repo, ref in pat.findall(f.read_text(encoding="utf-8")):
            uses.setdefault((repo, ref), [])
            if f.name not in uses[(repo, ref)]:
                uses[(repo, ref)].append(f.name)
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    rows: List[str] = []
    count = 0
    for (repo, ref), files in sorted(uses.items()):
        m = re.fullmatch(r"v(\d+)", ref)
        if not m:
            continue  # SHA or branch pin: nothing to compare
        try:
            rel = http_json(f"https://api.github.com/repos/{repo}/releases/latest", headers=headers)
        except Exception as e:
            rows.append(f"- `{repo}@{ref}`: lookup failed ({e})")
            continue
        tag = (rel or {}).get("tag_name", "")
        tm = re.match(r"v?(\d+)", tag)
        if tm and int(tm.group(1)) > int(m.group(1)):
            rows.append(f"- `{repo}@{ref}` → **@v{tm.group(1)}** (latest release {tag}) — used in {', '.join(files)}")
            count += 1
    return (rows or ["✅ Every action is on its latest major version."]), count


# ── main ─────────────────────────────────────────────────────────────────────
def section(title: str, fn) -> Tuple[List[str], int, Any]:
    try:
        res = fn()
        lines, n = res[0], res[1]
        extra = res[2] if len(res) > 2 else None
        return [f"## {title}", *lines, ""], n, extra
    except Exception as e:  # one failing check must not hide the others
        return [f"## {title}", f"⚠️ Check failed: {e}", ""], 0, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default="update_report.md")
    args = ap.parse_args()

    df = dockerfile_text()
    mpy = re.search(r"^FROM\s+python:(\d+\.\d+)", df, re.M)
    py_version = mpy.group(1) if mpy else "3.11"
    req_path = ROOT / "requirements.txt"
    base_lines = [r["raw"].split("#", 1)[0].strip() for r in read_requirements(req_path)] if req_path.exists() else []

    report: List[str] = [f"# Dependency check — {date.today()}", "",
                         f"Image Python: {py_version}. Newest *stable* releases only; pre-releases and yanked files are ignored.", ""]
    total = 0
    for title, fn in (
        ("requirements.txt", lambda: check_requirements(py_version)),
        ("Python", lambda: check_python(df, base_lines)),
        ("uv (Dockerfile)", lambda: check_uv(df)),
        ("Debian base (Dockerfile)", lambda: check_debian(df)),
        ("GitHub Actions", check_actions),
    ):
        lines, n, _ = section(title, fn)
        report += lines
        total += n
    report.append(f"**{total} item(s) need a decision.**" if total else "**Nothing to change.**")

    text = "\n".join(report) + "\n"
    Path(args.report).write_text(text, encoding="utf-8")
    print(text)
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as fh:
            fh.write(f"has_updates={'true' if total else 'false'}\ncount={total}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
