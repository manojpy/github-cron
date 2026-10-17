from __future__ import annotations
from pathlib import Path
import ast
import re
import sys

TEXT_EXTENSIONS = {
    ".py", ".pyx", ".md", ".yml", ".yaml", ".json", ".txt",
    ".rst", ".cfg", ".ini", ".toml", ".sh", ".bash",
}

TEXT_FILENAMES = {
    "Dockerfile", "Makefile", "requirements.txt",
    ".gitignore", ".gitattributes", ".editorconfig",
}

SKIP_DIR_PARTS = {
    ".git", "__pycache__", ".venv", "venv", "env",
    "node_modules", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".tox", ".eggs", "build", "dist",
}

CP1252_CONT = (
    "[\u0080-\u00bf"
    "\u20ac\u201a\u0192\u201e\u2026\u2020\u2021"
    "\u02c6\u2030\u0160\u2039\u0152\u017d"
    "\u2018\u2019\u201c\u201d"
    "\u2022\u2013\u2014\u02dc\u2122"
    "\u0161\u203a\u0153\u017e\u0178]"
)

MOJIBAKE_PATTERNS = [
    "\u00c3[\u0080-\u00bf]",
    "\u00c2[\u0080-\u00bf]",
    f"\u00e2\u20ac{CP1252_CONT}",
    "\u00e2\u0080[\u0080-\u00bf]",
    f"\u00f0\u0178{CP1252_CONT}{CP1252_CONT}",
    "\u00f0\u009f[\u0080-\u00bf][\u0080-\u00bf]",
    "\u00ef\u00bb\u00bf",
    f"[\u00e0-\u00ef]{CP1252_CONT}{CP1252_CONT}",
]
MOJIBAKE_RE = re.compile("|".join(MOJIBAKE_PATTERNS))

# Characters that look like nothing (or like a space) but break scripts, JSON
# and diffs. U+200D (zero-width joiner) and U+FE0F (variation selector) are
# allowed only inside emoji sequences, i.e. between two symbol characters.
INVISIBLE_CHARS = {
    "\u00a0": "NO-BREAK SPACE", "\u00ad": "SOFT HYPHEN",
    "\u180e": "MONGOLIAN VOWEL SEPARATOR", "\u200b": "ZERO WIDTH SPACE",
    "\u200c": "ZERO WIDTH NON-JOINER", "\u200e": "LEFT-TO-RIGHT MARK",
    "\u200f": "RIGHT-TO-LEFT MARK", "\u2028": "LINE SEPARATOR",
    "\u2029": "PARAGRAPH SEPARATOR", "\u202a": "BIDI EMBEDDING",
    "\u202b": "BIDI EMBEDDING", "\u202c": "BIDI EMBEDDING",
    "\u202d": "BIDI OVERRIDE", "\u202e": "BIDI OVERRIDE",
    "\u2060": "WORD JOINER", "\u2066": "BIDI ISOLATE", "\u2067": "BIDI ISOLATE",
    "\u2068": "BIDI ISOLATE", "\u2069": "BIDI ISOLATE", "\ufeff": "BOM / ZERO WIDTH NO-BREAK SPACE",
}
C1_CONTROL_RE = re.compile("[\u0080-\u009f]")
EMOJI_GLUE = {"\u200d", "\ufe0f"}
MOJIBAKE_ALLOWLIST: set[str] = set()

def iter_text_files(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIR_PARTS for part in p.parts):
            continue
        if p.suffix.lower() in TEXT_EXTENSIONS or p.name in TEXT_FILENAMES:
            yield p

def _glue_ok(line: str, i: int) -> bool:
    """A joiner / variation selector is fine only when it follows a non-ASCII
    symbol (and, for the joiner, also precedes one), i.e. inside an emoji."""
    prev_c = line[i - 1] if i > 0 else ""
    next_c = line[i + 1] if i + 1 < len(line) else ""
    prev_ok = bool(prev_c) and ord(prev_c) > 0x7F
    if line[i] == "\ufe0f":
        return prev_ok
    return prev_ok and bool(next_c) and ord(next_c) > 0x7F

def check_bytes_and_mojibake(root: Path) -> list[str]:
    problems: list[str] = []

    for p in iter_text_files(root):
        raw = p.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            problems.append(f"{p}:1: UTF-8 BOM at start of file (save as UTF-8 without BOM)")
        if b"\r\n" in raw or b"\r" in raw.replace(b"\r\n", b""):
            problems.append(f"{p}: Windows (CRLF) or old-Mac (CR) line endings - save with LF")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            problems.append(f"{p}: invalid UTF-8: {e}")
            continue

        for i, line in enumerate(text.split("\n"), 1):
            for j, ch in enumerate(line):
                if ch in EMOJI_GLUE:
                    if not _glue_ok(line, j):
                        problems.append(f"{p}:{i}:{j + 1}: stray U+{ord(ch):04X} outside an emoji: {line.strip()[:100]!r}")
                elif ch in INVISIBLE_CHARS and not (ch == "\ufeff" and i == 1 and j == 0):
                    problems.append(f"{p}:{i}:{j + 1}: invisible U+{ord(ch):04X} {INVISIBLE_CHARS[ch]}: {line.strip()[:100]!r}")
            if C1_CONTROL_RE.search(line):
                problems.append(f"{p}:{i}: C1 control character (usually mojibake): {line.strip()[:100]!r}")

        if "\ufffd" in text:
            for i, line in enumerate(text.splitlines(), 1):
                if "\ufffd" in line:
                    problems.append(
                        f"{p}:{i}: U+FFFD replacement char: "
                        f"{line.strip()[:120]}"
                    )

        if p.as_posix() in MOJIBAKE_ALLOWLIST:
            continue

        for i, line in enumerate(text.splitlines(), 1):
            matches = list(MOJIBAKE_RE.finditer(line))
            if not matches:
                continue

            max_shown = 3

            for m in matches[:max_shown]:
                start = max(0, m.start() - 20)
                end = min(len(line), m.end() + 20)

                left = "..." if start > 0 else ""
                right = "..." if end < len(line) else ""

                context = line[start:end].strip()

                problems.append(
                    f"{p}:{i}: mojibake {m.group()!r} near "
                    f"{left}{context}{right}"
                )

            remaining = len(matches) - max_shown
            if remaining > 0:
                problems.append(
                    f"{p}:{i}: {remaining} more mojibake match(es) on same line"
                )
    return problems

def _missing_kw(node: ast.Call, name: str) -> bool:
    return not any(k.arg == name for k in node.keywords)

def _open_missing_encoding(node: ast.Call) -> bool:
    if not _missing_kw(node, "encoding"):
        return False

    mode = None

    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
        mode = node.args[1].value

    for kw in node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            mode = kw.value.value

    if isinstance(mode, str) and "b" in mode:
        return False

    return True

def check_python_encodings(root: Path) -> list[str]:
    problems: list[str] = []

    for p in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIR_PARTS for part in p.parts):
            continue

        try:
            tree = ast.parse(
                p.read_text(encoding="utf-8"),
                filename=str(p),
            )
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            fn = node.func

            if isinstance(fn, ast.Name) and fn.id == "open":
                if _open_missing_encoding(node):
                    problems.append(
                        f"{p}:{node.lineno}: open() missing encoding= - "
                        f"add encoding='utf-8'"
                    )

            if isinstance(fn, ast.Attribute) and fn.attr in (
                "read_text",
                "write_text",
            ):
                if _missing_kw(node, "encoding"):
                    problems.append(
                        f"{p}:{node.lineno}: .{fn.attr}() missing encoding= - "
                        f"add encoding='utf-8'"
                    )

            if isinstance(fn, ast.Attribute) and fn.attr in (
                "run",
                "Popen",
                "check_output",
                "check_call",
                "call",
            ):
                kw_names = {k.arg for k in node.keywords if k.arg}

                if (
                    "text" in kw_names
                    or "universal_newlines" in kw_names
                ) and "encoding" not in kw_names:
                    problems.append(
                        f"{p}:{node.lineno}: subprocess.{fn.attr}"
                        f"(text=True) missing encoding= - add "
                        f"encoding='utf-8'"
                    )

    return problems

def main() -> int:
    root = Path(".")
    problems = check_bytes_and_mojibake(root)
    problems += check_python_encodings(root)

    if problems:
        print("BLOCKING: encoding / mojibake / UTF-8 issues")
        for prob in problems:
            print("  -", prob)
        return 1

    print(
        "All text files valid UTF-8, no U+FFFD, no mojibake, "
        "and every Python text I/O call declares UTF-8."
    )
    return 0

sys.exit(main())
