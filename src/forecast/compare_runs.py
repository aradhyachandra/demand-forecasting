"""Compare two folders file by file (used by the clean-clone check, docs/F6_reproducibility.md).

    python -m forecast.compare_runs <first dir> <second dir> [--out file.json]

Every file must be byte-identical, except that JSON files may differ in run-time fields only
(RUN_INFO_KEYS, at any depth: timestamps, timings, memory, git commit). For JSON files the dropped
paths and any remaining differing paths are listed, so every difference can be explained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

RUN_INFO_KEYS = {
    "generated_at", "git",                       # not written by this project, kept for safety
    "commit", "started_utc", "run_minutes",      # f4_holdout.json: which commit ran it, when, how long
    "resources",                                 # f4_backtest.json: run time and memory
    "build_seconds", "fit_seconds",              # f4_backtest.json training_tables: timings
}


def strip_run_info(obj):
    if isinstance(obj, dict):
        return {k: strip_run_info(v) for k, v in obj.items() if k not in RUN_INFO_KEYS}
    if isinstance(obj, list):
        return [strip_run_info(v) for v in obj]
    return obj


def json_differences(a, b, path: str = "") -> list[str]:
    """Dotted paths at which two parsed JSON values differ (keys missing on either side count)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in list(a) + [k for k in b if k not in a]:
            p = f"{path}.{k}" if path else str(k)
            out += json_differences(a[k], b[k], p) if k in a and k in b else [p]
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in json_differences(x, y, f"{path}.{i}" if path else str(i))]
    return [] if a == b else [path]


def _ignored(obj, path=""):
    """Paths of run-time fields present in obj."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            out += [p] if k in RUN_INFO_KEYS else _ignored(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += _ignored(v, f"{path}.{i}" if path else str(i))
    return out


def _files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in Path(root).rglob("*")
            if p.is_file() and p.name != ".DS_Store" and "__pycache__" not in p.parts}


def compare_dirs(first, second) -> list[dict]:
    first, second = Path(first), Path(second)
    a, b = _files(first), _files(second)
    rows = []
    for name in sorted(a | b):
        row = {"file": name}
        if name not in b:
            row["status"] = "MISSING_IN_SECOND"
        elif name not in a:
            row["status"] = "MISSING_IN_FIRST"
        else:
            x, y = (first / name).read_bytes(), (second / name).read_bytes()
            if x == y:
                row["status"] = "identical"
            elif name.endswith(".json"):
                jx, jy = json.loads(x), json.loads(y)
                diff = json_differences(strip_run_info(jx), strip_run_info(jy))
                row["status"] = "DIFFERENT" if diff else "same_ignoring_run_info"
                row["ignored_paths"] = sorted(set(_ignored(jx)) | set(_ignored(jy)))
                if diff:
                    row["differing_paths"] = diff
            else:
                row["status"] = "DIFFERENT"
        for side, root, names in (("first", first, a), ("second", second, b)):
            row[f"sha256_{side}"] = hashlib.sha256((root / name).read_bytes()).hexdigest() if name in names else None
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--out")
    args = parser.parse_args()
    rows = compare_dirs(args.first, args.second)
    for r in rows:
        print(f"{r['status']:<24} {r['file']}")
    bad = [r for r in rows if r["status"] not in ("identical", "same_ignoring_run_info")]
    print(f"{len(rows)} files, {len(rows) - len(bad)} match, {len(bad)} do not")
    if args.out:
        Path(args.out).write_text(json.dumps({"first": args.first, "second": args.second, "files": rows},
                                             indent=2) + "\n")


if __name__ == "__main__":
    main()
