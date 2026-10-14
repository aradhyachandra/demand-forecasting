"""Fingerprints of the raw files (sha256 and row counts), so a changed download is caught."""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def count_data_rows(path: Path) -> int:
    """Number of data rows (lines minus the header) in a CSV."""
    with open(path, "rb") as fh:
        lines = sum(block.count(b"\n") for block in iter(lambda: fh.read(1 << 20), b""))
    with open(path, "rb") as fh:
        fh.seek(-1, 2)
        ends_with_newline = fh.read(1) == b"\n"
    return lines - 1 + (0 if ends_with_newline else 1)


def fingerprint(paths: dict[str, Path]) -> dict[str, dict]:
    return {
        key: {
            "file": Path(p).name,
            "sha256": sha256_file(p),
            "data_rows": count_data_rows(p),
            "bytes": Path(p).stat().st_size,
        }
        for key, p in paths.items()
    }


def write_fingerprints(fp: dict, out: Path) -> None:
    header = (
        "# Fingerprints of the raw M5 files as downloaded from Kaggle (recorded in F1).\n"
        "# run_f1 stops if a file no longer matches. Do not edit by hand.\n"
    )
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.safe_dump(fp, fh, sort_keys=True)


def verify(paths: dict[str, Path], expected: dict[str, dict]) -> None:
    """Raise ValueError if any raw file's sha256 differs from the recorded one."""
    problems = []
    for key, p in paths.items():
        if key not in expected:
            problems.append(f"{key}: no recorded fingerprint")
            continue
        got = sha256_file(p)
        if got != expected[key]["sha256"]:
            problems.append(f"{key} ({Path(p).name}): sha256 {got[:12]}... != recorded "
                            f"{expected[key]['sha256'][:12]}...")
    if problems:
        raise ValueError("Raw files do not match config/raw_files.yaml:\n  " + "\n  ".join(problems))
