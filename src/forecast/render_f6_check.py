"""Render docs/F6_reproducibility.md from results/f6_clean_clone_check.json (the clean-clone check's
own record). Formats saved values only; the check itself is described in the doc.

    python -m forecast.render_f6_check
"""
from __future__ import annotations

import json
from pathlib import Path

from forecast import config
from forecast.report import markdown_table, render_template

ROOT = config.ROOT
RECORD = ROOT / "results" / "f6_clean_clone_check.json"
TEMPLATE = ROOT / "docs" / "templates" / "F6_reproducibility.md"
DOC = ROOT / "docs" / "F6_reproducibility.md"
MATCH = ("identical", "same_ignoring_run_info")


def folder_summary(rows: list[dict]) -> dict:
    out = {"files": len(rows), "identical": 0, "same_ignoring_run_info": 0, "different": 0, "missing": 0}
    for r in rows:
        if r["status"] in MATCH:
            out[r["status"]] += 1
        elif r["status"].startswith("MISSING"):
            out["missing"] += 1
        else:
            out["different"] += 1
    out["match"] = out["identical"] + out["same_ignoring_run_info"]
    return out


def context(rec: dict) -> tuple[dict, dict]:
    summary = {k: folder_summary(v) for k, v in rec["comparisons"].items()}
    folder_rows = [{"f": "README.md" if k == "." else f"{k}/", **s} for k, s in summary.items()]
    not_identical = [{"f": f"`{k}/{r['file']}`" if k != "." else f"`{r['file']}`", "s": r["status"],
                      "ignored": ", ".join(f"`{p}`" for p in r.get("ignored_paths", [])) or "",
                      "why": r.get("explanation", "")}
                     for k, rows in rec["comparisons"].items() for r in rows if r["status"] != "identical"]
    steps = [{"s": s["step"], "sec": s["seconds"], "rss": f"{s['max_resident_mb']:,.0f}",
              "fp": f"{s['peak_footprint_mb']:,.0f}"} for s in rec["steps"]]
    raw = [{"f": r["file"], "sha": f"`{r['sha256']}`", "ok": "yes" if r["matches_config"] else "NO"}
           for r in rec["raw_files"]]
    hr = rec["holdout_rerun"]
    tables = {
        "folders": markdown_table(folder_rows, [("f", "Folder"), ("files", "Files"), ("identical", "Byte-identical"),
                                                ("same_ignoring_run_info", "Same except run-time fields"),
                                                ("different", "Different"), ("missing", "Missing on one side")]),
        "not_identical": markdown_table(not_identical, [("f", "File"), ("s", "Result"),
                                                        ("ignored", "Run-time fields ignored"),
                                                        ("why", "Explanation")]) if not_identical else "None.",
        "steps": markdown_table(steps, [("s", "Step"), ("sec", "Seconds"), ("rss", "Max resident memory (MB)"),
                                        ("fp", "macOS peak footprint (MB)")]),
        "raw": markdown_table(raw, [("f", "File"), ("sha", "sha256"), ("ok", "Matches config/raw_files.yaml")]),
        "processed": markdown_table(
            [{"f": f"`{k}`", "w": f"`{v['working_copy'][:16]}`", "c": f"`{v['clone'][:16]}`",
              "same": "yes" if v["working_copy"] == v["clone"] else "NO"}
             for k, v in rec.get("processed_parquet_sha256", {}).items()],
            [("f", "File"), ("w", "Working copy"), ("c", "Clone"), ("same", "Identical")]),
        "holdout_paths": "\n".join(f"- `{p}`" for p in hr["comparison"]["ignored_paths"]) or "- none",
    }
    all_rows = [r for rows in rec["comparisons"].values() for r in rows]
    ctx = {"r": rec, "t": {
        "total_files": len(all_rows), "total_match": sum(r["status"] in MATCH for r in all_rows),
        "total_identical": sum(r["status"] == "identical" for r in all_rows),
        "results_files": summary["results"]["files"], "results_match": summary["results"]["match"],
        "results_identical": summary["results"]["identical"],
        "holdout_status": hr["comparison"]["status"],
        "holdout_differing": ", ".join(hr["comparison"].get("differing_paths", [])) or "none",
        "unit_passed": rec["tests"]["unit"]["passed"], "unit_failed": rec["tests"]["unit"]["failed"],
        "real_passed": rec["tests"]["realdata"]["passed"], "real_failed": rec["tests"]["realdata"]["failed"],
        "overlay_n": len(rec["clone"]["overlay_files"]),
        "overlay_list": ", ".join(f"`{f}`" for f in rec["clone"]["overlay_files"]),
    }}
    return ctx, tables


def render(record: Path = RECORD, template: Path = TEMPLATE) -> str:
    rec = json.loads(Path(record).read_text(encoding="utf-8"))
    ctx, tables = context(rec)
    return render_template(Path(template).read_text(encoding="utf-8"), ctx, tables)


def main() -> None:
    DOC.write_text(render(), encoding="utf-8")
    print(f"wrote {DOC.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
