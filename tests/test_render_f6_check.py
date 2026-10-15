"""Clean-clone write-up: counts come from the saved record; unknown placeholders fail."""
import json

import pytest

from forecast import render_f6_check as r6


def record():
    row = {"sha256_first": "a", "sha256_second": "a"}
    return {
        "clone": {"overlay_files": ["src/x.py"]},
        "raw_files": [{"file": "calendar.csv", "sha256": "abc", "matches_config": True}],
        "steps": [{"step": "run_f1", "seconds": 90, "max_resident_mb": 1500.4, "peak_footprint_mb": 2600.0}],
        "holdout_rerun": {"comparison": {"status": "same_ignoring_run_info", "ignored_paths": ["commit"]}},
        "tests": {"unit": {"passed": 10, "failed": 0}, "realdata": {"passed": 3, "failed": 0}},
        "comparisons": {
            "results": [{"file": "a.json", "status": "identical", **row},
                        {"file": "b.json", "status": "same_ignoring_run_info", "ignored_paths": ["resources"],
                         "explanation": "timings", **row}],
            "docs": [{"file": "F4.md", "status": "DIFFERENT", "explanation": "run time in text", **row},
                     {"file": "old.md", "status": "MISSING_IN_SECOND", **row}],
        },
    }


def test_folder_summary_counts():
    s = r6.folder_summary(record()["comparisons"]["docs"])
    assert s == {"files": 2, "identical": 0, "same_ignoring_run_info": 0, "different": 1, "missing": 1, "match": 0}


def test_render_uses_the_record(tmp_path):
    rec, tpl = tmp_path / "r.json", tmp_path / "t.md"
    rec.write_text(json.dumps(record()))
    tpl.write_text("{{t.results_match}} of {{t.results_files}}; unit {{t.unit_passed}}\n{{table:not_identical}}")
    out = r6.render(rec, tpl)
    assert out.startswith("2 of 2; unit 10")
    assert "| `results/b.json` | same_ignoring_run_info | `resources` | timings |" in out
    assert "| `docs/F4.md` | DIFFERENT |  | run time in text |" in out
    assert "a.json" not in out


def test_render_unknown_placeholder_fails(tmp_path):
    rec, tpl = tmp_path / "r.json", tmp_path / "t.md"
    rec.write_text(json.dumps(record()))
    tpl.write_text("{{t.nope}}")
    with pytest.raises(KeyError):
        r6.render(rec, tpl)
