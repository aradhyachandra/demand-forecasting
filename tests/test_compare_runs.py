"""Clean-clone comparison: files must match byte for byte; JSON may differ only in run-time fields."""
import json

from forecast.compare_runs import compare_dirs, json_differences, strip_run_info


def test_strip_run_info_drops_run_time_fields_at_any_depth():
    d = {"commit": "x", "started_utc": "t", "run_minutes": 1.0, "resources": {"peak_memory_gb": 2},
         "a": [{"build_seconds": 3, "fit_seconds": {"s1": 4}, "b": 1}], "setup": {"n": 2}}
    assert strip_run_info(d) == {"a": [{"b": 1}], "setup": {"n": 2}}


def test_json_differences_lists_paths():
    a = {"x": 1, "y": {"z": [1, 2]}, "only_a": 0}
    b = {"x": 1, "y": {"z": [1, 3]}}
    assert json_differences(a, b) == ["y.z.1", "only_a"]


def test_compare_dirs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        (d / "sub").mkdir(parents=True)
    (a / "r.json").write_text(json.dumps({"started_utc": "1", "x": 1}))
    (b / "r.json").write_text(json.dumps({"started_utc": "2", "x": 1}))
    (a / "w.json").write_text(json.dumps({"x": 1}))
    (b / "w.json").write_text(json.dumps({"x": 2}))
    (a / "t.csv").write_text("x\n1\n")
    (b / "t.csv").write_text("x\n2\n")
    (a / "sub" / "f.png").write_bytes(b"png")
    (b / "sub" / "f.png").write_bytes(b"png")
    (a / "only_a.csv").write_text("x\n")
    rows = {r["file"]: r for r in compare_dirs(a, b)}
    assert {k: r["status"] for k, r in rows.items()} == {
        "sub/f.png": "identical", "r.json": "same_ignoring_run_info", "w.json": "DIFFERENT",
        "t.csv": "DIFFERENT", "only_a.csv": "MISSING_IN_SECOND"}
    assert rows["r.json"]["ignored_paths"] == ["started_utc"]
    assert rows["w.json"]["differing_paths"] == ["x"]
