"""Report rendering: placeholders are filled only from the given context; unknown ones fail."""
import pytest

from forecast.report import markdown_table, render_template


def ctx():
    return {"t": {"rows": "1,234", "share": 0.1234, "nested": {"x": 7}}, "list": [{"a": 3}]}


def test_render_fills_values_from_context():
    assert render_template("Rows {{t.rows}}, x {{t.nested.x}}.", ctx()) == "Rows 1,234, x 7."


def test_render_percent_style():
    assert render_template("{{t.share|pct}}", ctx()) == "12.3%"


def test_render_indexes_into_lists():
    assert render_template("{{list.0.a}}", ctx()) == "3"


def test_render_unknown_placeholder_fails():
    with pytest.raises(KeyError, match="t.missing"):
        render_template("{{t.missing}}", ctx())


def test_render_named_tables_and_unknown_table_fails():
    assert render_template("A\n{{table:x}}\nB", ctx(), {"x": "| a |"}) == "A\n| a |\nB"
    with pytest.raises(KeyError, match="table:y"):
        render_template("{{table:y}}", ctx(), {})


def test_single_braces_are_left_alone():
    # mermaid diagrams and code use single braces; only {{...}} is a placeholder
    assert render_template("A{B} {x: 1}", ctx()) == "A{B} {x: 1}"


def test_markdown_table_aligns_numbers_and_formats():
    rows = [{"name": "TSB", "n": 1234, "rate": 0.25, "note": None}]
    out = markdown_table(rows, [("name", "Method"), ("n", "Series"), ("rate", "Rate", "pct"), ("note", "Note")])
    assert out.splitlines() == ["| Method | Series | Rate | Note |", "|---|---:|---:|---|",
                                "| TSB | 1,234 | 25.0% |  |"]
