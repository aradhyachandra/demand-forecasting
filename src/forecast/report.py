"""Fill F6 doc templates (docs/templates/REPORT.md, README.md) from saved results only.

A placeholder is {{a.b.c}} (a value in the context, {{a.b|pct}} as a percentage) or {{table:name}}
(a ready-made markdown table). An unknown placeholder raises, so a doc can never hold a number that is
not in the context. Single braces are left alone (mermaid, code).
"""
from __future__ import annotations

import re

PLACEHOLDER = re.compile(r"\{\{\s*([\w.\-: |]+?)\s*\}\}")


def fmt(value, style=None) -> str:
    if value is None:
        return ""
    if style == "pct":
        return f"{value:.1%}"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def markdown_table(rows: list[dict], columns: list[tuple]) -> str:
    """rows: list of dicts. columns: (key, header) or (key, header, "pct"). Numbers align right."""
    def numeric(key):
        return any(isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool) for r in rows)

    lines = ["| " + " | ".join(c[1] for c in columns) + " |",
             "|" + "|".join("---:" if numeric(c[0]) else "---" for c in columns) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c[0]), c[2] if len(c) > 2 else None) for c in columns) + " |")
    return "\n".join(lines)


def _lookup(context: dict, key: str, tables: dict) -> str:
    if key.startswith("table:"):
        name = key[len("table:"):]
        if name not in tables:
            raise KeyError(f"Placeholder {{{{{key}}}}} has no table")
        return tables[name]
    dotted, _, style = key.partition("|")
    value = context
    for part in dotted.split("."):
        if isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise KeyError(f"Placeholder {{{{{dotted}}}}} not found in the context")
    return fmt(value, style or None)


def render_template(text: str, context: dict, tables: dict | None = None) -> str:
    return PLACEHOLDER.sub(lambda m: _lookup(context, m.group(1), tables or {}), text)
