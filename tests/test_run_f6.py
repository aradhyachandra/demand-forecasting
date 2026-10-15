"""F6 report: helpers by hand, and the rendered report / README use only saved results, keep the
wording rules and answer the five business questions in five summary bullets."""
import re

import pytest

from forecast import run_f6
from forecast.config import ROOT


def test_change_and_wording_helpers():
    assert run_f6.change_pct(95, 100) == pytest.approx(-5)
    assert run_f6.less(-5.47) == "5.5% less" and run_f6.less(2.0) == "2.0% more"
    assert run_f6.signed(-3.26) == "-3.3%" and run_f6.signed(10.46) == "+10.5%"
    assert run_f6.money(1234.6) == "$1,235" and run_f6.money(None) == "n/a"


def test_sensitivity_summary_by_hand():
    def check(rank, tsb, sn):
        return {"assumption": "x", "value": 1, "ranking_by_total_cost": rank,
                "equal_fill": [{"fill_rate": 0.95, "lgbm_vs_tsb_pct": tsb, "lgbm_vs_seasonal_naive_pct": sn}]}
    s = run_f6.sensitivity_summary([check(["lgbm", "tsb"], -5, -10), check(["tsb", "lgbm"], -4, -12)], 0.95)
    assert s["n"] == 2 and s["lgbm_first"] == 1 and len(s["not_first"]) == 1
    assert s["always_less"] and s["tsb_min"] == -5 and s["tsb_max"] == -4
    assert s["seasonal_naive_min"] == -12 and s["seasonal_naive_max"] == -10


def test_weekly_errors_and_slice_bias():
    h = {"weekly_totals": [{"baseline": "lgbm", "forecast": 90, "actual": 100},
                           {"baseline": "lgbm", "forecast": 110, "actual": 100},
                           {"baseline": "tsb", "forecast": 1, "actual": 1}]}
    assert run_f6.weekly_errors(h, "lgbm") == pytest.approx([-10, 10])
    assert run_f6.slice_bias(h, "lgbm") == pytest.approx(0)


@pytest.fixture(scope="module")
def docs():
    return run_f6.render()


def test_render_fills_every_placeholder(docs):
    for name, text in docs.items():
        assert "{{" not in text and "}}" not in text, name


def test_headline_numbers_come_from_the_results(docs):
    r = run_f6.load_results()
    report = docs["docs/REPORT.md"]
    for m in ("lgbm", "tsb", "seasonal_naive"):
        assert f"{r['h']['overall'][m]['wrmsse']:.3f}" in report
    e = run_f6.equal_fill(r["f5"]["holdout"]["equal_fill"]["bias_adjusted"], 0.95)
    assert f"{abs(e['lgbm_vs_tsb_pct']):.1f}% less" in report
    assert f"{abs(e['lgbm_vs_seasonal_naive_pct']):.1f}% less" in report
    s = run_f6.sensitivity_summary(r["f5"]["sensitivity_checks"], 0.95)
    assert f"{len(s['not_first'])} of {s['n']}" in report
    assert f"{r['f2']['snap']['lift_pct']:+.1f}%" in report


def test_executive_summary_has_five_bullets(docs):
    report = docs["docs/REPORT.md"]
    section = report.split("## 1. Executive summary")[1].split("\n## ")[0]
    bullets = [line for line in section.splitlines() if line.startswith("- ") or re.match(r"^\d\. ", line)]
    assert len(bullets) == 5


def test_wording_rules(docs):
    for name, text in docs.items():
        assert not re.search(r"\bsavings?\b", text, re.I), f"{name}: inventory results are simulated outcomes"
        assert not re.search(r"\bcaused?\b|\bcauses\b", text, re.I), f"{name}: drivers are associations"


def test_every_figure_referenced_exists(docs):
    for name, text in docs.items():
        base = (ROOT / name).parent
        for path in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
            assert (base / path).exists(), f"{name}: {path}"
