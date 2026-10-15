"""F2 figures, saved as PNG under docs/figures/. Static images for the markdown docs, so values are
written on the bars instead of in tooltips."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, BLUE_LIGHT, RED, GRAY = "#2a78d6", "#9ec5f4", "#e34948", "#a3a29c"
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e6e5e0"


def _style(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _title(ax, text, sub=None):
    ax.set_title(text, fontsize=11, color=INK, loc="left", pad=18 if sub else 8)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8, color=MUTED)


def demand_class_map(classes: pd.DataFrame, summary: list[dict], cfg: dict, path, cv2_max: float = 3.0):
    """ADI (log scale) against CV², one dot per series, with the class cut-offs and counts."""
    df = classes[classes["demand_class"] != "too_few_sales"]
    y = df["cv2"].clip(upper=cv2_max)
    n_clipped = int((df["cv2"] > cv2_max).sum())

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(df["adi"], y, s=6, color=BLUE, alpha=0.25, linewidths=0)
    ax.set_xscale("log")
    ax.axvline(cfg["adi_cutoff"], color=INK, linewidth=1)
    ax.axhline(cfg["cv2_cutoff"], color=INK, linewidth=1)
    ax.set_ylim(0, cv2_max * 1.04)
    lo, hi = df["adi"].min() * 0.95, df["adi"].max() * 1.05
    ax.set_xlim(lo, hi)
    ax.set_xlabel("ADI: listed days per day with a sale (log scale)", fontsize=9, color=INK)
    ax.set_ylabel("CV²: variability of the non-zero daily sales", fontsize=9, color=INK)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))

    by = {s["demand_class"]: s for s in summary}
    spots = {"smooth": (lo * 1.02, 0.03, "left", "bottom"),
             "erratic": (lo * 1.02, cv2_max * 0.98, "left", "top"),
             "intermittent": (hi * 0.98, 0.03, "right", "bottom"),
             "lumpy": (hi * 0.98, cv2_max * 0.98, "right", "top")}
    for c, (x, yy, ha, va) in spots.items():
        s = by[c]
        ax.text(x, yy, f"{c}\n{s['n_series']:,} series ({s['share_series']:.0%})", ha=ha, va=va,
                fontsize=9, color=INK, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    _style(ax, grid_axis="both")
    _title(ax, "Demand classes of the training series (Syntetos-Boylan-Croston)",
           f"Cut-offs ADI {cfg['adi_cutoff']}, CV² {cfg['cv2_cutoff']}. {n_clipped} series with CV² above "
           f"{cv2_max:g} drawn at the top."
           + (f" {by['too_few_sales']['n_series']} with too few sales not shown." if by["too_few_sales"]["n_series"] else ""))
    _save(fig, path)


def _bar_labels(ax, bars, values, fmt, horizontal=False):
    for b, v in zip(bars, values):
        if horizontal:
            x = b.get_width()
            ax.text(x + (0.4 if x >= 0 else -0.4), b.get_y() + b.get_height() / 2, fmt(v),
                    ha="left" if x >= 0 else "right", va="center", fontsize=7.5, color=INK)
        else:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt(v), ha="center", va="bottom",
                    fontsize=7.5, color=INK)


def weekday_month_profile(weekday: pd.DataFrame, month: pd.DataFrame, path):
    """Bars = % above or below the average units per listed series-day (0 = average)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), sharey=True, gridspec_kw={"width_ratios": [7, 12]})
    wk = 100 * (weekday["index"] - 1)
    bars = a1.bar([w[:3] for w in weekday["weekday"]], wk, color=BLUE, width=0.7)
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x = np.arange(len(month))
    mo = 100 * (month["index"] - 1)
    a2.bar(x, mo, color=BLUE, width=0.7)
    a2.vlines(x, 100 * (month["min"] - 1), 100 * (month["max"] - 1), color=INK, linewidth=1)
    a2.set_xticks(x, [names[m - 1] for m in month["month"]])
    for b, v in zip(bars, wk):
        a1.text(b.get_x() + b.get_width() / 2, v + (0.6 if v >= 0 else -0.6), f"{v:+.0f}%", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK)
    for ax in (a1, a2):
        ax.axhline(0, color=INK, linewidth=1)
        _style(ax)
    a1.set_ylabel("% vs average units per listed series-day", fontsize=8.5, color=INK)
    _title(a1, "By weekday", "Non-event training days")
    _title(a2, "By month", f"Average over the {int(month['n_years'].iloc[0])} full years; line = lowest to highest year")
    _save(fig, path)


def snap_effect(snap: dict, by_dept: list[dict], path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [4, 7]})
    labels = ["All FOODS"] + [d["dept_id"] for d in by_dept]
    vals = [snap["lift_pct"]] + [d["lift_pct"] for d in by_dept]
    bars = a1.barh(labels[::-1], vals[::-1], color=[BLUE_LIGHT] * len(by_dept) + [BLUE], height=0.6)
    _bar_labels(a1, bars, vals[::-1], lambda v: f"{v:+.1f}%", horizontal=True)
    wk = snap["by_weekday"]
    bars = a2.bar([w["weekday"][:3] for w in wk], [w["lift_pct"] for w in wk], color=BLUE, width=0.7)
    _bar_labels(a2, bars, [w["lift_pct"] for w in wk], lambda v: f"{v:+.1f}%")
    _style(a1, grid_axis="x")
    _style(a2)
    a1.set_xlim(0, max(vals) * 1.3)
    a1.set_xlabel("SNAP days vs other days, % more units per listed series-day", fontsize=8.5, color=INK)
    _title(a1, "SNAP effect by department", "Non-event training days")
    _title(a2, "SNAP effect by weekday", "Same comparison within each weekday")
    _save(fig, path)


def event_effects(by_event: pd.DataFrame, path):
    df = by_event.sort_values("mean_lift_pct")
    fig, ax = plt.subplots(figsize=(8, 0.26 * len(df) + 1.2))
    colors = [BLUE if v >= 0 else RED for v in df["mean_lift_pct"]]
    labels = [f"{n}  ({t}, n={k})" for n, t, k in zip(df["event_name"], df["event_type"], df["n"])]
    bars = ax.barh(labels, df["mean_lift_pct"], color=colors, height=0.65)
    _bar_labels(ax, bars, df["mean_lift_pct"], lambda v: f"{v:+.0f}%", horizontal=True)
    ax.axvline(0, color=INK, linewidth=1)
    lo, hi = df["mean_lift_pct"].min(), df["mean_lift_pct"].max()
    ax.set_xlim(lo - 12, hi + 12)
    ax.set_xlabel("Event day vs same weekday +-4 weeks (same SNAP status), % difference in units "
                  "per listed series-day", fontsize=8, color=INK)
    ax.tick_params(axis="y", labelsize=7.5)
    _style(ax, grid_axis="x")
    _title(ax, "Event days compared with nearby ordinary days", "Mean over the years each event occurs in "
           "training (n); correlation, not proven cause")
    _save(fig, path)


def price_cut_effect(summary: dict, before_weeks: int, path):
    """Left: the configured before/after summary per cut size, with no-change windows in grey.
    Right: week-by-week units (indexed to the before weeks) and price around the cut week."""
    fig = plt.figure(figsize=(12, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[5, 7], height_ratios=[3, 2], wspace=0.28, hspace=0.45)
    a1, a2, a3 = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])

    rows = [(f"Cut {b['label']}\n(n={b['n_cuts']:,})", b["pooled_lift_pct"], BLUE) for b in summary["buckets"]
            if b["n_cuts"]]
    c = summary["control"]
    rows.append((f"No price change\n(n={c['n_windows']:,})", c["pooled_lift_pct"], GRAY))
    bars = a1.bar([r[0] for r in rows], [r[1] for r in rows], color=[r[2] for r in rows], width=0.6)
    for b, v in zip(bars, [r[1] for r in rows]):
        a1.text(b.get_x() + b.get_width() / 2, v, f"{v:+.1f}%", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK)
    a1.axhline(0, color=INK, linewidth=1)
    lo = min(r[1] for r in rows)
    hi = max(r[1] for r in rows)
    a1.set_ylim(min(lo, 0) * 1.25 - 1, max(hi, 0) * 1.25 + 1)
    a1.set_ylabel("Units per day: cut week + next week vs the\n4 weeks before (pooled), % change",
                  fontsize=8.5, color=INK)
    a1.tick_params(axis="x", labelsize=7.5)
    _style(a1)
    _title(a1, "Sales after a price cut (configured summary)", "Correlation, not proven cause")

    colors = {"5-10%": BLUE, "10-20%": "#eb6834", "20%+": "#1baf7a", "no change": GRAY}
    for label, prof in summary["profiles"].items():
        df = pd.DataFrame(prof).dropna()
        base = df[(df["week"] < 0) & (df["week"] >= -before_weeks)]["units_per_day"].mean()
        color = colors.get(label, INK)
        name = label if label == "no change" else f"cut {label}"
        a2.plot(df["week"], df["units_per_day"] / base, color=color, linewidth=2)
        a2.text(df["week"].iloc[-1] + 0.3, df["units_per_day"].iloc[-1] / base, name, color=INK,
                fontsize=7.5, va="center")
        a3.plot(df["week"], df["price_vs_before"], color=color, linewidth=2)
    for ax in (a2, a3):
        ax.axvline(0, color=MUTED, linewidth=1, linestyle=":")
        n = summary_weeks(summary)
        ax.set_xlim(-n - 0.3, n + 2.5)
        ax.set_xticks(range(-n, n + 1, 2))
        _style(ax, grid_axis="both")
    a2.axhline(1.0, color=INK, linewidth=1)
    a2.set_ylabel("Units/day (1 = the 4\nweeks before)", fontsize=8, color=INK)
    a3.set_ylabel("Price vs\nbefore", fontsize=8, color=INK)
    a3.set_xlabel("Weeks from the cut week (0)", fontsize=8.5, color=INK)
    _title(a2, "Week by week around the cut", "Pooled over all cuts of each size; grey = no price change")
    _save(fig, path)


def summary_weeks(summary: dict) -> int:
    return max(r["week"] for r in next(iter(summary["profiles"].values())))


# --- F3 -----------------------------------------------------------------------------------------

F3_SHORT = {"naive": "Naive", "seasonal_naive": "Seasonal naive", "moving_average": "28-day MA",
            "ses": "SES", "croston": "Croston", "sba": "SBA", "tsb": "TSB"}
ORANGE = "#eb6834"


def _baseline_bars(ax, metrics: dict, order: list[str], ref: str, key: str = "wrmsse"):
    """Horizontal bars, one per baseline (top to bottom in `order`); the best is dark blue, the
    reference grey, the rest light blue. Values written at the bar ends."""
    vals = [metrics[b][key] for b in order]
    best = order[int(np.argmin(vals))]
    colors = [BLUE if b == best else GRAY if b == ref else BLUE_LIGHT for b in order]
    bars = ax.barh([F3_SHORT[b] for b in order][::-1], vals[::-1], color=colors[::-1], height=0.65)
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=7.5, color=INK)
    ax.axvline(1.0, color=MUTED, linewidth=1, linestyle=":")
    ax.set_xlim(0, max(vals) * 1.22)
    _style(ax, grid_axis="x")


def f3_scaled_error_by_class(report: dict, path):
    """Revenue-weighted RMSSE per baseline: all series, then one panel per demand class."""
    order, ref = report["setup"]["baselines"], report["setup"]["reference_baseline"]
    panels = [("All series", report["overall"])] + [(c, m) for c, m in report["by_class"].items()]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 3.4), sharey=True)
    for ax, (name, m) in zip(axes, panels):
        _baseline_bars(ax, m, order, ref)
        n = m[order[0]]["n"]
        ax.set_title(f"{name}\n{n:,} series-origins", fontsize=9, color=INK, loc="left")
        ax.set_xticks([0, 0.5, 1.0])
    axes[0].text(0, 1.22, "Scaled error by baseline and demand class (WRMSSE, lower is better)",
                 transform=axes[0].transAxes, fontsize=11, color=INK)
    axes[0].text(0, 1.155, "Mean over the 4 backtest origins; dark blue = best in the panel, grey = seasonal "
                 "naive, dotted line = 1 (as bad as the in-sample one-step naive)",
                 transform=axes[0].transAxes, fontsize=8, color=MUTED)
    fig.subplots_adjust(wspace=0.12, top=0.78)
    _save(fig, path)


def f3_error_by_origin(report: dict, path):
    """Per baseline: WRMSSE at each of the origins (dots) and their range (line)."""
    order = report["setup"]["baselines"]
    origins = list(report["by_origin"])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, b in enumerate(order[::-1]):
        vals = [report["by_origin"][o][b]["wrmsse"] for o in origins]
        ax.hlines(i, min(vals), max(vals), color=BLUE_LIGHT, linewidth=4)
        ax.scatter(vals, [i] * len(vals), s=28, color=BLUE, zorder=3, edgecolors="white", linewidths=1)
        ax.text(max(vals) + 0.012, i, f"{min(vals):.3f} - {max(vals):.3f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(order)), [F3_SHORT[b] for b in order[::-1]])
    lo = min(report["by_origin"][o][b]["wrmsse"] for o in origins for b in order)
    hi = max(report["by_origin"][o][b]["wrmsse"] for o in origins for b in order)
    ax.set_xlim(lo - 0.05, hi + 0.12)
    ax.set_xlabel("WRMSSE at one origin (lower is better)", fontsize=8.5, color=INK)
    _style(ax, grid_axis="x")
    _title(ax, "How much each baseline's error moves between the 4 origins",
           "One dot per origin (" + ", ".join(f"d_{o}" for o in origins) + "); line = lowest to highest")
    _save(fig, path)


def f3_wins(report: dict, path):
    """Share of series-origin pairs where each baseline has a lower RMSE than the reference
    (wins), the same (ties) or a higher one (losses)."""
    w = report["wins_vs_reference"]["overall"]
    ref = report["wins_vs_reference"]["reference"]
    order = [b for b in report["setup"]["baselines"] if b != ref]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    y = [F3_SHORT[b] for b in order][::-1]
    left = np.zeros(len(order))
    for part, color, label in (("wins", BLUE, "beats seasonal naive"), ("ties", GRID, "tie"),
                               ("losses", ORANGE, "worse than seasonal naive")):
        share = np.array([w[b]["pairs"][part] / w[b]["pairs"]["n"] for b in order][::-1])
        bars = ax.barh(y, share, left=left, color=color, height=0.65, label=label,
                       edgecolor="white", linewidth=2)
        for bar, s in zip(bars, share):
            if s >= 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, f"{s:.0%}",
                        ha="center", va="center", fontsize=7.5, color="white" if color != GRID else INK)
        left += share
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=8)
    _style(ax, grid_axis="x")
    n = w[order[0]]["pairs"]["n"]
    _title(ax, "How often each baseline beats seasonal naive, series by series",
           f"{n:,} series-origin pairs; comparison on RMSE over the 28-day window")
    _save(fig, path)
