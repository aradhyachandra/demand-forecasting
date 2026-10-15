"""F6: findings report and README, rendered from saved results only.

    python -m forecast.run_f6

Reads results/*.json (F1-F5) and config/inventory.yaml (assumption reasons). Loads no data, fits and
simulates nothing; it formats saved numbers (differences between saved numbers are computed here) and
draws one summary figure from them. Writes docs/REPORT.md, README.md and docs/figures/f6_summary.png.
docs/REPORT.md and README.md come from docs/templates/; never edit them by hand.
"""
from __future__ import annotations

import json

from forecast import config, plots
from forecast.report import markdown_table, render_template

ROOT = config.ROOT
RESULTS = ROOT / "results"
TEMPLATES = ROOT / "docs" / "templates"
OUTPUTS = {"docs/REPORT.md": "REPORT.md", "README.md": "README.md"}
FIGURE = ROOT / "docs" / "figures" / "f6_summary.png"
POLICY = {"seasonal_naive": "Seasonal naive", "tsb": "TSB", "lgbm": "LightGBM"}
METHOD = {"naive": "Naive (last day)", "seasonal_naive": "Seasonal naive (same weekday last week)",
          "moving_average": "28-day moving average", "ses": "Simple exponential smoothing", "croston": "Croston",
          "sba": "SBA (bias-corrected Croston)", "tsb": "TSB", "lgbm": "**LightGBM (global model)**"}
CLASSES = ["smooth", "erratic", "intermittent", "lumpy"]
PRICE_FEATURES = ("price_last", "price_ratio")
CALENDAR_FEATURES = ("wday", "month", "mday", "snap", "event_name", "event_type", "days_to_event")


def load_results(results=RESULTS) -> dict:
    names = {"f1": "f1_data_report", "f2": "f2_demand_patterns", "f3": "f3_backtest", "f4": "f4_backtest",
             "h": "f4_holdout", "f5": "f5_inventory", "fsw": "f6_first_sale_week"}
    return {k: json.loads((results / f"{v}.json").read_text(encoding="utf-8")) for k, v in names.items()}


def change_pct(a: float, b: float) -> float:
    """a relative to b, in percent."""
    return 100 * (a / b - 1)


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def signed(x: float, d: int = 1) -> str:
    return f"{x:+.{d}f}%"


def less(x: float, d: int = 1) -> str:
    """-5.47 -> '5.5% less'."""
    return f"{abs(x):.{d}f}% {'less' if x < 0 else 'more'}"


def money(x) -> str:
    return "n/a" if x is None else f"${x:,.0f}"


def equal_fill(rows: list[dict], fill: float) -> dict:
    return next(e for e in rows if abs(e["fill_rate"] - fill) < 1e-9)


def sensitivity_summary(checks: list[dict], fill: float) -> dict:
    """How often the model-based policy is cheapest at a fixed target, which scenarios are not, and the
    range of its inventory gap at an equal fill rate."""
    not_first = [c for c in checks if c["ranking_by_total_cost"][0] != "lgbm"]
    gaps = {ref: [equal_fill(c["equal_fill"], fill)[f"lgbm_vs_{ref}_pct"] for c in checks]
            for ref in ("tsb", "seasonal_naive")}
    return {"n": len(checks), "lgbm_first": len(checks) - len(not_first), "not_first": not_first,
            "always_less": all(g < 0 for v in gaps.values() for g in v),
            **{f"{ref}_min": min(v) for ref, v in gaps.items()}, **{f"{ref}_max": max(v) for ref, v in gaps.items()}}


def weekly_errors(holdout: dict, method: str) -> list[float]:
    return [change_pct(w["forecast"], w["actual"]) for w in holdout["weekly_totals"] if w["baseline"] == method]


def slice_bias(holdout: dict, method: str) -> float:
    w = [x for x in holdout["weekly_totals"] if x["baseline"] == method]
    return change_pct(sum(x["forecast"] for x in w), sum(x["actual"] for x in w))


SCENARIO_LABEL = {"lead_time_days": "lead time {} day(s)", "review_period_days": "review every {} day(s)",
                  "starting_inventory_factor": "starting stock {} x S", "unmet_demand": "unmet demand: {}",
                  "holding_cost_rate_per_year": "holding cost {} of price a year",
                  "stockout_cost_share_of_price": "lost-sale cost {} of price"}


def scenario_name(check: dict) -> str:
    v = check["value"]
    if check["assumption"] in ("holding_cost_rate_per_year", "stockout_cost_share_of_price"):
        v = f"{v:.0%}"
    text = SCENARIO_LABEL[check["assumption"]].format(str(v).replace("_", " "))
    return text.replace("day(s)", "day" if v == 1 else "days")


def gap_range(a: float, b: float) -> str:
    """Two negative gaps -> '4.7% to 7.3% less'."""
    lo, hi = sorted((abs(a), abs(b)))
    if a < 0 and b < 0:
        return f"{lo:.1f}% to {hi:.1f}% less"
    return f"{signed(a)} to {signed(b)}"


def direction_text(sens: dict, robustness_fill: dict) -> str:
    rob = robustness_fill["lgbm_vs_tsb_pct"] < 0 and robustness_fill["lgbm_vs_seasonal_naive_pct"] < 0
    if rob and sens["always_less"]:
        return f"The direction holds on a pre-holdout window and in all {sens['n']} assumption scenarios"
    return (f"The direction {'holds' if rob else 'does not hold'} on a pre-holdout window and "
            f"{'holds' if sens['always_less'] else 'does not hold'} in every one of {sens['n']} assumption scenarios")


def class_text(h: dict, ref: str) -> str:
    gaps = {k: change_pct(h["by_class"][k]["lgbm"]["wrmsse"], h["by_class"][k][ref]["wrmsse"]) for k in CLASSES}
    worse = [k for k, g in gaps.items() if g >= 0]
    if worse:
        return f"The model does not beat TSB for {', '.join(worse)} items:"
    return f"The model beats TSB in every class, by the most on {min(gaps, key=gaps.get)} items:"


def context(r: dict, inv_cfg: dict) -> tuple[dict, dict]:
    f1, f2, f3, f4, h, f5 = (r[k] for k in ("f1", "f2", "f3", "f4", "h", "f5"))
    sc, hr = f1["slice_checks"], f1["holdout_reserved"]
    opt = next(o for o in f1["scope_options"] if o["state_id"] == f1["scope"]["state"]
               and o["cat_id"] == f1["scope"]["category"])
    cls = {c["demand_class"]: c for c in f2["classes"]}
    wd = {w["weekday"]: w["index"] for w in f2["weekday_profile"]}
    ev = {e["event_name"]: e for e in f2["events"]["by_event"]}
    snap_dept = {d["dept_id"]: d["lift_pct"] for d in f2["snap_by_dept"]}
    pcs = {x["zero_sales_week"]: x for x in f2["price_cuts"]["price_change_vs_sales"]}
    ho, bo = h["overall"], f4["overall"]
    ref = h["setup"]["best_baseline_from_backtest"]
    hold_baselines = [m for m in h["setup"]["methods"] if m != "lgbm"]
    best_hold = min(hold_baselines, key=lambda m: ho[m]["wrmsse"])
    gain = {g["feature"]: g["mean_share"] for g in f4["feature_importance_gain"]}
    wk = weekly_errors(h, "lgbm")
    smoothing = [m for m in f3["overall"] if m not in ("naive", "seasonal_naive")]

    c = f5["central"]
    hs, rs = f5["holdout"]["setup"], f5["robustness"]["setup"]
    main = {p: f5["holdout"]["main"]["bias_adjusted"][p]["overall"] for p in f5["policies"]}
    zsd = {p: f5["holdout"]["main"]["z_sd_only"][p]["overall"] for p in f5["policies"]}
    e = equal_fill(f5["holdout"]["equal_fill"]["bias_adjusted"], c["sl"])
    er = equal_fill(f5["robustness"]["equal_fill"]["bias_adjusted"], c["sl"])
    ez = equal_fill(f5["holdout"]["equal_fill"]["z_sd_only"], c["sl"])
    sens = sensitivity_summary(f5["sensitivity_checks"], c["sl"])
    mon = f5["holdout"]["money"]["lgbm"]
    csl = [main[p]["cycle_service_level"] for p in f5["policies"]]

    t = {
        # data and scope
        "state": f1["scope"]["state"], "category": f1["scope"]["category"], "n_series": f"{sc['n_series']:,}",
        "n_stores": opt["n_stores"], "n_items": f"{opt['n_items']:,}", "all_series": f"{f1['all_series']:,}",
        "first_date": sc["first_date"], "last_date": sc["last_date"], "n_days": f"{sc['n_days']:,}",
        "n_rows": f"{sc['n_rows']:,}", "total_units": f"{sc['total_units']:,}",
        "unlisted_share": pct(sc["unlisted_rows"] / sc["n_rows"]),
        "unlisted_after": f"{sc['unlisted_after_first_listing']:,}",
        "fsw_after": f"{r['fsw']['series_starting_after_first_week']:,}",
        "fsw_sell": f"{r['fsw']['sell_in_first_priced_week']:,}",
        "listed_zero_share": pct(sc["listed_zero_rows"] / sc["listed_rows"]),
        "hold_first_d": str(hr["first_d"]), "hold_last_d": str(hr["last_d"]), "hold_first_date": hr["first_date"],
        "hold_last_date": hr["last_date"], "hold_days": hr["days"], "origin": str(h["setup"]["forecast_origin"]),
        "raw_sales_sha": f1["raw_files"]["sales"]["sha256"][:16],
        # demand patterns
        "not_smooth": pct(1 - cls["smooth"]["share_series"], 0),
        "int_lumpy_series": pct(cls["intermittent"]["share_series"] + cls["lumpy"]["share_series"], 0),
        "smooth_series": pct(cls["smooth"]["share_series"], 0), "smooth_units": pct(cls["smooth"]["share_units"], 0),
        "median_adi": f"{f2['classes_median']['adi']:.1f}",
        "share_days_with_sale": pct(f2["classes_median"]["share_listed_days_with_sales"], 0),
        "sun": signed(100 * (wd["Sunday"] - 1), 0), "sat": signed(100 * (wd["Saturday"] - 1), 0),
        "wed": signed(100 * (wd["Wednesday"] - 1), 0),
        "snap": signed(f2["snap"]["lift_pct"]), "snap_f1": signed(snap_dept["FOODS_1"]),
        "snap_f2": signed(snap_dept["FOODS_2"]), "snap_f3": signed(snap_dept["FOODS_3"]),
        **{f"ev_{k}": signed(ev[k]["mean_lift_pct"], 0) for k in
           ("LaborDay", "PresidentsDay", "MemorialDay", "Thanksgiving", "NewYear", "Easter")},
        "ev_christmas": signed(ev["Christmas"]["mean_lift_pct"]),
        "ev_n_occ": f2["events"]["n_occurrences_used"],
        "price_change_sales_weeks": pct(pcs[False]["change_rate"]),
        "price_change_zero_weeks": f"{pcs[True]['n_changes']:,} of {pcs[True]['n_weeks']:,}",
        # accuracy
        "lgbm_h": f"{ho['lgbm']['wrmsse']:.3f}", "tsb_h": f"{ho[ref]['wrmsse']:.3f}",
        "sn_h": f"{ho['seasonal_naive']['wrmsse']:.3f}", "lgbm_b": f"{bo['lgbm']['wrmsse']:.3f}",
        "tsb_b": f"{bo[ref]['wrmsse']:.3f}", "sn_b": f"{bo['seasonal_naive']['wrmsse']:.3f}",
        "gap_tsb_h": less(change_pct(ho["lgbm"]["wrmsse"], ho[ref]["wrmsse"])),
        "gap_tsb_b": less(change_pct(bo["lgbm"]["wrmsse"], bo[ref]["wrmsse"])),
        "gap_sn_h": less(change_pct(ho["lgbm"]["wrmsse"], ho["seasonal_naive"]["wrmsse"])),
        "best_hold": METHOD[best_hold], "best_hold_w": f"{ho[best_hold]['wrmsse']:.3f}",
        "gap_best_hold": less(change_pct(ho["lgbm"]["wrmsse"], ho[best_hold]["wrmsse"])),
        "smooth_min": f"{min(f3['overall'][m]['wrmsse'] for m in smoothing):.3f}",
        "smooth_max": f"{max(f3['overall'][m]['wrmsse'] for m in smoothing):.3f}",
        "win_sn": pct(h["wins_vs_reference"]["overall"]["lgbm"]["pairs"]["share_wins"], 0),
        "win_sn_n": f"{h['wins_vs_reference']['overall']['lgbm']['pairs']['wins']:,}",
        "win_tsb": pct(h["wins_vs_best_baseline"]["overall"]["pairs"]["share_wins"], 0),
        "win_tsb_n": f"{h['wins_vs_best_baseline']['overall']['pairs']['wins']:,}",
        "n_scored": f"{h['coverage'][0]['n_scored']:,}",
        "lgbm_wape_h": pct(ho["lgbm"]["weekly_wape"]), "tsb_wape_h": pct(ho[ref]["weekly_wape"]),
        "bias_h": signed(slice_bias(h, "lgbm")), "bias_tsb_h": signed(slice_bias(h, ref)),
        "bias_b": signed(f4["slice_total_bias_pct"]["lgbm"]), "bias_h_abs": f"{abs(slice_bias(h, 'lgbm')):.1f}%",
        "bias_b_abs": f"{abs(f4['slice_total_bias_pct']['lgbm']):.1f}%",
        "wk_min": signed(min(wk)), "wk_max": signed(max(wk)), "wk_under": sum(x < 0 for x in wk), "wk_n": len(wk),
        "wk_low": (f"{abs(max(wk)):.1f}% to {abs(min(wk)):.1f}% low" if max(wk) < 0
                   else f"{signed(min(wk))} to {signed(max(wk))} off"),
        "easier_text": ("Every method scored better on the holdout than in the backtest"
                        if all(ho[m]["wrmsse"] < bo[m]["wrmsse"] for m in ho)
                        else "The methods did not all score better on the holdout than in the backtest"),
        "class_text": class_text(h, ref),
        "snap_wd_min": signed(min(x["lift_pct"] for x in f2["snap"]["by_weekday"])),
        "snap_wd_max": signed(max(x["lift_pct"] for x in f2["snap"]["by_weekday"])),
        "gain_recent": pct(gain["mean_28"] + gain["mean_7"], 0),
        "gain_price": pct(sum(gain[f] for f in PRICE_FEATURES), 2),
        "gain_calendar": pct(sum(gain[f] for f in CALENDAR_FEATURES), 1),
        "model_leaves": f4["chosen"]["num_leaves"], "model_trees": f4["chosen"]["rounds"],
        "holdout_commit": h["commit"][:7], "holdout_minutes": str(h["run_minutes"]),
        "grid_n": len(f4["grid"]), "grid_min": f"{min(g['wrmsse'] for g in f4['grid']):.3f}",
        "grid_max": f"{max(g['wrmsse'] for g in f4['grid']):.3f}",
        # inventory
        "lead": c["lead"], "review": c["review"], "sl": pct(c["sl"], 0), "hold_rate": pct(c["hold"], 0),
        "short_share": pct(c["short"], 0), "protection": hs["protection_days"],
        "n_reviews": len(hs["reviews_d"]), "measured_days": hs["measured_days"],
        "measured_first": str(hs["measured_d"][0]), "measured_last": str(hs["measured_d"][1]),
        "n_sim": f"{hs['n_simulated']:,}", "n_excl": hs["n_excluded_no_backtest"],
        "fill_sn": pct(main["seasonal_naive"]["fill_rate"]), "fill_tsb": pct(main["tsb"]["fill_rate"]),
        "fill_lgbm": pct(main["lgbm"]["fill_rate"]),
        "inv_lgbm": money(main["lgbm"]["avg_inv_money"]), "inv_sn": money(main["seasonal_naive"]["avg_inv_money"]),
        "inv_tsb": money(main["tsb"]["avg_inv_money"]),
        "inv_gap_sn": less(change_pct(main["lgbm"]["avg_inv_money"], main["seasonal_naive"]["avg_inv_money"])),
        "inv_gap_tsb": less(change_pct(main["lgbm"]["avg_inv_money"], main["tsb"]["avg_inv_money"])),
        "e_lgbm": money(e["inventory_money"]["lgbm"]), "e_tsb": money(e["inventory_money"]["tsb"]),
        "e_sn": money(e["inventory_money"]["seasonal_naive"]),
        "e_gap_tsb": less(e["lgbm_vs_tsb_pct"]), "e_gap_sn": less(e["lgbm_vs_seasonal_naive_pct"]),
        "er_gap_tsb": less(er["lgbm_vs_tsb_pct"]), "er_gap_sn": less(er["lgbm_vs_seasonal_naive_pct"]),
        "rob_origin": str(rs["forecast_origin"]), "rob_first": str(rs["forecast_days"][0]),
        "rob_last": str(rs["forecast_days"][1]),
        "rob_ranking": " > ".join(POLICY[p] for p in f5["robustness"]["ranking_by_total_cost"]),
        "hold_ranking": " > ".join(POLICY[p] for p in f5["holdout"]["ranking_by_total_cost"]),
        "cost_lgbm": money(main["lgbm"]["total_cost"]), "cost_tsb": money(main["tsb"]["total_cost"]),
        "cost_sn": money(main["seasonal_naive"]["total_cost"]),
        "n_sc": sens["n"], "sc_lgbm_first": sens["lgbm_first"], "sc_flips": len(sens["not_first"]),
        "sc_flip_list": "; ".join(f"{scenario_name(x)} ({POLICY[x['ranking_by_total_cost'][0]]} cheapest)"
                                  for x in sens["not_first"]),
        "sc_always": "every one" if sens["always_less"] else "not every one",
        "sc_tsb_range": gap_range(sens["tsb_max"], sens["tsb_min"]),
        "sc_sn_range": gap_range(sens["seasonal_naive_max"], sens["seasonal_naive_min"]),
        "direction_text": direction_text(sens, er),
        "central_cost_text": ("with the lowest simulated total cost" if f5["holdout"]["ranking_by_total_cost"][0] == "lgbm"
                              else "but not the lowest simulated total cost"),
        "csl_min": pct(min(csl)), "csl_max": pct(max(csl)), "csl_lgbm": pct(main["lgbm"]["cycle_service_level"]),
        "z_e_lgbm": money(ez["inventory_money"]["lgbm"]),
        "z_vs_main": less(change_pct(ez["inventory_money"]["lgbm"], e["inventory_money"]["lgbm"])),
        "z_fill_lgbm": pct(zsd["lgbm"]["fill_rate"]),
        # where the money is
        "top_share": pct(f5["concentration_share"], 0), "top_lost": pct(mon["top_share_of_lost_cost"], 0),
        "top_inv": pct(mon["top_share_of_inventory_money"], 0), "top_dem": pct(mon["top_share_of_demand"], 0),
        "n_lost": f"{mon['series_with_any_lost_sale']:,}", "lost_cost": money(mon["lost_cost"]),
        "int_lost": pct(mon["by_class"]["intermittent"]["lost_cost_share"], 0),
        "int_dem": pct(mon["by_class"]["intermittent"]["demand_share"], 0),
        "err_lost": pct(mon["by_class"]["erratic"]["lost_cost_share"], 0),
        "err_dem": pct(mon["by_class"]["erratic"]["demand_share"], 0),
        "smooth_lost": pct(mon["by_class"]["smooth"]["lost_cost_share"], 0),
        "smooth_dem": pct(mon["by_class"]["smooth"]["demand_share"], 0),
        "smooth_inv": pct(mon["by_class"]["smooth"]["inventory_share"], 0),
        "f1_lost": pct(mon["by_dept"]["FOODS_1"]["lost_cost_share"], 0),
        "f1_dem": pct(mon["by_dept"]["FOODS_1"]["demand_share"], 0),
        "f2_inv": pct(mon["by_dept"]["FOODS_2"]["inventory_share"], 0),
        "f2_dem": pct(mon["by_dept"]["FOODS_2"]["demand_share"], 0),
        "toprev_lost": pct(mon["by_revenue"]["top"]["lost_cost_share"], 0),
        "toprev_dem": pct(mon["by_revenue"]["top"]["demand_share"], 0),
    }
    return {"t": t}, tables(r, inv_cfg)


def tables(r: dict, inv_cfg: dict) -> dict:
    f2, f4, h, f5 = r["f2"], r["f4"], r["h"], r["f5"]
    ho, bo = h["overall"], f4["overall"]
    ref = h["setup"]["best_baseline_from_backtest"]
    pols, c = f5["policies"], f5["central"]

    accuracy = [{"m": METHOD[m], "h": f"{ho[m]['wrmsse']:.3f}", "b": f"{bo[m]['wrmsse']:.3f}",
                 "wape": pct(ho[m]["weekly_wape"]), "bias": signed(slice_bias(h, m))} for m in h["setup"]["methods"]]
    by_class = [{"c": k, "n": h["by_class"][k]["lgbm"]["n"], "lgbm": f"{h['by_class'][k]['lgbm']['wrmsse']:.3f}",
                 "tsb": f"{h['by_class'][k][ref]['wrmsse']:.3f}",
                 "sn": f"{h['by_class'][k]['seasonal_naive']['wrmsse']:.3f}",
                 "gh": signed(change_pct(h["by_class"][k]["lgbm"]["wrmsse"], h["by_class"][k][ref]["wrmsse"])),
                 "gb": signed(change_pct(f4["by_class"][k]["lgbm"]["wrmsse"], f4["by_class"][k][ref]["wrmsse"]))}
                for k in CLASSES]
    classes = [{"c": x["demand_class"], "n": x["n_series"], "s": pct(x["share_series"]), "u": pct(x["share_units"])}
               for x in f2["classes"] if x["n_series"]]
    events = sorted(f2["events"]["by_event"], key=lambda x: -x["mean_lift_pct"])
    ev_rows = [{"e": x["event_name"], "n": x["n"], "m": signed(x["mean_lift_pct"]),
                "r": f"{signed(x['min_lift_pct'])} to {signed(x['max_lift_pct'])}"} for x in events[:4] + events[-4:]]
    central = [{"p": POLICY[p], "f": pct(o["fill_rate"]), "csl": pct(o["cycle_service_level"]),
                "inv": money(o["avg_inv_money"]), "hc": money(o["holding_cost"]), "lc": money(o["lost_cost"]),
                "tc": money(o["total_cost"])}
               for p in pols for o in [f5["holdout"]["main"]["bias_adjusted"][p]["overall"]]]

    def ef_rows(rows):
        return [{"f": pct(x["fill_rate"], 0), **{p: money(x["inventory_money"][p]) for p in pols},
                 "vs_sn": signed(x["lgbm_vs_seasonal_naive_pct"]) if "lgbm_vs_seasonal_naive_pct" in x else "n/a",
                 "vs_tsb": signed(x["lgbm_vs_tsb_pct"]) if "lgbm_vs_tsb_pct" in x else "n/a"} for x in rows]
    ef_cols = [("f", "Fill rate"), *[(p, POLICY[p]) for p in pols], ("vs_sn", "Model vs seasonal naive"),
               ("vs_tsb", "Model vs TSB")]

    sens_rows = []
    for ch in f5["sensitivity_checks"]:
        if ch["is_central"] and ch["assumption"] != "lead_time_days":
            continue                                   # the central scenario is listed once
        e = equal_fill(ch["equal_fill"], c["sl"])
        sens_rows.append({"s": "central scenario" if ch["is_central"] else scenario_name(ch),
                          "rank": " > ".join(POLICY[p] for p in ch["ranking_by_total_cost"]),
                          "sn": signed(e["lgbm_vs_seasonal_naive_pct"]), "tsb": signed(e["lgbm_vs_tsb_pct"])})

    mon = f5["holdout"]["money"]
    where = []
    for level, keys in (("by_class", CLASSES), ("by_dept", sorted(mon["lgbm"]["by_dept"]))):
        for k in keys:
            m = mon["lgbm"][level][k]
            where.append({"g": k, "d": pct(m["demand_share"], 0), "l": pct(m["lost_cost_share"], 0),
                          "i": pct(m["inventory_share"], 0)})
    for k, name in (("top", f"top {f5['top_revenue_share']:.0%} by revenue"), ("rest", "other items")):
        m = mon["lgbm"]["by_revenue"][k]
        where.append({"g": name, "d": pct(m["demand_share"], 0), "l": pct(m["lost_cost_share"], 0),
                      "i": pct(m["inventory_share"], 0)})

    a = f5["assumptions"]
    reasons = {"lead_time_days": "lead_time_reason", "review_period_days": "review_period_reason",
               "holding_cost_rate_per_year": "holding_cost_reason", "stockout_cost_share_of_price": "stockout_cost_reason",
               "unmet_demand": "unmet_demand_reason", "starting_inventory_factor": "starting_inventory_reason"}
    names = {"lead_time_days": "Lead time (days)", "review_period_days": "Review period (days)",
             "holding_cost_rate_per_year": "Holding cost (share of price per year)",
             "stockout_cost_share_of_price": "Cost of a lost sale (share of price)", "unmet_demand": "Unmet demand",
             "starting_inventory_factor": "Starting stock (x order-up-to level)"}
    assumptions = [{"a": names[k], "c": str(a[k]["central"]).replace("_", " "),
                    "r": ", ".join(str(v).replace("_", " ") for v in a[k]["scenarios"]),
                    "why": " ".join(inv_cfg[reasons[k]].split())} for k in names]
    assumptions.append({"a": "Target cycle service level", "c": pct(c["sl"], 0),
                        "r": ", ".join(f"{v:.1%}" for v in f5["service_level_sweep"]),
                        "why": " ".join(inv_cfg["service_level_reason"].split())})

    readme = [{"m": METHOD[m], "h": f"{ho[m]['wrmsse']:.3f}", "b": f"{bo[m]['wrmsse']:.3f}"}
              for m in ("seasonal_naive", ref, "lgbm")]
    e = equal_fill(f5["holdout"]["equal_fill"]["bias_adjusted"], c["sl"])
    for row, p in zip(readme, ("seasonal_naive", ref, "lgbm")):
        row["inv"] = money(e["inventory_money"][p])
        row["fill"] = pct(f5["holdout"]["main"]["bias_adjusted"][p]["overall"]["fill_rate"])

    return {
        "classes": markdown_table(classes, [("c", "Demand class"), ("n", "Item-stores"), ("s", "Share of item-stores"),
                                            ("u", "Share of units sold")]),
        "events": markdown_table(ev_rows, [("e", "Event"), ("n", "Years"), ("m", "Mean difference vs nearby days"),
                                           ("r", "Lowest to highest year")]),
        "accuracy": markdown_table(accuracy, [("m", "Method"), ("h", "Holdout WRMSSE"), ("b", "Backtest WRMSSE (4 windows)"),
                                              ("wape", "Holdout weekly WAPE"), ("bias", "Holdout slice-total bias")]),
        "by_class": markdown_table(by_class, [("c", "Demand class"), ("n", "Item-stores"), ("lgbm", "LightGBM"),
                                              ("tsb", "TSB"), ("sn", "Seasonal naive"),
                                              ("gh", "Model vs TSB, holdout"), ("gb", "Model vs TSB, backtest")]),
        "central": markdown_table(central, [("p", "Policy"), ("f", "Fill rate"), ("csl", "Cycle service level"),
                                            ("inv", "Avg inventory"), ("hc", "Holding cost"), ("lc", "Lost-sales cost"),
                                            ("tc", "Total cost")]),
        "equal_fill": markdown_table(ef_rows(f5["holdout"]["equal_fill"]["bias_adjusted"]), ef_cols),
        "equal_fill_check": markdown_table(ef_rows(f5["robustness"]["equal_fill"]["bias_adjusted"]), ef_cols),
        "sensitivity": markdown_table(sens_rows, [("s", "Scenario (one assumption moved)"),
                                                  ("rank", "Cheapest at the 95% target"),
                                                  ("sn", "Model inventory vs seasonal naive at 95% fill"),
                                                  ("tsb", "vs TSB")]),
        "where": markdown_table(where, [("g", "Group"), ("d", "Share of demand units"), ("l", "Share of lost-sales cost"),
                                        ("i", "Share of inventory money")]),
        "assumptions": markdown_table(assumptions, [("a", "Assumption"), ("c", "Central"), ("r", "Range tested"),
                                                    ("why", "Reason (config/inventory.yaml)")]),
        "readme_results": markdown_table(readme, [("m", "Method"), ("h", "Holdout WRMSSE"), ("b", "Backtest WRMSSE"),
                                                  ("fill", "Fill rate at the 95% target"),
                                                  ("inv", "Inventory needed for 95% fill")]),
    }


def render(results=RESULTS) -> dict:
    """{output path relative to the repo: rendered text}, from saved results only."""
    ctx, tabs = context(load_results(results), config.load("inventory"))
    return {out: render_template((TEMPLATES / tpl).read_text(encoding="utf-8"), ctx, tabs)
            for out, tpl in OUTPUTS.items()}


def main() -> None:
    r = load_results()
    plots.f6_summary(r["h"], r["f5"], FIGURE)
    for out, text in render().items():
        (ROOT / out).write_text(text, encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
