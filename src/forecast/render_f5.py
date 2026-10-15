"""Format docs/F5_inventory.md from results/f5_inventory.json only (no data loaded, nothing simulated).

    python -m forecast.run_f5 --render
"""
from __future__ import annotations

ORDER_CLASSES = ["smooth", "erratic", "intermittent", "lumpy"]


def money(v):
    return "n/a" if v is None else f"${v:,.0f}"


def pct(v, d=1):
    return "n/a" if v is None else f"{100 * v:.{d}f}%"


def signed(v, d=1):
    return "n/a" if v is None else f"{v:+.{d}f}%"


def less_more(v, d=1):
    """-10.4 -> '10.4% less'; None -> 'n/a'."""
    return "n/a" if v is None else f"{abs(v):.{d}f}% {'less' if v < 0 else 'more'}"


def _ef(rows, f):
    return next(e for e in rows if abs(e["fill_rate"] - f) < 1e-9)


def values(r: dict) -> dict:
    pols, lab = r["policies"], r["labels"]
    c = r["central"]
    h, rb = r["holdout"], r["robustness"]
    hs, rs = h["setup"], rb["setup"]
    mid = r["compare_fill_rates"][len(r["compare_fill_rates"]) // 2]
    main = h["main"]["bias_adjusted"]
    ba = {p: main[p]["overall"] for p in pols}

    def main_rows(w, method):
        return "\n".join(
            f"| {lab[p]} | {pct(o['fill_rate'])} | {pct(o['cycle_service_level'])} | {pct(o['stockout_day_share'])} | "
            f"{o['avg_inv_units']:,.0f} | {money(o['avg_inv_money'])} | {money(o['holding_cost'])} | "
            f"{money(o['lost_cost'])} | **{money(o['total_cost'])}** |"
            for p in pols for o in [r[w]["main"][method][p]["overall"]])

    def ef_rows(w, method):
        out = []
        for e in r[w]["equal_fill"][method]:
            need = e["inventory_money"]
            out.append(f"| {pct(e['fill_rate'], 0)} | " + " | ".join(money(need[p]) for p in pols) +
                       f" | {signed(e.get('lgbm_vs_seasonal_naive_pct'))} | {signed(e.get('lgbm_vs_tsb_pct'))} |")
        return "\n".join(out)

    e_mid = _ef(h["equal_fill"]["bias_adjusted"], mid)
    er_mid = _ef(rb["equal_fill"]["bias_adjusted"], mid)
    fills = ", ".join(f"{lab[p]} {pct(ba[p]['fill_rate'])}" for p in pols)
    csls = ", ".join(f"{lab[p]} {pct(ba[p]['cycle_service_level'])}" for p in pols)
    spread_fill = max(ba[p]["fill_rate"] for p in pols) - min(ba[p]["fill_rate"] for p in pols)
    fill_word = "almost the same" if spread_fill < 0.01 else "different"
    inv_gap_sn = 100 * (ba["lgbm"]["avg_inv_money"] / ba["seasonal_naive"]["avg_inv_money"] - 1)
    inv_gap_tsb = 100 * (ba["lgbm"]["avg_inv_money"] / ba["tsb"]["avg_inv_money"] - 1)
    headline = (
        f"At the central scenario ({c['sl']:.0%} target, lead time {c['lead']} days, a review every "
        f"{c['review']} days) the three policies reach {fill_word} fill rate on the holdout "
        f"({fills}), "
        f"and the model-based policy holds {less_more(inv_gap_sn)} average inventory than the seasonal-naive "
        f"rule and {less_more(inv_gap_tsb)} than TSB. Read off the full curves at the same {mid:.0%} fill rate, "
        f"it needs {money(e_mid['inventory_money']['lgbm'])} of stock against "
        f"{money(e_mid['inventory_money']['seasonal_naive'])} (seasonal naive, "
        f"{signed(e_mid.get('lgbm_vs_seasonal_naive_pct'))}) and {money(e_mid['inventory_money']['tsb'])} (TSB, "
        f"{signed(e_mid.get('lgbm_vs_tsb_pct'))}).")

    # sensitivity
    checks = r["sensitivity_checks"]
    n_sc = len(checks)
    lgbm_first = sum(ch["ranking_by_total_cost"][0] == "lgbm" for ch in checks)
    gaps_tsb = [_ef(ch["equal_fill"], mid).get("lgbm_vs_tsb_pct") for ch in checks]
    gaps_sn = [_ef(ch["equal_fill"], mid).get("lgbm_vs_seasonal_naive_pct") for ch in checks]
    gaps_tsb = [g for g in gaps_tsb if g is not None]
    gaps_sn = [g for g in gaps_sn if g is not None]
    sens = {(x["assumption"], str(x["value"]), x["policy"]): x for x in r["sensitivity"]}

    def sens_row(ch):
        e = _ef(ch["equal_fill"], mid)
        tc = " / ".join(money(sens[(ch["assumption"], str(ch["value"]), p)]["total_cost"]) for p in pols)
        fr = " / ".join(pct(sens[(ch["assumption"], str(ch["value"]), p)]["fill_rate"]) for p in pols)
        mark = " (central)" if ch["is_central"] else ""
        return (f"| {ch['assumption']} | {str(ch['value']).replace('_', ' ')}{mark} | {ch['reviews']} / "
                f"{ch['measured_days']} | {fr} | {tc} | {' > '.join(lab[p] for p in ch['ranking_by_total_cost'])} | "
                f"{signed(e.get('lgbm_vs_seasonal_naive_pct'))} | {signed(e.get('lgbm_vs_tsb_pct'))} |")
    flips = [ch for ch in checks if ch["ranking_by_total_cost"][0] != "lgbm"]
    flip_text = ("The model-based policy has the lowest total cost in every scenario." if not flips else
                 f"The model-based policy has the lowest simulated total cost in {lgbm_first} of {n_sc} scenarios. "
                 "It is not first in: " + "; ".join(
                     f"{ch['assumption']} = {ch['value']} ({lab[ch['ranking_by_total_cost'][0]]} first)" for ch in flips) +
                 ". " + ("In each of these the cheaper policy reaches a higher fill rate than the model at the same "
                 "nominal target (" + "; ".join(
                     f"{lab[ch['ranking_by_total_cost'][0]]} "
                     f"{pct(sens[(ch['assumption'], str(ch['value']), ch['ranking_by_total_cost'][0])]['fill_rate'])} vs "
                     f"LightGBM {pct(sens[(ch['assumption'], str(ch['value']), 'lgbm')]['fill_rate'])}" for ch in flips) +
                 "), because its wider backtest errors give it a bigger safety stock, and the lost sales it avoids are "
                 "worth more than the extra stock costs. A fixed target therefore does not compare like with like; the "
                 "comparison at equal fill rate below does." if all(
                     sens[(ch['assumption'], str(ch['value']), ch['ranking_by_total_cost'][0])]['fill_rate'] >
                     sens[(ch['assumption'], str(ch['value']), 'lgbm')]['fill_rate'] for ch in flips) else
                 "Not all of these come with a higher fill rate for the winner; see the table."))
    survive = (f"At an equal {mid:.0%} fill rate the model needs less inventory than both baselines in every one of "
               f"the {len(gaps_tsb)} scenarios: {min(gaps_tsb):+.1f}% to {max(gaps_tsb):+.1f}% against TSB and "
               f"{min(gaps_sn):+.1f}% to {max(gaps_sn):+.1f}% against seasonal naive."
               if gaps_tsb and max(gaps_tsb) < 0 and max(gaps_sn) < 0 else
               f"At an equal {mid:.0%} fill rate the model does not need less inventory in every scenario "
               f"(against TSB {min(gaps_tsb):+.1f}% to {max(gaps_tsb):+.1f}%).")

    # by group
    def class_rows(w):
        m = r[w]["main"]["bias_adjusted"]
        out = []
        for g in ORDER_CLASSES:
            if g not in m[pols[0]]["by_class"]:
                continue
            row = [m[p]["by_class"][g] for p in pols]
            out.append(f"| {g} | {row[0]['n_series']:,} | " + " | ".join(pct(x["fill_rate"]) for x in row) + " | " +
                       " | ".join(money(x["avg_inv_money"]) for x in row) + " |")
        return "\n".join(out)

    def group_rows(level, names):
        out = []
        for g in names:
            row = [main[p][level][g] for p in pols]
            out.append(f"| {g} | {row[0]['n_series']:,} | " + " | ".join(pct(x["fill_rate"]) for x in row) + " | " +
                       " | ".join(money(x["avg_inv_money"]) for x in row) + " | " +
                       " | ".join(money(x["total_cost"]) for x in row) + " |")
        return "\n".join(out)

    # money
    mo = h["money"]
    conc_rows = "\n".join(
        f"| {lab[p]} | {money(mo[p]['lost_cost'])} | {pct(mo[p]['top_share_of_lost_cost'], 0)} | "
        f"{mo[p]['series_with_any_lost_sale']:,} | {money(mo[p]['inventory_money'])} | "
        f"{pct(mo[p]['top_share_of_inventory_money'], 0)} | {pct(mo[p]['top_share_of_demand'], 0)} |" for p in pols)
    mclass_rows = "\n".join(
        f"| {g} | {pct(mo[pols[0]]['by_class'][g]['demand_share'], 0)} | " +
        " | ".join(pct(mo[p]["by_class"][g]["lost_cost_share"], 0) for p in pols) + " | " +
        " | ".join(pct(mo[p]["by_class"][g]["inventory_share"], 0) for p in pols) + " |"
        for g in ORDER_CLASSES if g in mo[pols[0]]["by_class"])
    mrev_rows = "\n".join(
        f"| {'top ' + pct(r['top_revenue_share'], 0) + ' by revenue' if g == 'top' else 'other items'} | "
        f"{pct(mo[pols[0]]['by_revenue'][g]['demand_share'], 0)} | " +
        " | ".join(pct(mo[p]["by_revenue"][g]["lost_cost_share"], 0) for p in pols) + " | " +
        " | ".join(pct(mo[p]["by_revenue"][g]["inventory_share"], 0) for p in pols) + " |" for g in ("top", "rest"))
    lgbm_cls = mo["lgbm"]["by_class"]
    worst = max(lgbm_cls, key=lambda g: lgbm_cls[g]["lost_cost_share"] - lgbm_cls[g]["demand_share"])
    money_text = (
        f"Lost sales are very concentrated: under the model-based policy the top {pct(r['concentration_share'], 0)} of "
        f"item-stores carry {pct(mo['lgbm']['top_share_of_lost_cost'], 0)} of the lost-sales cost (only "
        f"{mo['lgbm']['series_with_any_lost_sale']:,} of {hs['n_simulated']:,} item-stores lose any sale), while "
        f"inventory money is spread much like demand (top {pct(r['concentration_share'], 0)}: "
        f"{pct(mo['lgbm']['top_share_of_inventory_money'], 0)} of inventory, "
        f"{pct(mo['lgbm']['top_share_of_demand'], 0)} of demand). By class, {worst} items have the largest lost-sales "
        f"share relative to their demand ({pct(lgbm_cls[worst]['lost_cost_share'], 0)} of lost-sales cost vs "
        f"{pct(lgbm_cls[worst]['demand_share'], 0)} of demand). The top revenue items lose comparatively little "
        f"({pct(mo['lgbm']['by_revenue']['top']['lost_cost_share'], 0)} of lost-sales cost for "
        f"{pct(mo['lgbm']['by_revenue']['top']['demand_share'], 0)} of demand units).")

    # bias handling
    zs = {p: h["main"]["z_sd_only"][p]["overall"] for p in pols}
    zr = {p: rb["main"]["z_sd_only"][p]["overall"] for p in pols}
    br = {p: rb["main"]["bias_adjusted"][p]["overall"] for p in pols}
    ez = _ef(h["equal_fill"]["z_sd_only"], mid)["inventory_money"]
    ezr = _ef(rb["equal_fill"]["z_sd_only"], mid)["inventory_money"]
    ebr = er_mid["inventory_money"]
    pure_better = all(ez[p] is not None and e_mid["inventory_money"][p] is not None and
                      ez[p] < e_mid["inventory_money"][p] for p in pols)
    pure_better_r = all(ezr[p] is not None and ebr[p] is not None and ezr[p] < ebr[p] for p in pols)
    bias_rows = "\n".join(
        f"| {lab[p]} | {pct(ba[p]['fill_rate'])} / {pct(zs[p]['fill_rate'])} | "
        f"{pct(ba[p]['cycle_service_level'])} / {pct(zs[p]['cycle_service_level'])} | "
        f"{money(e_mid['inventory_money'][p])} / {money(ez[p])} | "
        f"{money(ebr[p])} / {money(ezr[p])} |" for p in pols)
    ss = h["safety_stock"]
    ss_rows = "\n".join(
        f"| {lab[p]} | {ss[p]['mean_window_error_total']:+,.0f} | {pct(ss[p]['share_series_under_forecast'], 0)} | "
        f"{ss[p]['sd_window_error_median']:.2f} | {ss[p]['safety_stock_units']:,.0f} | {money(ss[p]['safety_stock_money'])} |"
        for p in pols)
    bias_text = (
        (f"In both windows the simpler z x sd version reaches the {mid:.0%} fill rate with less stock than the "
         "main bias-adjusted version, for every policy. " if pure_better and pure_better_r else
         "The two versions do not order the same way in both windows. ") +
        "A likely reason: the per-series mean error from a few short backtest windows is mostly noise, so "
        "a series that happened to be under-forecast in the backtest gets extra stock it does not need later, "
        "and one that was over-forecast gets cut (TO VERIFY). The main method was fixed before the simulation and stays the headline; the comparison is "
        "reported, not swapped in (see NOTES: a pooled or shrunk bias term could be tested on the backtest windows).")
    csl_gap = [100 * (ba[p]["cycle_service_level"] - c["sl"]) for p in pols]
    csl_text = (f"The achieved cycle service level is below the {c['sl']:.0%} target for every policy "
                f"({csls}; "
                f"{min(csl_gap):+.1f} to {max(csl_gap):+.1f} points). A likely reason (TO VERIFY in NOTES): the "
                "normal z assumes bell-shaped errors, while summed errors of mostly intermittent items are skewed, "
                "so the normal quantile under-covers the right tail; the spread also comes from only a few backtest "
                "windows. The fill rate counts units, so a cycle that misses a few of many units lowers it only a "
                "little; that is why it is higher than the cycle service level."
                if max(csl_gap) < 0 else
                f"The achieved cycle service level ranges {min(csl_gap):+.1f} to {max(csl_gap):+.1f} points from the target.")

    # robustness
    rob_rank = rb["ranking_by_total_cost"]
    rob_rows = "\n".join(
        f"| {lab[p]} | {pct(ba[p]['fill_rate'])} / {pct(br[p]['fill_rate'])} | "
        f"{money(ba[p]['avg_inv_money'])} / {money(br[p]['avg_inv_money'])} | "
        f"{money(ba[p]['total_cost'])} / {money(br[p]['total_cost'])} | "
        f"{money(e_mid['inventory_money'][p])} / {money(ebr[p])} |" for p in pols)
    same_rank = rob_rank == h["ranking_by_total_cost"]
    rob_text = (
        f"On the pre-holdout window the ranking by total cost is {' > '.join(lab[p] for p in rob_rank)}"
        f"{', the same as on the holdout' if same_rank else ', different from the holdout'}. At {mid:.0%} fill rate "
        f"the model needs {less_more(er_mid.get('lgbm_vs_seasonal_naive_pct'))} inventory than seasonal naive and "
        f"{less_more(er_mid.get('lgbm_vs_tsb_pct'))} than TSB (holdout: "
        f"{less_more(e_mid.get('lgbm_vs_seasonal_naive_pct'))} and {less_more(e_mid.get('lgbm_vs_tsb_pct'))}). "
        + ("So the conclusion does not rest on the one holdout window."
           if (er_mid.get("lgbm_vs_tsb_pct") or 0) < 0 and (er_mid.get("lgbm_vs_seasonal_naive_pct") or 0) < 0
           else "The conclusion is not the same on this window."))

    a = r["assumptions"]

    def rng(k):
        return ", ".join(str(v).replace("_", " ") for v in a[k]["scenarios"])

    return {
        "headline": headline,
        "n_series": f"{hs['n_series']:,}", "n_sim": f"{hs['n_simulated']:,}", "n_ex": hs["n_excluded_no_backtest"],
        "origin": hs["forecast_origin"], "fd0": hs["forecast_days"][0], "fd1": hs["forecast_days"][1],
        "spread_origins": ", ".join(f"d_{o}" for o in hs["spread_origins"]),
        "rob_spread_origins": ", ".join(f"d_{o}" for o in rs["spread_origins"]),
        "rob_origin": rs["forecast_origin"], "rfd0": rs["forecast_days"][0], "rfd1": rs["forecast_days"][1],
        "rob_n_sim": f"{rs['n_simulated']:,}", "rob_n_ex": rs["n_excluded_no_backtest"],
        "lead": c["lead"], "review": c["review"], "P": hs["protection_days"], "sl": f"{c['sl']:.0%}",
        "reviews_d": ", ".join(f"d_{d}" for d in hs["reviews_d"]),
        "m0": hs["measured_d"][0], "m1": hs["measured_d"][1], "n_measured": hs["measured_days"],
        "demand_measured": f"{hs['demand_measured_units']:,.0f}",
        "hold": f"{c['hold']:.0%}", "short": f"{c['short']:.0%}", "start": c["start_factor"],
        "unmet": c["unmet"].replace("_", " "),
        "lead_rng": rng("lead_time_days"), "review_rng": rng("review_period_days"),
        "start_rng": rng("starting_inventory_factor"), "unmet_rng": rng("unmet_demand"),
        "hold_rng": rng("holding_cost_rate_per_year"), "short_rng": rng("stockout_cost_share_of_price"),
        "sl_rng": ", ".join(f"{v:.1%}".replace(".0%", "%") for v in r["service_level_sweep"]),
        "compare_fill": ", ".join(f"{f:.0%}" for f in r["compare_fill_rates"]), "mid": f"{mid:.0%}",
        "main_rows": main_rows("holdout", "bias_adjusted"), "csl_text": csl_text,
        "ef_rows": ef_rows("holdout", "bias_adjusted"), "ef_rows_rob": ef_rows("robustness", "bias_adjusted"),
        "ef_rows_z": ef_rows("holdout", "z_sd_only"),
        "rank": " > ".join(lab[p] for p in h["ranking_by_total_cost"]),
        "class_rows": class_rows("holdout"),
        "dept_rows": group_rows("by_dept", sorted(main[pols[0]]["by_dept"])),
        "rev_rows": group_rows("by_revenue", ["top", "rest"]).replace("| top |", f"| top {pct(r['top_revenue_share'], 0)} by revenue |").replace("| rest |", "| other items |"),
        "n_sc": n_sc, "sens_rows": "\n".join(sens_row(ch) for ch in checks), "flip_text": flip_text, "survive": survive,
        "conc_rows": conc_rows, "mclass_rows": mclass_rows, "mrev_rows": mrev_rows, "money_text": money_text,
        "conc": pct(r["concentration_share"], 0),
        "bias_rows": bias_rows, "bias_text": bias_text, "ss_rows": ss_rows,
        "rob_rows": rob_rows, "rob_text": rob_text,
        "pol_cols": " | ".join(lab[p] for p in pols), "pol_sep": " | ".join("---" for _ in pols),
    }
