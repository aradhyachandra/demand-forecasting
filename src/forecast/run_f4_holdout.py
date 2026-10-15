"""F4 part B: the one-shot holdout run. DO NOT run until the frozen part A is committed and approved.

    python -m forecast.run_f4_holdout

Before reading any data it refuses to run if the git working tree is not clean (so the code and the
frozen config/model_final.yaml are exactly a commit) or if results/f4_holdout.json already exists
(the holdout is scored once). It records the commit hash, checks that config/model.yaml is the one
frozen in model_final.yaml, trains the chosen setting on everything up to the last training day
(the recipe in model_final.yaml), forecasts the holdout days and scores the model and every F3
baseline from the same origin with the F3 metrics and weights.

Writes results/f4_holdout.json and, for F5, data/processed/f4_holdout_forecasts.parquet.
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from forecast import backtest as bt
from forecast import config, run_f3, run_f4
from forecast import model

ROOT = config.ROOT


class HoldoutRunError(RuntimeError):
    """The holdout run refused to start."""


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                              text=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise HoldoutRunError(f"git failed in {repo}: {e}") from e


def preflight(repo: Path, out: Path) -> str:
    """Refuse a dirty tree (modified, staged or untracked files; ignored files do not count) or an
    existing result. Returns the commit hash the run is made from."""
    dirty = _git(repo, "status", "--porcelain").strip()
    if dirty:
        raise HoldoutRunError("the git working tree is not clean; commit the frozen setup first:\n" + dirty)
    if Path(out).exists():
        raise HoldoutRunError(f"{out} already exists: the holdout has been scored once and is not re-run")
    return _git(repo, "rev-parse", "HEAD").strip()


def main(root: Path = ROOT) -> dict:
    root = Path(root)
    out = root / "results" / "f4_holdout.json"
    commit = preflight(root, out)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.time()

    cdir = root / "config"
    data_cfg, bt_cfg = config.load("data", cdir), config.load("backtest", cdir)
    cfg, final = config.load("model", cdir), config.load("model_final", cdir)
    cls_cfg = config.load("demand_patterns", cdir)["classification"]
    holdout_days = config.load("holdout", cdir)["holdout_days"]
    state = config.load("scope", cdir)["state"]
    if run_f4.sha256(cdir / "model.yaml") != final["model_yaml_sha256"]:
        raise HoldoutRunError("config/model.yaml changed after the freeze (sha256 differs from model_final.yaml)")
    backtest = json.loads((root / "results" / "f4_backtest.json").read_text(encoding="utf-8"))
    best_base = backtest["best_baseline"]["name"]   # fixed on the backtest, not chosen on the holdout

    processed = root / data_cfg["processed_dir"]
    data = model.load_data(processed / data_cfg["slice_parquet"], holdout_days, cfg, allow_holdout=True,
                           calendar_csv=config.raw_paths(data_cfg, root)["calendar"], state=state,
                           cls_cfg=cls_cfg, memory_limit=data_cfg["duckdb"]["memory_limit"],
                           threads=data_cfg["duckdb"]["threads"])
    H, name = bt_cfg["horizon_days"], cfg["model_name"]
    origin = int(final["final_run"]["forecast_origin"])
    assert origin == data.first_holdout - 1, "the frozen forecast origin is not the last training day"
    assert final["final_run"]["forecast_days"] == [origin + 1, origin + H]

    # training: no holdout flag, so any training row touching the holdout raises
    X, y, meta = model.training_table(data, origin, cfg, bt_cfg)
    assert [int(meta["origin"].min()), int(meta["origin"].max())] == final["final_run"]["training_origins"]
    rows = int(len(X))
    params = model.lgb_params(cfg, final["params"])
    booster = model.fit(X, y, model.feature_names(cfg), params, int(final["rounds"]))
    del X, y, meta
    fc = model.forecast(data, booster, origin, cfg, bt_cfg, [int(final["rounds"])], allow_holdout=True)[int(final["rounds"])]

    captured, classes = {}, {}

    def extra(o, inp):
        captured[o] = {"eligible": inp["eligible"], "forecasts": inp["forecasts"]}
        classes[o] = inp["demand_class"]
        return {name: fc}

    one = {**bt_cfg, "n_origins": 1}   # the panel ends on the last holdout day, so the one origin is d_origin
    per_series, coverage, weekly = bt.evaluate(data.panel, one, cls_cfg, extra=extra)
    assert [c["origin"] for c in coverage] == [origin]
    order = bt_cfg["baselines"] + [name]
    summary = run_f3.summarize(per_series, {**bt_cfg, "baselines": order})
    pair = per_series[per_series["baseline"].isin([best_base, name])]
    weekly["error_pct"] = 100 * (weekly["forecast"] / weekly["actual"] - 1)

    long = run_f4.forecast_long(data.panel, captured, {origin: fc}, name, H, classes)
    long.to_parquet(processed / "f4_holdout_forecasts.parquet", index=False)

    report = {
        "commit": commit, "started_utc": started, "model_final": final,
        "setup": {"forecast_origin": origin, "holdout_days": [origin + 1, origin + H], "training_rows": rows,
                  "methods": order, "best_baseline_from_backtest": best_base},
        "coverage": coverage, **summary,
        "wins_vs_best_baseline": {"reference": best_base,
                                  "overall": bt.win_counts(pair, best_base, bt_cfg["tie_tolerance"])[name]},
        "weekly_totals": json.loads(weekly.to_json(orient="records")),
        "f5_forecasts": {"file": f"{data_cfg['processed_dir']}/f4_holdout_forecasts.parquet", "rows": int(len(long))},
        "run_minutes": round((time.time() - t0) / 60, 1),
    }
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    r = main()
    print(yaml.safe_dump({"commit": r["commit"], "overall_wrmsse": {b: m["wrmsse"] for b, m in r["overall"].items()}}))
