"""Read the YAML files in config/. Every assumption and scope choice lives there, not in code."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
SQL_DIR = ROOT / "sql"


def load(name: str, config_dir: Path = CONFIG_DIR) -> dict:
    """Load config/<name>.yaml as a dict."""
    path = Path(config_dir) / f"{name}.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def raw_paths(data_cfg: dict, root: Path = ROOT) -> dict[str, Path]:
    """Absolute paths of the three raw files, keyed sales / calendar / prices."""
    raw_dir = Path(root) / data_cfg["raw_dir"]
    return {key: raw_dir / fname for key, fname in data_cfg["files"].items()}
