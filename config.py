from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def load_config(path=None):
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
