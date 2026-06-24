import json
from pathlib import Path


DEFAULT_CONFIG_PATH = "config.json"


def load_config(path=DEFAULT_CONFIG_PATH):
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def project_path(*parts):
    return Path(__file__).resolve().parents[1].joinpath(*parts)
