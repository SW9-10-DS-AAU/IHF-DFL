import yaml
from pathlib import Path
from types import SimpleNamespace

_config = None  # cache

def get_config():
    global _config
    if _config is None:
        path = Path(__file__).resolve()
        config_path = path.parent.parent.parent.parent / "config" / "config.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)
            _config = _to_namespace(data)
    return _config

def get_print_config():
    return get_config().printing

def get_contracts_config():
    return get_config().contracts

def _to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_to_namespace(i) for i in d]
    return d