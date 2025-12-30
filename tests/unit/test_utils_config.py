import builtins
from types import SimpleNamespace

import importlib

import openfl.utils.config as config


def test_get_config_caches_result(monkeypatch):
    # Reset cache
    config._config = None

    # Load config twice and ensure same object is reused
    first = config.get_config()
    second = config.get_config()

    assert first is second
    # Validate expected fields from sample config
    assert hasattr(first, "printing")
    assert hasattr(first, "contracts")


def test_to_namespace_handles_nested_structures():
    nested = {
        "outer": {"inner": 1},
        "list": [{"value": 2}, 3],
    }

    ns = config._to_namespace(nested)

    assert isinstance(ns.outer, SimpleNamespace) # Check that dictionaries are converted into SimpleNamespace
    assert ns.outer.inner == 1
    assert isinstance(ns.list[0], SimpleNamespace)
    assert ns.list[0].value == 2 # Check that nested dictionaries are also processed
    assert ns.list[1] == 3
