import importlib
import os
import pathlib

import pytest

import openfl.utils.require_env as require_env


def reset_env_state():
    require_env.env_loaded = False


def test_require_env_var_returns_value(monkeypatch):
    reset_env_state()
    monkeypatch.setenv("TEST_VAR", "present")
    monkeypatch.setattr(require_env, "load_env", lambda: None)

    assert require_env.require_env_var("TEST_VAR") == "present"


def test_require_env_var_exits_when_missing(monkeypatch):
    reset_env_state()
    monkeypatch.delenv("MISSING_VAR", raising=False)
    monkeypatch.setattr(require_env, "load_env", lambda: None)

    with pytest.raises(SystemExit) as excinfo:
        require_env.require_env_var("MISSING_VAR")

    assert excinfo.value.code == 1


def test_load_env_uses_expected_path(monkeypatch, tmp_path):
    paths = {}

    def fake_load_dotenv(path):
        paths["path"] = path

    monkeypatch.setattr(require_env, "load_dotenv", fake_load_dotenv)
    monkeypatch.setenv("ENV", "custom")

    # Point __file__ to a temporary location to control parents[3]
    custom_file = tmp_path / "a" / "b" / "c" / "require_env.py"
    monkeypatch.setattr(require_env, "__file__", str(custom_file))

    require_env.load_env()

    expected = custom_file.parents[3] / ".env" / ".env.custom"
    assert paths["path"] == expected
