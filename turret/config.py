"""YAML config loading with dotted-key CLI overrides and attribute access."""
from __future__ import annotations

import ast
import copy
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


class Cfg(dict):
    """dict with attribute access, recursively."""

    def __getattr__(self, key):
        try:
            v = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return v

    def __setattr__(self, key, value):
        self[key] = value

    def __deepcopy__(self, memo):
        return Cfg(copy.deepcopy(dict(self), memo))


def _wrap(obj):
    if isinstance(obj, dict):
        return Cfg({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def _parse_value(s: str):
    if s.lower() in ("null", "none"):
        return None
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Cfg:
    """Load default.yaml, merge a user file on top, then apply key.path=value overrides."""
    with open(DEFAULT_PATH) as f:
        data = yaml.safe_load(f)
    if path and Path(path).resolve() != DEFAULT_PATH:
        with open(path) as f:
            _merge(data, yaml.safe_load(f) or {})
    for item in overrides or []:
        key, _, val = item.partition("=")
        node = data
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(val.strip())
    return _wrap(data)
