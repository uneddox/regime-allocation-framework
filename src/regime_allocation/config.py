from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .features import FINANCIAL_FEATURE_COLUMNS, MACRO_FEATURE_COLUMNS

DEFAULT_CONFIG: dict[str, Any] = {
    "macro": {
        "states": 4,
        "features": MACRO_FEATURE_COLUMNS,
        "starts": 20,
        "min_history": 8,
        "sticky_strength": 8.0,
        "ensemble_top_n": 5,
        "seed": 42,
        "forecast_horizons": [1, 2, 4],
    },
    "financial": {
        "states": 5,
        "features": FINANCIAL_FEATURE_COLUMNS,
        "starts": 20,
        "min_history": 8,
        "sticky_strength": 8.0,
        "ensemble_top_n": 5,
        "seed": 42,
        "forecast_horizons": [1, 2, 4],
    },
    "allocation": {
        "benchmark": {"equity": 0.65, "bond": 0.35},
        "macro_equity_by_state": [0.75, 0.65, 0.55, 0.45],
        "financial_conviction_by_state": [0.50, 0.75, 0.50, 1.00, 0.75],
    },
    "bond_sleeve": {
        "rules": [
            {"name": "hedge_friendly", "max": 0.10, "bond_share": 1.00, "cash_share": 0.00},
            {"name": "hedge_weakened", "min": 0.10, "max": 0.20, "bond_share": 0.90, "cash_share": 0.10},
            {"name": "hedge_poor", "min": 0.20, "bond_share": 0.80, "cash_share": 0.20},
        ]
    },
    "country_factor": {
        "rolling_window": 252,
        "strength_percentiles": [0.33, 0.67],
        "active_budget": {"weak": 0.00, "medium": 0.05, "strong": 0.10},
    },
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    raw = {} if path is None else yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    config = _merge(DEFAULT_CONFIG, raw)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if int(config["macro"]["states"]) != 4:
        raise ValueError("production compatibility requires macro.states=4")
    if int(config["financial"]["states"]) != 5:
        raise ValueError("production compatibility requires financial.states=5")
    if list(config["macro"]["features"]) != MACRO_FEATURE_COLUMNS:
        raise ValueError("macro feature specification differs from the locked production model")
    if list(config["financial"]["features"]) != FINANCIAL_FEATURE_COLUMNS:
        raise ValueError("financial feature specification differs from the locked production model")
    allocation = config["allocation"]
    if len(allocation["macro_equity_by_state"]) != 4:
        raise ValueError("macro allocation requires four state weights")
    if len(allocation["financial_conviction_by_state"]) != 5:
        raise ValueError("financial allocation requires five state scalers")
    benchmark = allocation["benchmark"]
    if abs(float(benchmark["equity"]) + float(benchmark["bond"]) - 1.0) > 1e-12:
        raise ValueError("benchmark weights must sum to one")
    for rule in config["bond_sleeve"]["rules"]:
        if abs(float(rule["bond_share"]) + float(rule["cash_share"]) - 1.0) > 1e-12:
            raise ValueError(f"invalid bond sleeve shares: {rule['name']}")
