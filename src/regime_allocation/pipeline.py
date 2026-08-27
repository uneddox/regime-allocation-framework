from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .bond import apply_bond_sleeve
from .config import load_config
from .continuity import extend_locked_result, load_baseline_bundle, save_baseline_bundle
from .country import apply_country_modifier, run_country_factor
from .features import load_prepared_features, prepare_financial_features, prepare_macro_features
from .financial import (
    combine_macro_financial,
    financial_summary,
    fit_financial,
    fit_financial_feature_frame,
)
from .hmm import HMMResult, forecast_probabilities
from .macro import fit_macro, macro_summary, macro_targets


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _probability_frame(model_df: pd.DataFrame, result: HMMResult) -> pd.DataFrame:
    output = model_df.copy()
    for state in range(result.n_states):
        output[f"p_filtered_state_{state}"] = result.gamma_filtered[:, state]
        output[f"p_smoothed_state_{state}"] = result.gamma_smoothed[:, state]
    output["state_filtered"] = result.gamma_filtered.argmax(axis=1)
    output["state_smoothed"] = result.gamma_smoothed.argmax(axis=1)
    return output


def build_baseline_bundles(
    macro_path: str | Path,
    financial_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    financial_prepared: bool = False,
) -> dict[str, str]:
    config = load_config(config_path)
    output = Path(output_dir)
    macro_input = pd.read_csv(macro_path)
    financial_input = pd.read_csv(financial_path)
    macro_result, macro_model, macro_features = fit_macro(macro_input, config)
    if financial_prepared:
        financial_features = load_prepared_features(financial_input, list(config["financial"]["features"]))
        financial_result, financial_model = fit_financial_feature_frame(financial_features, config)
    else:
        financial_result, financial_model, financial_features = fit_financial(financial_input, config)
    macro_cutoff = macro_model.index[-1]
    financial_cutoff = financial_model.index[-1]
    save_baseline_bundle(
        output / "macro",
        macro_cutoff,
        list(config["macro"]["features"]),
        macro_features.loc[
            (macro_features.index <= macro_cutoff)
            & macro_features["quarter_complete"]
            & macro_features["feature_ready"]
        ],
        macro_model,
        macro_result,
    )
    save_baseline_bundle(
        output / "financial",
        financial_cutoff,
        list(config["financial"]["features"]),
        financial_features.loc[
            (financial_features.index <= financial_cutoff) & financial_features["feature_ready"]
        ],
        financial_model,
        financial_result,
    )
    return {"macro": str(output / "macro"), "financial": str(output / "financial")}


def _macro_run(
    monthly: pd.DataFrame, config: dict[str, Any], baseline_dir: str | Path | None
) -> tuple[HMMResult, pd.DataFrame, pd.DataFrame]:
    if baseline_dir is None:
        return fit_macro(monthly, config)
    features = prepare_macro_features(monthly)
    eligible = features.loc[features["quarter_complete"] & features["feature_ready"]].copy()
    result, model = extend_locked_result(load_baseline_bundle(baseline_dir), eligible)
    return result, model, features


def _financial_run(
    prices: pd.DataFrame,
    config: dict[str, Any],
    baseline_dir: str | Path | None,
    prepared: bool,
) -> tuple[HMMResult, pd.DataFrame, pd.DataFrame]:
    if prepared:
        features = load_prepared_features(prices, list(config["financial"]["features"]))
        if baseline_dir is None:
            result, model = fit_financial_feature_frame(features, config)
            return result, model, features
        eligible = features.loc[features["feature_ready"]].copy()
        result, model = extend_locked_result(load_baseline_bundle(baseline_dir), eligible)
        return result, model, features
    if baseline_dir is None:
        return fit_financial(prices, config)
    features = prepare_financial_features(prices)
    eligible = features.loc[features["feature_ready"]].copy()
    result, model = extend_locked_result(load_baseline_bundle(baseline_dir), eligible)
    return result, model, features


def run_pipeline(
    macro_path: str | Path,
    financial_path: str | Path,
    country_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    country_weights_path: str | Path | None = None,
    macro_baseline_dir: str | Path | None = None,
    financial_baseline_dir: str | Path | None = None,
    financial_prepared: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    macro_result, macro_model, _ = _macro_run(pd.read_csv(macro_path), config, macro_baseline_dir)
    financial_result, financial_model, financial_features = _financial_run(
        pd.read_csv(financial_path), config, financial_baseline_dir, financial_prepared
    )
    macro_state_summary = macro_summary(macro_model, macro_result)
    financial_state_summary = financial_summary(financial_model, financial_result)
    macro_allocation = macro_targets(macro_result, macro_model, config)
    combined = combine_macro_financial(macro_allocation, financial_result, financial_model, config)
    bond_targets = apply_bond_sleeve(combined, financial_state_summary, financial_features, config)
    benchmark_weights = pd.read_csv(country_weights_path) if country_weights_path else None
    country_table, country_metadata = run_country_factor(
        pd.read_csv(country_path), config, benchmark_weights=benchmark_weights
    )
    country_portfolio = apply_country_modifier(bond_targets, country_table, country_metadata)

    macro_probability = _probability_frame(macro_model, macro_result)
    financial_probability = _probability_frame(financial_model, financial_result)
    macro_forecast = forecast_probabilities(macro_result, list(config["macro"]["forecast_horizons"]))
    financial_forecast = forecast_probabilities(
        financial_result, list(config["financial"]["forecast_horizons"])
    )
    macro_forecast.insert(0, "base_date", macro_model.index[-1].date().isoformat())
    financial_forecast.insert(0, "base_date", financial_model.index[-1].date().isoformat())
    outputs = {
        "macro_regime_probabilities.csv": macro_probability,
        "macro_regime_summary.csv": macro_state_summary,
        "macro_regime_forecast.csv": macro_forecast,
        "macro_allocation.csv": macro_allocation,
        "financial_regime_probabilities.csv": financial_probability,
        "financial_regime_summary.csv": financial_state_summary,
        "financial_regime_forecast.csv": financial_forecast,
        "macro_financial_allocation.csv": combined,
        "bond_sleeve_allocation.csv": bond_targets,
        "country_factor_weights.csv": country_table,
        "country_modifier_portfolio_weights.csv": country_portfolio,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output / filename, index="probabilities" in filename, index_label="date")

    latest_macro = macro_result.gamma_filtered[-1]
    latest_financial = financial_result.gamma_filtered[-1]
    summary = {
        "framework_version": "0.2.0-production-parity",
        "model_spec": {
            "macro": config["macro"],
            "financial": config["financial"],
            "continuity": {
                "macro_locked": macro_baseline_dir is not None,
                "financial_locked": financial_baseline_dir is not None,
            },
        },
        "probability_policy": {
            "operations": "filtered",
            "ex_post_explanation_only": "smoothed",
            "forecast": "top-N ensemble mean pi_t and transition matrix",
        },
        "macro": {
            "signal_date": macro_model.index[-1].date().isoformat(),
            "state": int(np.argmax(latest_macro)),
            "confidence": float(np.max(latest_macro)),
        },
        "financial": {
            "signal_date": financial_model.index[-1].date().isoformat(),
            "state": int(np.argmax(latest_financial)),
            "confidence": float(np.max(latest_financial)),
        },
        "bond_sleeve": bond_targets.iloc[0][
            ["bond_sleeve_regime", "stock_bond_corr_4q_assumption", "corr_source"]
        ].to_dict(),
        "country_factor": country_metadata,
        "latest_allocation": bond_targets.iloc[0].to_dict(),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    return summary
