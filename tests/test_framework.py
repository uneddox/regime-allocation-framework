from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml

from regime_allocation.bond import apply_bond_sleeve, classify_bond_sleeve
from regime_allocation.config import load_config
from regime_allocation.continuity import extend_locked_result, load_baseline_bundle
from regime_allocation.features import (
    FINANCIAL_FEATURE_COLUMNS,
    MACRO_FEATURE_COLUMNS,
    FinancialBasketSpec,
    prepare_financial_features,
    prepare_macro_features,
)
from regime_allocation.financial import financial_summary, fit_financial
from regime_allocation.hmm import forecast_probabilities
from regime_allocation.macro import fit_macro
from regime_allocation.pipeline import build_baseline_bundles, run_pipeline
from regime_allocation.sample import generate_sample_data


def quick_config(tmp_path):
    config = load_config()
    config["macro"]["starts"] = 3
    config["macro"]["ensemble_top_n"] = 2
    config["financial"]["starts"] = 3
    config["financial"]["ensemble_top_n"] = 2
    config["country_factor"]["rolling_window"] = 40
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config, path


def test_locked_model_specification() -> None:
    config = load_config()
    assert config["macro"]["states"] == 4
    assert config["macro"]["features"] == MACRO_FEATURE_COLUMNS
    assert config["financial"]["states"] == 5
    assert config["financial"]["features"] == FINANCIAL_FEATURE_COLUMNS
    assert config["macro"]["starts"] == config["financial"]["starts"] == 20
    assert config["macro"]["ensemble_top_n"] == config["financial"]["ensemble_top_n"] == 5
    assert config["allocation"]["benchmark"] == {"equity": 0.65, "bond": 0.35}
    assert config["allocation"]["macro_equity_by_state"] == [0.75, 0.65, 0.55, 0.45]
    assert config["allocation"]["financial_conviction_by_state"] == [
        0.50,
        0.75,
        0.50,
        1.00,
        0.75,
    ]


def test_exact_feature_schemas(tmp_path) -> None:
    paths = generate_sample_data(tmp_path / "data")
    macro = prepare_macro_features(pd.read_csv(paths["macro"]))
    financial = prepare_financial_features(pd.read_csv(paths["financial"]))
    assert macro[MACRO_FEATURE_COLUMNS].dropna().shape[0] > 24
    assert financial[FINANCIAL_FEATURE_COLUMNS].dropna().shape[0] > 24
    assert set(FinancialBasketSpec().required_columns()).issubset(
        pd.read_csv(paths["financial"], nrows=1).columns
    )


def test_multistart_ensemble_is_deterministic(tmp_path) -> None:
    paths = generate_sample_data(tmp_path / "data")
    config, _ = quick_config(tmp_path)
    first, _, _ = fit_macro(pd.read_csv(paths["macro_baseline"]), config)
    second, _, _ = fit_macro(pd.read_csv(paths["macro_baseline"]), config)
    np.testing.assert_allclose(first.transmat, second.transmat, atol=0, rtol=0)
    np.testing.assert_allclose(
        first.forecast_ensemble["pi_t_mean"], second.forecast_ensemble["pi_t_mean"], atol=0, rtol=0
    )
    assert first.forecast_ensemble["n_members"] == 2


def test_continuity_locks_parameters_and_history(tmp_path) -> None:
    paths = generate_sample_data(tmp_path / "data")
    _, config_path = quick_config(tmp_path)
    bundle_paths = build_baseline_bundles(
        paths["macro_baseline"], paths["financial_baseline"], tmp_path / "bundles", config_path
    )
    bundle = load_baseline_bundle(bundle_paths["macro"])
    full_features = prepare_macro_features(pd.read_csv(paths["macro"]))
    eligible = full_features.loc[full_features["quarter_complete"] & full_features["feature_ready"]]
    extended, model = extend_locked_result(bundle, eligible)
    np.testing.assert_allclose(extended.transmat, bundle.result.transmat, atol=0, rtol=0)
    np.testing.assert_allclose(extended.means, bundle.result.means, atol=0, rtol=0)
    np.testing.assert_allclose(
        extended.gamma_filtered[: len(bundle.result.gamma_filtered)],
        bundle.result.gamma_filtered,
        atol=0,
        rtol=0,
    )
    assert len(model) > len(bundle.model_df)
    assert not np.allclose(
        extended.forecast_ensemble["pi_t_mean"], bundle.result.forecast_ensemble["pi_t_mean"]
    )


def test_future_bond_sleeve_uses_state_average(tmp_path) -> None:
    paths = generate_sample_data(tmp_path / "data")
    config, _ = quick_config(tmp_path)
    result, model, features = fit_financial(pd.read_csv(paths["financial_baseline"]), config)
    summary = financial_summary(model, result)
    allocations = pd.DataFrame(
        [
            {
                "type": "current_operating_h1",
                "horizon_quarters": 0,
                "financial_state": 0,
                "final_equity_weight_pre_valuation": 0.6,
                "final_bond_weight_pre_valuation": 0.4,
            },
            {
                "type": "forecast",
                "horizon_quarters": 1,
                "financial_state": int(summary.iloc[0]["state"]),
                "final_equity_weight_pre_valuation": 0.6,
                "final_bond_weight_pre_valuation": 0.4,
            },
        ]
    )
    output = apply_bond_sleeve(allocations, summary, features, config)
    assert output.iloc[0]["corr_source"].startswith("latest realized")
    assert output.iloc[1]["corr_source"].startswith("state-average")
    assert np.allclose(output["final_total_weight_check"], 1.0)
    rules = config["bond_sleeve"]["rules"]
    assert classify_bond_sleeve(0.10, rules)["name"] == "hedge_friendly"


def test_end_to_end_continuity_pipeline(tmp_path) -> None:
    paths = generate_sample_data(tmp_path / "data")
    _, config_path = quick_config(tmp_path)
    bundles = build_baseline_bundles(
        paths["macro_baseline"], paths["financial_baseline"], tmp_path / "bundles", config_path
    )
    output = tmp_path / "output"
    summary = run_pipeline(
        paths["macro"],
        paths["financial"],
        paths["country"],
        output,
        config_path,
        macro_baseline_dir=bundles["macro"],
        financial_baseline_dir=bundles["financial"],
    )
    assert summary["framework_version"] == "0.2.0-production-parity"
    assert summary["model_spec"]["continuity"] == {
        "macro_locked": True,
        "financial_locked": True,
    }
    assert (output / "country_modifier_portfolio_weights.csv").exists()
    country = pd.read_csv(output / "country_modifier_portfolio_weights.csv")
    equity = pd.read_csv(output / "bond_sleeve_allocation.csv").set_index("horizon_quarters")
    country_sums = country.groupby("horizon_quarters")["portfolio_country_weight"].sum()
    np.testing.assert_allclose(
        country_sums.sort_index().to_numpy(),
        equity.loc[country_sums.sort_index().index, "final_equity_weight"].to_numpy(),
    )
    assert json.loads((output / "summary.json").read_text())["framework_version"].startswith("0.2")


def test_forecast_uses_ensemble(tmp_path) -> None:
    paths = generate_sample_data(tmp_path / "data")
    config, _ = quick_config(tmp_path)
    result, _, _ = fit_macro(pd.read_csv(paths["macro_baseline"]), config)
    forecast = forecast_probabilities(result, [1, 2, 4])
    expected = result.forecast_ensemble["pi_t_mean"] @ result.forecast_ensemble["transmat_mean"]
    np.testing.assert_allclose(
        forecast.loc[0, [f"p_state_{state}" for state in range(4)]].to_numpy(float), expected
    )
