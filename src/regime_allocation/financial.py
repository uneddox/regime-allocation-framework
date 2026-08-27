from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from .features import (
    FINANCIAL_REPORT_COLUMNS,
    FinancialBasketSpec,
    prepare_financial_features,
)
from .hmm import HMMResult, fit_regime_multistart, forecast_probabilities

FINANCIAL_LABELS = [
    "Stress / USD up / Credit stress",
    "Selective risk-on / USD firm",
    "Mixed transition / Commodity weak",
    "Risk on",
    "Hot reflation / Commodity surge",
]


def relabel_financial(result: HMMResult, feature_df: pd.DataFrame) -> HMMResult:
    rows = []
    for state in range(result.n_states):
        weights = result.gamma_smoothed[:, state]
        rows.append(
            {
                "state": state,
                "equity": float(np.average(feature_df["eq_global_yoy"], weights=weights)),
                "commodity": float(np.average(feature_df["commodity_yoy"], weights=weights)),
                "usd": float(np.average(feature_df["usd_broad_yoy"], weights=weights)),
            }
        )
    order = (
        pd.DataFrame(rows).sort_values(["equity", "commodity", "usd", "state"])["state"].astype(int).tolist()
    )
    fitted = None
    if result.fitted_params:
        fitted = {
            "startprob": result.fitted_params["startprob"][order],
            "transmat": result.fitted_params["transmat"][np.ix_(order, order)],
            "means": result.fitted_params["means"][order],
            "covars": result.fitted_params["covars"][order],
        }
    return replace(
        result,
        startprob=result.startprob[order],
        transmat=result.transmat[np.ix_(order, order)],
        means=result.means[order],
        covars=result.covars[order],
        gamma_smoothed=result.gamma_smoothed[:, order],
        gamma_filtered=result.gamma_filtered[:, order],
        state_order=order,
        fitted_params=fitted,
    )


def fit_financial(
    prices: pd.DataFrame,
    config: dict[str, Any],
    init_params: dict | None = None,
    basket_spec: FinancialBasketSpec | None = None,
) -> tuple[HMMResult, pd.DataFrame, pd.DataFrame]:
    features = prepare_financial_features(prices, basket_spec)
    result, model_df = fit_financial_feature_frame(features, config, init_params)
    return result, model_df, features


def fit_financial_feature_frame(
    features: pd.DataFrame,
    config: dict[str, Any],
    init_params: dict | None = None,
) -> tuple[HMMResult, pd.DataFrame]:
    eligible = features.loc[features["feature_ready"]].copy()
    spec = config["financial"]
    result, model_df = fit_regime_multistart(
        eligible,
        list(spec["features"]),
        relabel_financial,
        k_min=int(spec["states"]),
        k_max=int(spec["states"]),
        n_starts=int(spec["starts"]),
        min_history=int(spec["min_history"]),
        sticky_strength=float(spec["sticky_strength"]),
        ensemble_top_n=int(spec["ensemble_top_n"]),
        init_params=init_params,
        seed_base=int(spec["seed"]),
    )
    return result, model_df


def financial_summary(model_df: pd.DataFrame, result: HMMResult) -> pd.DataFrame:
    states = result.gamma_filtered.argmax(axis=1)
    rows = []
    for state in range(result.n_states):
        subset = model_df.loc[states == state]
        if subset.empty:
            continue
        row: dict[str, float | int] = {"state": state, "obs": len(subset)}
        row.update(
            {
                f"avg_{column}": float(subset[column].mean())
                for column in FINANCIAL_REPORT_COLUMNS
                if column in subset
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("state")


def combine_macro_financial(
    macro_targets: pd.DataFrame,
    result: HMMResult,
    model_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    spec = config["financial"]
    forecasts = forecast_probabilities(result, list(spec["forecast_horizons"]))
    latest = result.gamma_filtered[-1]
    state_by_horizon: dict[int, tuple[int, float]] = {0: (int(np.argmax(latest)), float(np.max(latest)))}
    state_by_horizon.update(
        {
            int(row.horizon_quarters): (int(row.predicted_state), float(row.pred_confidence))
            for row in forecasts.itertuples()
        }
    )
    benchmark = config["allocation"]["benchmark"]
    baseline_equity = float(benchmark["equity"])
    baseline_bond = float(benchmark["bond"])
    scalers = config["allocation"]["financial_conviction_by_state"]
    financial_date = model_df.index[-1].date().isoformat()
    rows = []
    for macro in macro_targets.itertuples():
        financial_state, financial_confidence = state_by_horizon[int(macro.horizon_quarters)]
        scaler = float(scalers[financial_state])
        equity_tilt = float(macro.equity_weight) - baseline_equity
        bond_tilt = float(macro.bond_weight) - baseline_bond
        rows.append(
            {
                "type": macro.type,
                "macro_signal_date": macro.base_date,
                "financial_signal_date": financial_date,
                "horizon_quarters": int(macro.horizon_quarters),
                "macro_state": int(macro.state),
                "macro_bucket": macro.allocation_bucket,
                "macro_confidence": float(macro.confidence),
                "macro_equity_weight": float(macro.equity_weight),
                "macro_bond_weight": float(macro.bond_weight),
                "financial_state": financial_state,
                "financial_label": FINANCIAL_LABELS[financial_state],
                "financial_confidence": financial_confidence,
                "financial_scaler": scaler,
                "final_equity_weight_pre_valuation": baseline_equity + scaler * equity_tilt,
                "final_bond_weight_pre_valuation": baseline_bond + scaler * bond_tilt,
                "active_equity_tilt_vs_65_35": scaler * equity_tilt,
                "active_bond_tilt_vs_65_35": scaler * bond_tilt,
            }
        )
    return pd.DataFrame(rows)
