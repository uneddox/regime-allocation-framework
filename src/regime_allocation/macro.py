from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from .features import prepare_macro_features
from .hmm import HMMResult, fit_regime_multistart, forecast_probabilities

MACRO_BUCKETS = [
    "Risk-on growth",
    "Neutral / transition",
    "Disinflation slowdown",
    "Inflation stress / deep defense",
]


def relabel_macro(result: HMMResult, feature_df: pd.DataFrame) -> HMMResult:
    rows = []
    for state in range(result.n_states):
        weights = result.gamma_smoothed[:, state]
        if weights.sum() <= 1e-8:
            inflation = float(feature_df["inflation_yoy"].mean())
            growth = float(feature_df["growth_yoy"].mean())
        else:
            inflation = float(np.average(feature_df["inflation_yoy"], weights=weights))
            growth = float(np.average(feature_df["growth_yoy"], weights=weights))
        rows.append({"state": state, "inflation": inflation, "growth": growth})
    order = pd.DataFrame(rows).sort_values(["inflation", "growth", "state"])["state"].astype(int).tolist()
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


def fit_macro(
    monthly: pd.DataFrame,
    config: dict[str, Any],
    init_params: dict | None = None,
) -> tuple[HMMResult, pd.DataFrame, pd.DataFrame]:
    features = prepare_macro_features(monthly)
    eligible = features.loc[features["quarter_complete"] & features["feature_ready"]].copy()
    spec = config["macro"]
    result, model_df = fit_regime_multistart(
        eligible,
        list(spec["features"]),
        relabel_macro,
        k_min=int(spec["states"]),
        k_max=int(spec["states"]),
        n_starts=int(spec["starts"]),
        min_history=int(spec["min_history"]),
        sticky_strength=float(spec["sticky_strength"]),
        ensemble_top_n=int(spec["ensemble_top_n"]),
        init_params=init_params,
        seed_base=int(spec["seed"]),
    )
    return result, model_df, features


def macro_summary(model_df: pd.DataFrame, result: HMMResult) -> pd.DataFrame:
    states = result.gamma_filtered.argmax(axis=1)
    rows = []
    for state in range(result.n_states):
        subset = model_df.loc[states == state]
        if subset.empty:
            continue
        rows.append(
            {
                "state": state,
                "obs": len(subset),
                "avg_inflation_yoy": float(subset["inflation_yoy"].mean()),
                "avg_inflation_gap_regime": float(subset["inflation_gap_regime"].mean()),
                "avg_inflation_momentum": float(subset["inflation_momentum"].mean()),
                "avg_growth_yoy": float(subset["growth_yoy"].mean()),
                "avg_growth_gap_regime": float(subset["growth_gap_regime"].mean()),
                "avg_growth_qoq_annualized": float(subset["growth_qoq_annualized"].mean()),
                "avg_unrate_level": float(subset["unrate_level"].mean()),
                "avg_unrate_change": float(subset["unrate_change"].mean()),
                "avg_payems_qoq": float(subset["payems_qoq"].mean()),
                "avg_gap": float(subset["growth_infl_gap"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("state")


def macro_targets(result: HMMResult, model_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    spec = config["macro"]
    forecasts = forecast_probabilities(result, list(spec["forecast_horizons"]))
    base_date = model_df.index[-1].date().isoformat()
    weights = config["allocation"]["macro_equity_by_state"]
    rows = []
    h1 = forecasts.loc[forecasts["horizon_quarters"] == 1].iloc[0]
    h1_state = int(h1["predicted_state"])
    rows.append(
        {
            "type": "current_operating_h1",
            "base_date": base_date,
            "horizon_quarters": 0,
            "state": h1_state,
            "allocation_bucket": MACRO_BUCKETS[h1_state],
            "equity_weight": float(weights[h1_state]),
            "bond_weight": 1.0 - float(weights[h1_state]),
            "confidence": float(h1["pred_confidence"]),
        }
    )
    for row in forecasts.itertuples():
        state = int(row.predicted_state)
        rows.append(
            {
                "type": "forecast",
                "base_date": base_date,
                "horizon_quarters": int(row.horizon_quarters),
                "state": state,
                "allocation_bucket": MACRO_BUCKETS[state],
                "equity_weight": float(weights[state]),
                "bond_weight": 1.0 - float(weights[state]),
                "confidence": float(row.pred_confidence),
            }
        )
    return pd.DataFrame(rows)
