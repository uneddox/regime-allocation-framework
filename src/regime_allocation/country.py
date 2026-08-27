from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _country_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "date" not in frame.columns:
        raise ValueError("country input must contain a date column")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    return result.dropna(subset=["date"]).sort_values("date").set_index("date").astype(float)


def _weighted_row_mean(frame: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    aligned = weights.reindex_like(frame)
    valid = frame.notna() & aligned.notna()
    numerator = frame.where(valid, 0.0).mul(aligned.where(valid, 0.0)).sum(axis=1)
    denominator = aligned.where(valid, 0.0).sum(axis=1)
    return numerator.div(denominator.replace(0.0, np.nan))


def run_country_factor(
    levels: pd.DataFrame,
    config: dict[str, Any],
    benchmark_weights: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prices = _country_frame(levels)
    returns = prices.pct_change(fill_method=None)
    if benchmark_weights is None:
        weights = pd.DataFrame(1.0 / prices.shape[1], index=prices.index, columns=prices.columns)
    else:
        weights = (
            _country_frame(benchmark_weights)
            .reindex(index=prices.index, columns=prices.columns)
            .ffill()
            .bfill()
        )
        weights = weights.div(weights.sum(axis=1), axis=0)
    global_return = _weighted_row_mean(returns, weights)
    effects = returns.sub(global_return, axis=0)
    window = int(config["country_factor"]["rolling_window"])
    rolling = effects.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(252.0)
    total_volatility = _weighted_row_mean(rolling, weights).dropna()
    if total_volatility.empty:
        raise ValueError("not enough country observations for the configured rolling window")
    percentile = float(total_volatility.rank(pct=True).iloc[-1])
    low, high = [float(value) for value in config["country_factor"]["strength_percentiles"]]
    strength = "weak" if percentile < low else "medium" if percentile < high else "strong"
    active_budget = float(config["country_factor"]["active_budget"][strength])
    latest_effect = effects.iloc[-1].dropna()
    signal = latest_effect - latest_effect.mean()
    denominator = float(signal.abs().sum())
    signal = signal * 0.0 if denominator <= 0 else signal / denominator
    benchmark = weights.iloc[-1].reindex(signal.index)
    tilted = (benchmark + active_budget * signal).clip(lower=0.0)
    tilted = tilted / tilted.sum()
    table = pd.DataFrame(
        {
            "country": tilted.index,
            "benchmark_weight": benchmark.values,
            "signal": signal.values,
            "tilted_weight": tilted.values,
            "active_weight": (tilted - benchmark).values,
        }
    ).sort_values("tilted_weight", ascending=False)
    metadata = {
        "signal_date": prices.index[-1].date().isoformat(),
        "rolling_window": window,
        "total_volatility": float(total_volatility.iloc[-1]),
        "strength_percentile": percentile,
        "strength": strength,
        "active_budget": active_budget,
    }
    return table, metadata


def apply_country_modifier(
    portfolio_targets: pd.DataFrame,
    country_table: pd.DataFrame,
    country_metadata: dict[str, Any],
) -> pd.DataFrame:
    """Map equity-sleeve country weights into total-portfolio weights for every horizon."""
    rows = []
    for target in portfolio_targets.itertuples():
        equity_weight = float(target.final_equity_weight)
        for country in country_table.itertuples():
            rows.append(
                {
                    "type": target.type,
                    "horizon_quarters": int(target.horizon_quarters),
                    "macro_signal_date": target.macro_signal_date,
                    "financial_signal_date": target.financial_signal_date,
                    "country_factor_signal_date": country_metadata["signal_date"],
                    "country_factor_strength": country_metadata["strength"],
                    "country_factor_strength_percentile": country_metadata["strength_percentile"],
                    "country_active_budget": country_metadata["active_budget"],
                    "country": country.country,
                    "equity_sleeve_benchmark_weight": float(country.benchmark_weight),
                    "equity_sleeve_tilted_weight": float(country.tilted_weight),
                    "equity_sleeve_active_weight": float(country.active_weight),
                    "portfolio_country_weight": equity_weight * float(country.tilted_weight),
                }
            )
    return pd.DataFrame(rows)
