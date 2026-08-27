from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import FinancialBasketSpec


def generate_sample_data(output_dir: str | Path, seed: int = 42) -> dict[str, Path]:
    """Create synthetic full-schema data, including every production financial basket column."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    monthly_dates = pd.date_range("1995-01-31", periods=360, freq="ME")
    cycle = np.sin(np.arange(monthly_dates.size) / 18.0)
    inflation = 0.002 + 0.0015 * cycle + rng.normal(0.0, 0.0005, monthly_dates.size)
    production = (
        0.002
        + 0.003 * np.cos(np.arange(monthly_dates.size) / 22.0)
        + rng.normal(0.0, 0.004, monthly_dates.size)
    )
    macro = pd.DataFrame(
        {
            "sasdate": monthly_dates,
            "CPIAUCSL": 100.0 * np.cumprod(1.0 + inflation),
            "INDPRO": 90.0 * np.cumprod(1.0 + production),
            "UNRATE": 5.0 - 0.8 * cycle + rng.normal(0.0, 0.15, monthly_dates.size),
            "PAYEMS": 100000.0 * np.cumprod(1.0 + 0.0015 + 0.001 * cycle),
        }
    )

    quarter_dates = pd.date_range("1995-03-31", periods=120, freq="QE")
    quarter_cycle = np.sin(np.arange(quarter_dates.size) / 7.0)
    spec = FinancialBasketSpec()
    financial = pd.DataFrame({"date": quarter_dates})

    def level(drift: float, cycle_beta: float, volatility: float) -> np.ndarray:
        returns = drift + cycle_beta * quarter_cycle + rng.normal(0.0, volatility, quarter_dates.size)
        return 100.0 * np.cumprod(1.0 + returns)

    for column in spec.us_equity + spec.global_equity:
        financial[column] = level(0.025, 0.025, 0.025)
    for column in spec.safe_bonds:
        financial[column] = level(0.010, -0.006, 0.010)
    for column in spec.credit:
        financial[column] = level(0.014, 0.006, 0.013)
    for column in spec.commodity:
        financial[column] = level(0.012, 0.022, 0.030)
    financial[spec.dxy] = level(0.002, -0.004, 0.010)
    for column in [spec.eurusd, spec.jpyusd, spec.gbpusd, spec.chfusd]:
        financial[column] = level(0.001, 0.003, 0.008)

    business_dates = pd.bdate_range("2018-01-01", periods=1600)
    common = rng.normal(0.00025, 0.008, (business_dates.size, 1))
    idiosyncratic = rng.normal(0.0, 0.006, (business_dates.size, 6))
    drifts = np.asarray([0.00015, 0.00010, 0.00008, 0.00012, 0.00013, 0.00011])
    country = pd.DataFrame(
        100.0 * np.cumprod(1.0 + common + idiosyncratic + drifts, axis=0),
        columns=["US", "Eurozone", "UK", "Japan", "India", "China"],
    )
    country.insert(0, "date", business_dates)

    paths = {
        "macro": output / "macro_monthly.csv",
        "macro_baseline": output / "macro_monthly_baseline.csv",
        "financial": output / "financial_baskets.csv",
        "financial_baseline": output / "financial_baskets_baseline.csv",
        "country": output / "country_levels.csv",
    }
    macro.to_csv(paths["macro"], index=False)
    macro.iloc[:-24].to_csv(paths["macro_baseline"], index=False)
    financial.to_csv(paths["financial"], index=False)
    financial.iloc[:-8].to_csv(paths["financial_baseline"], index=False)
    country.to_csv(paths["country"], index=False)
    return paths
