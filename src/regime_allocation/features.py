from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

MACRO_FEATURE_COLUMNS = [
    "inflation_yoy",
    "growth_yoy",
    "inflation_gap_regime",
    "unrate_level",
]
FINANCIAL_FEATURE_COLUMNS = [
    "eq_global_yoy",
    "credit_excess_yoy",
    "commodity_yoy",
    "usd_broad_yoy",
    "bond_vol_4q",
    "cross_asset_dispersion_1q",
    "stock_bond_corr_4q",
]
MACRO_REPORT_COLUMNS = [
    "inflation_yoy",
    "growth_yoy",
    "inflation_gap_regime",
    "growth_gap_regime",
    "growth_qoq_annualized",
    "inflation_regime_base_5y",
    "growth_regime_base_5y",
    "inflation_momentum",
    "unrate_level",
    "unrate_change",
    "payems_qoq",
]
FINANCIAL_REPORT_COLUMNS = [
    "eq_us_yoy",
    "eq_global_yoy",
    "bond_safe_yoy",
    "credit_excess_yoy",
    "commodity_yoy",
    "usd_broad_yoy",
    "usd_major_yoy",
    "equity_vol_4q",
    "bond_vol_4q",
    "commodity_vol_4q",
    "fx_vol_4q",
    "stock_bond_gap_yoy",
    "commodity_bond_gap_yoy",
    "equity_drawdown_4q",
    "cross_asset_dispersion_1q",
    "stock_bond_corr_4q",
    "stock_bond_corr_8q",
]


@dataclass(frozen=True)
class FinancialBasketSpec:
    safe_bonds: tuple[str, ...] = (
        "LBUSTRUU Index",
        "LUATTRUU Index",
        "LD20TRUU Index",
        "LUTLTRUU Index",
        "LBUTTRUU Index",
        "LUMSTRUU Index",
        "LBEATREU Index",
    )
    credit: tuple[str, ...] = ("LF98TRUU Index", "LUACTRUU Index", "I00732US Index")
    us_equity: tuple[str, ...] = (
        "SPX INDEX",
        "CCMP INDEX",
        "M1US000V Index",
        "M1US000G Index",
    )
    global_equity: tuple[str, ...] = (
        "MXAU INDEX",
        "MXCA INDEX",
        "UKX INDEX",
        "MXEU INDEX",
        "MXJP INDEX",
        "MXEF INDEX",
        "MXBR INDEX",
        "MXCN INDEX",
        "MXIN INDEX",
        "KOSPI INDEX",
        "MXTW INDEX",
    )
    commodity: tuple[str, ...] = (
        "BCOMTR Index",
        "XAU Curncy",
        "XAG Curncy",
        "XPT Curncy",
        "XPD Curncy",
    )
    dxy: str = "DXY Index"
    eurusd: str = "EURUSD curncy"
    jpyusd: str = "JPYUSD curncy"
    gbpusd: str = "GBPUSD curncy"
    chfusd: str = "CHFUSD curncy"

    def required_columns(self) -> list[str]:
        return sorted(
            set(
                self.safe_bonds
                + self.credit
                + self.us_equity
                + self.global_equity
                + self.commodity
                + (self.dxy, self.eurusd, self.jpyusd, self.gbpusd, self.chfusd)
            )
        )


def normalize_macro_input(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "sasdate" not in data.columns and "date" in data.columns:
        data = data.rename(columns={"date": "sasdate"})
    aliases = {
        "cpi": "CPIAUCSL",
        "industrial_production": "INDPRO",
        "unemployment": "UNRATE",
        "payrolls": "PAYEMS",
    }
    data = data.rename(columns={key: value for key, value in aliases.items() if key in data.columns})
    if "sasdate" not in data.columns:
        raise ValueError("macro input requires date or sasdate")
    data = data.loc[data["sasdate"].astype(str) != "Transform:"].copy()
    data["sasdate"] = pd.to_datetime(data["sasdate"], errors="coerce")
    data = data.dropna(subset=["sasdate"]).sort_values("sasdate").set_index("sasdate")
    required = ["CPIAUCSL", "INDPRO", "UNRATE", "PAYEMS"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"missing macro columns: {missing}")
    data[required] = data[required].apply(pd.to_numeric, errors="coerce")
    return data


def prepare_macro_features(monthly: pd.DataFrame) -> pd.DataFrame:
    data = (
        normalize_macro_input(monthly) if not isinstance(monthly.index, pd.DatetimeIndex) else monthly.copy()
    )
    quarter_last = data.resample("QE").last()
    counts = data[["CPIAUCSL", "INDPRO", "UNRATE", "PAYEMS"]].resample("QE").count().min(axis=1)
    result = pd.DataFrame(index=quarter_last.index)
    result["month_count"] = counts
    result["quarter_complete"] = result["month_count"] >= 3
    result["inflation_yoy"] = (quarter_last["CPIAUCSL"] / quarter_last["CPIAUCSL"].shift(4) - 1.0) * 100.0
    result["inflation_momentum"] = result["inflation_yoy"] - result["inflation_yoy"].shift(1)
    result["growth_yoy"] = (quarter_last["INDPRO"] / quarter_last["INDPRO"].shift(4) - 1.0) * 100.0
    result["inflation_regime_base_5y"] = result["inflation_yoy"].rolling(20, min_periods=12).median()
    result["growth_regime_base_5y"] = result["growth_yoy"].rolling(20, min_periods=12).median()
    result["inflation_gap_regime"] = result["inflation_yoy"] - result["inflation_regime_base_5y"]
    result["growth_gap_regime"] = result["growth_yoy"] - result["growth_regime_base_5y"]
    result["unrate_level"] = data["UNRATE"].resample("QE").mean()
    result["unrate_change"] = result["unrate_level"].diff()
    result["growth_qoq_annualized"] = (
        (quarter_last["INDPRO"] / quarter_last["INDPRO"].shift(1)) ** 4 - 1.0
    ) * 100.0
    result["payems_qoq"] = (quarter_last["PAYEMS"] / quarter_last["PAYEMS"].shift(1) - 1.0) * 100.0
    result["growth_infl_gap"] = result["growth_yoy"] - result["inflation_yoy"]
    result["feature_ready"] = result[MACRO_FEATURE_COLUMNS].notna().all(axis=1)
    return result


def normalize_financial_prices(frame: pd.DataFrame, spec: FinancialBasketSpec) -> pd.DataFrame:
    prices = frame.copy()
    if "date" in prices.columns:
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
        prices = prices.dropna(subset=["date"]).set_index("date")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("financial price input requires a DatetimeIndex or date column")
    prices = prices.sort_index()
    missing = [column for column in spec.required_columns() if column not in prices.columns]
    if missing:
        raise ValueError(f"missing financial basket columns: {missing}")
    prices[spec.required_columns()] = prices[spec.required_columns()].apply(pd.to_numeric, errors="coerce")
    return prices


def _equal_weight_return_index(prices: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    """Build a synthetic index from equally weighted constituent period returns.

    The basket is rebalanced to equal constituent weights at every observation.
    Averaging raw index levels would instead weight constituents by their arbitrary
    quoted level.
    """
    constituent_returns = prices[list(columns)].astype(float).pct_change(fill_method=None)
    basket_returns = constituent_returns.mean(axis=1, skipna=False)
    if not basket_returns.empty:
        basket_returns.iloc[0] = 0.0
    return (1.0 + basket_returns).cumprod() * 100.0


def _rolling_drawdown(levels: pd.Series, window: int = 4) -> pd.Series:
    return (levels / levels.rolling(window, min_periods=window).max() - 1.0) * 100.0


def prepare_financial_features(
    price_frame: pd.DataFrame, spec: FinancialBasketSpec | None = None
) -> pd.DataFrame:
    spec = spec or FinancialBasketSpec()
    prices = normalize_financial_prices(price_frame, spec)
    composite = pd.DataFrame(index=prices.index)
    composite["eq_us"] = _equal_weight_return_index(prices, spec.us_equity)
    composite["eq_global"] = _equal_weight_return_index(prices, spec.global_equity)
    composite["bond_safe"] = _equal_weight_return_index(prices, spec.safe_bonds)
    composite["credit"] = _equal_weight_return_index(prices, spec.credit)
    composite["commodity"] = _equal_weight_return_index(prices, spec.commodity)
    composite["dxy"] = prices[spec.dxy].astype(float)
    composite["usdeur"] = 1.0 / prices[spec.eurusd].astype(float)
    composite["usdjpy"] = 1.0 / prices[spec.jpyusd].astype(float)
    composite["usdgbp"] = 1.0 / prices[spec.gbpusd].astype(float)
    composite["usdchf"] = 1.0 / prices[spec.chfusd].astype(float)
    quarterly_return = composite.pct_change(fill_method=None)
    result = pd.DataFrame(index=composite.index)
    result["quarter_complete"] = pd.Series(result.index.is_quarter_end, index=result.index)
    result["eq_us_yoy"] = composite["eq_us"].pct_change(4, fill_method=None) * 100.0
    result["eq_global_yoy"] = composite["eq_global"].pct_change(4, fill_method=None) * 100.0
    result["bond_safe_yoy"] = composite["bond_safe"].pct_change(4, fill_method=None) * 100.0
    result["credit_excess_yoy"] = (
        composite["credit"].pct_change(4, fill_method=None)
        - composite["bond_safe"].pct_change(4, fill_method=None)
    ) * 100.0
    result["commodity_yoy"] = composite["commodity"].pct_change(4, fill_method=None) * 100.0
    result["usd_broad_yoy"] = composite["dxy"].pct_change(4, fill_method=None) * 100.0
    result["usd_major_yoy"] = pd.DataFrame(
        {
            name: composite[name].pct_change(4, fill_method=None) * 100.0
            for name in ["usdeur", "usdjpy", "usdgbp", "usdchf"]
        }
    ).mean(axis=1)
    result["equity_vol_4q"] = quarterly_return["eq_global"].rolling(4, min_periods=4).std(ddof=0) * 200.0
    result["bond_vol_4q"] = quarterly_return["bond_safe"].rolling(4, min_periods=4).std(ddof=0) * 200.0
    result["commodity_vol_4q"] = quarterly_return["commodity"].rolling(4, min_periods=4).std(ddof=0) * 200.0
    result["fx_vol_4q"] = (
        quarterly_return[["dxy", "usdeur", "usdjpy", "usdgbp", "usdchf"]].std(axis=1, ddof=0) * 200.0
    )
    result["stock_bond_gap_yoy"] = result["eq_global_yoy"] - result["bond_safe_yoy"]
    result["commodity_bond_gap_yoy"] = result["commodity_yoy"] - result["bond_safe_yoy"]
    result["equity_drawdown_4q"] = _rolling_drawdown(composite["eq_global"], 4)
    result["cross_asset_dispersion_1q"] = pd.DataFrame(
        {
            "eq": quarterly_return["eq_global"] * 100.0,
            "bond": quarterly_return["bond_safe"] * 100.0,
            "commodity": quarterly_return["commodity"] * 100.0,
            "usd": quarterly_return["dxy"] * 100.0,
        }
    ).std(axis=1, ddof=0)
    result["stock_bond_corr_4q"] = (
        quarterly_return["eq_global"].rolling(4, min_periods=4).corr(quarterly_return["bond_safe"])
    )
    result["stock_bond_corr_8q"] = (
        quarterly_return["eq_global"].rolling(8, min_periods=6).corr(quarterly_return["bond_safe"])
    )
    result["feature_ready"] = result[FINANCIAL_FEATURE_COLUMNS].notna().all(axis=1)
    return result


def load_prepared_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    data = frame.copy()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["date"]).sort_values("date").set_index("date")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("prepared feature input requires date")
    for column in data.columns:
        if column != "source_tag":
            data[column] = pd.to_numeric(data[column], errors="ignore")
    if "quarter_complete" not in data:
        data["quarter_complete"] = True
    data["feature_ready"] = data[feature_columns].notna().all(axis=1)
    return data
