from __future__ import annotations

from typing import Any

import pandas as pd


def classify_bond_sleeve(correlation: float, rules: list[dict[str, Any]]) -> dict[str, Any]:
    for rule in rules:
        lower = rule.get("min") is None or correlation > float(rule["min"])
        upper = rule.get("max") is None or correlation <= float(rule["max"])
        if lower and upper:
            return rule
    raise ValueError(f"no bond sleeve rule matches correlation={correlation}")


def apply_bond_sleeve(
    allocation: pd.DataFrame,
    financial_summary: pd.DataFrame,
    financial_features: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    valid = financial_features.dropna(subset=["stock_bond_corr_4q"])
    if valid.empty:
        raise ValueError("stock-bond correlation is unavailable")
    latest_date = valid.index[-1].date().isoformat()
    latest_correlation = float(valid.iloc[-1]["stock_bond_corr_4q"])
    state_correlations = dict(
        zip(
            financial_summary["state"].astype(int),
            financial_summary["avg_stock_bond_corr_4q"].astype(float),
            strict=True,
        )
    )
    rows = []
    for row in allocation.to_dict(orient="records"):
        if int(row["horizon_quarters"]) == 0:
            correlation = latest_correlation
            source = f"latest realized stock_bond_corr_4q at {latest_date}"
        else:
            correlation = float(state_correlations[int(row["financial_state"])])
            source = "state-average stock_bond_corr_4q for predicted financial state"
        rule = classify_bond_sleeve(correlation, config["bond_sleeve"]["rules"])
        total_bond = float(row["final_bond_weight_pre_valuation"])
        total_equity = float(row["final_equity_weight_pre_valuation"])
        aggregate_bond = total_bond * float(rule["bond_share"])
        cash = total_bond * float(rule["cash_share"])
        rows.append(
            {
                **row,
                "stock_bond_corr_4q_assumption": correlation,
                "corr_source": source,
                "bond_sleeve_regime": str(rule["name"]),
                "aggregate_bond_share_of_bond_sleeve": float(rule["bond_share"]),
                "cash_share_of_bond_sleeve": float(rule["cash_share"]),
                "final_equity_weight": total_equity,
                "final_aggregate_bond_weight": aggregate_bond,
                "final_cash_weight": cash,
                "final_total_weight_check": total_equity + aggregate_bond + cash,
            }
        )
    return pd.DataFrame(rows)
