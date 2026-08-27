from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from regime_allocation.config import load_config
from regime_allocation.continuity import filter_step as public_filter_step
from regime_allocation.features import prepare_financial_features, prepare_macro_features
from regime_allocation.financial import fit_financial
from regime_allocation.macro import fit_macro


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assert_close(label: str, actual, expected, atol: float = 1e-10) -> None:
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=atol, rtol=0)
    print(f"PASS {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the public core with the local operating code.")
    parser.add_argument("--production-dir", type=Path, required=True)
    parser.add_argument("--macro", type=Path, required=True)
    parser.add_argument("--financial", type=Path, required=True)
    parser.add_argument("--starts", type=int, default=5)
    args = parser.parse_args()
    os.environ.update(
        {
            "REGIME_FIXED_K": "4",
            "REGIME_FEATURE_COLUMNS": "inflation_yoy,growth_yoy,inflation_gap_regime,unrate_level",
            "FINANCIAL_FIXED_K": "5",
            "FINANCIAL_FEATURE_COLUMNS": (
                "eq_global_yoy,credit_excess_yoy,commodity_yoy,usd_broad_yoy,"
                "bond_vol_4q,cross_asset_dispersion_1q,stock_bond_corr_4q"
            ),
            "REGIME_DETERMINISTIC": "1",
            "FORCE_BACKEND": "numpy",
            "CPU_MAX_WORKERS": "1",
            "REGIME_FORECAST_ENSEMBLE_TOP_N": str(min(5, args.starts)),
        }
    )
    original_cwd = Path.cwd()
    production_macro = load_module("inflation_regime_hmm", args.production_dir / "inflation_regime_hmm.py")
    os.chdir(original_cwd)
    macro_raw = pd.read_csv(args.macro)
    production_monthly = macro_raw.copy()
    production_monthly["sasdate"] = pd.to_datetime(production_monthly["sasdate"])
    production_monthly = production_monthly.set_index("sasdate")
    production_features = production_macro.prepare_macro_features(production_monthly)
    public_features = prepare_macro_features(macro_raw)
    pd.testing.assert_frame_equal(
        public_features[production_features.columns], production_features, check_freq=False
    )
    print("PASS macro features")
    eligible = production_features.loc[
        production_features["quarter_complete"] & production_features["feature_ready"]
    ]
    production_macro_result, _ = production_macro.fit_regime_model(
        eligible,
        k_min=4,
        k_max=4,
        n_starts=args.starts,
        min_history=8,
        backend="numpy",
    )
    config = load_config()
    config["macro"]["starts"] = args.starts
    config["macro"]["ensemble_top_n"] = min(5, args.starts)
    config["financial"]["starts"] = args.starts
    config["financial"]["ensemble_top_n"] = min(5, args.starts)
    public_macro_result, _, _ = fit_macro(macro_raw, config)
    for field in ["startprob", "transmat", "means", "covars", "gamma_filtered", "gamma_smoothed"]:
        assert_close(
            f"macro {field}", getattr(public_macro_result, field), getattr(production_macro_result, field)
        )
    assert_close(
        "macro ensemble pi",
        public_macro_result.forecast_ensemble["pi_t_mean"],
        production_macro_result.forecast_ensemble["pi_t_mean"],
    )
    assert_close(
        "macro ensemble transition",
        public_macro_result.forecast_ensemble["transmat_mean"],
        production_macro_result.forecast_ensemble["transmat_mean"],
    )
    production_warm, _ = production_macro.fit_regime_model(
        eligible,
        k_min=4,
        k_max=4,
        n_starts=args.starts,
        min_history=8,
        backend="numpy",
        init_params=production_macro_result.fitted_params,
    )
    public_warm, _, _ = fit_macro(macro_raw, config, init_params=production_macro_result.fitted_params)
    assert_close("macro warm-start transition", public_warm.transmat, production_warm.transmat)
    assert_close(
        "macro warm-start ensemble pi",
        public_warm.forecast_ensemble["pi_t_mean"],
        production_warm.forecast_ensemble["pi_t_mean"],
    )
    production_macro_continuity = load_module(
        "production_macro_continuity", args.production_dir / "macro_regime_continuity.py"
    )
    os.chdir(original_cwd)
    macro_scaled = production_macro.expanding_zscore(
        eligible[production_macro.FEATURE_COLUMNS], min_history=8
    ).dropna()
    macro_observation = macro_scaled.iloc[-1].to_numpy(float)
    production_member = {
        key: np.asarray(value, dtype=float).copy()
        for key, value in production_macro_result.forecast_ensemble["members"][0].items()
        if key in {"transmat", "means", "covars", "pi_t"}
    }
    public_member = {key: value.copy() for key, value in production_member.items()}
    assert_close(
        "macro continuity filter",
        public_filter_step(public_member, macro_observation),
        production_macro_continuity.filter_step(production_member, macro_observation),
    )

    production_financial = load_module(
        "financial_regime_hmm", args.production_dir / "financial_regime_hmm.py"
    )
    os.chdir(original_cwd)
    financial_raw = pd.read_csv(args.financial, parse_dates=["date"]).set_index("date")
    production_financial_features = production_financial.prepare_financial_features(financial_raw)
    public_financial_features = prepare_financial_features(financial_raw)
    pd.testing.assert_frame_equal(
        public_financial_features[production_financial_features.columns],
        production_financial_features,
        check_freq=False,
    )
    print("PASS financial features")
    eligible_financial = production_financial_features.loc[production_financial_features["feature_ready"]]
    production_financial_result, _ = production_financial.fit_regime_model(
        eligible_financial,
        k_min=5,
        k_max=5,
        n_starts=args.starts,
        min_history=8,
        backend="numpy",
    )
    public_financial_result, _, _ = fit_financial(financial_raw, config)
    for field in ["startprob", "transmat", "means", "covars", "gamma_filtered", "gamma_smoothed"]:
        assert_close(
            f"financial {field}",
            getattr(public_financial_result, field),
            getattr(production_financial_result, field),
        )
    assert_close(
        "financial ensemble pi",
        public_financial_result.forecast_ensemble["pi_t_mean"],
        production_financial_result.forecast_ensemble["pi_t_mean"],
    )
    assert_close(
        "financial ensemble transition",
        public_financial_result.forecast_ensemble["transmat_mean"],
        production_financial_result.forecast_ensemble["transmat_mean"],
    )
    production_financial_continuity = load_module(
        "production_financial_continuity",
        args.production_dir / "financial_regime_continuity.py",
    )
    os.chdir(original_cwd)
    financial_scaled = production_financial.expanding_zscore(
        eligible_financial[production_financial.FEATURE_COLUMNS], min_history=8
    ).dropna()
    financial_observation = financial_scaled.iloc[-1].to_numpy(float)
    production_member = {
        key: np.asarray(value, dtype=float).copy()
        for key, value in production_financial_result.forecast_ensemble["members"][0].items()
        if key in {"transmat", "means", "covars", "pi_t"}
    }
    public_member = {key: value.copy() for key, value in production_member.items()}
    assert_close(
        "financial continuity filter",
        public_filter_step(public_member, financial_observation),
        production_financial_continuity.filter_step(production_member, financial_observation),
    )
    print("ALL PRODUCTION PARITY CHECKS PASSED")


if __name__ == "__main__":
    main()
