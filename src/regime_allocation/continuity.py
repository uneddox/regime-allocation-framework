from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .hmm import HMMResult, expanding_zscore


@dataclass
class BaselineBundle:
    cutoff: pd.Timestamp
    feature_columns: list[str]
    baseline_features: pd.DataFrame
    model_df: pd.DataFrame
    result: HMMResult
    input_sha256: str


def frame_sha256(frame: pd.DataFrame) -> str:
    normalized = frame.sort_index().to_csv(index=True, float_format="%.17g").encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def _arrays(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value, dtype=float) for key, value in payload.items()}


def _result_payload(result: HMMResult) -> dict[str, Any]:
    ensemble = None
    if result.forecast_ensemble:
        ensemble = {
            "n_members": int(result.forecast_ensemble["n_members"]),
            "transmat_mean": np.asarray(result.forecast_ensemble["transmat_mean"]).tolist(),
            "pi_t_mean": np.asarray(result.forecast_ensemble["pi_t_mean"]).tolist(),
            "members": [
                {
                    key: (np.asarray(value).tolist() if key != "bic" else float(value))
                    for key, value in member.items()
                }
                for member in result.forecast_ensemble["members"]
            ],
        }
    return {
        "n_states": result.n_states,
        "loglik": result.loglik,
        "bic": result.bic,
        "startprob": result.startprob.tolist(),
        "transmat": result.transmat.tolist(),
        "means": result.means.tolist(),
        "covars": result.covars.tolist(),
        "gamma_smoothed": result.gamma_smoothed.tolist(),
        "gamma_filtered": result.gamma_filtered.tolist(),
        "state_order": result.state_order,
        "selection_table": result.selection_table,
        "fitted_params": {
            key: np.asarray(value).tolist() for key, value in (result.fitted_params or {}).items()
        },
        "forecast_ensemble": ensemble,
    }


def _payload_result(payload: dict[str, Any]) -> HMMResult:
    ensemble = payload.get("forecast_ensemble")
    if ensemble:
        ensemble = {
            "n_members": int(ensemble["n_members"]),
            "transmat_mean": np.asarray(ensemble["transmat_mean"], dtype=float),
            "pi_t_mean": np.asarray(ensemble["pi_t_mean"], dtype=float),
            "members": [
                {
                    key: (float(value) if key == "bic" else np.asarray(value, dtype=float))
                    for key, value in member.items()
                }
                for member in ensemble["members"]
            ],
        }
    return HMMResult(
        n_states=int(payload["n_states"]),
        loglik=float(payload["loglik"]),
        bic=float(payload["bic"]),
        startprob=np.asarray(payload["startprob"], dtype=float),
        transmat=np.asarray(payload["transmat"], dtype=float),
        means=np.asarray(payload["means"], dtype=float),
        covars=np.asarray(payload["covars"], dtype=float),
        gamma_smoothed=np.asarray(payload["gamma_smoothed"], dtype=float),
        gamma_filtered=np.asarray(payload["gamma_filtered"], dtype=float),
        state_order=[int(value) for value in payload["state_order"]],
        selection_table=payload["selection_table"],
        fitted_params=_arrays(payload.get("fitted_params", {})),
        forecast_ensemble=ensemble,
    )


def save_baseline_bundle(
    directory: str | Path,
    cutoff: pd.Timestamp,
    feature_columns: list[str],
    baseline_features: pd.DataFrame,
    model_df: pd.DataFrame,
    result: HMMResult,
) -> BaselineBundle:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    features_path = output / "baseline_features.csv"
    model_path = output / "model_frame.csv"
    result_path = output / "model_result.json"
    baseline_features.to_csv(features_path, index_label="date", float_format="%.17g")
    persisted_features = pd.read_csv(features_path, parse_dates=["date"]).set_index("date")
    input_hash = frame_sha256(persisted_features)
    model_df.to_csv(model_path, index_label="date", float_format="%.17g")
    result_path.write_text(json.dumps(_result_payload(result), indent=2), encoding="utf-8")
    manifest = {
        "format_version": 1,
        "cutoff": pd.Timestamp(cutoff).date().isoformat(),
        "feature_columns": feature_columns,
        "input_sha256": input_hash,
        "baseline_features_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
        "model_frame_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "model_result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "state_parameters_locked": True,
        "historical_probabilities_locked": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return BaselineBundle(
        pd.Timestamp(cutoff), feature_columns, baseline_features, model_df, result, input_hash
    )


def load_baseline_bundle(directory: str | Path) -> BaselineBundle:
    source = Path(directory)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    features_path = source / "baseline_features.csv"
    model_path = source / "model_frame.csv"
    result_path = source / "model_result.json"
    for path, field in (
        (features_path, "baseline_features_sha256"),
        (model_path, "model_frame_sha256"),
        (result_path, "model_result_sha256"),
    ):
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != manifest[field]:
            raise RuntimeError(f"Locked baseline file changed: {path.name}")
    baseline_features = pd.read_csv(features_path, parse_dates=["date"]).set_index("date")
    if frame_sha256(baseline_features) != manifest["input_sha256"]:
        raise RuntimeError("Locked baseline feature content changed")
    model_df = pd.read_csv(model_path, parse_dates=["date"]).set_index("date")
    result = _payload_result(json.loads(result_path.read_text(encoding="utf-8")))
    return BaselineBundle(
        cutoff=pd.Timestamp(manifest["cutoff"]),
        feature_columns=list(manifest["feature_columns"]),
        baseline_features=baseline_features,
        model_df=model_df,
        result=result,
        input_sha256=str(manifest["input_sha256"]),
    )


def filter_step(member: dict[str, Any], observation: np.ndarray) -> np.ndarray:
    means = np.asarray(member["means"], dtype=float)
    covars = np.clip(np.asarray(member["covars"], dtype=float), 1e-4, None)
    transmat = np.asarray(member["transmat"], dtype=float)
    prior = np.asarray(member["pi_t"], dtype=float) @ transmat
    difference = observation[None, :] - means
    log_emission = -0.5 * (
        np.sum(np.log(2 * np.pi * covars), axis=1) + np.sum((difference * difference) / covars, axis=1)
    )
    log_posterior = np.log(np.clip(prior, 1e-300, None)) + log_emission
    posterior = np.exp(log_posterior - np.max(log_posterior))
    posterior /= posterior.sum()
    member["pi_t"] = posterior
    return posterior


def extend_locked_result(
    bundle: BaselineBundle,
    hybrid_features: pd.DataFrame,
    *,
    min_history: int = 8,
) -> tuple[HMMResult, pd.DataFrame]:
    if bundle.result.forecast_ensemble is None:
        raise ValueError("continuity requires a baseline forecast ensemble")
    historical = hybrid_features.loc[hybrid_features.index <= bundle.cutoff].copy()
    expected = bundle.baseline_features.reindex(columns=historical.columns)
    try:
        pd.testing.assert_frame_equal(
            historical,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
            check_freq=False,
            check_names=False,
        )
    except AssertionError as exc:
        raise RuntimeError("Historical features differ from the locked append-only baseline") from exc
    scaled = expanding_zscore(hybrid_features[bundle.feature_columns], min_history=min_history)
    z_columns = [f"{column}_z" for column in bundle.feature_columns]
    extended = hybrid_features.join(scaled.add_suffix("_z"), how="left")
    additions = extended.loc[extended.index > bundle.cutoff].dropna(subset=z_columns).copy()
    if additions.empty:
        return bundle.result, bundle.model_df.copy()

    best_member = {
        "transmat": bundle.result.transmat.copy(),
        "means": bundle.result.means.copy(),
        "covars": bundle.result.covars.copy(),
        "pi_t": bundle.result.gamma_filtered[-1].copy(),
    }
    ensemble_members = [
        {
            key: (float(value) if key == "bic" else np.asarray(value, dtype=float).copy())
            for key, value in member.items()
        }
        for member in bundle.result.forecast_ensemble["members"]
    ]
    best_posteriors = []
    ensemble_posteriors = []
    for _, row in additions.iterrows():
        observation = row[z_columns].to_numpy(dtype=float)
        best_posteriors.append(filter_step(best_member, observation))
        ensemble_posteriors.append(
            np.mean([filter_step(member, observation) for member in ensemble_members], axis=0)
        )
    ensemble = {
        "n_members": len(ensemble_members),
        "transmat_mean": np.mean([member["transmat"] for member in ensemble_members], axis=0),
        "pi_t_mean": np.asarray(ensemble_posteriors[-1], dtype=float),
        "members": ensemble_members,
    }
    result = replace(
        bundle.result,
        gamma_filtered=np.vstack([bundle.result.gamma_filtered, best_posteriors]),
        gamma_smoothed=np.vstack([bundle.result.gamma_smoothed, best_posteriors]),
        forecast_ensemble=ensemble,
    )
    return result, pd.concat([bundle.model_df, additions], axis=0)
