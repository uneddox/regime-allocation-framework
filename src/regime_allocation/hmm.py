from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

EPS = 1e-8
MODEL_RANDOM_SEED = 42


def logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True) + EPS)
    return result.squeeze() if axis is None else np.squeeze(result, axis=axis)


@dataclass
class HMMResult:
    n_states: int
    loglik: float
    bic: float
    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray
    covars: np.ndarray
    gamma_smoothed: np.ndarray
    gamma_filtered: np.ndarray
    state_order: list[int]
    selection_table: list[dict]
    fitted_params: dict | None = None
    forecast_ensemble: dict | None = None


class GaussianHMMDiag:
    """Production-parity NumPy implementation of the diagonal Gaussian HMM."""

    def __init__(
        self,
        n_states: int,
        n_iter: int = 250,
        tol: float = 1e-4,
        init_params: dict | None = None,
        sticky_strength: float = 8.0,
        seed: int | None = None,
    ) -> None:
        self.n_states = int(n_states)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.init_params = init_params
        self.sticky_strength = float(sticky_strength)
        self.seed = MODEL_RANDOM_SEED if seed is None else int(seed)
        self.rng = np.random.default_rng(self.seed)

    def export_params(self) -> dict[str, np.ndarray]:
        return {
            "startprob": self.startprob.copy(),
            "transmat": self.transmat.copy(),
            "means": self.means.copy(),
            "covars": self.covars.copy(),
        }

    def _valid_init_params(self, shape: tuple[int, int]) -> bool:
        if not self.init_params:
            return False
        features = shape[1]
        try:
            return (
                self.init_params["startprob"].shape == (self.n_states,)
                and self.init_params["transmat"].shape == (self.n_states, self.n_states)
                and self.init_params["means"].shape == (self.n_states, features)
                and self.init_params["covars"].shape == (self.n_states, features)
            )
        except (KeyError, AttributeError):
            return False

    def _initialize(self, x: np.ndarray) -> None:
        observations, _ = x.shape
        if self._valid_init_params(x.shape):
            self.means = self.init_params["means"].copy()
            self.covars = np.clip(self.init_params["covars"].copy(), 1e-4, None)
            self.startprob = np.clip(self.init_params["startprob"].copy(), EPS, None)
            self.startprob /= self.startprob.sum()
            self.transmat = np.clip(self.init_params["transmat"].copy(), EPS, None)
            self.transmat /= self.transmat.sum(axis=1, keepdims=True)
            return
        selected = self.rng.choice(observations, self.n_states, replace=False)
        self.means = x[selected].copy()
        self.covars = np.tile(np.var(x, axis=0) + 1e-3, (self.n_states, 1))
        self.startprob = self.rng.dirichlet(np.ones(self.n_states))
        self.transmat = np.full((self.n_states, self.n_states), 0.1 / max(self.n_states - 1, 1))
        np.fill_diagonal(self.transmat, 0.9)

    def _log_emission(self, x: np.ndarray) -> np.ndarray:
        output = np.zeros((x.shape[0], self.n_states))
        for state in range(self.n_states):
            variance = np.clip(self.covars[state], 1e-4, None)
            difference = x - self.means[state]
            output[:, state] = -0.5 * (
                np.sum(np.log(2 * np.pi * variance)) + np.sum((difference * difference) / variance, axis=1)
            )
        return output

    def _forward_backward(self, log_emission: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        observations, states = log_emission.shape
        log_start = np.log(np.clip(self.startprob, EPS, None))
        log_transition = np.log(np.clip(self.transmat, EPS, None))
        alpha = np.zeros((observations, states))
        alpha[0] = log_start + log_emission[0]
        for index in range(1, observations):
            alpha[index] = log_emission[index] + logsumexp(alpha[index - 1][:, None] + log_transition, axis=0)
        beta = np.zeros((observations, states))
        for index in range(observations - 2, -1, -1):
            beta[index] = logsumexp(
                log_transition + log_emission[index + 1][None, :] + beta[index + 1][None, :],
                axis=1,
            )
        loglik = float(logsumexp(alpha[-1], axis=0))
        gamma_smoothed = np.exp(alpha + beta - loglik)
        gamma_filtered = np.exp(alpha - logsumexp(alpha, axis=1)[:, None])
        xi = np.zeros((observations - 1, states, states))
        for index in range(observations - 1):
            xi[index] = np.exp(
                alpha[index][:, None]
                + log_transition
                + log_emission[index + 1][None, :]
                + beta[index + 1][None, :]
                - loglik
            )
        return gamma_smoothed, gamma_filtered, xi, loglik

    def fit(self, x: np.ndarray) -> GaussianHMMDiag:
        matrix = np.asarray(x, dtype=float)
        self._initialize(matrix)
        previous = -np.inf
        for _ in range(self.n_iter):
            emission = self._log_emission(matrix)
            smoothed, filtered, xi, loglik = self._forward_backward(emission)
            self.startprob = np.clip(filtered[0], EPS, None)
            self.startprob /= self.startprob.sum()
            transitions = np.sum(xi, axis=0) + np.eye(self.n_states) * self.sticky_strength + EPS
            self.transmat = transitions / transitions.sum(axis=1, keepdims=True)
            weights = smoothed.sum(axis=0) + EPS
            self.means = (smoothed.T @ matrix) / weights[:, None]
            for state in range(self.n_states):
                difference = matrix - self.means[state]
                self.covars[state] = (smoothed[:, state][:, None] * difference * difference).sum(
                    axis=0
                ) / weights[state]
                self.covars[state] = np.clip(self.covars[state], 1e-4, None)
            if abs(loglik - previous) < self.tol:
                break
            previous = loglik
        self.loglik_ = loglik
        self.gamma_smoothed_ = smoothed
        self.gamma_filtered_ = filtered
        return self


def candidate_state_range(n_obs: int, k_min: int = 2, k_max: int = 5, min_obs_per_state: int = 8) -> range:
    upper = min(k_max, max(k_min, n_obs // min_obs_per_state))
    return range(k_min, max(k_min, upper) + 1)


def derive_start_seed(seed_base: int, n_obs: int, n_states: int, start_idx: int) -> int:
    return int(
        (int(seed_base) * 1_000_003 + int(n_obs) * 97 + int(n_states) * 193 + int(start_idx) * 389)
        % (2**32 - 1)
    )


def expanding_zscore(frame: pd.DataFrame, min_history: int = 8) -> pd.DataFrame:
    mean = frame.expanding(min_periods=min_history).mean()
    std = frame.expanding(min_periods=min_history).std(ddof=0).replace(0, np.nan)
    return (frame - mean) / (std + EPS)


def _raw_result(model: GaussianHMMDiag, n_states: int, start_idx: int) -> HMMResult:
    return HMMResult(
        n_states=n_states,
        loglik=float(model.loglik_),
        bic=float("nan"),
        startprob=model.startprob.copy(),
        transmat=model.transmat.copy(),
        means=model.means.copy(),
        covars=model.covars.copy(),
        gamma_smoothed=model.gamma_smoothed_.copy(),
        gamma_filtered=model.gamma_filtered_.copy(),
        state_order=list(range(n_states)),
        selection_table=[{"start_idx": start_idx}],
        fitted_params=model.export_params(),
    )


def fit_regime_multistart(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    relabel: Callable[[HMMResult, pd.DataFrame], HMMResult],
    *,
    k_min: int,
    k_max: int,
    n_starts: int = 20,
    min_history: int = 8,
    sticky_strength: float = 8.0,
    ensemble_top_n: int = 5,
    init_params: dict | None = None,
    seed_base: int = MODEL_RANDOM_SEED,
) -> tuple[HMMResult, pd.DataFrame]:
    scaled = expanding_zscore(feature_df[feature_columns], min_history=min_history)
    z_columns = [f"{column}_z" for column in feature_columns]
    model_df = feature_df.join(scaled.add_suffix("_z"), how="left").dropna(subset=z_columns)
    matrix = model_df[z_columns].to_numpy(dtype=float)
    observations, features = matrix.shape
    if observations < 24:
        raise ValueError("Not enough quarterly observations to fit the HMM.")

    selection_table: list[dict] = []
    best_result: HMMResult | None = None
    for n_states in candidate_state_range(observations, k_min=k_min, k_max=k_max):
        starts: list[HMMResult] = []
        for start_idx in range(n_starts):
            seed = derive_start_seed(seed_base, observations, n_states, start_idx)
            model = GaussianHMMDiag(
                n_states=n_states,
                n_iter=300,
                tol=1e-4,
                init_params=init_params if start_idx == 0 else None,
                sticky_strength=sticky_strength,
                seed=seed,
            ).fit(matrix)
            starts.append(relabel(_raw_result(model, n_states, start_idx), model_df))

        ranked: list[HMMResult] = []
        for start_idx, result in enumerate(starts):
            parameter_count = (n_states - 1) + n_states * (n_states - 1) + 2 * n_states * features
            bic = -2 * result.loglik + parameter_count * math.log(observations)
            result = replace(result, bic=float(bic))
            ranked.append(result)
            selection_table.append(
                {
                    "n_states": n_states,
                    "start_idx": start_idx,
                    "loglik": result.loglik,
                    "bic": float(bic),
                    "backend": "numpy",
                }
            )
        ranked.sort(key=lambda item: item.bic)
        best_for_k = ranked[0]
        members = ranked[: min(int(ensemble_top_n), len(ranked))]
        ensemble_members = [
            {
                "startprob": item.startprob.copy(),
                "transmat": item.transmat.copy(),
                "means": item.means.copy(),
                "covars": item.covars.copy(),
                "pi_t": item.gamma_filtered[-1].copy(),
                "bic": float(item.bic),
            }
            for item in members
        ]
        best_for_k = replace(
            best_for_k,
            forecast_ensemble={
                "n_members": len(ensemble_members),
                "transmat_mean": np.mean([item["transmat"] for item in ensemble_members], axis=0),
                "pi_t_mean": np.mean([item["pi_t"] for item in ensemble_members], axis=0),
                "members": ensemble_members,
            },
        )
        if best_result is None or best_for_k.bic < best_result.bic:
            best_result = best_for_k

    if best_result is None:
        raise RuntimeError("HMM fitting produced no result")
    return replace(best_result, selection_table=selection_table), model_df


def forecast_probabilities(result: HMMResult, horizons: list[int]) -> pd.DataFrame:
    if result.forecast_ensemble:
        current = np.asarray(result.forecast_ensemble["pi_t_mean"], dtype=float)
        transition = np.asarray(result.forecast_ensemble["transmat_mean"], dtype=float)
    else:
        current = result.gamma_filtered[-1]
        transition = result.transmat
    rows = []
    for horizon in horizons:
        probability = current @ np.linalg.matrix_power(transition, int(horizon))
        row = {
            "horizon_quarters": int(horizon),
            "predicted_state": int(np.argmax(probability)),
            "pred_confidence": float(np.max(probability)),
            "prob_entropy": float(
                -np.sum(np.clip(probability, EPS, 1.0) * np.log(np.clip(probability, EPS, 1.0)))
            ),
            "confidence_band": (
                "meaningful"
                if int(horizon) == 1
                else "reference"
                if int(horizon) == 2
                else "directional_only"
            ),
        }
        row.update({f"p_state_{state}": float(value) for state, value in enumerate(probability)})
        rows.append(row)
    return pd.DataFrame(rows)
