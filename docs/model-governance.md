# Model governance

## Locked controls

- Macro K4 and its ordered four-feature specification
- Financial K5 and its ordered seven-feature specification
- Seed 42 and deterministic per-start seed derivation
- Twenty starts, sticky strength 8.0, five-member ensemble
- State relabel rules
- Filtered/smoothed usage policy
- Baseline features, parameters, histories, and file hashes
- Allocation mappings and correlation thresholds

A change to any locked control is a model version change, not a routine data update.

## Required validation

- Run unit and end-to-end continuity tests.
- Run the local production parity tool when the operating source is available.
- Confirm baseline hashes and append-only history.
- Compare forecast probabilities with persistence and unconditional baselines.
- Report log loss and Brier score, not only hard-state accuracy.
- Review state occupancy, transition stability, and ensemble dispersion.
- Verify that live and backtest signals use filtered probabilities.

## Limitations

Diagonal emissions omit within-state covariance. Revised macro data may not represent information available in real
time. Financial baskets can change meaning when a data provider substitutes a series. Markov forecasts assume
stable transitions. State labels remain model constructs rather than directly observed facts. Portfolio mappings
are governed rules and do not establish expected returns or suitability.
