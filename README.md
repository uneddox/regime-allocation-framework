# Regime Allocation Framework

An independently packaged, production-compatible implementation of a four-stage allocation process:

1. Macro regime
2. Financial regime
3. Bond sleeve
4. Country factor

The numerical core follows the operating model: diagonal Gaussian emissions, sticky transitions, deterministic
multi-start fitting, warm-start support, state relabeling, top-fit forecast ensembles, and append-only continuity.
The repository excludes private observations and operational infrastructure.

## Process

```text
Monthly macro levels ─ Macro K4 HMM ────────────┐
                                                ├─ Macro target × Financial conviction
Cross-asset baskets ─ Financial K5 HMM ─────────┘
                                │
                                ├─ state-conditioned stock/bond correlation ─ Bond/Cash sleeve
                                │
Country index levels ─ country-specific effects ─ equity-sleeve country weights
                                                        │
                                                        └─ total-portfolio country weights
```

## Locked calculation specification

| Control | Macro | Financial |
|---|---:|---:|
| States | 4 | 5 |
| Starts | 20 | 20 |
| Sticky transition strength | 8.0 | 8.0 |
| Forecast ensemble | Best 5 BIC-ranked fits | Best 5 BIC-ranked fits |
| Random seed | 42 with observation/state/start seed derivation | Same |
| Operational probability | Filtered | Filtered |
| Forecast horizons | H+1, H+2, H+4 quarters | H+1, H+2, H+4 quarters |

Macro features are inflation YoY, industrial-production growth YoY, inflation relative to its trailing five-year
median, and unemployment level. Financial features are global-equity YoY, credit excess YoY, commodity YoY,
broad-dollar YoY, four-quarter bond volatility, cross-asset dispersion, and four-quarter stock-bond correlation.

Multi-asset baskets are constructed by equally weighting constituent returns at each observation and compounding
them into synthetic indices. Raw quoted index levels are not averaged.

## Continuity

`baseline` freezes the baseline model frame, parameters, filtered/smoothed histories, state labels, ensemble
members, and file hashes. A later `run` validates that history and processes only post-cutoff observations through
the locked parameters. It does not silently refit history.

For each new observation and ensemble member:

```text
prior(t)     = filtered(t-1) × transition
posterior(t) ∝ prior(t) × GaussianEmission(observation(t))
```

## Reproducible synthetic example

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

regime-allocation sample --output sample-data

regime-allocation baseline \
  --macro sample-data/macro_monthly_baseline.csv \
  --financial sample-data/financial_baskets_baseline.csv \
  --config configs/default.yaml \
  --output baseline-bundles

regime-allocation run \
  --macro sample-data/macro_monthly.csv \
  --financial sample-data/financial_baskets.csv \
  --country sample-data/country_levels.csv \
  --macro-baseline baseline-bundles/macro \
  --financial-baseline baseline-bundles/financial \
  --config configs/default.yaml \
  --output outputs
```

All generated sample observations are synthetic. No vendor or internal observation is committed.

When the locked Financial baseline and post-cutoff basket use different licensed providers, pass the already
constructed append-only feature ledger with `--financial-prepared`. The historical portion must match the baseline
bundle exactly; only rows after the cutoff may be appended.

## Production parity check

Organizations with the operating source available locally can compare both implementations on the same synthetic
input:

```bash
PYTHONPATH=src python tools/check_production_parity.py \
  --production-dir /path/to/operating/regime/code \
  --macro sample-data/macro_monthly_baseline.csv \
  --financial sample-data/financial_baskets_baseline.csv \
  --starts 5
```

The check covers transformations, best-fit parameters, filtered and smoothed probabilities, warm starts,
continuity filtering, and forecast-ensemble inputs.

## Allocation policy

The compatibility configuration preserves the operating calculation: 65/35 benchmark, macro state equity targets
of 75%, 65%, 55%, and 45%, and financial conviction scalers of 0.50, 0.75, 0.50, 1.00, and 0.75. These are model
rules, not claims of universal optimality or personal investment advice.

## Documentation

- [Changelog](CHANGELOG.md)
- [Methodology](docs/methodology.md)
- [Data contracts](docs/data-contracts.md)
- [Production parity](docs/production-parity.md)
- [Model governance](docs/model-governance.md)
- [Public/private boundary](docs/public-private-boundary.md)

## License

Code is released under the MIT License. Data remain subject to their original provider licenses. This repository
does not redistribute or relicense third-party datasets.
