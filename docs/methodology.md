# Methodology

## Common HMM engine

Both regime models use a diagonal Gaussian hidden Markov model. The EM loop uses forward-backward probabilities,
variance floors of `1e-4`, and a diagonal transition pseudo-count of `8.0`. Features are standardized with an
expanding mean and population standard deviation using a minimum history of eight quarters.

Each fit derives its random seed from the base seed, observation count, state count, and start number. Twenty
starts are fitted deterministically. The first start may receive governed initial parameters; subsequent starts
remain independent. BIC ranks fits. The five best relabeled fits form the forecast ensemble.

Operational and backtest signals use filtered probabilities. Smoothed probabilities use future observations and
are retained for ex-post explanation only.

## Macro regime

Monthly `CPIAUCSL`, `INDPRO`, `UNRATE`, and `PAYEMS` levels are sampled by calendar quarter. A quarter is complete
only when every required series has at least three monthly observations. Features are:

1. CPI year-over-year inflation
2. Industrial-production year-over-year growth
3. Inflation minus its trailing 20-quarter median, requiring 12 quarters
4. Quarterly average unemployment level

Four states are fitted. Every start is relabeled by ascending smoothed-probability-weighted inflation and then
growth. The operating macro allocation uses the H+1 ensemble forecast as its current actionable state.

## Financial regime

The input is the operating cross-asset basket schema. Equal-weight level composites are built for US equities,
global equities, safe bonds, credit, and commodities, along with DXY and four inverse major-currency pairs.
Seventeen report features are calculated. The locked K5 model uses seven of them:

1. Global-equity YoY
2. Credit return minus safe-bond return, YoY
3. Commodity YoY
4. DXY YoY
5. Four-quarter safe-bond volatility
6. One-quarter cross-asset dispersion
7. Four-quarter stock-bond correlation

States are relabeled by ascending weighted-average global-equity return, commodity return, and broad-dollar
return. The current Financial state is the latest filtered state; future states use the ensemble transition
forecast.

## Forecast ensemble

For every relabeled start, the last filtered vector and transition matrix are retained. The model averages the
five best vectors and matrices. Forecast horizon `h` is:

```text
ensemble_pi_t × ensemble_transition^h
```

H+1 is meaningful, H+2 is a reference, and H+4 is directional only.

## Continuity

A baseline bundle stores the exact baseline feature frame, standardized model frame, parameters, historical
probabilities, state order, ensemble members, and SHA-256 hashes. On update, historical features through the
cutoff must match within `1e-12`. Parameters and historical probabilities remain frozen. Only later observations
are normalized using expanding statistics and passed through the locked one-step filter for the best fit and every
ensemble member.

## Allocation combination

Macro state targets are 75/25, 65/35, 55/45, and 45/55. Financial conviction scales the active difference from
the 65/35 benchmark:

```text
equity = 0.65 + financial_scaler × (macro_equity − 0.65)
bond   = 0.35 + financial_scaler × (macro_bond   − 0.35)
```

## Bond sleeve

At horizon zero, the latest realized four-quarter stock-bond correlation is used. At forecast horizons, the
average correlation of the predicted Financial state is used. Correlation at or below 0.10 allocates the complete
bond sleeve to aggregate bonds; `(0.10, 0.20]` allocates 90/10 to bonds/cash; above 0.20 allocates 80/20.

## Country factor

Country-index returns are decomposed into a benchmark-weighted global return and country-specific effects. Rolling
annualized country-effect volatility is ranked through history. Weak, medium, and strong readings map to 0%, 5%,
and 10% active budgets. The normalized latest effect tilts benchmark country weights, which are then multiplied by
each horizon's final equity allocation to obtain total-portfolio country weights.
