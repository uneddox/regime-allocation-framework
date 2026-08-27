# Data contracts

All inputs are UTF-8 CSV files. No data downloader is coupled to the model core.

## Macro monthly input

Required columns are `sasdate`, `CPIAUCSL`, `INDPRO`, `UNRATE`, and `PAYEMS`. Public aliases `date`, `cpi`,
`industrial_production`, `unemployment`, and `payrolls` are accepted and normalized before calculation.

## Financial basket input

The first column is `date`; observations are quarter-end or provisional quarter snapshots. Required basket members
are defined in `FinancialBasketSpec`:

- Safe bonds: `LBUSTRUU`, `LUATTRUU`, `LD20TRUU`, `LUTLTRUU`, `LBUTTRUU`, `LUMSTRUU`, `LBEATREU`
- Credit: `LF98TRUU`, `LUACTRUU`, `I00732US`
- US equity: `SPX`, `CCMP`, `M1US000V`, `M1US000G`
- Global equity: `MXAU`, `MXCA`, `UKX`, `MXEU`, `MXJP`, `MXEF`, `MXBR`, `MXCN`, `MXIN`, `KOSPI`, `MXTW`
- Commodity: `BCOMTR`, gold, silver, platinum, and palladium
- FX: `DXY`, `EURUSD`, `JPYUSD`, `GBPUSD`, and `CHFUSD`

The precise full column strings are visible in `src/regime_allocation/features.py`. Identifiers define the schema;
no observations from these series are included.

## Prepared Financial continuity input

`--financial-prepared` accepts a dated feature CSV instead of basket levels. It must contain the seven locked model
features plus any report features required by the Bond sleeve, particularly `stock_bond_corr_4q`. This mode
supports an operating transition where the locked historical proxy source and post-cutoff licensed basket source
differ. The caller constructs an append-only ledger; the framework verifies the historical portion against the
baseline bundle before filtering new rows.

## Country input

`date` is followed by one positive index-level column per country. Optional benchmark weights use the same column
names. Missing benchmark dates are forward-filled and all rows are normalized to one.

## Baseline bundle

Each bundle contains:

- `baseline_features.csv`
- `model_frame.csv`
- `model_result.json`
- `manifest.json`

The manifest records cutoff, feature order, lock policy, and SHA-256 hashes. Any file mutation causes loading to
fail. Current history must also match the stored baseline before new observations can be filtered.
