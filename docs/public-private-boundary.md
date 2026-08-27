# Public/private boundary

## Public calculation layer

- Exact deterministic NumPy HMM calculation path
- Multi-start, warm-start, relabeling, BIC ranking, and top-fit ensemble
- Baseline bundle hashes and append-only continuity filtering
- Macro and Financial transformations
- Macro/Financial allocation combination
- Horizon-specific Bond sleeve and Country modifier
- Synthetic full-schema generator, tests, and documentation

## Private operating layer

- Bloomberg and other licensed observations, workbooks, screenshots, and exports
- Internal emails, reports, attachments, databases, and search indexes
- Actual frozen baseline data and production portfolio outputs
- Credentials, SMTP, recipients, IP addresses, hostnames, and absolute workstation paths
- Schedulers, alerting, and internal report-generation workflows

The public package never imports or calls the private scheduler. Provider adapters produce the documented CSV or
prepared-feature contracts, while the calculation path remains unchanged.
