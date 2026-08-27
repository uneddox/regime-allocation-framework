# Production parity

The parity tool loads the operating modules only when their path is supplied locally. It never copies operating
data or paths into the repository.

On deterministic synthetic full-schema inputs, the check compares:

- Macro and Financial feature frames
- Start probabilities, transition matrices, means, and variances
- Filtered and smoothed probability histories
- Warm-start transition and ensemble results
- Ensemble mean filtered vectors and transition matrices
- Macro and Financial one-step continuity filters

The expected tolerance is absolute `1e-10` with zero relative tolerance. The public CI cannot access private
operating source, so it runs deterministic behavioral and continuity-lock tests. The direct parity check is run in
the private review environment before a release.
