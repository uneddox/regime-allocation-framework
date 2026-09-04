# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-09-05

### Changed

- Replaced raw index-level averaging with return-equal-weighted financial baskets.
- Rebalanced basket constituents to equal weights at every observation.
- Compounded the equal-weighted period returns into synthetic basket indices.
- Updated the methodology, data contract, package metadata, and reported framework version.

### Added

- Added regression coverage proving that arbitrary quoted index levels do not affect constituent weights.

### Migration

- Regenerate Financial baseline bundles and downstream outputs before using version 0.3.0 in production.
- Do not reuse Financial baseline bundles created with version 0.2.0 raw-level basket construction.

## [0.2.0] - 2026-08-26

### Added

- Published the initial standalone macro, Financial, Bond sleeve, and Country factor allocation framework.
- Added deterministic multi-start HMM fitting, append-only continuity, synthetic data, tests, and documentation.

[Unreleased]: https://github.com/uneddox/regime-allocation-framework/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/uneddox/regime-allocation-framework/compare/6680f2d...v0.3.0
[0.2.0]: https://github.com/uneddox/regime-allocation-framework/commit/6680f2d
