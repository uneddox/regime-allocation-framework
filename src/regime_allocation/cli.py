from __future__ import annotations

import argparse
import json

from .pipeline import build_baseline_bundles, run_pipeline
from .sample import generate_sample_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="regime-allocation")
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="generate deterministic synthetic full-schema inputs")
    sample.add_argument("--output", default="sample-data")
    sample.add_argument("--seed", type=int, default=42)
    baseline = commands.add_parser("baseline", help="fit and freeze locked baseline bundles")
    baseline.add_argument("--macro", required=True)
    baseline.add_argument("--financial", required=True)
    baseline.add_argument("--config", default="configs/default.yaml")
    baseline.add_argument("--output", default="baseline-bundles")
    baseline.add_argument("--financial-prepared", action="store_true")
    run = commands.add_parser("run", help="run the four-stage production-compatible process")
    run.add_argument("--macro", required=True)
    run.add_argument("--financial", required=True)
    run.add_argument("--country", required=True)
    run.add_argument("--country-weights")
    run.add_argument("--macro-baseline")
    run.add_argument("--financial-baseline")
    run.add_argument("--financial-prepared", action="store_true")
    run.add_argument("--config", default="configs/default.yaml")
    run.add_argument("--output", default="outputs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "sample":
        result = generate_sample_data(args.output, args.seed)
    elif args.command == "baseline":
        result = build_baseline_bundles(
            args.macro,
            args.financial,
            args.output,
            args.config,
            financial_prepared=args.financial_prepared,
        )
    else:
        result = run_pipeline(
            macro_path=args.macro,
            financial_path=args.financial,
            country_path=args.country,
            country_weights_path=args.country_weights,
            macro_baseline_dir=args.macro_baseline,
            financial_baseline_dir=args.financial_baseline,
            financial_prepared=args.financial_prepared,
            config_path=args.config,
            output_dir=args.output,
        )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
