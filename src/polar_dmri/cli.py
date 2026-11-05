"""Command-line interface: ``polar-dmri optimize|evaluate|models``."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .config import load_run_config
from .evaluate import compare_protocols, write_summary_csv
from .models import available_models
from .optimize import optimize_protocol
from .protocol import PGSEProtocol


def _print_protocol(protocol: PGSEProtocol) -> None:
    print(f"{'delta [ms]':>11} {'Delta [ms]':>11} {'G [mT/m]':>10} {'b [s/mm2]':>11}")
    rows = zip(protocol.delta, protocol.big_delta, protocol.G, protocol.b_values)
    for delta, big_delta, G, b in rows:
        print(f"{delta * 1e3:11.2f} {big_delta * 1e3:11.2f} {G * 1e3:10.1f} {b:11.0f}")
    print(f"Common TE: {protocol.te * 1e3:.1f} ms")


def _optimize(args: argparse.Namespace) -> int:
    config = load_run_config(args.config)
    output_dir = args.output_dir or Path("runs") / (
        f"{config.model.name}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    result = optimize_protocol(
        config.model,
        config.substrates,
        config.n_shells,
        constraints=config.constraints,
        noise=config.noise,
        n_noise_realizations=config.n_noise_realizations,
        loss_parameters=config.loss_parameters,
        weights=config.weights,
        settings=config.optimizer,
        fit_settings=config.fit,
        include_b0=config.include_b0,
        seed=args.seed if args.seed is not None else config.seed,
        n_jobs=args.jobs,
        output_dir=output_dir,
    )
    print(f"\nStopped after {result.n_iterations} generations: {result.stop_reason}")
    print(f"Best loss: {result.loss:.6g}")
    _print_protocol(result.protocol)
    print(f"Outputs: {result.output_dir}")
    return 0


def _parse_protocol_argument(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    return (name, Path(path)) if separator else (Path(value).stem, Path(value))


def _evaluate(args: argparse.Namespace) -> int:
    config = load_run_config(args.config)
    protocols = {}
    for value in args.protocol:
        name, path = _parse_protocol_argument(value)
        if name in protocols:
            raise ValueError(f"Duplicate protocol name {name!r}; use NAME=PATH to distinguish.")
        protocols[name] = PGSEProtocol.load_csv(path, include_b0=config.include_b0)
        violation = config.constraints.violation(protocols[name])
        if violation > 1e-18:
            print(f"Warning: protocol {name!r} violates the configured limits.", file=sys.stderr)

    results = compare_protocols(
        config.model,
        protocols,
        config.substrates,
        noise=config.noise,
        n_noise_realizations=args.n_noise or config.evaluation_noise_realizations,
        fit_settings=config.fit,
        seed=args.seed,
        n_jobs=args.jobs,
    )

    print(f"{'protocol':<20} {'parameter':<10} {'bias':>11} {'std':>11} {'rmse':>11} {'norm. rmse':>11}")
    for name, result in results.items():
        for parameter, stats in result.summary().items():
            print(
                f"{name:<20} {parameter:<10} {stats['bias']:11.3e} {stats['std']:11.3e} "
                f"{stats['rmse']:11.3e} {stats['normalized_rmse']:11.4f}"
            )
        print(f"{name:<20} {'loss':<10} {result.loss(config.loss_parameters, config.weights):11.5f}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_summary_csv(results, output_dir / "summary.csv")
        for name, result in results.items():
            result.save_csv(output_dir / f"{name}_fits.csv")
        print(f"Outputs: {output_dir}")
    return 0


def _models(_: argparse.Namespace) -> int:
    print("\n".join(available_models()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polar-dmri",
        description="Optimize and evaluate diffusion MRI protocols by parameter recovery.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    optimize = commands.add_parser("optimize", help="search for an optimal protocol")
    optimize.add_argument("config", help="YAML configuration file")
    optimize.add_argument("--output-dir", help="new or empty directory for run outputs")
    optimize.add_argument("--seed", type=int, help="overrides the seed in the configuration")
    optimize.add_argument("--jobs", type=int, default=-1, help="worker processes (default: all)")
    optimize.set_defaults(handler=_optimize)

    evaluate = commands.add_parser("evaluate", help="compare protocols under matched noise")
    evaluate.add_argument("config", help="YAML configuration file")
    evaluate.add_argument(
        "--protocol", action="append", required=True, metavar="[NAME=]CSV",
        help="protocol CSV file; repeat to compare several",
    )  # fmt: skip
    evaluate.add_argument("--n-noise", type=int, help="noise realizations per substrate")
    evaluate.add_argument("--output-dir", help="directory for summary and per-fit CSV files")
    evaluate.add_argument("--seed", type=int, default=0, help="noise seed (default: 0)")
    evaluate.add_argument("--jobs", type=int, default=-1, help="worker processes (default: all)")
    evaluate.set_defaults(handler=_evaluate)

    models = commands.add_parser("models", help="list available signal models")
    models.set_defaults(handler=_models)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
