import argparse
from pathlib import Path

from app.services.analyzer import InteractionAnalyzer
from app.services.map_updater import UserMapUpdater
from app.services.profile_compactor import UserProfileCompactor
from app.services.settings import load_llm_settings
from app.services.viability_evaluator import PersonalizationViabilityEvaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project maintenance commands.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    viability_parser = subparsers.add_parser(
        "viability-benchmark",
        help="Run the personalization viability benchmark and export reports.",
    )
    viability_parser.add_argument("--rounds-per-user", type=int, default=4)
    viability_parser.add_argument("--report-dir", type=Path, default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "viability-benchmark":
        raise ValueError(f"Unsupported command: {args.command}")

    evaluator = PersonalizationViabilityEvaluator(
        analyzer=InteractionAnalyzer(),
        updater=UserMapUpdater(),
        profile_compactor=UserProfileCompactor(),
        settings=load_llm_settings(),
        report_dir=args.report_dir,
    )
    result = evaluator.run_experiment(rounds_per_user=args.rounds_per_user)

    print(
        f"viability supports={result.supports_viability} "
        f"alignment_lift={result.mean_alignment_lift:.4f} "
        f"token_delta={result.cost_benefit.mean_token_cost_delta:.8f}"
    )
    for artifact in result.exported_artifacts:
        print(f"{artifact.format}: {artifact.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())