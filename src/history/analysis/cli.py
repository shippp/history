import logging
from pathlib import Path
from typing import Callable

import click

from history.analysis.steps.landcover import run_landcover
from history.analysis.steps.std_dem import run_std_dem
from history.cli_common import configure_logging, load_config
from history.config import Config

logger = logging.getLogger(__name__)

STEP_RUNNERS: dict[str, Callable[[Config], None]] = {
    "std_dem": run_std_dem,
    "landcover": run_landcover
}

STEPS: list[str] = list(STEP_RUNNERS.keys()) + ["all"]

@click.group()
def cli() -> None:
    """Analysis."""

@cli.command("run")
@click.argument("step", nargs=-1, required=True, type=click.Choice(STEPS))
@click.option("--config", "config_path", default=Path("config.toml"), type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Path to config.toml (default: ./config.toml)")
@click.option("--overwrite", is_flag=True, default=False, help="Force recompute of existing data outputs (overrides config)")
@click.option("--overwrite-plots", "overwrite_plots", is_flag=True, default=False, help="Force regeneration of existing plots (overrides config)")
@click.option("--dry-run", "dry_run", is_flag=True, default=False, help="Print actions without executing them (overrides config)")
@click.option("--no-plots", "no_plots", is_flag=True, default=False, help="Skip plot generation for this step")
@click.option("--max-workers", "max_workers", type=int, default=None, help="Number of parallel workers (overrides config)")
@click.option("-v", "--verbose", "verbose", count=True, help="Increase verbosity (-v INFO, -vv DEBUG)")
def cmd_run(
    step: tuple[str, ...],
    config_path: Path,
    overwrite: bool,
    overwrite_plots: bool,
    dry_run: bool,
    no_plots: bool,
    max_workers: int | None,
    verbose: int,
) -> None:
    """Run one or more Analysis steps, always in pipeline order regardless of the order given on the command line."""
    configure_logging(verbose, cli_logger_name=__name__)
    config = load_config(config_path, overwrite, overwrite_plots, dry_run, no_plots, max_workers)

    steps_to_run = list(STEP_RUNNERS) if "all" in step else [name for name in STEP_RUNNERS if name in step]

    for name in steps_to_run:
        logger.info(f"Running step: {name}")
        STEP_RUNNERS[name](config)
        logger.info(f"Step `{name}` finished")


def main():
    """Entry point for the ``history-analysis`` command."""
    cli()


if __name__ == "__main__":
    main()
