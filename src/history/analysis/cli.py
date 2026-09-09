import dataclasses
import logging
from pathlib import Path
import sys
from typing import Callable

import click

from history.postprocessing.config import Config

logger = logging.getLogger(__name__)

STEP_RUNNERS: dict[str, Callable[[Config], None]] = {
    "std_dem": print
}

STEPS: list[str] = list(STEP_RUNNERS.keys()) + ["all"]

@click.group()
def cli() -> None:
    """Analysis."""

@cli.command("run")
@click.argument("step", type=click.Choice(STEPS))
@click.option("--config", "config_path", default=Path("config.toml"), type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Path to config.toml (default: ./config.toml)")
@click.option("--overwrite", is_flag=True, default=False, help="Force recompute of existing data outputs (overrides config)")
@click.option("--overwrite-plots", "overwrite_plots", is_flag=True, default=False, help="Force regeneration of existing plots (overrides config)")
@click.option("--dry-run", "dry_run", is_flag=True, default=False, help="Print actions without executing them (overrides config)")
@click.option("--no-plots", "no_plots", is_flag=True, default=False, help="Skip plot generation for this step")
@click.option("--max-workers", "max_workers", type=int, default=None, help="Number of parallel workers (overrides config)")
@click.option("-v", "--verbose", "verbose", count=True, help="Increase verbosity (-v INFO, -vv DEBUG)")
def cmd_run(
    step: str,
    config_path: Path,
    overwrite: bool,
    overwrite_plots: bool,
    dry_run: bool,
    no_plots: bool,
    max_workers: int | None,
    verbose: int,
) -> None:
    """Run one or more Analysis steps."""
    _configure_logging(verbose)
    config = _load_config(config_path, overwrite, overwrite_plots, dry_run, no_plots, max_workers)

    if step == "all":
        for name, runner in STEP_RUNNERS.items():
            logger.info(f"Running step: {name}")
            runner(config)
            logger.info(f"Step `{name}` finished")

    else:
        STEP_RUNNERS[step](config)



def _load_config(
    config_path: Path,
    overwrite: bool,
    overwrite_plots: bool,
    dry_run: bool,
    no_plots: bool,
    max_workers: int | None,
) -> Config:
    """Load ``Config`` from the TOML file and apply any CLI flag overrides."""
    config = Config.from_toml_file(config_path)

    overrides = {}
    if overwrite:
        overrides["overwrite"] = True
    if overwrite_plots:
        overrides["overwrite_plots"] = True
    if dry_run:
        overrides["dry_run"] = True
    if no_plots:
        overrides["no_plots"] = True
    if max_workers is not None:
        overrides["max_workers"] = max_workers

    if overrides:
        config = dataclasses.replace(config, **overrides)

    return config


def _configure_logging(verbosity: int) -> None:
    """Set the ``history`` logger level based on the ``-v`` / ``-vv`` count."""
    import os

    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(verbosity, logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.addHandler(stdout_handler)

    # Only add a separate stderr handler when stdout and stderr go to different places (e.g. Slurm).
    # In an interactive terminal both point to the same fd, which would cause duplicates.
    try:
        stdout_stderr_differ = os.fstat(sys.stdout.fileno()) != os.fstat(sys.stderr.fileno())
    except Exception:
        stdout_stderr_differ = False

    if stdout_stderr_differ:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(fmt)
        stderr_handler.setLevel(logging.WARNING)
        root.addHandler(stderr_handler)

    logging.getLogger("history").setLevel(level)

    # Always print high level info for CLI steps (start, finish), regardless of verbose option.
    logging.getLogger("history.postprocessing.cli").setLevel(logging.INFO)


def main():
    """Entry point for the ``history-analysis`` command."""
    cli()


if __name__ == "__main__":
    main()
