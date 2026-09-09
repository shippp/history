import dataclasses
import logging
import os
import sys
from pathlib import Path

from history.config import Config


def configure_logging(verbosity: int, cli_logger_name: str) -> None:
    """Set the ``history`` logger level based on the ``-v`` / ``-vv`` count.

    ``cli_logger_name`` is always forced to INFO so a CLI's own step
    start/finish messages print regardless of verbosity.
    """
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
    logging.getLogger(cli_logger_name).setLevel(logging.INFO)


def load_config(
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
