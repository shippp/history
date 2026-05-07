"""
Command-line interface for the post-processing pipeline.

Two subcommands are available:

``history-postprocess create <output_dir>``
    Scaffold a new working directory by copying the config template into it.
    Edit the generated ``config.toml`` to point to your data before running
    any pipeline step.

``history-postprocess run <STEP> --config <path/to/config.toml>``
    Execute one or all pipeline steps in order. Available steps:

    uncompress  Extract compressed submission archives into the extracted dir.
    symlinks    Index submissions, parse filenames, and create typed symlinks.
    point2dem   Convert dense point clouds to DEMs via PDAL; integrate any
                user-provided DEMs by reprojecting them on the reference grid.
    coregister  Coregister raw DEMs to the reference using Nuth–Kaab + vertical
                shift.
    ddem        Compute differential DEMs before and after coregistration.
    std_dem     Build one standard-deviation DEM per (site, dataset) group from
                all coregistered DEMs.
    landcover   Compute and plot landcover-stratified statistics on dDEMs and
                STD DEMs.
    all         Run all steps in the order listed above.

Verbosity is controlled with ``-v`` (INFO) or ``-vv`` (DEBUG).
"""

import argparse
import dataclasses
import logging
import shutil
import sys
from pathlib import Path
from history.postprocessing.config import Config

logger = logging.getLogger(__name__)

_TEMPLATE_CONFIG = Path(__file__).parent / "config.exemple.toml"

RUN_STEPS = ["uncompress", "symlinks", "point2dem", "coregister", "ddem", "std_dem", "landcover", "all"]


def _configure_logging(verbosity: int) -> None:
    """Set the ``history`` logger level based on the ``-v`` / ``-vv`` count."""
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(verbosity, logging.DEBUG)
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S", stream=sys.stderr)
    logging.getLogger("history").setLevel(level)


def _load_config(args: argparse.Namespace) -> Config:
    """Load ``Config`` from the TOML file and apply any CLI flag overrides."""
    config = Config.from_toml_file(Path(args.config))

    overrides = {}
    if args.overwrite:
        overrides["overwrite"] = True
    if args.dry_run:
        overrides["dry_run"] = True
    if args.no_plots:
        overrides["no_plots"] = True
    if args.max_workers is not None:
        overrides["max_workers"] = args.max_workers

    if overrides:
        config = dataclasses.replace(config, **overrides)

    return config


def cmd_create(args: argparse.Namespace) -> None:
    """
    Scaffold a new post-processing working directory.

    Creates ``output_dir`` and copies the config template into it as
    ``config.toml``. The user must then edit that file to set the correct
    paths before running any pipeline step.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dest = output_dir / "config.toml"
    shutil.copy(_TEMPLATE_CONFIG, dest)

    print(f"Created '{output_dir}'")
    print(f"Config template copied to '{dest}'")
    print("Edit config.toml to point to your data before running the pipeline.")


def _run_uncompress(config: Config) -> None:
    """Extract all compressed submission archives into the extracted directory."""
    from history.postprocessing.pipeline import uncompress_all_submissions

    uncompress_all_submissions(
        config.raw_dir,
        config.extracted_dir,
        overwrite=config.overwrite,
        max_workers=config.max_workers,
    )


def _run_symlinks(config: Config) -> None:
    """Index submissions, parse filenames, and create typed symlink directories."""
    from history.postprocessing.pipeline import index_submissions_and_link_files, plot_symlinks

    index_submissions_and_link_files(
        config.extracted_dir,
        config.proc_dir.symlinks_dir,
        overwrite=config.overwrite,
    )

    if not config.no_plots:
        plot_symlinks(config.proc_dir.symlinks_dir, config.plot_dir)


def _run_point2dem(config: Config) -> None:
    """Convert dense point clouds to DEMs via PDAL, and integrate any user-provided DEMs."""
    from history.postprocessing.pipeline import process_pointclouds_to_dems, add_provided_dems, plot_point2dem

    dense_pc_dir = config.proc_dir.symlinks_dir / "dense_pointclouds"
    pointcloud_files = list(dense_pc_dir.glob("*.las")) + list(dense_pc_dir.glob("*.laz"))

    process_pointclouds_to_dems(
        pointcloud_files=pointcloud_files,
        output_directory=config.proc_dir.raw_dems_dir,
        references_data=config.references_data_mapping,
        pdal_exec_path=config.pdal_exec_path,
        overwrite=config.overwrite,
        dry_run=config.dry_run,
        max_workers=config.max_workers,
    )

    dems_symlink_dir = config.proc_dir.symlinks_dir / "dems"
    if dems_symlink_dir.exists():
        dem_files = list(dems_symlink_dir.glob("*.tif"))
        if dem_files:
            add_provided_dems(
                dem_files=dem_files,
                output_dir=config.proc_dir.raw_dems_dir,
                references_data=config.references_data_mapping,
                overwrite=config.overwrite,
            )

    if not config.no_plots:
        plot_point2dem(config.proc_dir.raw_dems_dir, config.plot_dir, config.max_workers)


def _run_coregister(config: Config) -> None:
    """Coregister raw DEMs to the reference using Nuth–Kaab + vertical shift."""
    from history.postprocessing.pipeline import coregister_dems, plot_coregistration

    coregister_dems(
        dem_files=list(config.proc_dir.raw_dems_dir.glob("*-DEM.tif")),
        output_dir=config.proc_dir.coreg_dems_dir,
        references_data=config.references_data_mapping,
        overwrite=config.overwrite,
        max_workers=config.max_workers,
    )

    if not config.no_plots:
        plot_coregistration(config.proc_dir.coreg_dems_dir, config.plot_dir, config.max_workers)


def _run_ddem(config: Config) -> None:
    """Compute differential DEMs against the reference, before and after coregistration."""
    from history.postprocessing.pipeline import generate_ddems, plot_ddems

    generate_ddems(
        dem_files=list(config.proc_dir.raw_dems_dir.glob("*-DEM.tif")),
        output_dir=config.proc_dir.before_coreg_ddems_dir,
        references_data=config.references_data_mapping,
        overwrite=config.overwrite,
        max_workers=config.max_workers,
    )

    generate_ddems(
        dem_files=list(config.proc_dir.coreg_dems_dir.glob("*-DEM.tif")),
        output_dir=config.proc_dir.after_coreg_ddems_dir,
        references_data=config.references_data_mapping,
        overwrite=config.overwrite,
        max_workers=config.max_workers,
    )

    if not config.no_plots:
        plot_ddems(
            config.proc_dir.before_coreg_ddems_dir,
            config.proc_dir.after_coreg_ddems_dir,
            config.plot_dir,
            overwrite=config.overwrite,
            max_workers=config.max_workers,
        )


def _run_std_dem(config: Config) -> None:
    """Build one standard-deviation DEM per (site, dataset) group from all coregistered DEMs."""
    from history.postprocessing.pipeline import create_std_dems, plot_std_dems

    create_std_dems(
        dem_files=list(config.proc_dir.coreg_dems_dir.glob("*-DEM.tif")),
        output_dir=config.proc_dir.std_dems_dir,
        overwrite=config.overwrite,
    )

    if not config.no_plots:
        plot_std_dems(config.proc_dir.std_dems_dir, config.plot_dir)


def _run_landcover(config: Config) -> None:
    """Compute and plot landcover-stratified statistics on dDEMs and STD DEMs."""
    from history.postprocessing.pipeline import plot_landcover

    if not config.no_plots:
        plot_landcover(
            config.proc_dir.after_coreg_ddems_dir,
            config.proc_dir.std_dems_dir,
            config.references_data_mapping,
            config.plot_dir,
            config.max_workers,
        )


_STEP_RUNNERS = {
    "uncompress": _run_uncompress,
    "symlinks": _run_symlinks,
    "point2dem": _run_point2dem,
    "coregister": _run_coregister,
    "ddem": _run_ddem,
    "std_dem": _run_std_dem,
    "landcover": _run_landcover,
}


def cmd_run(args: argparse.Namespace) -> None:
    """
    Execute one or all post-processing pipeline steps.

    Loads the config from the TOML file specified by ``--config``, applies any
    CLI flag overrides, then dispatches to the appropriate step runner(s).
    When ``step`` is ``"all"``, every step in ``_STEP_RUNNERS`` is executed in
    insertion order.
    """
    config = _load_config(args)
    step = args.step

    if step == "all":
        for name, runner in _STEP_RUNNERS.items():
            logger.info(f"Running step: {name}")
            runner(config)
    else:
        _STEP_RUNNERS[step](config)


def build_parser() -> argparse.ArgumentParser:
    """
    Build and return the argument parser for the ``history-postprocess`` CLI.

    Returns
    -------
    argparse.ArgumentParser
        Parser with two subcommands: ``create`` and ``run``.
    """
    parser = argparse.ArgumentParser(prog="history-postprocess", description="Postprocessing")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v INFO, -vv DEBUG)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- create ---
    create_parser = subparsers.add_parser("create", help="Initialize a new postprocessing directory")
    create_parser.add_argument("output_dir", help="Directory to create")
    create_parser.set_defaults(func=cmd_create)

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Run one or more postprocessing steps")
    run_parser.add_argument("step", choices=RUN_STEPS, metavar="STEP",
                            help=f"Step to run: {{{', '.join(RUN_STEPS)}}}")
    run_parser.add_argument("--config", required=True, metavar="PATH", help="Path to config.toml")
    run_parser.add_argument("--overwrite", action="store_true", default=False,
                            help="Force overwrite of existing outputs (overrides config)")
    run_parser.add_argument("--dry-run", action="store_true", default=False, dest="dry_run",
                            help="Print actions without executing them (overrides config)")
    run_parser.add_argument("--no-plots", action="store_true", default=False, dest="no_plots",
                            help="Skip plot generation for this step")
    run_parser.add_argument("--max-workers", type=int, default=None, metavar="N", dest="max_workers",
                            help="Number of parallel workers (overrides config)")
    run_parser.set_defaults(func=cmd_run)

    return parser


def main() -> None:
    """Entry point for the ``history-postprocess`` command."""
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
