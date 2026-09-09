"""
Command-line interface for the post-processing pipeline.

Two subcommands are available:

``history-postprocess create <output_dir>``
    Scaffold a new working directory by copying the config template into it.
    Edit the generated ``config.toml`` to point to your data before running
    any pipeline step.

``history-postprocess run <STEP> --config <path/to/config.toml>``
    Execute one or all pipeline steps in order. Available steps:

    uncompress     Extract compressed submission archives into the extracted dir.
    symlinks       Index submissions, parse filenames, and create typed symlinks.
    check_planned  Check extracted submissions against the planned submissions sheet.
    sparse_viz     Generate sparse point cloud mosaics colored by elevation
                   difference with the reference DEM.
    provided_dem     Plot a mosaic of user-provided DEMs against the reference DEM,
                   grouped by (site, dataset).
    point2dem      Convert dense point clouds to DEMs via PDAL; integrate any
                   user-provided DEMs by reprojecting them on the reference grid.
    coregister     Coregister raw DEMs to the reference using Nuth–Kaab + vertical
                   shift.
    ddem           Compute differential DEMs before and after coregistration.
    std_dem        Build one standard-deviation DEM per (site, dataset) group from
                   all coregistered DEMs.
    landcover      Compute and plot landcover-stratified statistics on dDEMs and
                   STD DEMs.
    generate_pdf   Assemble all pipeline output plots into a single PDF report.
    all            Run all steps in the order listed above.

``history-postprocess status --config <path/to/config.toml>``
    Print a quick file-count overview of every processing directory, to see at
    a glance how far the pipeline has progressed.

Verbosity is controlled with ``-v`` (INFO) or ``-vv`` (DEBUG).
"""

import dataclasses
import logging
import shutil
import sys
from pathlib import Path

import click

from history.postprocessing.config import Config
from history.postprocessing.pipeline import generate_sparse_pointcloud_viz, report_symlinks

logger = logging.getLogger(__name__)

_TEMPLATE_CONFIG = Path(__file__).parent / "config.exemple.toml"

RUN_STEPS = ["uncompress", "symlinks", "check_planned", "sparse_viz", "provided_dem", "point2dem", "coregister", "ddem", "std_dem", "landcover", "generate_pdf", "all"]

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


@click.group()
def cli() -> None:
    """Postprocessing."""


@cli.command("create")
@click.argument("output_dir", type=click.Path(path_type=Path))
def cmd_create(output_dir: Path) -> None:
    """Initialize a new postprocessing directory.

    Creates OUTPUT_DIR and copies the config template into it as
    ``config.toml``. The user must then edit that file to set the correct
    paths before running any pipeline step.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    dest = output_dir / "config.toml"
    shutil.copy(_TEMPLATE_CONFIG, dest)

    print(f"Created '{output_dir}'")
    print(f"Config template copied to '{dest}'")
    print("Edit config.toml to point to your data before running the pipeline.")


def _count(directory: Path, pattern: str = "*", kind: str = "file") -> int:
    """Count entries of *kind* ('file' or 'dir') matching *pattern* in *directory*, or 0 if it doesn't exist."""
    if not directory.exists():
        return 0
    is_match = Path.is_file if kind == "file" else Path.is_dir
    return sum(1 for p in directory.glob(pattern) if is_match(p))


@cli.command("status")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Path to config.toml")
def cmd_status(config_path: Path) -> None:
    """Print a quick file-count overview of the processing directory tree."""
    config = Config.from_toml_file(config_path)
    proc_dir = config.proc_dir

    rows = [
        ("raw archives", config.raw_dir, "*", "file"),
        ("extracted submissions", config.extracted_dir, "*", "dir"),
        ("symlinks/dense_pointclouds", proc_dir.symlinks_dir / "dense_pointclouds", "*", "file"),
        ("symlinks/sparse_pointclouds", proc_dir.symlinks_dir / "sparse_pointclouds", "*", "file"),
        ("symlinks/extrinsics", proc_dir.symlinks_dir / "extrinsics", "*", "file"),
        ("symlinks/intrinsics", proc_dir.symlinks_dir / "intrinsics", "*", "file"),
        ("symlinks/dems", proc_dir.symlinks_dir / "dems", "*", "file"),
        ("raw_dems", proc_dir.raw_dems_dir, "*-DEM.tif", "file"),
        ("coregistered_dems", proc_dir.coreg_dems_dir, "*-DEM.tif", "file"),
        ("ddems/before_coregistration", proc_dir.before_coreg_ddems_dir, "*-DDEM.tif", "file"),
        ("ddems/after_coregistration", proc_dir.after_coreg_ddems_dir, "*-DDEM.tif", "file"),
        ("std_dems", proc_dir.std_dems_dir, "*.tif", "file"),
        ("plots", config.plot_dir, "**/*.png", "file"),
    ]

    label_width = max(len(label) for label, _, _, _ in rows)
    print(f"Status for '{config_path}'")
    for label, directory, pattern, kind in rows:
        count = _count(directory, pattern, kind)
        print(f"  {label.ljust(label_width)} : {count}")


def _run_uncompress(config: Config) -> None:
    """Extract all compressed submission archives into the extracted directory."""
    from history.postprocessing.pipeline import uncompress_all_submissions, cleanup_orphaned_extracted

    cleanup_orphaned_extracted(config.raw_dir, config.extracted_dir)
    uncompress_all_submissions(
        config.raw_dir,
        config.extracted_dir,
        overwrite=config.overwrite,
        max_workers=config.max_workers,
    )


def _run_symlinks(config: Config) -> None:
    """Index submissions, parse filenames, and create typed symlink directories."""
    from history.postprocessing import io
    from history.postprocessing.pipeline import create_symlinks, plot_symlinks

    df = io.scan_submissions(config.extracted_dir, filename_renames=config.filename_renames)
    io.validate_submissions(df)
    create_symlinks(df, config.proc_dir.symlinks_dir, overwrite=True)  # force overwriting the symlinks to avoid issues
    report_symlinks(config)

    if not config.no_plots:
        plot_symlinks(config, submissions_df=df)


def _run_check_planned(config: Config) -> None:
    """Check extracted results against planned submissions sheet."""
    from history.postprocessing import io
    from history.postprocessing.pipeline import check_planned_submissions, plot_planned_submissions

    df = io.scan_submissions(config.extracted_dir, filename_renames=config.filename_renames)
    planned_outfile = config.proc_dir.base_dir / "planned_submissions.csv"
    check_planned_submissions(df.index.tolist(), planned_outfile)

    if not config.no_plots:
        plot_planned_submissions(config, planned_outfile)

        
def _run_sparse_viz(config: Config) -> None:
    """Generate viz for sparse point cloud"""
    generate_sparse_pointcloud_viz(config)


def _run_provided_dem(config: Config) -> None:
    """Create visualisation for provided DEMs"""
    from history.postprocessing.pipeline import generate_provided_dem_viz
    generate_provided_dem_viz(config)

def _run_point2dem(config: Config) -> None:
    """Convert dense point clouds to DEMs via PDAL, and integrate any user-provided DEMs."""
    from history.postprocessing.pipeline import process_pointclouds_to_dems, add_provided_dems, plot_point2dem, cleanup_orphaned_raw_dems
    from history.utils import log_to_file

    cleanup_orphaned_raw_dems(config.proc_dir.raw_dems_dir, config.proc_dir.symlinks_dir)

    dense_pc_dir = config.proc_dir.symlinks_dir / "dense_pointclouds"
    pointcloud_files = list(dense_pc_dir.glob("*.las")) + list(dense_pc_dir.glob("*.laz"))

    logs_dir = config.proc_dir.raw_dems_dir / "logs"
    with log_to_file(logs_dir, logging.getLogger("history.postprocessing")) as log_path:
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

    logger.info(f"point2dem log saved at {log_path}")

    if not config.no_plots:
        logger.info("Plotting raw DEMs mosaics")
        plot_point2dem(config)


def _run_coregister(config: Config) -> None:
    """Coregister raw DEMs to the reference using Nuth–Kaab + vertical shift."""
    from history.postprocessing.pipeline import coregister_dems, plot_coregistration, cleanup_orphaned_coreg_dems

    cleanup_orphaned_coreg_dems(config.proc_dir.coreg_dems_dir, config.proc_dir.raw_dems_dir)

    coregister_dems(
        dem_files=list(config.proc_dir.raw_dems_dir.glob("*-DEM.tif")),
        output_dir=config.proc_dir.coreg_dems_dir,
        references_data=config.references_data_mapping,
        overwrite=config.overwrite,
        max_workers=config.max_workers,
    )

    if not config.no_plots:
        logger.info("Plotting coregistered DEMs mosaics and figures")
        plot_coregistration(config)


def _run_ddem(config: Config) -> None:
    """Compute differential DEMs against the reference, before and after coregistration."""
    from history.postprocessing.pipeline import generate_ddems, plot_ddems, cleanup_orphaned_ddems

    cleanup_orphaned_ddems(config.proc_dir.before_coreg_ddems_dir, config.proc_dir.raw_dems_dir)
    cleanup_orphaned_ddems(config.proc_dir.after_coreg_ddems_dir, config.proc_dir.coreg_dems_dir)

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
        logger.info("Plotting dDEMs mosaics")
        plot_ddems(config)


def _run_std_dem(config: Config) -> None:
    """Build one standard-deviation DEM per (site, dataset) group from all coregistered DEMs."""
    from history.postprocessing.pipeline import create_std_dems, plot_std_dems

    create_std_dems(
        dem_files=list(config.proc_dir.coreg_dems_dir.glob("*-DEM.tif")),
        output_dir=config.proc_dir.std_dems_dir,
        overwrite=config.overwrite,
    )

    if not config.no_plots:
        plot_std_dems(config)


def _run_landcover(config: Config) -> None:
    """Compute and plot landcover-stratified statistics on dDEMs and STD DEMs."""
    from history.postprocessing.pipeline import plot_landcover

    if not config.no_plots:
        plot_landcover(config)

    logger.info("Step `landcover` finished")


def _run_generate_pdf(config: Config) -> None:
    """Assemble all pipeline output PNGs into a single PDF report."""
    from history.postprocessing.pdf_report import generate_pdf_report

    generate_pdf_report(
        extracted_dir=config.extracted_dir,
        plot_dir=config.plot_dir,
        filename_renames=config.filename_renames,
        orientation=config.pdf_orientation,
        overwrite=config.overwrite,
    )


_STEP_RUNNERS = {
    "uncompress": _run_uncompress,
    "symlinks": _run_symlinks,
    "check_planned": _run_check_planned,
    "sparse_viz": _run_sparse_viz,
    "provided_dem": _run_provided_dem,
    "point2dem": _run_point2dem,
    "coregister": _run_coregister,
    "ddem": _run_ddem,
    "std_dem": _run_std_dem,
    "landcover": _run_landcover,
    "generate_pdf": _run_generate_pdf,
}


@cli.command("run")
@click.argument("step", type=click.Choice(RUN_STEPS))
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Path to config.toml")
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
    """Run one or more postprocessing steps.

    STEP is one of: uncompress, symlinks, check_planned, sparse_viz, provided_dem, point2dem,
    coregister, ddem, std_dem, landcover, generate_pdf, all.
    """
    _configure_logging(verbose)
    config = _load_config(config_path, overwrite, overwrite_plots, dry_run, no_plots, max_workers)

    if step == "all":
        for name, runner in _STEP_RUNNERS.items():
            logger.info(f"Running step: {name}")
            runner(config)
            logger.info(f"Step `{name}` finished")

    else:
        _STEP_RUNNERS[step](config)


def main() -> None:
    """Entry point for the ``history-postprocess`` command."""
    cli()


if __name__ == "__main__":
    main()
