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
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(verbosity, logging.DEBUG)
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S", stream=sys.stderr)
    logging.getLogger("history").setLevel(level)


def _load_config(args: argparse.Namespace):
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dest = output_dir / "config.toml"
    shutil.copy(_TEMPLATE_CONFIG, dest)

    print(f"Created '{output_dir}'")
    print(f"Config template copied to '{dest}'")
    print("Edit config.toml to point to your data before running the pipeline.")


def _run_uncompress(config: Config) -> None:
    from history.postprocessing.pipeline import uncompress_all_submissions

    uncompress_all_submissions(
        config.raw_dir,
        config.extracted_dir,
        overwrite=config.overwrite,
        max_workers=config.max_workers,
    )


def _run_symlinks(config: Config) -> None:
    from history.postprocessing.pipeline import index_submissions_and_link_files

    index_submissions_and_link_files(
        config.extracted_dir,
        config.proc_dir.symlinks_dir,
        overwrite=config.overwrite,
    )

    if not config.no_plots:
        from history.postprocessing.statistics import compute_pcs_statistics_df
        from history.postprocessing.visualization import barplot_var

        pointcloud_files = list((config.proc_dir.symlinks_dir / "dense_pointclouds").iterdir())
        df = compute_pcs_statistics_df(pointcloud_files)
        barplot_var(df, config.plot_dir / "pointcloud_point_count.png", "point_count", "Point count in dense point-cloud file")


def _run_point2dem(config: Config) -> None:
    from history.postprocessing.pipeline import process_pointclouds_to_dems, add_provided_dems

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
        from history.postprocessing.statistics import compute_dems_statistics_df
        from history.postprocessing.visualization import barplot_var, generate_dems_mosaic

        df = compute_dems_statistics_df(config.proc_dir.raw_dems_dir.glob("*-DEM.tif"), max_workers=config.max_workers)
        barplot_var(df, config.plot_dir / "raw_dem_voids.png", "percent_nodata", "Raw DEM nodata percent")
        for (site, dataset), group in df.groupby(["site", "dataset"]):
            output_path = config.plot_dir / f"{site}_{dataset}" / "mosaic" / "mosaic_raw_dem.png"
            vmin, vmax = group["min"].median(), group["max"].median()
            generate_dems_mosaic(group["file"].to_dict(), output_path, vmin, vmax, f"({site} {dataset}) Mosaic Raw DEMs")


def _run_coregister(config: Config) -> None:
    from history.postprocessing.pipeline import coregister_dems

    dem_files = list(config.proc_dir.raw_dems_dir.glob("*-DEM.tif"))
    coregister_dems(
        dem_files=dem_files,
        output_dir=config.proc_dir.coreg_dems_dir,
        references_data=config.references_data_mapping,
        overwrite=config.overwrite,
        max_workers=config.max_workers,
    )

    if not config.no_plots:
        from history.postprocessing.statistics import compute_dems_statistics_df, get_coregistration_statistics_df
        from history.postprocessing.visualization import generate_dems_mosaic, generate_plot_coreg_shifts

        df = compute_dems_statistics_df(config.proc_dir.coreg_dems_dir.glob("*-DEM.tif"), max_workers=config.max_workers)
        for (site, dataset), group in df.groupby(["site", "dataset"]):
            output_path = config.plot_dir / f"{site}_{dataset}" / "mosaic" / "mosaic_coreg_dem.png"
            vmin, vmax = group["min"].median(), group["max"].median()
            generate_dems_mosaic(group["file"].to_dict(), output_path, vmin, vmax, f"({site} {dataset}) Mosaic Coregistered DEMs")

        df_shifts = get_coregistration_statistics_df(config.proc_dir.coreg_dems_dir.glob("*-DEM.tif"))
        for (site, dataset), group in df_shifts.groupby(["site", "dataset"]):
            output_path = config.plot_dir / f"{site}_{dataset}" / "coregistration_shifts.png"
            generate_plot_coreg_shifts(group, output_path, f"({site} {dataset}) Coregistration shifts")


def _run_ddem(config: Config) -> None:
    from history.postprocessing.pipeline import generate_ddems

    raw_dem_files = list(config.proc_dir.raw_dems_dir.glob("*-DEM.tif"))
    generate_ddems(
        dem_files=raw_dem_files,
        output_dir=config.proc_dir.before_coreg_ddems_dir,
        references_data=config.references_data_mapping,
        overwrite=config.overwrite,
        max_workers=config.max_workers,
    )

    coreg_dem_files = list(config.proc_dir.coreg_dems_dir.glob("*-DEM.tif"))
    generate_ddems(
        dem_files=coreg_dem_files,
        output_dir=config.proc_dir.after_coreg_ddems_dir,
        references_data=config.references_data_mapping,
        overwrite=config.overwrite,
        max_workers=config.max_workers,
    )

    if not config.no_plots:
        import pandas as pd
        from history.postprocessing.statistics import compute_dems_statistics_df
        from history.postprocessing.visualization import (
            barplot_var,
            generate_coregistration_individual_plots,
            generate_ddems_mosaic,
            generate_hillshades_mosaic,
            generate_plot_nmad_before_vs_after,
            generate_slopes_mosaic,
        )

        ddem_before_df = compute_dems_statistics_df(
            config.proc_dir.before_coreg_ddems_dir.glob("*-DDEM.tif"), "ddem_before_", config.max_workers
        )
        ddem_after_df = compute_dems_statistics_df(
            config.proc_dir.after_coreg_ddems_dir.glob("*-DDEM.tif"), "ddem_after_", config.max_workers
        )
        df = pd.concat([ddem_before_df, ddem_after_df]).groupby(level=0).first()

        barplot_var(
            df,
            config.plot_dir / "nmad_after_coregistration.png",
            "ddem_after_nmad",
            "NMAD of Altitude differences with ref DEM after coregistration by code",
        )

        for (site, dataset), group in df.groupby(["site", "dataset"]):
            sub_dir = config.plot_dir / f"{site}_{dataset}"
            generate_plot_nmad_before_vs_after(
                group,
                sub_dir / "nmad_before_vs_after_coregistration.png",
                f"({site} {dataset}) NMAD of DEM differences before vs after coregistration",
            )
            generate_coregistration_individual_plots(group, sub_dir / "coregistrations", config.overwrite)

            ddem_files_dict = group["ddem_after_file"].dropna().to_dict()
            generate_ddems_mosaic(ddem_files_dict, sub_dir / "mosaic" / "mosaic_ddem.png", f"({site} {dataset}) Mosaic of DDEMs after coregistration")
            generate_slopes_mosaic(ddem_files_dict, sub_dir / "mosaic" / "mosaic_slopes_ddem.png", f"({site} {dataset}) Mosaic slopes of DDEMs after coregistration")
            generate_hillshades_mosaic(ddem_files_dict, sub_dir / "mosaic" / "mosaic_hillshades_ddem.png", f"({site} {dataset}) Mosaic hillshades of DDEMs after coregistration")


def _run_std_dem(config: Config) -> None:
    from history.postprocessing.pipeline import create_std_dem
    from history.postprocessing.io import parse_filename

    coreg_dem_files = list(config.proc_dir.coreg_dems_dir.glob("*-DEM.tif"))

    groups: dict[tuple[str, str], list[Path]] = {}
    for file in coreg_dem_files:
        try:
            _, metadatas = parse_filename(file)
            key = (metadatas["site"], metadatas["dataset"])
            groups.setdefault(key, []).append(file)
        except ValueError:
            logger.warning(f"Cannot parse filename for std_dem grouping: {file.name}")

    config.proc_dir.std_dems_dir.mkdir(exist_ok=True, parents=True)
    for (site, dataset), dem_files in groups.items():
        output_path = config.proc_dir.std_dems_dir / f"{site}_{dataset}_std_dem.tif"
        create_std_dem(
            dem_files=dem_files,
            output_path=output_path,
            overwrite=config.overwrite,
        )

    if not config.no_plots:
        from history.postprocessing.visualization import generate_std_dem_plots

        for file in config.proc_dir.std_dems_dir.glob("*.tif"):
            subdir = file.stem.replace("_std_dem", "")
            output_path = config.plot_dir / subdir / file.with_suffix(".png").name
            generate_std_dem_plots(file, output_path)


def _run_landcover(config: Config) -> None:
    from history.postprocessing.statistics import compute_landcover_statistics, compute_landcover_statistics_on_std_dems

    landcover_df = compute_landcover_statistics(
        config.proc_dir.after_coreg_ddems_dir.glob("*-DDEM.tif"),
        config.references_data_mapping,
        config.max_workers,
    )
    std_lc_df = compute_landcover_statistics_on_std_dems(
        config.proc_dir.std_dems_dir.glob("*.tif"),
        config.references_data_mapping,
        config.max_workers,
    )

    if not config.no_plots:
        from history.postprocessing.visualization import (
            generate_landcover_grouped_boxplot,
            generate_landcover_grouped_boxplot_from_std_dems,
            generate_landcover_nmad,
        )

        for (site, dataset), group in landcover_df.groupby(["site", "dataset"]):
            sub_dir = config.plot_dir / f"{site}_{dataset}"
            generate_landcover_grouped_boxplot(
                group,
                sub_dir / "landcover_grouped_boxplot.png",
                f"({site} {dataset}) Boxplot of Altitude difference with ref DEM by code/landcover",
            )
            generate_landcover_nmad(
                group,
                sub_dir / "landcover_nmad.png",
                f"({site} {dataset}) NMAD of Altitude difference with ref DEM by code/landcover",
            )

        generate_landcover_grouped_boxplot_from_std_dems(std_lc_df, config.plot_dir / "landcover_boxplot_from_std_dems.png")


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
    config = _load_config(args)
    step = args.step

    if step == "all":
        for name, runner in _STEP_RUNNERS.items():
            logger.info(f"Running step: {name}")
            runner(config)
    else:
        _STEP_RUNNERS[step](config)


def build_parser() -> argparse.ArgumentParser:
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


def main():
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
