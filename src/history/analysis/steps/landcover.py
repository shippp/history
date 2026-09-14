"""
Landcover step: computes landcover-stratified statistics on dDEMs and STD DEMs,
and generates the associated plots.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import geoutils as gu
import numpy as np
import pandas as pd
import rasterio
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from tqdm import tqdm

from history.config import Config, ReferencesConfig
from history.postprocessing.io import FILE_CODE_MAPPING, is_output_up_to_date, parse_filename

logger = logging.getLogger(__name__)

# code -> label
LANDCOVER_MAPPING = {
    10: "tree cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built-up",
    60: "bare / sparse vegetation",
    70: "snow and ice",
    80: "permanent water bodies",
    90: "herbaceous wetland",
    95: "mangroves",
    100: "moss and lichen",
}
# label -> code
LANDCOVER_MAPPING_INV = {v: k for k, v in LANDCOVER_MAPPING.items()}


#######################################################################################################################
##                                                  STATISTICS
#######################################################################################################################


def get_raster_statistics_by_landcover(
    dem_file: str | Path, metadata_key: str = "landcover_stats"
) -> list[dict[str, Any]] | None:
    """
    Retrieves precomputed landcover-based statistics from a raster's metadata.

    Args:
        dem_file (str | Path): Path to the raster file containing the statistics in its metadata.
        metadata_key (str, optional): Metadata tag name storing the statistics. Defaults to "landcover_stats".

    Returns:
        list[dict[str, Any]] | None: A list of dictionaries with statistics for each landcover class
        if the metadata exists; otherwise, returns None.
    """

    with rasterio.open(dem_file, "r") as src:
        tags = src.tags(1)
        if metadata_key in tags:
            # Already present → return parsed JSON
            return json.loads(tags[metadata_key])
        else:
            return None


def compute_raster_statistics_by_landcover(
    raster_file: str | Path,
    landcover_file: str | Path,
    metadata_key: str = "landcover_stats",
) -> list[dict[str, Any]]:
    """
    Computes statistics of a raster dataset grouped by landcover classes.

    This function calculates descriptive statistics (mean, median, quartiles, NMAD, min, max, std)
    for each landcover class present in a landcover raster, considering only valid (unmasked) pixels
    overlapping with the input raster. The results are stored as metadata in the input raster.

    Args:
        raster_file (str | Path): Path to the input raster file to analyze.
        landcover_file (str | Path): Path to the landcover raster file.
        metadata_key (str, optional): Metadata tag name to store the computed statistics.
            Defaults to "landcover_stats".

    Returns:
        list[dict[str, Any]]: A list of dictionaries containing statistics for each landcover class.
    """

    # open the first raster
    raster = gu.Raster(raster_file)

    # open the landcover reprojected on the first raster
    # here we use nearest resampling to preserve class
    landcover = gu.Raster(landcover_file).reproject(raster, resampling="nearest", silent=True)

    # Combine masks to keep only valid pixels in both arrays
    combined_mask = (~raster.data.mask) & (~landcover.data.mask)

    # remove masked values
    raster_valid = raster.data.data[combined_mask]
    landcover_valid = landcover.data.data[combined_mask]

    stats = []
    for c in np.unique(landcover_valid):
        values = raster_valid[landcover_valid == c]
        if len(values) == 0:
            continue
        percent = (len(values) / len(raster_valid)) * 100
        stats.append(
            {
                "landcover_class": int(c),
                "landcover_label": LANDCOVER_MAPPING.get(int(c), "unknown"),
                "count": len(values),
                "percent": percent,
                "mean": float(np.nanmean(values)),
                "median": float(np.nanmedian(values)),
                "q1": float(np.nanpercentile(values, 25)),
                "q3": float(np.nanpercentile(values, 75)),
                "nmad": float(gu.stats.nmad(values)),
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
                "std": float(np.nanstd(values)),
            }
        )
    # Write JSON to raster metadata
    with rasterio.open(raster_file, "r+") as src:
        src.update_tags(1, **{metadata_key: json.dumps(stats)})
    return stats


def compute_landcover_statistics(
    dem_files: Iterable[str | Path], references_data: ReferencesConfig, max_workers: int | None = None
) -> pd.DataFrame:
    """
    Compute landcover-based raster statistics for a collection of DEM files.

    This function processes a list of DEM file paths and ensures that each file has
    corresponding landcover statistics. If statistics already exist (retrieved via
    `get_raster_statistics_by_landcover`), they are reused. Otherwise, the function
    retrieves the appropriate landcover raster from `references_data` and computes
    the statistics in parallel using a thread pool.

    The workflow is as follows:
        1. Parse each DEM filename to extract metadata (code, site, dataset).
        2. Check whether landcover statistics already exist for the DEM.
        3. If not, queue the DEM for computation using the matching landcover file.
        4. Compute missing statistics in parallel.
        5. Aggregate all results into a flat pandas DataFrame.

    Parameters
    ----------
    dem_files : Iterable[str | Path]
        List of DEM file paths to process.
    references_data : ReferencesConfig
        Object providing access to reference datasets, in particular landcover rasters.
    max_workers : int, optional
        Maximum number of worker threads to use for parallel computation.
        If None, the default number of workers is chosen by `ThreadPoolExecutor`.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row contains the DEM identifier (code, site, dataset)
        along with the computed landcover statistics.

    Notes
    -----
    - Errors encountered during file parsing or computation are logged and skipped.
    - The resulting DataFrame is in a flattened “records” format for easy analysis.
    """
    dem_files: list[Path] = [Path(f) for f in dem_files]
    args_dict = {}
    stats_dict = {}

    for file in dem_files:
        try:
            code, metadatas = parse_filename(file)

            # extract site and dataset from the filename
            site, dataset = metadatas["site"], metadatas["dataset"]

            # create a key with this infos
            key = (code, site, dataset)

            # get existing landcover statistics
            stats = get_raster_statistics_by_landcover(file)

            if stats is None:
                # get the corresponding landcover file
                landcover_file = references_data.get_landcover(site, dataset)

                args_dict[key] = (file, landcover_file)
            else:
                stats_dict[key] = stats

        except Exception as e:
            tqdm.write(f"[ERROR] Error while processing {file} : {e}")
            continue

    if args_dict:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(compute_raster_statistics_by_landcover, file, landcover_file): key
                for key, (file, landcover_file) in args_dict.items()
            }

            for fut in tqdm(as_completed(futures), desc="Landcover statistics", total=len(futures)):
                key = futures[fut]
                try:
                    stats_dict[key] = fut.result()
                except Exception as e:
                    tqdm.write(f"[ERROR] Error while computing landcover statistics for {key[0]} : {e}")

    # flatten the stats_dict to convert it into a df an return it
    records = []
    for (code, site, dataset), stats in stats_dict.items():
        records += [{"code": code, "site": site, "dataset": dataset, **elem} for elem in stats]
    return pd.DataFrame(records)


def compute_landcover_statistics_on_std_dems(
    dem_files: Iterable[str | Path], references_data: ReferencesConfig, max_workers: int | None = None
) -> pd.DataFrame:
    """
    Compute landcover-based statistics for a collection of standardized DEM (STD DEM) files.

    This function processes a set of STD DEM file paths and ensures that each file has
    associated landcover statistics. For each DEM file, the function attempts to detect
    the corresponding site and dataset directly from the filename using the
    `FILE_CODE_MAPPING`. If statistics already exist (via `get_raster_statistics_by_landcover`),
    they are reused. Otherwise, the matching landcover raster is retrieved from
    `references_data`, and the missing statistics are computed in parallel.

    The workflow is as follows:
        1. Detect site and dataset identifiers from each DEM filename.
        2. Check if landcover statistics already exist for the DEM.
        3. If missing, schedule computation using the corresponding landcover raster.
        4. Perform computations in parallel using a thread pool.
        5. Aggregate all computed statistics into a flattened pandas DataFrame.

    Parameters
    ----------
    dem_files : Iterable[str | Path]
        Iterable of paths to standardized DEM files to process.
    references_data : ReferencesConfig
        Object providing access to reference datasets, including landcover rasters.
    max_workers : int, optional
        Maximum number of worker threads to use during parallel computation.
        If None, the default value used by `ThreadPoolExecutor` is applied.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row corresponds to a landcover statistics record,
        including the STD DEM file path, site, dataset, and statistical values.

    Notes
    -----
    - Files for which the site or dataset cannot be identified are skipped with an error message.
    - All errors during computation are logged and do not interrupt the rest of the process.
    - The returned DataFrame is in a flattened “records” format suited for analysis or export.
    """
    dem_files: list[Path] = [Path(f) for f in dem_files]
    stats_dict = {}
    args_dict: dict[tuple[Path, str, str], tuple[Path, Path]] = {}

    for file in dem_files:
        site = next((s for s in FILE_CODE_MAPPING["site"].values() if s in file.name), None)
        dataset = next((d for d in FILE_CODE_MAPPING["dataset"].values() if d in file.name), None)

        if site is None or dataset is None:
            tqdm.write(f"[ERROR] Can't determine the site, dataset of this file : {file}.")
            continue

        key = (file, site, dataset)
        stats = get_raster_statistics_by_landcover(file)
        if stats is None:
            landcover_file = references_data.get_landcover(site, dataset)
            args_dict[key] = (file, landcover_file)
        else:
            stats_dict[key] = stats

    if args_dict:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(compute_raster_statistics_by_landcover, file, landcover_file): key
                for key, (file, landcover_file) in args_dict.items()
            }
            for fut in tqdm(as_completed(futures), desc="Landcover Statistics (STD DEM)", total=len(futures)):
                key = futures[fut]
                try:
                    stats_dict[key] = fut.result()
                except Exception as e:
                    tqdm.write(f"[ERROR] Error while computing landcover statistics for {key[0]} : {e}")
                    continue

    records = []
    for (std_dem_file, site, dataset), stats in stats_dict.items():
        records += [{"std_dem_file": str(std_dem_file), "site": site, "dataset": dataset, **elem} for elem in stats]
    return pd.DataFrame(records)


#######################################################################################################################
##                                                  VISUALIZATION
#######################################################################################################################


def _plot_grouped_boxplot(df: pd.DataFrame, category_col: str, hue_col: str, y_label: str = "", title: str = ""):
    """
    Create a grouped boxplot showing median and quartile statistics per category and hue group.
    Thread-safe version (no pyplot).

    Args:
        df (pd.DataFrame): Input DataFrame with columns:
            - category_col: categorical x-axis variable
            - hue_col: subgroup variable defining box colors
            - 'median', 'q1', 'q3'
        category_col (str): Column name defining x-axis categories
        hue_col (str): Column name defining subgroups (hues)
        y_label (str): Y-axis label
        title (str): Plot title

    Returns:
        matplotlib.figure.Figure: The generated figure (not displayed or saved)
    """
    category_labels = df[category_col].unique()
    hues = df[hue_col].unique()
    n_category = len(category_labels)
    n_hue = len(hues)
    width = 0.8 / n_hue
    x_base = np.arange(n_category)

    fig = Figure(figsize=(12, 6))
    ax = fig.subplots(1, 1)

    color_map = {hue: f"C{i}" for i, hue in enumerate(hues)}  # assign colors

    for i, hue in enumerate(hues):
        positions = x_base - 0.4 + i * width + width / 2
        box_data = []
        for lc_label in category_labels:
            lc_group = df[(df[category_col] == lc_label) & (df[hue_col] == hue)]
            if lc_group.empty:
                box_data.append({"med": 0, "q1": 0, "q3": 0, "whislo": 0, "whishi": 0, "fliers": []})
            else:
                box_data.append(
                    {
                        "med": lc_group["median"].mean(),
                        "q1": lc_group["q1"].mean(),
                        "q3": lc_group["q3"].mean(),
                        "whislo": None,
                        "whishi": None,
                        "fliers": [],
                    }
                )
        ax.bxp(
            box_data,
            positions=positions,
            widths=width,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(facecolor=color_map[hue]),
            medianprops=dict(color="black"),
        )

    # X-axis category labels
    ax.set_xticks(x_base)
    ax.set_xticklabels(category_labels, rotation=45, ha="right")

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Create custom legend
    legend_handles = [Patch(facecolor=color_map[hue], label=str(hue)) for hue in hues]
    ax.legend(handles=legend_handles, title=hue_col, bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.tight_layout()
    return fig


def generate_landcover_grouped_boxplot(
    landcover_df: pd.DataFrame, output_path: str | Path, title: str = "", overwrite: bool = False, inputs=None
) -> None:
    """
    Generate a grouped boxplot of NMAD (or other elevation differences) per landcover class.

    The function orders codes by their mean NMAD, appends the mean percentage of each landcover
    class to the labels, and creates a grouped boxplot using these labels. The resulting plot is
    saved to `output_path`.

    Parameters
    ----------
    landcover_df : pd.DataFrame
        DataFrame containing at least the columns 'code', 'landcover_label', 'nmad', and 'percent'.
    output_path : str | Path
        Path where the generated plot will be saved; parent directories are created if needed.
    title : str, optional
        Title of the plot (default is an empty string).

    Returns
    -------
    None
        The plot is saved to the specified path without returning a value.
    """
    if not overwrite:
        check = is_output_up_to_date(inputs, output_path) if inputs is not None else Path(output_path).exists()
        if check:
            logger.debug(f"File {output_path} is up to date -> skipping.")
            return

    # order group with the mean of nmad
    code_order = landcover_df.groupby("code")["nmad"].mean().sort_values().index
    ordered_df = landcover_df.set_index("code").loc[code_order].reset_index()

    percent_means = ordered_df.groupby("landcover_label")["percent"].transform("mean")
    ordered_df["landcover_label"] = (
        ordered_df["landcover_label"].astype(str) + " (" + percent_means.round(2).astype(str) + " %)"
    )

    fig = _plot_grouped_boxplot(
        ordered_df,
        "landcover_label",
        "code",
        y_label="Altitude difference (m)",
        title=title,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)


def generate_landcover_boxplot(
    landcover_df: pd.DataFrame, output_path: str | Path, overwrite: bool = False, inputs=None
) -> None:
    """
    Generate a boxplot of mean elevation statistics grouped by landcover class.

    For each landcover class, the function computes the mean of median, Q1, and Q3 values
    across all rasters, then creates a boxplot with class labels annotated by the mean
    percentage of pixels. The plot is saved to `output_path`.

    Parameters
    ----------
    landcover_df : pd.DataFrame
        DataFrame containing at least the columns 'landcover_label', 'median', 'q1', 'q3', and 'percent'.
    output_path : str | Path
        Path where the generated plot will be saved; parent directories are created if needed.

    Returns
    -------
    None
        The plot is saved to the specified path without returning a value.
    """
    if not overwrite:
        check = is_output_up_to_date(inputs, output_path) if inputs is not None else Path(output_path).exists()
        if check:
            logger.debug(f"File {output_path} is up to date -> skipping.")
            return

    box_data = []
    labels = []
    for lc_label, lc_group in landcover_df.groupby("landcover_label"):
        box_data.append(
            {
                "med": lc_group["median"].mean(),
                "q1": lc_group["q1"].mean(),
                "q3": lc_group["q3"].mean(),
                "whislo": None,
                "whishi": None,
                "fliers": [],
            }
        )
        labels.append(f"{lc_label} ({lc_group['percent'].mean():.2f}%)")
    fig = Figure(figsize=(10, 6))
    ax = fig.subplots(1, 1)
    ax.bxp(box_data, showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Altitude difference (m)")
    ax.set_title(f"Boxplot of mean values from {len(landcover_df['code'].unique())} raster(s) by landcover class")
    fig.tight_layout()

    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path)


def generate_landcover_nmad(
    landcover_df: pd.DataFrame, output_path: str | Path, title: str = "", overwrite: bool = False, inputs=None
) -> None:
    """
    Generate a grouped barplot of NMAD values for each raster, separated by landcover class.

    Each landcover class is represented on the x-axis, with bars showing the NMAD value
    per raster code. Class labels are annotated with the mean percentage of pixels.
    The plot is styled with a grid and legend and saved to the specified output path.

    Parameters
    ----------
    landcover_df : pd.DataFrame
        DataFrame containing columns 'landcover_label', 'code', 'nmad', and 'percent'.
    output_path : str | Path
        File path where the plot will be saved; parent directories are created if needed.
    title : str, optional
        Title for the plot. Default is an empty string.

    Returns
    -------
    None
        The plot is saved to the specified path.
    """
    if not overwrite:
        check = is_output_up_to_date(inputs, output_path) if inputs is not None else Path(output_path).exists()
        if check:
            logger.debug(f"File {output_path} is up to date -> skipping.")
            return

    df_plot = landcover_df.pivot_table(
        index="landcover_label",  # x axe
        columns="code",  # one color per bar
        values="nmad",  # values to show
        aggfunc="mean",  # aff func in case
    )
    # Compute mean percent per landcover_label
    percent_means = landcover_df.groupby("landcover_label")["percent"].mean()

    # Replace index labels with label + mean percent
    df_plot.index = [f"{label} ({percent_means[label]:.2f}%)" for label in df_plot.index]

    fig = Figure(figsize=(12, 7))
    ax = fig.subplots(1, 1)

    # plot the grouped barplot
    df_plot.plot(kind="bar", ax=ax)

    # Style and labels
    ax.set_ylabel("NMAD")
    ax.set_xticklabels(df_plot.index, rotation=45, ha="right")
    ax.legend(title="Code", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()

    # Save
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path)


def generate_landcover_grouped_boxplot_from_std_dems(
    std_landcover_df: pd.DataFrame, output_path: str | Path, overwrite: bool = False, inputs=None
) -> None:
    """
    Generate and save a grouped boxplot of altitude standard deviations (STD)
    by landcover class across dataset–site combinations.

    This function:
    - Combines 'dataset' and 'site' columns into a single 'dataset_site' identifier.
    - Sorts dataset–site groups by their mean NMAD values for consistent ordering.
    - Appends the mean percentage of each landcover class to its label for clarity.
    - Plots a grouped boxplot using `_plot_grouped_boxplot()` to visualize
      the distribution of altitude STD values per landcover class and dataset–site group.
    - Saves the resulting figure to the specified output path.

    Args:
        landcover_df (pd.DataFrame): DataFrame containing landcover and DEM statistics.
            Required columns:
                - 'dataset': Dataset identifier.
                - 'site': Site identifier.
                - 'landcover_label': Name of the landcover class.
                - 'percent': Percentage of the landcover class area.
                - 'nmad': NMAD (Normalized Median Absolute Deviation) or STD metric.
        output_path (str | Path): File path where the generated boxplot image will be saved.

    Returns:
        None

    Notes:
        - The function relies on `_plot_grouped_boxplot()` for the visualization.
        - The y-axis represents altitude standard deviation in meters.
        - The landcover labels include mean percentage values for readability.
    """
    if not overwrite:
        check = is_output_up_to_date(inputs, output_path) if inputs is not None else Path(output_path).exists()
        if check:
            logger.debug(f"File {output_path} is up to date -> skipping.")
            return

    df = std_landcover_df.copy()

    # first group the dataset + site
    df["dataset_site"] = df["dataset"].astype(str) + " " + df["site"].astype(str)

    # next order the df with nmad mean per group of dataset + site
    order = df.groupby("dataset_site")["nmad"].mean().sort_values().index
    df = df.set_index("dataset_site").loc[order].reset_index()

    # add the percent to each landcover_label
    percent_means = df.groupby("landcover_label")["percent"].transform("mean")
    df["landcover_label"] = df["landcover_label"].astype(str) + " (" + percent_means.round(2).astype(str) + " %)"

    # plot the bgrouped boxplot
    fig = _plot_grouped_boxplot(
        df,
        "landcover_label",
        "dataset_site",
        y_label="Altitude STD (m)",
        title="Boxplot of altitude STD by landcover class for each dataset + site groups",
    )
    fig.savefig(output_path)


#######################################################################################################################
##                                                  RUNNER
#######################################################################################################################


def run_landcover(config: Config) -> None:
    """Compute and plot landcover-stratified statistics on dDEMs and STD DEMs."""
    if config.no_plots:
        logger.info("Step `landcover` finished")
        return

    after_coreg_ddems_dir = config.proc_dir.after_coreg_ddems_dir
    std_dems_dir = config.proc_dir.std_dems_dir

    all_ddem_files = list(after_coreg_ddems_dir.glob("*-DDEM.tif"))
    ddem_files_by_group: dict[tuple, list[Path]] = {}
    for f in all_ddem_files:
        try:
            _, meta = parse_filename(f)
            key = (meta["site"], meta["dataset"])
            ddem_files_by_group.setdefault(key, []).append(f)
        except Exception:
            pass

    landcover_df = compute_landcover_statistics(all_ddem_files, config.references_data_mapping, config.max_workers)
    std_lc_df = compute_landcover_statistics_on_std_dems(std_dems_dir.glob("*.tif"), config.references_data_mapping, config.max_workers)

    for (site, dataset), group in landcover_df.groupby(["site", "dataset"]):
        sub_dir = config.plot_dir / f"{site}_{dataset}"
        group_inputs = ddem_files_by_group.get((site, dataset))
        generate_landcover_grouped_boxplot(group, sub_dir / "landcover_grouped_boxplot.png", f"({site} {dataset}) Boxplot of Altitude difference with ref DEM by code/landcover",
                                            overwrite=config.overwrite_plots, inputs=group_inputs)
        generate_landcover_nmad(group, sub_dir / "landcover_nmad.png", f"({site} {dataset}) NMAD of Altitude difference with ref DEM by code/landcover",
                                 overwrite=config.overwrite_plots, inputs=group_inputs)

    std_dem_files = list(std_dems_dir.glob("*.tif"))
    generate_landcover_grouped_boxplot_from_std_dems(std_lc_df, config.plot_dir / "landcover_boxplot_from_std_dems.png",
                                                      overwrite=config.overwrite_plots, inputs=std_dem_files)

    logger.info("Step `landcover` finished")
