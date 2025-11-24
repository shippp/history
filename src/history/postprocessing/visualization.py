"""
Contains functions to generate post-processing Visualization
"""

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LightSource
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from tqdm import tqdm

from history.postprocessing.io import parse_filename

#######################################################################################################################
##                                                  MOSAIC VISUALIZATION
#######################################################################################################################


def generate_all_mosaics(df: pd.DataFrame, output_dir: str | Path, max_workers: int | None = None) -> None:
    """
    Generate all mosaic plots (DEMs, dDEMs, slopes, and hillshades) for each
    unique (site, dataset) combination in the input DataFrame.

    This function dispatches the mosaic generation tasks to multiple processes
    using a ProcessPoolExecutor. For each group of DEM-related files, it runs
    the corresponding mosaic generation functions (DEM mosaic, dDEM mosaic,
    slope mosaic, and hillshade mosaic) and stores the results in a structured
    output directory. A tqdm progress bar is displayed to track processing
    progress.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing DEM and dDEM file paths as well as `site`
        and `dataset` columns used to group the data.
    output_dir : str or Path
        Directory where all mosaic images will be saved. Subdirectories are
        automatically created for each (site, dataset) pair.
    max_workers : int, optional
        Maximum number of processes to use. If None, the default value of
        ProcessPoolExecutor is used.

    Notes
    -----
    - The function handles all submissions asynchronously and waits for every
      task to complete.
    - If a mosaic generation task raises an exception, the error is caught and
      reported, but the processing of other mosaics continues.
    - Output filenames are automatically derived from the column names.
    """
    output_dir = Path(output_dir)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for (site, dataset), group in df.groupby(["site", "dataset"]):
            group_dir = output_dir / f"{site}_{dataset}" / "mosaic"
            title_prefix = f"({site} - {dataset})"

            for colname in ["raw_dem_file", "coreg_dem_file"]:
                futures.append(
                    executor.submit(
                        generate_dems_mosaic,
                        group,
                        group_dir / f"mosaic_{colname[:-5]}.png",
                        colname,
                        title=f"{title_prefix} Mosaic {colname}",
                    )
                )

            for colname in ["ddem_before_file", "ddem_after_file"]:
                futures.append(
                    executor.submit(
                        generate_ddems_mosaic,
                        group,
                        group_dir / f"mosaic_{colname[:-5]}_coreg.png",
                        colname,
                        title=f"{title_prefix} Mosaic {colname}",
                    )
                )

                futures.append(
                    executor.submit(
                        generate_slopes_mosaic,
                        group,
                        group_dir / f"mosaic_slopes_{colname[:-5]}_coreg.png",
                        colname,
                        title=f"{title_prefix} Mosaic slopes {colname}",
                    )
                )
                futures.append(
                    executor.submit(
                        generate_hillshades_mosaic,
                        group,
                        group_dir / f"mosaic_hillshades_{colname[:-5]}_coreg.png",
                        colname,
                        title=f"{title_prefix} Mosaic hillshades {colname}",
                    )
                )

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating mosaics"):
            try:
                fut.result()
            except Exception as e:
                print(f"Error of plot generating : {e}")


def generate_dems_mosaic(
    df: pd.DataFrame, output_path: str | Path, colname: str = "raw_dem_file", title: str = ""
) -> None:
    df_dropped = df.dropna(subset=colname)

    vmin = df_dropped[colname.replace("file", "min")].median()
    vmax = df_dropped[colname.replace("file", "max")].median()
    with _generate_mosaic_figure_and_axes(len(df_dropped), output_path) as (fig, axes):
        for i, (code, row) in enumerate(df_dropped.iterrows()):
            dem = _read_raster_with_max_size(row[colname])
            axes[i].imshow(dem, cmap="terrain", vmin=vmin, vmax=vmax)
            axes[i].set_title(code)

        # add the global color bar
        cbar = fig.colorbar(
            ScalarMappable(cmap="terrain", norm=plt.Normalize(vmin=vmin, vmax=vmax)),
            ax=axes,
            orientation="vertical",
        )
        cbar.set_label("Altitude (m)")
        fig.suptitle(title, fontsize=16)


def generate_ddems_mosaic(
    df: pd.DataFrame,
    output_path: str | Path,
    colname: str = "ddem_before_file",
    vmin: float = -10,
    vmax: float = 10,
    title: str = "",
) -> None:
    df_dropped = df.dropna(subset=colname)

    with _generate_mosaic_figure_and_axes(len(df_dropped), output_path) as (fig, axes):
        for i, (code, row) in enumerate(df_dropped.iterrows()):
            dem = _read_raster_with_max_size(row[colname])
            dem = np.clip(dem, vmin, vmax)

            axes[i].imshow(dem, cmap="coolwarm", vmin=vmin, vmax=vmax)
            nmad = row[colname.replace("file", "nmad")]
            axes[i].set_title(f"{code}\nNMAD:{nmad:.2f}")

        # add the global color bar
        cbar = fig.colorbar(
            ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=vmin, vmax=vmax)),
            ax=axes,
            orientation="vertical",
        )
        cbar.set_label("Altitude difference (m)")
        fig.suptitle(title, fontsize=16)


def generate_slopes_mosaic(
    df: pd.DataFrame,
    output_path: str | Path,
    colname: str = "ddem_before_file",
    vmin: float = 0,
    vmax: float = 15,
    title: str = "",
) -> None:
    df_dropped = df.dropna(subset=colname)

    with _generate_mosaic_figure_and_axes(len(df_dropped), output_path) as (fig, axes):
        for i, (code, row) in enumerate(df_dropped.iterrows()):
            dem = _read_raster_with_max_size(row[colname])

            with rasterio.open(row[colname]) as src:
                dx, dy = src.res

            # compute slope in degree
            grad_y, grad_x = np.gradient(dem, dy, dx)
            # Ignore runtime warnings only inside this block
            with np.errstate(invalid="ignore", divide="ignore"):
                slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
            slope_dem = np.degrees(slope_rad)
            slope_dem = np.clip(slope_dem, vmin, vmax)

            axes[i].imshow(slope_dem, cmap="terrain", vmin=vmin, vmax=vmax)
            axes[i].axis("off")
            axes[i].set_title(code)

        # add the global color bar
        cbar = fig.colorbar(
            ScalarMappable(cmap="terrain", norm=plt.Normalize(vmin=vmin, vmax=vmax)), ax=axes, orientation="vertical"
        )
        cbar.set_label("Slope (Degree)")

        fig.suptitle(title, fontsize=16)


def generate_hillshades_mosaic(
    df: pd.DataFrame,
    output_path: str | Path,
    colname: str = "ddem_before_file",
    vmin: float = 0,
    vmax: float = 1,
    title: str = "",
) -> None:
    df_dropped = df.dropna(subset=colname)

    with _generate_mosaic_figure_and_axes(len(df_dropped), output_path) as (fig, axes):
        for i, (code, row) in enumerate(df_dropped.iterrows()):
            dem = _read_raster_with_max_size(row[colname])

            with rasterio.open(row[colname]) as src:
                dx, dy = src.res

            ls = LightSource(azdeg=315, altdeg=45)  # azimuth, sun altitude
            hillshade = ls.hillshade(dem, vert_exag=1, dx=dx, dy=dy)
            clean = np.asarray(hillshade).copy()

            axes[i].imshow(hillshade, cmap="gray", vmin=0, vmax=np.nanpercentile(clean, 99))
            axes[i].axis("off")
            axes[i].set_title(code)

        # add the global color bar
        cbar = fig.colorbar(
            ScalarMappable(cmap="gray", norm=plt.Normalize(vmin=vmin, vmax=vmax)),
            ax=axes,
            orientation="vertical",
        )
        cbar.set_label("Hillshade")
        fig.suptitle(title, fontsize=16)


def generate_std_dem_plots(dem_path: str | Path, output_path: str | Path) -> None:
    """
    Generate and save a heatmap of the elevation standard deviation from a DEM.

    The function reads the DEM raster, computes the standard deviation values, and
    creates a plot using a 'viridis' colormap. The color scale is capped at the 90th
    percentile to reduce the effect of extreme values. The figure is saved to `output_path`.

    Parameters
    ----------
    dem_path : str | Path
        Path to the DEM raster file.
    output_path : str | Path
        File path where the generated plot will be saved. Parent directories are created if needed.

    Returns
    -------
    None
        Saves the plot to `output_path` without returning a value.
    """
    # create the output directory if needed
    dem_path = Path(dem_path)

    std_dem = _read_raster_with_max_size(dem_path)

    # create the plot and save them at output_plot_file
    vmax = np.nanquantile(std_dem, 0.9)
    fig = Figure()
    ax = fig.subplots(1, 1)
    im = ax.imshow(std_dem, cmap="viridis", vmax=vmax)
    ax.set_title(f"{dem_path.stem}")
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Elevation standard deviation (m)")

    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path)


#######################################################################################################################
##                                                  STATISTICS VISUALIZATION
#######################################################################################################################


def barplot_var(global_df: pd.DataFrame, output_path: str | Path, colname: str, title: str = "") -> None:
    """
    Generate a grouped bar plot for a specified column in a DataFrame and save it to a file.

    The function groups the data by `site` and `dataset`, sorts values within each group,
    assigns distinct colors per group, and creates a vertical bar plot with proper labels,
    title, and legend. The resulting figure is saved to `output_path`.

    Parameters
    ----------
    global_df : pd.DataFrame
        DataFrame containing the data. Must have columns `site`, `dataset`, and the target `colname`.
    output_path : str | Path
        File path where the generated plot will be saved. Parent directories are created if needed.
    colname : str
        Name of the column in `global_df` to plot as bar heights.
    title : str, optional
        Title for the plot (default is empty).

    Returns
    -------
    None
        Saves the plot to `output_path` without returning a value.

    Notes
    -----
    - Each unique combination of `site` and `dataset` is treated as a separate color group.
    - The x-axis labels correspond to the DataFrame index, rotated for readability.
    """
    df = global_df.dropna(subset=[colname]).copy(True)

    # Créer la colonne groupe
    df["group"] = df["site"] + "_" + df["dataset"]

    # Trier par groupe puis point_count
    df_sorted = df.sort_values(["group", colname], ascending=[True, True])

    # Couleurs par groupe
    unique_groups = df_sorted["group"].unique()
    color_map = {g: f"C{i}" for i, g in enumerate(unique_groups)}

    # Plot
    fig = Figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)

    bars = []
    labels = []
    x = range(len(df_sorted))

    # iterate on numpy arrays (fast and robust)
    values = df_sorted[colname].to_numpy()
    groups = df_sorted["group"].to_numpy()

    for i, (val, grp) in enumerate(zip(values, groups)):
        bar = ax.bar(i, val, color=color_map[grp])
        # add one legend entry per group (use the Rectangle, not the BarContainer)
        if grp not in labels:
            bars.append(bar[0])
            labels.append(grp)

    # Mise en forme
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted.index, rotation=90, ha="right")
    ax.set_ylabel(colname)
    ax.set_title(title)
    ax.legend(bars, labels, title="Groupes")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path)


def generate_plot_coreg_shifts(df: pd.DataFrame, output_path: str | Path, title: str = "") -> None:
    """
    Generate a bar plot of coregistration shifts (X, Y, Z) from a DataFrame and save it to a file.

    The function drops rows with missing shift values, sorts them by the mean absolute shift,
    and creates a vertical bar plot with values labeled on top of each bar. The resulting
    figure is saved to `output_path`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the columns "coreg_shift_x", "coreg_shift_y", and "coreg_shift_z".
    output_path : str | Path
        File path where the generated plot will be saved. Parent directories are created if needed.
    title : str, optional
        Title of the plot (default is empty).

    Returns
    -------
    None
        Saves the plot to `output_path` without returning a value.

    Notes
    -----
    - Rows with NaN values in any of the shift columns are ignored.
    - Shifts are displayed in meters with labels above each bar.
    """
    colnames = ["coreg_shift_x", "coreg_shift_y", "coreg_shift_z"]
    dropped_df = (
        df.dropna(subset=colnames)
        .assign(mean_abs_shift=df[colnames].abs().mean(axis=1))
        .sort_values(by="mean_abs_shift")
        .drop(columns="mean_abs_shift")
    )
    if len(dropped_df) == 0:
        return

    fig = Figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    dropped_df[colnames].plot(kind="bar", ax=ax)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", label_type="edge", padding=3)
    ax.set_ylabel("Coregistration shifts (meters)")

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()

    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path)


def generate_plot_nmad_before_vs_after(df: pd.DataFrame, output_path: str | Path, title: str = "") -> None:
    """
    Generate a bar plot comparing NMAD values before and after DEM coregistration.

    The function filters rows with valid 'ddem_before_nmad' and 'ddem_after_nmad' values,
    sorts them by the after-coregistration NMAD, and plots both sets of values side by side.
    Bars are annotated with their numeric values. The figure is saved to `output_path`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the columns 'ddem_before_nmad' and 'ddem_after_nmad'.
    output_path : str | Path
        Path where the generated plot will be saved. Parent directories are created if needed.
    title : str, optional
        Title of the plot (default is an empty string).

    Returns
    -------
    None
        The plot is saved to the specified path without returning a value.
    """
    colnames = ["ddem_before_nmad", "ddem_after_nmad"]
    dropped_df = df.dropna(subset=colnames).sort_values(by="ddem_after_nmad")
    if len(dropped_df) == 0:
        return

    fig = Figure(figsize=(12, 7))
    ax = fig.add_subplot(1, 1, 1)
    dropped_df[colnames].plot(kind="bar", ax=ax)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", padding=3)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()

    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path)


#######################################################################################################################
##                                                  STATISTICS LANDCOVER VISUALIZATION
#######################################################################################################################


def generate_landcover_grouped_boxplot(landcover_df: pd.DataFrame, output_path: str | Path, title: str = "") -> None:
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


def generate_landcover_boxplot(landcover_df: pd.DataFrame, output_path: str | Path) -> None:
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


def generate_landcover_nmad(landcover_df: pd.DataFrame, output_path: str | Path, title: str = "") -> None:
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


def generate_landcover_grouped_boxplot_from_std_dems(std_landcover_df: pd.DataFrame, output_path: str | Path) -> None:
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
##                                                  OTHER VISUALIZATION
#######################################################################################################################


def visualize_files_presence_map(directories: list[str | Path]) -> None:
    """
    Create a visual presence/absence map of files across multiple directories.

    This function scans each provided directory, parses filenames to extract a unique
    'code', and marks whether a file exists for that code in each directory. The resulting
    boolean matrix is then visualized as a heatmap, where True indicates file presence
    and False indicates absence.

    Parameters
    ----------
    directories : list[str | Path]
        List of directories to scan for files. Only files with parsable codes are considered.

    Returns
    -------
    None
        The function generates a plot showing the presence/absence map; it does not return a value.
    """
    directories: list[Path] = [Path(d) for d in directories]
    df = pd.DataFrame()
    df.index.name = "code"

    for directory in directories:
        if directory.is_dir():
            for file in directory.iterdir():
                if file.is_file():
                    code, _ = parse_filename(file)

                    df.at[code, directory.name] = True

    df = df.astype(pd.BooleanDtype()).fillna(False).sort_index()
    _plot_boolean_df(df, cell_height=0.2)


def generate_coregistration_individual_plots(
    df: pd.DataFrame, output_directory: str | Path, overwrite: bool = False, vmin: float = -10, vmax: float = 10
) -> None:
    """
    Generate side-by-side plots of dDEMs before and after coregistration for each code.

    For each row in the DataFrame containing paths to 'ddem_before_file' and 'ddem_after_file',
    this function reads the rasters, clips values to [vmin, vmax], and creates a two-panel
    image showing the dDEM before and after coregistration with summary statistics (mean, median, NMAD)
    in the titles. Each plot is saved as a PNG in the specified output directory.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'ddem_before_file' and 'ddem_after_file' columns, plus statistics.
    output_directory : str | Path
        Directory where the plots will be saved. Created if it does not exist.
    overwrite : bool, optional
        If True, existing plots will be overwritten. Default is False.
    vmin : float, optional
        Minimum value for color clipping in the plots. Default is -10.
    vmax : float, optional
        Maximum value for color clipping in the plots. Default is 10.

    Returns
    -------
    None
        Saves individual PNG plots for each code; does not return anything.
    """
    # create the output directory if needed
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    dropped_df = df.dropna(subset=["ddem_before_file", "ddem_after_file"])

    for code, row in dropped_df.iterrows():
        output_path = output_directory / f"{code}.png"

        if not output_path.exists() or overwrite:
            # open the raw dDEM and the coregistered dDEM
            ddem_before = _read_raster_with_max_size(row["ddem_before_file"])
            ddem_after = _read_raster_with_max_size(row["ddem_after_file"])

            ddem_before = np.clip(ddem_before, vmin, vmax)
            ddem_after = np.clip(ddem_after, vmin, vmax)

            # create the figure
            fig = Figure(figsize=(10, 5), constrained_layout=True)
            axes = fig.subplots(1, 2)

            # add the dDEMs and their titles
            axes[0].imshow(ddem_before, cmap="coolwarm", vmin=vmin, vmax=vmax)
            axes[0].axis("off")
            axes[0].set_title(
                f"dDEM before coregistration \n(mean: {row['ddem_before_mean']:.3f}, med: {row['ddem_before_median']:.3f}, nmad: {row['ddem_before_nmad']:.3f})"
            )

            axes[1].imshow(ddem_after, cmap="coolwarm", vmin=vmin, vmax=vmax)
            axes[1].axis("off")
            axes[1].set_title(
                f"dDEM after coregistration \n(mean: {row['ddem_after_mean']:.3f}, med: {row['ddem_after_median']:.3f}, nmad: {row['ddem_after_nmad']:.3f})"
            )

            # add a global color bar
            cbar = fig.colorbar(
                ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=vmin, vmax=vmax)),
                ax=axes,
                orientation="vertical",
                fraction=0.03,
                pad=0.02,
            )
            cbar.set_label("Altitude difference(m)")

            fig.savefig(output_path)


#######################################################################################################################
##                                                  PRIVATE FUNCTIONS
#######################################################################################################################


@contextmanager
def _generate_mosaic_figure_and_axes(
    n: int, output_path: str | Path
) -> Generator[tuple[Figure, list[Axes]], None, None]:
    try:
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
        fig = Figure(figsize=(4 * ncols, 4 * nrows), constrained_layout=True)
        axes = fig.subplots(nrows, ncols)
        axes = list(axes.flatten()) if isinstance(axes, np.ndarray) else [axes]

        for i in range(ncols * nrows):
            axes[i].axis("off")

        yield fig, axes
    finally:
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(output_path)
        del fig


def _plot_boolean_df(
    df: pd.DataFrame,
    title: str = "Boolean Matrix",
    output_path: str | None = None,
    show: bool = True,
    cell_width: float = 0.6,
    cell_height: float = 0.4,
    min_width: float = 6,
    min_height: float = 4,
) -> None:
    """
    Plot a boolean DataFrame as a black/white matrix using pcolormesh,
    with automatic figure size based on the DataFrame shape.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing only boolean values.
    title : str, optional
        Title of the plot.
    output_path : str or None, optional
        If provided, the plot is saved to the given file path.
    show : bool, optional
        Whether to display the plot.
    cell_width : float, optional
        Width (in inches) allocated per column.
    cell_height : float, optional
        Height (in inches) allocated per row.
    min_width : float, optional
        Minimum figure width in inches.
    min_height : float, optional
        Minimum figure height in inches.
    """
    # Convert boolean DataFrame to integer matrix (1=True, 0=False)
    matrix = df.astype(int).values

    # Compute figure size based on DataFrame shape
    n_rows, n_cols = df.shape
    fig_width = max(min_width, n_cols * cell_width)
    fig_height = max(min_height, n_rows * cell_height)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Use a binary colormap for black/white representation
    cmap = plt.get_cmap("binary")
    ax.pcolormesh(matrix, cmap=cmap, edgecolors="grey", linewidth=1, shading="auto")

    # Set tick positions
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_xticklabels(df.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(df.index, fontsize=9)

    # Add visible grid
    ax.set_xticks(np.arange(n_cols), minor=True)
    ax.set_yticks(np.arange(n_rows), minor=True)
    ax.grid(which="minor", color="grey", linestyle="-", linewidth=0.8, alpha=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Title and layout
    ax.set_title(title, fontsize=14, weight="bold")
    fig.tight_layout()

    if output_path:
        plt.savefig(output_path)

    if show:
        plt.show()
    else:
        plt.close()


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


def _read_raster_with_max_size(file: str, maxsize: int = 2000):
    with rasterio.open(file) as src:
        h, w = src.height, src.width
        max_dim = max(h, w)
        reduction_factor = math.ceil(max_dim / maxsize)

        new_height = h // reduction_factor
        new_width = w // reduction_factor

        dem = src.read(
            1,
            out_shape=(new_height, new_width),
            resampling=Resampling.nearest,
            masked=True,
        )

    return dem
