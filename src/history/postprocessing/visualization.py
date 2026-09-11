"""
Contains functions to generate post-processing Visualization
"""

import math
from contextlib import contextmanager
import logging
from pathlib import Path
from typing import Generator, Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LightSource
from matplotlib.figure import Figure
from rasterio.enums import Resampling

from history.postprocessing.io import is_output_up_to_date, parse_filename

logger = logging.getLogger(__name__)

#######################################################################################################################
##                                                  MOSAIC VISUALIZATION
#######################################################################################################################


def generate_dems_mosaic(
    dem_files_dict: dict[str, list[str | Path]],
    output_path: str | Path,
    vmin: float,
    vmax: float,
    title: str = "",
    overwrite: bool = False,
) -> None:
    """
    Generate a mosaic figure composed of multiple DEM images.

    This function creates a multi-panel visualization where each subplot displays
    a DEM (Digital Elevation Model). The DEMs are provided as a dictionary in which
    each key corresponds to a subplot title, and each value is a file path (or list
    of paths) to the DEM data. A global colorbar is added for consistent elevation
    scaling across all panels.

    Parameters
    ----------
    dem_files_dict : dict[str, list[str | Path]]
        A dictionary mapping subplot titles to one or multiple raster file paths
        representing the DEMs to display.
    output_path : str | Path
        Path where the generated mosaic figure will be saved. The parent
        directories must already exist or be creatable.
    vmin : float
        Minimum elevation value used to normalize the DEM color scaling.
    vmax : float
        Maximum elevation value used to normalize the DEM color scaling.
    title : str, optional
        A global title displayed above the entire mosaic figure. Default is an
        empty string.
    overwrite : bool, optional
        Set to True to force overwriting existing output. Default is False.

    Returns
    -------
    None
        The function saves the generated figure to ``output_path`` and does not
        return any value.
    """
    if not overwrite and is_output_up_to_date(list(dem_files_dict.values()), output_path):
        logger.debug(f"File {output_path} is up to date -> skipping.")
        return

    with _generate_mosaic_figure_and_axes(len(dem_files_dict), output_path) as (fig, axes):
        for i, (subtitle, file) in enumerate(sorted(dem_files_dict.items())):
            dem = _read_raster_with_max_size(file)

            axes[i].imshow(dem, cmap="terrain", vmin=vmin, vmax=vmax)
            axes[i].set_title(subtitle)

        # add the global color bar
        cbar = fig.colorbar(
            ScalarMappable(cmap="terrain", norm=plt.Normalize(vmin=vmin, vmax=vmax)),
            ax=axes,
            orientation="vertical",
        )
        cbar.set_label("Altitude (m)")
        fig.suptitle(title, fontsize=16)


def generate_ddems_mosaic(
    ddem_files_dict: dict[str, list[str | Path]],
    output_path: str | Path,
    vmin: float = -10,
    vmax: float = 10,
    title: str = "",
    overwrite: bool = False,
) -> None:
    """
    Generate a mosaic of dDEM (differential DEM) rasters and save it as an image.

    This function creates a multi-panel figure where each subplot displays one
    dDEM raster provided in the input dictionary. Values are clipped to the
    specified range before visualization, and all panels share a common
    colorbar using the *coolwarm* colormap. The output mosaic is written to
    the path indicated by `output_path`.

    Parameters
    ----------
    ddem_files_dict : dict[str, list[str | Path]]
        A dictionary mapping subplot titles to one or more raster file paths
        representing the dDEM to display.
    output_path : str | Path
        Path where the final mosaic figure will be saved.
    vmin : float, optional
        Minimum value for clipping and colormap normalization. Default is -10.
    vmax : float, optional
        Maximum value for clipping and colormap normalization. Default is 10.
    title : str, optional
        Global title for the mosaic figure. Default is an empty string.
    overwrite : bool, optional
        Set to True to force overwriting existing output. Default is False.

    Returns
    -------
    None
        The function saves the generated mosaic to `output_path` and does not
        return any value.
    """
    if not overwrite and is_output_up_to_date(list(ddem_files_dict.values()), output_path):
        logger.debug(f"File {output_path} is up to date -> skipping.")
        return

    with _generate_mosaic_figure_and_axes(len(ddem_files_dict), output_path) as (fig, axes):
        for i, (subtitle, file) in enumerate(sorted(ddem_files_dict.items())):
            try:
                dem = _read_raster_with_max_size(file)
                dem = np.clip(dem, vmin, vmax)

                axes[i].imshow(dem, cmap="coolwarm", vmin=vmin, vmax=vmax)
                axes[i].set_title(subtitle)
            except Exception as e:
                logger.error(f"Issue plotting file {file}: {e}")
                continue

        # add the global color bar
        cbar = fig.colorbar(
            ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=vmin, vmax=vmax)),
            ax=axes,
            orientation="vertical",
        )
        cbar.set_label("Altitude difference (m)")
        fig.suptitle(title, fontsize=16)


def generate_slopes_mosaic(
    dem_files_dict: dict[str, list[str | Path]],
    output_path: str | Path,
    vmin: float = 0,
    vmax: float = 40,
    title: str = "",
    overwrite: bool = False,
) -> None:
    """
    Generate a mosaic of slope maps derived from elevation rasters.

    This function computes slope (in degrees) for each raster provided in the
    input dictionary and arranges the resulting maps into a multi-panel mosaic.
    Slopes are calculated using the gradient of the DEM, converted to degrees,
    clipped to the specified range, and visualized using the *terrain* colormap.
    All panels share a global vertical colorbar, and the final mosaic is saved
    to the specified output path.

    Parameters
    ----------
    dem_files_dict : dict[str, list[str | Path]]
        A dictionary mapping subplot titles to one or more elevation raster files
        from which slopes will be computed.
    output_path : str | Path
        Path where the final slope mosaic figure will be saved.
    vmin : float, optional
        Minimum value for clipping and colormap normalization. Default is 0.
    vmax : float, optional
        Maximum value for clipping and colormap normalization. Default is 15.
    title : str, optional
        Global title for the mosaic figure. Default is an empty string.
    overwrite : bool, optional
        Set to True to force overwriting existing output. Default is False.

    Returns
    -------
    None
        The function saves the generated mosaic to `output_path` and does not
        return any value.
    """
    if not overwrite and is_output_up_to_date(list(dem_files_dict.values()), output_path):
        logger.debug(f"File {output_path} is up to date -> skipping.")
        return

    with _generate_mosaic_figure_and_axes(len(dem_files_dict), output_path) as (fig, axes):
        for i, (subtitle, file) in enumerate(sorted(dem_files_dict.items())):
            dem = _read_raster_with_max_size(file)

            with rasterio.open(file) as src:
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
            axes[i].set_title(subtitle)

        # add the global color bar
        cbar = fig.colorbar(
            ScalarMappable(cmap="terrain", norm=plt.Normalize(vmin=vmin, vmax=vmax)), ax=axes, orientation="vertical"
        )
        cbar.set_label("Slope (Degree)")

        fig.suptitle(title, fontsize=16)


def generate_hillshades_mosaic(
    dem_files_dict: dict[str, list[str | Path]],
    output_path: str | Path,
    vmin: float = 0,
    vmax: float = 1,
    title: str = "",
    overwrite: bool = False,
) -> None:
    """
    Generate a mosaic of hillshade visualizations from elevation rasters.

    This function computes hillshades for each raster provided in the input
    dictionary and arranges the resulting shaded-relief images into a multi-panel
    mosaic. Hillshades are generated using a fixed illumination geometry
    (azimuth 315°, altitude 45°), clipped based on the 99th percentile to
    enhance contrast, and displayed using a grayscale colormap. A global
    vertical colorbar is added, and the final figure is saved to the specified
    output path.

    Parameters
    ----------
    dem_files_dict : dict[str, list[str | Path]]
        A dictionary mapping subplot titles to elevation raster file paths
        from which hillshades will be computed.
    output_path : str | Path
        Path where the final mosaic figure will be saved.
    vmin : float, optional
        Minimum value for colormap normalization. Default is 0.
    vmax : float, optional
        Maximum value for colormap normalization. Default is 1.
    title : str, optional
        Global title for the hillshade mosaic. Default is an empty string.
    overwrite : bool, optional
        Set to True to force overwriting existing output. Default is False.

    Returns
    -------
    None
        The function saves the generated hillshade mosaic to `output_path`.
    """
    if not overwrite and is_output_up_to_date(list(dem_files_dict.values()), output_path):
        logger.debug(f"File {output_path} is up to date -> skipping.")
        return

    with _generate_mosaic_figure_and_axes(len(dem_files_dict), output_path) as (fig, axes):
        for i, (subtitle, file) in enumerate(sorted(dem_files_dict.items())):
            dem = _read_raster_with_max_size(file)

            with rasterio.open(file) as src:
                dx, dy = src.res

            ls = LightSource(azdeg=315, altdeg=45)  # azimuth, sun altitude
            hillshade = ls.hillshade(dem, vert_exag=1, dx=dx, dy=dy)
            clean = np.asarray(hillshade).copy()

            axes[i].imshow(hillshade, cmap="gray", vmin=0, vmax=np.nanpercentile(clean, 99))
            axes[i].axis("off")
            axes[i].set_title(subtitle)

        # add the global color bar
        cbar = fig.colorbar(
            ScalarMappable(cmap="gray", norm=plt.Normalize(vmin=vmin, vmax=vmax)),
            ax=axes,
            orientation="vertical",
        )
        cbar.set_label("Hillshade")
        fig.suptitle(title, fontsize=16)


def generate_sparse_pointclouds_mosaic(
    diff_files_dict: dict[str, str | Path],
    output_path: str | Path,
    title: str = "",
    overwrite: bool = False,
    vmin: float = -10,
    vmax: float = 10,
) -> None:
    """
    Generate a mosaic of sparse point clouds, each colored by its elevation difference with a
    reference DEM.

    ``diff_files_dict`` maps each code to a point cloud diff file (as produced by
    ``pipeline.generate_sparsecloud_diff``), whose Z values already hold the elevation
    difference with the reference DEM. One subplot is created per point cloud, computed and
    drawn one at a time (rather than precomputing all point clouds' data upfront) to keep
    memory usage low regardless of how many point clouds are in the mosaic.
    """
    if not overwrite and is_output_up_to_date(list(diff_files_dict.values()), output_path):
        logger.debug(f"File {output_path} is up to date -> skipping.")
        return

    # local import: keeps laspy out of this module's top-level imports.
    import laspy

    logger.info(f"Generating mosaic of sparse point clouds with {len(diff_files_dict)} input files...")

    with _generate_mosaic_figure_and_axes(len(diff_files_dict), output_path) as (fig, axes):
        for i, (code, file) in enumerate(sorted(diff_files_dict.items())):
            try:
                las = laspy.read(file)
                x, y, dz = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
                axes[i].scatter(x, y, c=dz, s=1, cmap="coolwarm", vmin=vmin, vmax=vmax)
                axes[i].set_aspect("equal")
                axes[i].set_title(code)
            except Exception as e:
                logger.error(f"Issue plotting point cloud {file}: {e}")
                continue

        # add the global color bar
        cbar = fig.colorbar(
            ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=vmin, vmax=vmax)),
            ax=axes,
            orientation="vertical",
        )
        cbar.set_label("Elevation difference (m)")
        fig.suptitle(title, fontsize=16)


def generate_provided_dems_mosaic(
    diff_files_dict: dict[str, Path],
    output_path: Path,
    title: str = "",
    overwrite: bool = False,
    vmin: float = -10,
    vmax: float = 10,
) -> None:
    """
    Plot a mosaic of elevation differences between each provided DEM and the reference DEM.

    ``diff_files_dict`` maps each code to a precomputed diff raster (as produced by
    ``pipeline.generate_dem_diff``), so this function only reads and plots them.
    """
    if not overwrite and is_output_up_to_date([*diff_files_dict.values()], output_path):
        logger.debug(f"File {output_path} is up to date -> skipping.")
        return

    import geoutils as gu

    logger.info(f"Generating mosaic of provided DEMs with {len(diff_files_dict)} input files...")

    with _generate_mosaic_figure_and_axes(len(diff_files_dict), output_path) as (fig, axes):
        for i, (code, file) in enumerate(diff_files_dict.items()):
            try:
                ddem = gu.Raster(file)

                axes[i].imshow(ddem.data, cmap="coolwarm", vmin=vmin, vmax=vmax, interpolation="bilinear")

                axes[i].set_aspect("equal")
                axes[i].set_title(code)
            except Exception as e:
                logger.error(f"Issure plotting provided DEM {file} : {e}")
                continue

        # add the global color bar
        cbar = fig.colorbar(
            ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=vmin, vmax=vmax)),
            ax=axes,
            orientation="vertical",
        )
        cbar.set_label("Elevation difference (m)")
        fig.suptitle(title, fontsize=16)


#######################################################################################################################
##                                                  EXTRINSICS VISUALIZATION
#######################################################################################################################


def generate_extrinsics_position_mosaic(
    extrinsics_df: pd.DataFrame,
    initial_extrinsics_df: pd.DataFrame,
    output_path: str | Path,
    title: str = "",
    overwrite: bool = False,
) -> None:
    """
    Generate a mosaic of camera XY positions per submission, relative to each camera's initial position.

    One subplot per submission; each point is a camera plotted at (x_map - initial_x_map,
    y_map - initial_y_map), so the initial (provided) camera position sits at the origin.
    ``extrinsics_df`` and ``initial_extrinsics_df`` are expected to cover a single (site, dataset)
    group; ``initial_extrinsics_df`` is the camera position file provided alongside the raw images
    (e.g. ``camera_model_extrinsics.csv``).
    """
    if not overwrite and Path(output_path).exists():
        logger.debug(f"File {output_path} is up to date -> skipping.")
        return

    merged = _merge_extrinsics_with_initial(extrinsics_df, initial_extrinsics_df)
    if merged.empty:
        logger.warning("No extrinsics data could be matched with initial positions -> skipping plot.")
        return

    codes = sorted(merged["code"].unique())

    with _generate_mosaic_figure_and_axes(len(codes), output_path) as (fig, axes):
        for i, code in enumerate(codes):
            ax = axes[i]
            ax.axis("on")
            group = merged.loc[merged["code"] == code]
            ax.scatter(group["dx"], group["dy"], s=15, alpha=0.7, edgecolor="black", linewidth=0.3)
            ax.axhline(0, color="grey", linewidth=0.8)
            ax.axvline(0, color="grey", linewidth=0.8)
            ax.set_box_aspect(1)
            ax.set_title(code, fontsize=9)

        fig.supxlabel("X (m)")
        fig.supylabel("Y (m)")
        fig.suptitle(title, fontsize=16)


def generate_extrinsics_z_boxplot(
    extrinsics_df: pd.DataFrame,
    initial_extrinsics_df: pd.DataFrame,
    output_path: str | Path,
    title: str = "",
    overwrite: bool = False,
) -> None:
    """
    Generate a boxplot of per-camera altitude shifts (optimized minus initial Z), one box per submission.

    ``extrinsics_df`` and ``initial_extrinsics_df`` are expected to cover a single (site, dataset)
    group, see :func:`generate_extrinsics_position_mosaic`.
    """
    if not overwrite and Path(output_path).exists():
        logger.debug(f"File {output_path} is up to date -> skipping.")
        return

    merged = _merge_extrinsics_with_initial(extrinsics_df, initial_extrinsics_df)
    if merged.empty:
        logger.warning("No extrinsics data could be matched with initial positions -> skipping plot.")
        return

    codes = sorted(merged["code"].unique())
    data = [merged.loc[merged["code"] == code, "dz"].to_numpy() for code in codes]

    fig = Figure(figsize=(max(6, len(codes) * 0.5), 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.boxplot(data)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xticklabels(codes, rotation=90, ha="right")
    ax.set_ylabel("Altitude shift: optimized minus initial (m)")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path)


def generate_intrinsics_boxplots(
    intrinsics_df: pd.DataFrame,
    variables: Sequence[tuple[str, str]],
    output_path: str | Path,
    title: str = "",
    overwrite: bool = False,
) -> None:
    """
    Generate one boxplot per intrinsics variable, side by side in a single figure, one point per code.

    ``intrinsics_df`` is expected to cover a single (site, dataset) group, indexed by submission
    ``code``. ``variables`` is a list of ``(column, ylabel)`` pairs, e.g.
    ``[("focal_length", "Focal length (mm)"), ("pixel_pitch", "Pixel pitch (mm)")]``; each becomes
    its own axis so more variables can be added without changing the plot layout logic.
    """
    if not overwrite and Path(output_path).exists():
        logger.debug(f"File {output_path} is up to date -> skipping.")
        return

    codes = intrinsics_df.index.tolist()
    color_by_code = dict(zip(codes, plt.get_cmap("tab20")(np.linspace(0, 1, len(codes)))))

    fig = Figure(figsize=(2 * len(variables), 8))
    axes = fig.subplots(1, len(variables), squeeze=False)[0]

    for ax, (column, ylabel) in zip(axes, variables):
        values = intrinsics_df[column].dropna()
        if values.empty:
            logger.warning(f"No {column} data found -> skipping subplot.")
            ax.set_axis_off()
            continue
        ax.boxplot(values, showfliers=False)
        for code, value in values.items():
            ax.scatter(1, value, s=40, alpha=0.8, color=color_by_code[code], edgecolor="black", linewidth=0.3, label=code)
        ax.set_xticks([])
        ax.set_ylabel(ylabel)

    handles, labels = {}, []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in handles:
                handles[label] = handle
                labels.append(label)
    fig.legend([handles[label] for label in labels], labels, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path, bbox_inches="tight")


def generate_principal_point_scatter(
    intrinsics_df: pd.DataFrame,
    output_path: str | Path,
    title: str = "",
    overwrite: bool = False,
) -> None:
    """
    Generate a scatter plot of the principal point spread across submissions, one point per code.

    ``intrinsics_df`` is expected to cover a single (site, dataset) group, indexed by
    submission ``code`` with ``principal_point_x_mm``/``principal_point_y_mm`` columns,
    see :func:`generate_extrinsics_z_boxplot`.
    """
    if not overwrite and Path(output_path).exists():
        logger.debug(f"File {output_path} is up to date -> skipping.")
        return

    points = intrinsics_df[["principal_point_x_mm", "principal_point_y_mm"]].dropna()
    if points.empty:
        logger.warning("No principal point data found -> skipping plot.")
        return

    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(points)))

    fig = Figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    for (code, row), color in zip(points.iterrows(), colors):
        ax.scatter(
            row["principal_point_x_mm"],
            row["principal_point_y_mm"],
            s=40,
            alpha=0.8,
            color=color,
            edgecolor="black",
            linewidth=0.3,
            label=code,
        )

    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("Principal point X (mm)")
    ax.set_ylabel("Principal point Y (mm)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path)


#######################################################################################################################
##                                                  STATISTICS VISUALIZATION
#######################################################################################################################


def barplot_var(
    global_df: pd.DataFrame, output_path: str | Path, colname: str, title: str = "", overwrite: bool = False
) -> None:
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
    file_cols = [c for c in global_df.columns if c == "file" or c.endswith("_file")]
    inputs = pd.concat([global_df[c].dropna() for c in file_cols]).values if file_cols else None
    if not overwrite:
        check = (
            is_output_up_to_date(inputs, output_path)
            if inputs is not None and len(inputs) > 0
            else Path(output_path).exists()
        )
        if check:
            logger.debug(f"File {output_path} is up to date -> skipping.")
            return

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


def generate_plot_coreg_shifts(
    df: pd.DataFrame, output_path: str | Path, title: str = "", overwrite: bool = False, inputs=None
) -> None:
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
    if not overwrite:
        check = is_output_up_to_date(inputs, output_path) if inputs is not None else Path(output_path).exists()
        if check:
            logger.debug(f"File {output_path} is up to date -> skipping.")
            return

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


def generate_plot_nmad_before_vs_after(
    df: pd.DataFrame, output_path: str | Path, title: str = "", overwrite: bool = False
) -> None:
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
    if not overwrite:
        file_cols = [c for c in ("ddem_before_file", "ddem_after_file") if c in df.columns]
        inputs = pd.concat([df[c].dropna() for c in file_cols]).values if file_cols else None
        check = (
            is_output_up_to_date(inputs, output_path)
            if inputs is not None and len(inputs) > 0
            else Path(output_path).exists()
        )
        if check:
            logger.debug(f"File {output_path} is up to date -> skipping.")
            return

    colnames = ["ddem_before_nmad", "ddem_after_nmad"]
    dropped_df = df.dropna(subset=colnames).sort_values(by="ddem_after_nmad")
    if len(dropped_df) == 0:
        return

    fig = Figure(figsize=(12, 7))
    ax = fig.add_subplot(1, 1, 1)
    dropped_df[colnames].plot(kind="bar", ax=ax)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", padding=3)
    ax.set_ylabel("DDEM NMAD")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()

    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path)


#######################################################################################################################
##                                                  OTHER VISUALIZATION
#######################################################################################################################


def visualize_files_presence_map(
    directories: list[str | Path], output_path: str | Path | None = None, overwrite: bool = False
) -> None:
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
    output_path : str or Path or None, optional
        If provided the plot is saved there; otherwise it is displayed interactively.
    """
    directories: list[Path] = [Path(d) for d in directories if d.is_dir()]

    if not overwrite and output_path is not None:
        input_files = [f for d in directories for f in d.iterdir() if f.is_file()]
        if is_output_up_to_date(input_files, output_path):
            logger.debug(f"File {output_path} is up to date -> skipping.")
            return
    rows: dict[str, dict] = {}

    for directory in directories:
        for file in directory.iterdir():
            if file.is_file():
                code, metadata = parse_filename(file)
                row = rows.setdefault(code, {"site": metadata["site"], "dataset": metadata["dataset"]})
                row[directory.name] = True

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "code"
    col_labels = [c for c in df.columns if c not in ("site", "dataset")]
    df[col_labels] = df[col_labels].astype(pd.BooleanDtype()).fillna(False)

    group_cols = [c for c in ("site", "dataset") if c in df.columns and df[c].notna().any()]
    groups = (
        [(key, grp[col_labels].sort_index()) for key, grp in df.groupby(group_cols, sort=True)]
        if group_cols
        else [("all", df[col_labels].sort_index())]
    )

    n_groups = len(groups)
    ncols_fig = min(3, n_groups)
    nrows_fig = math.ceil(n_groups / ncols_fig)

    cell_w, cell_h = 1.2, 0.28
    subplot_w = max(3.5, len(col_labels) * cell_w + 1.5)
    subplot_h = max(2.0, max(len(grp) for _, grp in groups) * cell_h + 1.2)
    fig, axes = plt.subplots(
        nrows_fig, ncols_fig, figsize=(ncols_fig * subplot_w + 1.0, nrows_fig * subplot_h + 0.6), squeeze=False
    )
    for ax in axes.ravel():
        ax.axis("off")

    for idx, (key, grp) in enumerate(groups):
        ax = axes[idx // ncols_fig][idx % ncols_fig]
        ax.axis("on")
        _plot_boolean_df(grp, ax=ax, title=" / ".join(key) if isinstance(key, tuple) else str(key))

    fig.suptitle("Submission file presence", fontsize=13, weight="bold", y=1.01)
    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")

    if output_path is None:
        plt.show()
    else:
        plt.close()


def visualize_files_size_map(df: pd.DataFrame, output_path: str | Path | None = None, overwrite: bool = False) -> None:
    """
    Create a visual file-size matrix for submission files.

    Rows are submission codes; columns are the file-type columns recognised by
    ``scan_submissions`` (dense/sparse point clouds, DEM, orthoimage).  Each cell
    is colour-coded by size and annotated with a human-readable label (e.g. "1.2 GB").
    Missing files are shown in light grey with a dash.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by ``scan_submissions``, indexed by submission code.
        Expected columns (subset): ``dense_pointcloud_file``, ``sparse_pointcloud_file``,
        ``dem_file``, ``orthoimage_file``.
    output_path : str or Path or None, optional
        If provided the plot is saved there; otherwise it is displayed interactively.
    """
    if not overwrite and output_path is not None:
        file_cols = [c for c in df.columns if c.endswith("_file")]
        inputs = pd.concat([df[c].dropna() for c in file_cols]).values if file_cols else None
        check = (
            is_output_up_to_date(inputs, output_path)
            if inputs is not None and len(inputs) > 0
            else Path(output_path).exists()
        )
        if check:
            logger.debug(f"File {output_path} is up to date -> skipping.")
            return

    _SIZE_COLS = {
        "dense_pointcloud_file": "dense PC",
        "sparse_pointcloud_file": "sparse PC",
        "dem_file": "DEM",
        "orthoimage_file": "ortho",
    }

    def _fmt_size(n_bytes: float) -> str:
        for unit, threshold in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
            if n_bytes >= threshold:
                return f"{n_bytes / threshold:.1f} {unit}"
        return f"{n_bytes:.0f} B"

    def _fill_size_arrays(group_df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, list[list[str]]]:
        values = np.full((len(group_df), len(cols)), np.nan)
        labels = [["—"] * len(cols) for _ in range(len(group_df))]
        for j, col in enumerate(cols):
            for i, (_, row) in enumerate(group_df.iterrows()):
                path = row.get(col)
                if pd.notna(path) and Path(path).is_file():
                    size = Path(path).stat().st_size
                    values[i, j] = size
                    labels[i][j] = _fmt_size(size)
        return values, labels

    def _draw_matrix(
        ax: plt.Axes, group_df: pd.DataFrame, size_values: np.ndarray, text_labels: list, col_labels: list, title: str
    ) -> None:
        n_rows, n_cols = size_values.shape
        cmap = plt.get_cmap("YlOrRd")
        missing_color = np.array([0.827, 0.827, 0.827, 1.0])  # lightgrey

        # Build an RGBA image where each column is normalised independently.
        rgba = np.ones((n_rows, n_cols, 4))
        for j in range(n_cols):
            col_vals = size_values[:, j]
            valid = col_vals[~np.isnan(col_vals)]
            col_vmin = float(valid.min()) if len(valid) else 0.0
            col_vmax = float(valid.max()) if len(valid) else 1.0
            col_range = col_vmax - col_vmin if col_vmax != col_vmin else 1.0
            for i in range(n_rows):
                if np.isnan(col_vals[i]):
                    rgba[i, j] = missing_color
                else:
                    rgba[i, j] = cmap((col_vals[i] - col_vmin) / col_range)

        ax.imshow(rgba, aspect="auto", origin="lower", extent=(0, n_cols, 0, n_rows))
        # Grid lines
        for x in range(n_cols + 1):
            ax.axvline(x, color="grey", linewidth=0.8)
        for y in range(n_rows + 1):
            ax.axhline(y, color="grey", linewidth=0.8)

        for i in range(n_rows):
            for j in range(n_cols):
                ax.text(j + 0.5, i + 0.5, text_labels[i][j], ha="center", va="center", fontsize=7.5)
        ax.set_xticks(np.arange(n_cols) + 0.5)
        ax.set_yticks(np.arange(n_rows) + 0.5)
        ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(group_df.index, fontsize=8)
        ax.set_title(title, fontsize=10, weight="bold")

    present_cols = [c for c in _SIZE_COLS if c in df.columns]
    col_labels = [_SIZE_COLS[c] for c in present_cols]

    group_cols = [c for c in ("site", "dataset") if c in df.columns]
    if group_cols:
        groups = [(key, grp) for key, grp in df.groupby(group_cols, sort=True)]
    else:
        groups = [("all", df)]

    n_groups = len(groups)
    ncols_fig = min(3, n_groups)
    nrows_fig = math.ceil(n_groups / ncols_fig)

    # Each sub-table: width fixed by number of file columns; height by number of rows.
    cell_w, cell_h = 1.4, 0.30
    subplot_w = max(3.5, len(present_cols) * cell_w + 1.5)
    subplot_h = max(2.0, max(len(grp) for _, grp in groups) * cell_h + 1.2)
    fig_width = ncols_fig * subplot_w + 1.0
    fig_height = nrows_fig * subplot_h + 0.6

    fig, axes = plt.subplots(nrows_fig, ncols_fig, figsize=(fig_width, fig_height), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for idx, (key, grp) in enumerate(groups):
        ax = axes[idx // ncols_fig][idx % ncols_fig]
        ax.axis("on")
        size_values, text_labels = _fill_size_arrays(grp, present_cols)
        title = " / ".join(key) if isinstance(key, tuple) else str(key)
        _draw_matrix(ax, grp, size_values, text_labels, col_labels, title)

    fig.suptitle("Submission file sizes", fontsize=13, weight="bold", y=1.01)
    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")

    if output_path is None:
        plt.show()
    else:
        plt.close()


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

        input_files = [row["ddem_before_file"], row["ddem_after_file"]]
        if not overwrite and is_output_up_to_date(input_files, output_path):
            logger.debug(f"File {output_path} is up to date -> skipping.")
            continue

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
        # enforce a minimum width so the suptitle fits even for a single-panel mosaic
        fig = Figure(figsize=(max(4 * ncols, 6), 4 * nrows), constrained_layout=True)
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
    ax: Axes | None = None,
) -> None:
    """
    Plot a boolean DataFrame as a black/white matrix using pcolormesh,
    with automatic figure size based on the DataFrame shape.
    If *ax* is provided, draws into that axes and skips figure creation/saving.
    """
    matrix = df.astype(int).values
    n_rows, n_cols = df.shape

    if ax is None:
        fig_width = max(min_width, n_cols * cell_width)
        fig_height = max(min_height, n_rows * cell_height)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        standalone = True
    else:
        fig = None
        standalone = False

    cmap = plt.get_cmap("binary")
    ax.pcolormesh(matrix, cmap=cmap, edgecolors="grey", linewidth=1, shading="auto")

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_xticklabels(df.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(df.index, fontsize=9)

    ax.set_xticks(np.arange(n_cols), minor=True)
    ax.set_yticks(np.arange(n_rows), minor=True)
    ax.grid(which="minor", color="grey", linestyle="-", linewidth=0.8, alpha=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title(title, fontsize=10 if not standalone else 14, weight="bold")

    if standalone:
        fig.tight_layout()
        if output_path:
            plt.savefig(output_path)
        if show:
            plt.show()
        else:
            plt.close()


def _merge_extrinsics_with_initial(extrinsics_df: pd.DataFrame, initial_extrinsics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join a (site, dataset) group's submitted cameras with their initial positions on ``image_file_name``.

    Adds ``dx``/``dy``/``dz`` columns holding the shift between the submitted (x_map, y_map, alt)
    and the initial ones.
    """
    initial_df = initial_extrinsics_df.rename(columns=lambda c: c.strip().lower().replace(" ", "_"))
    merged = extrinsics_df.reset_index().merge(
        initial_df[["image_file_name", "x_map", "y_map", "alt"]],
        on="image_file_name",
        how="inner",
        suffixes=("", "_initial"),
    )
    merged["dx"] = merged["x_map"] - merged["x_map_initial"]
    merged["dy"] = merged["y_map"] - merged["y_map_initial"]
    merged["dz"] = merged["alt"] - merged["alt_initial"]
    return merged


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
