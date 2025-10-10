import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LightSource
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from tqdm import tqdm

from history.postprocessing.statistics import create_std_dem

#######################################################################################################################
##                                                  MOSAIC VISUALIZATION
#######################################################################################################################


def generate_dems_mosaic(
    df: pd.DataFrame, output_dir: str | Path, max_cols_dict: dict[tuple[str, str], int] = {}
) -> None:
    mapping = [("raw_dem", "raw-DEMs"), ("coreg_dem", "coreg-DEMs")]
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for name, prefix in mapping:
        df_dropped = df.dropna(subset=[f"{name}_file"]).loc[df[f"{name}_percent_nodata"] < 95]
        pbar = tqdm(total=len(df_dropped), desc=f"mosaicing {prefix}")

        for (dataset, site), group in df_dropped.groupby(["dataset", "site"]):
            vmin = group[f"{name}_min"].median()
            vmax = group[f"{name}_max"].median()
            max_cols = max_cols_dict.get((dataset, site), 4)

            # create the subplot with a good grid
            n = len(group)
            ncols = min(max_cols, n)
            nrows = (n + max_cols - 1) // max_cols
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), constrained_layout=True)
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

            for i, (code, row) in enumerate(group.iterrows()):
                dem = _read_raster_with_max_size(row[f"{name}_file"])

                axes[i].imshow(dem, cmap="terrain", vmin=vmin, vmax=vmax)
                axes[i].axis("off")
                axes[i].set_title(code)
                pbar.update(1)

            # hidden empty plot
            for j in range(i + 1, len(axes)):
                axes[j].axis("off")

            # add the global color bar
            cbar = fig.colorbar(
                ScalarMappable(cmap="terrain", norm=plt.Normalize(vmin=vmin, vmax=vmax)),
                ax=axes,
                orientation="vertical",
                fraction=0.03,
                pad=0.02,
            )
            cbar.set_label("Altitude (m)")

            plt.suptitle(f"{prefix} : {dataset} - {site}", fontsize=16)

            # save or not and plot or not
            plt.savefig(output_dir / f"{prefix}-mosaic-{dataset}-{site}.png")
            plt.close()

        pbar.close()


def generate_ddems_mosaic(
    df: pd.DataFrame,
    output_dir: str | Path,
    max_cols_dict: dict[tuple[str, str], int] = {},
    vmin: float = -10,
    vmax: float = 10,
) -> None:
    mapping = [("ddem_before_file", "DDEM-before-coreg"), ("ddem_after_file", "DDEM-after-coreg")]
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for colname, prefix in mapping:
        df_dropped = df.dropna(subset=[colname])
        pbar = tqdm(total=len(df_dropped), desc=f"mosaicing {prefix}")

        for (dataset, site), group in df_dropped.groupby(["dataset", "site"]):
            max_cols = max_cols_dict.get((dataset, site), 4)
            # create the subplot with a good grid
            n = len(group)
            ncols = min(max_cols, n)
            nrows = (n + max_cols - 1) // max_cols
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), constrained_layout=True)
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

            for i, (code, row) in enumerate(group.iterrows()):
                dem = _read_raster_with_max_size(row[colname])
                axes[i].imshow(dem, cmap="coolwarm", vmin=vmin, vmax=vmax)
                axes[i].axis("off")
                axes[i].set_title(code)
                pbar.update(1)

            # hidden empty plot
            for j in range(i + 1, len(axes)):
                axes[j].axis("off")

            # add the global color bar
            cbar = fig.colorbar(
                ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=vmin, vmax=vmax)),
                ax=axes,
                orientation="vertical",
                fraction=0.03,
                pad=0.02,
            )
            cbar.set_label("Altitude difference (m)")

            plt.suptitle(f"{prefix} {dataset} - {site}", fontsize=16)
            plt.savefig(output_dir / f"{prefix}-{dataset}-{site}.png")
            plt.close()
        pbar.close()


def generate_slopes_mosaic(
    df: pd.DataFrame,
    output_dir: str | Path,
    max_cols_dict: dict[tuple[str, str], int] = {},
    vmin: float = 0,
    vmax: float = 15,
) -> None:
    mapping = [("ddem_before_file", "slope-before-coreg"), ("ddem_after_file", "slope-after-coreg")]
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for colname, prefix in mapping:
        df_dropped = df.dropna(subset=[colname])
        pbar = tqdm(total=len(df_dropped), desc=f"mosaicing {prefix}")

        for (dataset, site), group in df_dropped.groupby(["dataset", "site"]):
            max_cols = max_cols_dict.get((dataset, site), 4)
            # create the subplot with a good grid
            n = len(group)
            ncols = min(max_cols, n)
            nrows = (n + max_cols - 1) // max_cols
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), constrained_layout=True)
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

            for i, (code, row) in enumerate(group.iterrows()):
                dem = _read_raster_with_max_size(row[colname])

                with rasterio.open(row[colname]) as src:
                    dx, dy = src.res

                # compute slope in degree
                grad_y, grad_x = np.gradient(dem, dy, dx)
                slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
                slope_dem = np.degrees(slope_rad)

                axes[i].imshow(slope_dem, cmap="terrain", vmin=vmin, vmax=vmax)
                axes[i].axis("off")
                axes[i].set_title(code)
                pbar.update(1)

            # hidden empty plot
            for j in range(i + 1, len(axes)):
                axes[j].axis("off")

            # add the global color bar
            cbar = fig.colorbar(
                ScalarMappable(cmap="terrain", norm=plt.Normalize(vmin=vmin, vmax=vmax)),
                ax=axes,
                orientation="vertical",
                fraction=0.03,
                pad=0.02,
            )
            cbar.set_label("Slope (Degree)")

            plt.suptitle(f"{prefix} {dataset} - {site}", fontsize=16)
            plt.savefig(output_dir / f"{prefix}-{dataset}-{site}.png")
            plt.close()
        pbar.close()


def generate_hillshades_mosaic(
    df: pd.DataFrame,
    output_dir: str | Path,
    max_cols_dict: dict[tuple[str, str], int] = {},
    vmin: float = 0,
    vmax: float = 1,
) -> None:
    mapping = [("ddem_before_file", "hillshades-before-coreg"), ("ddem_after_file", "hillshades-after-coreg")]
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    for colname, prefix in mapping:
        df_dropped = df.dropna(subset=[colname])
        pbar = tqdm(total=len(df_dropped), desc=f"mosaicing {prefix}")

        for (dataset, site), group in df_dropped.groupby(["dataset", "site"]):
            max_cols = max_cols_dict.get((dataset, site), 4)
            # create the subplot with a good grid
            n = len(group)
            ncols = min(max_cols, n)
            nrows = (n + max_cols - 1) // max_cols
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), constrained_layout=True)
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

            for i, (code, row) in enumerate(group.iterrows()):
                dem = _read_raster_with_max_size(row[colname])

                with rasterio.open(row[colname]) as src:
                    dx, dy = src.res

                ls = LightSource(azdeg=315, altdeg=45)  # azimuth, sun altitude
                hillshade = ls.hillshade(dem, vert_exag=1, dx=dx, dy=dy)

                axes[i].imshow(hillshade, cmap="gray", vmin=vmin, vmax=vmax)
                axes[i].axis("off")
                axes[i].set_title(code)
                pbar.update(1)

            # hidden empty plot
            for j in range(i + 1, len(axes)):
                axes[j].axis("off")

            # add the global color bar
            cbar = fig.colorbar(
                ScalarMappable(cmap="gray", norm=plt.Normalize(vmin=vmin, vmax=vmax)),
                ax=axes,
                orientation="vertical",
                fraction=0.03,
                pad=0.02,
            )
            cbar.set_label("Hillshade")
            plt.suptitle(f"{prefix} {dataset} - {site}", fontsize=16)

            plt.savefig(output_dir / f"{prefix}-{dataset}-{site}.png")
            plt.close()
        pbar.close()


def generate_mosaic_std_dems(global_df: pd.DataFrame, output_directory: str | Path) -> None:
    """
    Generate per-pixel standard deviation DEMs and corresponding plots
    for each (dataset, site) pair in the provided dataframe.

    This function groups input DEMs by their 'dataset' and 'site' identifiers,
    computes a standard deviation (STD) DEM for each group using the
    `create_std_dem` function, and generates a visualization of the resulting
    STD raster. The resulting GeoTIFF and PNG files are saved in the specified
    output directory.

    Parameters
    ----------
    global_df : pandas.DataFrame
        A dataframe containing metadata for DEMs, including at least the columns:
        - 'dataset' : Name of the dataset or campaign.
        - 'site' : Name or ID of the geographic site.
        - 'coreg_dem_file' : Path to each coregistered DEM file.
    output_directory : str or Path
        Path to the directory where the STD DEMs and plots will be saved.
        The directory is created if it does not exist.

    Returns
    -------
    None
        The function writes the standard deviation DEMs (.tif) and
        corresponding visualization plots (.png) to the output directory.

    Notes
    -----
    - Each group of DEMs (same dataset and site) is used to compute a pixel-wise
      standard deviation raster via `create_std_dem`.
    - A plot of the standard deviation map is generated using a 90th percentile
      color scale cap (`vmax = np.nanquantile(std_dem, 0.9)`).
    - The colorbar indicates elevation variability in meters.

    Example
    -------
    >>> generate_std_dems_by_dataset_site(global_df, "outputs/std_dems/")
    >>> # Produces files like:
    >>> # outputs/std_dems/std-dem-datasetA-site1.tif
    >>> # outputs/std_dems/std-plot-datasetA-site1.png
    """
    # create the output directory if needed
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    # drop empty raw_dem_file rows
    df_dropped = global_df.dropna(subset=["coreg_dem_file"])
    for (dataset, site), group in df_dropped.groupby(["dataset", "site"]):
        mnt_files = group["coreg_dem_file"].tolist()
        tmp_dem_file = output_directory / "tmp-dem.tif"
        output_plot_file = output_directory / f"std-plot-{dataset}-{site}.png"

        # create the std tif
        create_std_dem(mnt_files, tmp_dem_file)

        # open the previously created dem
        std_dem = _read_raster_with_max_size(tmp_dem_file)

        # remove the tmp dem
        tmp_dem_file.unlink()

        # create the plot and save them at output_plot_file
        vmax = np.nanquantile(std_dem, 0.9)
        plt.imshow(std_dem, cmap="viridis", vmax=vmax)
        plt.title(f"DEM Standard Deviation Map for ({dataset}-{site})")
        plt.axis("off")

        cbar = plt.colorbar()
        cbar.set_label("Elevation standard deviation (m)", rotation=270, labelpad=15)

        plt.savefig(output_plot_file)
        plt.close()


#######################################################################################################################
##                                                  STATISTICS VISUALIZATION
#######################################################################################################################


def barplot_var(global_df: pd.DataFrame, output_directory: str, colname: str, title: str = "") -> None:
    df = global_df.dropna(subset=[colname]).copy(True)

    # Créer la colonne groupe
    df["group"] = df["site"] + "_" + df["dataset"]

    # Trier par groupe puis point_count
    df_sorted = df.sort_values(["group", colname], ascending=[True, True])

    # Couleurs par groupe
    unique_groups = df_sorted["group"].unique()
    color_map = {g: f"C{i}" for i, g in enumerate(unique_groups)}

    # Plot
    fig, ax = plt.subplots(figsize=(15, 8))

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

    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, f"global_{colname}.png"))
    plt.close()


def generate_barplot_group_var(global_df: pd.DataFrame, output_directory: str, colname: str, title: str = "") -> None:
    droped_df = global_df.dropna(subset=[colname])
    for (dataset, site), group in droped_df.groupby(["dataset", "site"]):
        df_sorted = group.sort_values(colname, ascending=True)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot bar chart
        ax.bar(df_sorted.index.astype(str), df_sorted[colname], color="skyblue")

        # Titles and labels
        plot_title = f"{title} - {dataset} - {site}" if title else f"{dataset} - {site}"
        ax.set_title(plot_title, fontsize=14)
        ax.set_ylabel(colname, fontsize=12)

        # Rotate x labels to 90
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right")

        # Save figure
        os.makedirs(output_directory, exist_ok=True)
        outfile = os.path.join(output_directory, f"barplot-{colname}-{dataset}-{site}.png")
        plt.savefig(outfile, bbox_inches="tight")

        # Close figure to avoid displaying and consuming memory
        plt.close(fig)


def plot_coregistration_shifts(global_df: pd.DataFrame, output_dir: str | Path) -> None:
    df_droped = global_df.dropna(subset=["coreg_shift_z"])
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    for (dataset, site), group in df_droped.groupby(["dataset", "site"]):
        ordered_group = group.sort_values("coreg_shift_z")
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10), gridspec_kw={"height_ratios": [1, 3]})

        colors = ["red", "green", "blue"]
        labels = ["shift x", "shift y", "shift z"]

        for col, color, label in zip(["coreg_shift_x", "coreg_shift_y", "coreg_shift_z"], colors, labels):
            ax1.scatter(ordered_group.index, ordered_group[col], color=color, s=50, label=label)
            ax2.scatter(ordered_group.index, ordered_group[col], color=color, s=50, label=label)

        ax1.set_ylim(20, 100)  # upper zone = outliers
        ax2.set_ylim(-20, 20)

        # break the axes
        ax1.spines["bottom"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax1.tick_params(bottom=False)

        # add zigzags for the cut
        d = 0.5  # zigzag prop
        kwargs = dict(
            marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color="k", mec="k", mew=1, clip_on=False
        )

        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

        # add a grid for the 2 axes
        for ax in (ax1, ax2):
            ax.grid(True, linestyle="--", alpha=0.6)

        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(output_dir / f"coregistration_shifts_{dataset}_{site}")
        plt.close()


def generate_nmad_groupby(df: pd.DataFrame, output_dir: str | Path) -> None:
    """
    Generate grouped bar plots of NMAD before and after coregistration for each dataset and site.
    Saves one plot per (dataset, site) combination.
    """
    droped_df = df.dropna(subset=["nmad_after_coreg", "nmad_before_coreg"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for (dataset, site), group in droped_df.groupby(["dataset", "site"]):
        ordered_group = group.sort_values("nmad_before_coreg")

        # Build the bar positions (one group per "code")
        codes = ordered_group.index
        x = range(len(codes))
        width = 0.35

        # Plot
        fig, ax = plt.subplots(figsize=(15, 5))

        bars_before = ax.bar(
            [pos - width / 2 for pos in x],
            ordered_group["nmad_before_coreg"],
            width=width,
            label="Before coregistration",
        )
        bars_after = ax.bar(
            [pos + width / 2 for pos in x],
            ordered_group["nmad_after_coreg"],
            width=width,
            label="After coregistration",
        )

        # Add value labels above bars
        ax.bar_label(bars_before, fmt="%.2f", padding=3)
        ax.bar_label(bars_after, fmt="%.2f", padding=3)

        # Labels and style
        ax.set_xticks(x)
        ax.set_xticklabels(codes, rotation=90)
        ax.set_ylabel("NMAD")
        ax.set_title(f"NMAD Before/After Coregistration\nDataset: {dataset}, Site: {site}")
        ax.legend()

        # Save figure
        plt.tight_layout()
        plt.savefig(output_dir / f"nmad_{dataset}_{site}.png", dpi=300)
        plt.close()


#######################################################################################################################
##                                                  STATISTICS LANDCOVER VISUALIZATION
#######################################################################################################################


def generate_landcover_grouped_boxplot_by_dataset_site(
    landcover_df: pd.DataFrame, output_directory: str | Path
) -> None:
    """
    Generate grouped boxplots of precomputed landcover statistics by dataset and site.

    This function creates grouped boxplots for each (dataset, site) pair in the provided
    DataFrame. Each group of boxes corresponds to a landcover class, and each box within a group
    represents a different raster code. The plot uses precomputed summary statistics (median,
    Q1, Q3) instead of raw pixel data.

    Steps:
        1. Group the input DataFrame by (dataset, site).
        2. Within each subgroup:
            - Sort raster codes by their mean NMAD value (ascending order).
            - Compute average Q1, Q3, and median for each (landcover_label, code) pair.
            - Build grouped boxplots (one color per code).
            - Annotate each x-axis label with the mean percentage of landcover coverage.
        3. Save one PNG file per (dataset, site) combination.

    Args:
        landcover_df (pd.DataFrame):
            A DataFrame containing precomputed statistics per landcover class and raster.
            Expected columns include:
            ['dataset', 'site', 'code', 'landcover_label', 'median', 'q1', 'q3', 'percent', 'nmad'].
        output_directory (str | Path):
            Directory where the resulting boxplot figures will be saved.
            It will be created if it does not already exist.

    Returns:
        None
        The function saves one grouped boxplot per (dataset, site) as PNG files in the output directory.

    Notes:
        - The boxplots are based on aggregated statistics, not raw pixel data.
        - Raster codes are sorted by their mean NMAD value for better comparison.
        - Colors are automatically assigned per code and reflected in the legend.
        - Outliers are not displayed (`showfliers=False`).

    Example:
        >>> generate_landcover_grouped_boxplot_by_dataset_site(landcover_df, "outputs/plots/")
        # Produces grouped boxplots comparing elevation differences across landcover classes
        # and raster codes, saved in the 'outputs/plots/' directory.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    for (dataset, site), group in landcover_df.groupby(["dataset", "site"]):
        # order group with the mean of nmad
        code_order = group.groupby("code")["nmad"].mean().sort_values().index
        group = group.set_index("code").loc[code_order].reset_index()

        landcover_labels = group["landcover_label"].unique()
        codes = group["code"].unique()
        n_labels = len(landcover_labels)
        n_codes = len(codes)
        width = 0.8 / n_codes
        x_base = np.arange(n_labels)

        fig, ax = plt.subplots(figsize=(12, 6))

        color_map = {code: f"C{i}" for i, code in enumerate(codes)}  # assign colors

        for i, code in enumerate(codes):
            positions = x_base - 0.4 + i * width + width / 2
            box_data = []
            for lc_label in landcover_labels:
                lc_group = group[(group["landcover_label"] == lc_label) & (group["code"] == code)]
                if lc_group.empty:
                    box_data.append({"med": 0, "q1": 0, "q3": 0, "whislo": 0, "whishi": 0, "fliers": []})
                else:
                    box_data.append(
                        {
                            "med": lc_group["median"].mean(),
                            "q1": lc_group["q1"].mean(),
                            "q3": lc_group["q3"].mean(),
                            "whislo": lc_group["q1"].mean(),
                            "whishi": lc_group["q3"].mean(),
                            "fliers": [],
                        }
                    )
            ax.bxp(
                box_data,
                positions=positions,
                widths=width,
                showfliers=False,
                patch_artist=True,
                boxprops=dict(facecolor=color_map[code]),
            )

        # X-axis labels
        percent_means = group.groupby("landcover_label")["percent"].mean()
        ax.set_xticks(x_base)
        ax.set_xticklabels([f"{lc} ({percent_means[lc]:.1f}%)" for lc in landcover_labels], rotation=45, ha="right")

        ax.set_ylabel("Altitude difference (m)")
        ax.set_title(f"Grouped boxplot by landcover and code ({dataset}-{site})")

        # Create custom legend
        legend_handles = [Patch(facecolor=color_map[code], label=str(code)) for code in codes]
        ax.legend(handles=legend_handles, title="Code", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_directory / f"grouped-boxplot-{dataset}-{site}.png")
        plt.close()


def generate_landcover_boxplot_by_dataset_site(landcover_df: pd.DataFrame, output_directory: str | Path) -> None:
    """
    Generate and save boxplots of landcover-based raster statistics for each (dataset, site) pair.

    This function groups the provided landcover statistics DataFrame by dataset and site,
    then computes summary statistics (median, Q1, Q3) for each landcover class within that group.
    Using these aggregated values, it generates boxplots representing the distribution of
    raster-derived metrics (e.g., elevation difference) by landcover class. Each boxplot is
    saved as a PNG image in the specified output directory.

    Steps:
        1. Ensure the output directory exists.
        2. Group the DataFrame by (dataset, site).
        3. For each subgroup:
            - Compute mean values of median, Q1, and Q3 for each landcover class.
            - Create a Matplotlib boxplot using precomputed statistics.
            - Annotate x-axis labels with landcover names and coverage percentage.
            - Save the resulting figure as a PNG file named:
              'landcover-boxplot-{dataset}-{site}.png'.

    Args:
        landcover_df (pd.DataFrame):
            A DataFrame containing per-landcover statistics.
            Expected columns include:
            ['dataset', 'site', 'code', 'landcover_label', 'median', 'q1', 'q3', 'percent'].
        output_directory (str | Path):
            Path to the folder where the boxplot images will be saved.
            The directory will be created if it does not exist.

    Returns:
        None
        The function saves one PNG file per (dataset, site) combination in the output directory.

    Notes:
        - The boxplots are built from pre-aggregated statistics (not raw pixel values).
        - Outliers are not displayed (`showfliers=False`).
        - Each plot title includes the dataset, site, and number of rasters considered.

    Example:
        >>> generate_landcover_boxplot_by_dataset_site(landcover_df, "outputs/plots/")
        # Generates PNG boxplots grouped by dataset and site in the specified directory.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    for (dataset, site), group in landcover_df.groupby(["dataset", "site"]):
        box_data = []
        labels = []
        for lc_label, lc_group in group.groupby("landcover_label"):
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
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bxp(box_data, showfliers=False)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Altitude difference (m)")
        ax.set_title(
            f"Boxplot of mean values from {len(group['code'].unique())} raster(s) by landcover class ({dataset}-{site})"
        )
        plt.tight_layout()
        plt.savefig(output_directory / f"landcover-boxplot-{dataset}-{site}.png")
        plt.close()


def generate_landcover_nmad_by_dataset_site(landcover_df: pd.DataFrame, output_directory: str | Path) -> None:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    for (dataset, site), group in landcover_df.groupby(["dataset", "site"]):
        df_plot = group.pivot_table(
            index="landcover_label",  # x axe
            columns="code",  # one color per bar
            values="nmad",  # values to show
            aggfunc="mean",  # aff func in case
        )
        # Compute mean percent per landcover_label
        percent_means = group.groupby("landcover_label")["percent"].mean()

        # Replace index labels with label + mean percent
        df_plot.index = [f"{label} ({percent_means[label]:.2f}%)" for label in df_plot.index]

        # plot the grouped barplot
        df_plot.plot(kind="bar", figsize=(10, 6))

        plt.ylabel("NMAD")
        plt.title(f"NMAD by landcover class and raster code ({dataset}-{site})")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Code", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        plt.savefig(output_directory / f"barplot-nmad-{dataset}-{site}.png")
        plt.close()


#######################################################################################################################
##                                                  OTHER VISUALIZATION
#######################################################################################################################


def plot_files_recap(filepaths_df: pd.DataFrame, output_path: str | None = None, show: bool = True) -> None:
    data = []
    for code, row in filepaths_df.iterrows():
        row_dict = {
            "code": code,
            "Sparse pointcloud": not pd.isna(row["sparse_pointcloud_file"]),
            "Dense pointcloud": not pd.isna(row["dense_pointcloud_file"]),
            "Extrinsics": not pd.isna(row["extrinsics_camera_file"]),
            "Intrinsics": not pd.isna(row["intrinsics_camera_file"]),
            "Raw DEM": not pd.isna(row["raw_dem_file"]),
            "Coregistered DEM": not pd.isna(row["coreg_dem_file"]),
        }
        data.append(row_dict)

    new_df = pd.DataFrame(data).set_index("code")

    # Convertir en matrice 0/1
    matrix = new_df.astype(int).values

    fig, ax = plt.subplots(figsize=(12, 6))

    # Utiliser pcolormesh pour avoir un meilleur contrôle des bordures
    cmap = plt.get_cmap("binary")  # blanc = 0, noir = 1
    c = ax.pcolormesh(matrix, cmap=cmap, edgecolors="grey", linewidth=1, shading="auto")

    # Ajouter ticks
    ax.set_xticks(np.arange(matrix.shape[1]) + 0.5)
    ax.set_yticks(np.arange(matrix.shape[0]) + 0.5)
    ax.set_xticklabels(new_df.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(new_df.index, fontsize=9)

    # Grille (par-dessus pour rester visible même sur les cases noires)
    ax.set_xticks(np.arange(matrix.shape[1]), minor=True)
    ax.set_yticks(np.arange(matrix.shape[0]), minor=True)
    ax.grid(which="minor", color="grey", linestyle="-", linewidth=0.8, alpha=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Amélioration visuelle
    ax.set_title("Files Recap", fontsize=14, weight="bold")
    fig.tight_layout()
    if output_path:
        plt.savefig(output_path)

    if show:
        plt.show()
    else:
        plt.close()


def generate_coregistration_individual_plots(
    global_df: pd.DataFrame, output_directory: str | Path, overwrite: bool = True, vmin: float = -10, vmax: float = 10
) -> None:
    """
    Generate and save before/after coregistration plots for each DEM pair in the provided dataset.

    This function iterates over all entries in the input DataFrame that contain valid coregistered DEM files.
    For each entry, it loads both the pre- and post-coregistration dDEM rasters, creates a side-by-side
    comparison plot, and saves the figure as a PNG file in the specified output directory.

    Parameters
    ----------
    global_df : pandas.DataFrame
        Input DataFrame containing metadata and file paths for DEM coregistration results.
        Must include the following columns:
        - "coreg_dem_file"
        - "ddem_before_file"
        - "ddem_after_file"
        - "mean_before_coreg"
        - "median_before_coreg"
        - "nmad_before_coreg"
        - "mean_after_coreg"
        - "median_after_coreg"
        - "nmad_after_coreg"
    output_directory : str or Path
        Directory where the generated PNG plots will be saved.
    overwrite : bool, optional
        If True, existing plots will be overwritten. Default is True.
    vmin : float, optional
        Minimum value for the color scale in the plots. Default is -10.
    vmax : float, optional
        Maximum value for the color scale in the plots. Default is 10.

    Returns
    -------
    None
        The function does not return anything. It writes plot files to disk.

    Notes
    -----
    - Each plot shows two panels: the dDEM before and after coregistration, with a shared colorbar.
    - The filenames of the plots are derived from the DataFrame index (e.g., "<index>_coreg_plot.png").
    - A progress bar is displayed using `tqdm` to indicate processing progress.

    Examples
    --------
    >>> generate_coregistration_individual_plots(global_df, "outputs/plots", overwrite=False)
    Generating coregistration plots:  45%|████▌     | 9/20 [00:04<00:05,  2.10it/s]
    """
    # create the output directory if needed
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    # filter every row with a valid coregistered dems
    filtered_df = global_df.loc[~global_df["coreg_dem_file"].isna()]

    for code, row in tqdm(filtered_df.iterrows(), desc="Generating coregistration plots", total=len(filtered_df)):
        output_path = output_directory / f"{code}_coreg_plot.png"

        if not output_path.exists() or overwrite:
            # open the raw dDEM and the coregistered dDEM
            ddem_before = _read_raster_with_max_size(row["ddem_before_file"])
            ddem_after = _read_raster_with_max_size(row["ddem_after_file"])

            # create the figure
            fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

            # add the dDEMs and their titles
            axes[0].imshow(ddem_before, cmap="coolwarm", vmin=vmin, vmax=vmax)
            axes[0].axis("off")
            axes[0].set_title(
                f"dDEM before coregistration \n(mean: {row['mean_before_coreg']:.3f}, med: {row['median_before_coreg']:.3f}, nmad: {row['nmad_before_coreg']:.3f})"
            )

            axes[1].imshow(ddem_after, cmap="coolwarm", vmin=vmin, vmax=vmax)
            axes[1].axis("off")
            axes[1].set_title(
                f"dDEM after coregistration \n(mean: {row['mean_after_coreg']:.3f}, med: {row['median_after_coreg']:.3f}, nmad: {row['nmad_after_coreg']:.3f})"
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

            plt.savefig(output_path)
            plt.close()


#######################################################################################################################
##                                                  PRIVATE FUNCTIONS
#######################################################################################################################


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
