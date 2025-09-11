import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LightSource
from rasterio.enums import Resampling
from tqdm import tqdm


def plot_postprocess_state(global_df: pd.DataFrame) -> None:
    data = []
    for code, row in global_df.iterrows():
        row_dict = {
            "code": code,
            "Has pointcloud": not pd.isna(row["pointcloud_file"]),
            "Has raw DEM": not pd.isna(row["raw_dem_file"]),
            "Raw DEM valid": row["raw_dem_percent_nodata"] < 95,
            "Has coregistered DEM": not pd.isna(row["coreg_dem_file"]),
        }
        data.append(row_dict)

    df = pd.DataFrame(data).set_index("code")

    # Convertir en matrice 0/1
    matrix = df.astype(int).values

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, cmap="Greys", aspect="auto")

    # Ajouter les ticks (labels)
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.index)

    # Lignes de grille pour simuler cases
    ax.set_xticks(np.arange(-0.5, len(df.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(df.index), 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    plt.show()


def generate_raw_dems_mosaic(
    global_df: pd.DataFrame,
    output_directory: str,
    max_cols: int = 4,
) -> None:
    df_without_void = global_df.dropna(subset=["raw_dem_file"]).loc[global_df["raw_dem_percent_nodata"] < 95]

    pbar = tqdm(total=len(df_without_void), desc="Mosaicing raw DEMs")
    for (dataset, site), group in df_without_void.groupby(["dataset", "site"]):
        vmin = group["raw_dem_min"].median()
        vmax = group["raw_dem_max"].median()

        # create the subplot with a good grid
        n = len(group)
        ncols = min(max_cols, n)
        nrows = (n + max_cols - 1) // max_cols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), constrained_layout=True)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i, (code, row) in enumerate(group.iterrows()):
            dem = _read_raster_with_max_size(row["raw_dem_file"])

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

        plt.suptitle(f"raw - DEMs : {dataset} - {site}", fontsize=16)

        # save or not and plot or not
        os.makedirs(output_directory, exist_ok=True)
        plt.savefig(os.path.join(output_directory, f"raw-DEMs-{dataset}-{site}.png"))
        plt.close()

    pbar.close()


def generate_coregistered_dems_mosaic(
    global_df: pd.DataFrame,
    output_directory: str,
    max_cols: int = 4,
) -> None:
    df_without_void = global_df.dropna(subset=["coreg_dem_file"]).loc[global_df["coreg_dem_percent_nodata"] < 95]

    pbar = tqdm(total=len(df_without_void), desc="Mosaicing coregistered DEMs")
    for (dataset, site), group in df_without_void.groupby(["dataset", "site"]):
        vmin = group["coreg_dem_min"].median()
        vmax = group["coreg_dem_max"].median()

        # create the subplot with a good grid
        n = len(group)
        ncols = min(max_cols, n)
        nrows = (n + max_cols - 1) // max_cols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), constrained_layout=True)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i, (code, row) in enumerate(group.iterrows()):
            dem = _read_raster_with_max_size(row["coreg_dem_file"])

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

        plt.suptitle(f"Coregistered - DEMs : {dataset} - {site}", fontsize=16)

        # save the plot and don't show it
        os.makedirs(output_directory, exist_ok=True)
        plt.savefig(os.path.join(output_directory, f"coregisterd-DEMs-{dataset}-{site}.png"))
        plt.close()
    pbar.close()


def generate_ddems_mosaic(
    global_df: pd.DataFrame, output_directory: str, vmin: float = -10, vmax: float = 10, max_cols: int = 4
) -> None:
    mapping = [("ddem_before_file", "DDEM-before-coreg"), ("ddem_after_file", "DDEM-after-coreg")]
    for colname, prefix in mapping:
        df_dropped = global_df.dropna(subset=[colname])
        pbar = tqdm(total=len(df_dropped), desc=f"mosaicing {prefix}")

        for (dataset, site), group in df_dropped.groupby(["dataset", "site"]):
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
            os.makedirs(output_directory, exist_ok=True)
            plt.savefig(os.path.join(output_directory, f"{prefix}-{dataset}-{site}.png"))
            plt.close()
        pbar.close()


def generate_slopes_mosaic(
    global_df: pd.DataFrame, output_directory: str, vmin: float = 0, vmax: float = 15, max_cols: int = 4
) -> None:
    mapping = [("ddem_before_file", "slope-before-coreg"), ("ddem_after_file", "slope-after-coreg")]
    for colname, prefix in mapping:
        df_dropped = global_df.dropna(subset=[colname])
        pbar = tqdm(total=len(df_dropped), desc=f"mosaicing {prefix}")

        for (dataset, site), group in df_dropped.groupby(["dataset", "site"]):
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
            os.makedirs(output_directory, exist_ok=True)
            plt.savefig(os.path.join(output_directory, f"{prefix}-{dataset}-{site}.png"))
            plt.close()
        pbar.close()


def generate_hillshades_mosaic(
    global_df: pd.DataFrame, output_directory: str, vmin: float = 0, vmax: float = 1, max_cols: int = 4
) -> None:
    mapping = [("ddem_before_file", "hillshades-before-coreg"), ("ddem_after_file", "hillshades-after-coreg")]
    for colname, prefix in mapping:
        df_dropped = global_df.dropna(subset=[colname])
        pbar = tqdm(total=len(df_dropped), desc=f"mosaicing {prefix}")

        for (dataset, site), group in df_dropped.groupby(["dataset", "site"]):
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

            os.makedirs(output_directory, exist_ok=True)
            plt.savefig(os.path.join(output_directory, f"{prefix}-{dataset}-{site}.png"))
            plt.close()
        pbar.close()


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


def plot_coregistration_shifts(global_df: pd.DataFrame, output_directory: str) -> None:
    df_droped = global_df.dropna(subset=["coreg_shift_z"])
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
        plt.savefig(os.path.join(output_directory, f"coregistration_shifts_{dataset}_{site}"))
        plt.close()


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
