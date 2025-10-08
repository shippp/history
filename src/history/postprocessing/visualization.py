import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LightSource
from rasterio.enums import Resampling
from tqdm import tqdm


class PostProcessPlot:
    def __init__(self, df: pd.DataFrame, output_dir: str | Path):
        self._df = df
        self._dir = Path(output_dir)
        self._max_cols = {
            ("aerial", "casa_grande"): 5,
            ("aerial", "iceland"): 4,
            ("kh9mc", "casa_gande"): 4,
            ("kh9mc", "iceland"): 4,
            ("kh9pc", "casa_grande"): 4,
            ("kh9pc", "iceland"): 4,
        }
        self._dir_stats = self._dir / "statistics"
        self._dir_stats.mkdir(exist_ok=True, parents=True)

    def generate_all_mosaics(self) -> None:
        generate_dems_mosaic(self._df, self._dir / "mosaic-DEMs", self._max_cols)
        generate_ddems_mosaic(self._df, self._dir / "mosaic-DDEMs", self._max_cols)
        generate_hillshades_mosaic(self._df, self._dir / "mosaic-hillshades", self._max_cols)
        generate_slopes_mosaic(self._df, self._dir / "mosaic-slopes", self._max_cols)

    def generate_all_stat_plots(self) -> None:
        stat_dir = self._dir / "statistics"
        generate_nmad_groupby(self._df, stat_dir / "nmad")
        barplot_var(self._df, stat_dir, "dense_point_count", "Point count")
        barplot_var(self._df, stat_dir, "nmad_before_coreg", "NMAD before coregistration")
        plot_coregistration_shifts(self._df, stat_dir / "coreg_shifts")


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


def barplot_var(global_df: pd.DataFrame, output_directory: str, colname: str, title: str = "", filter_outliers: bool = False) -> None:
    df = global_df.dropna(subset=[colname]).copy(True)

    # Créer la colonne groupe
    df["group"] = df["site"] + "_" + df["dataset"]

    # Trier par groupe puis point_count
    df_sorted = df.sort_values(["group", colname], ascending=[True, True])

    # Filter large outliers
    if filter_outliers:
        df_sorted = outlier_filtering(df_sorted, colname)

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


def outlier_filtering(df: pd.DataFrame, colname: str) -> pd.DataFrame:
    """
    Filter outliers in the df DataFrame.
     
    Lines are considered as outliers when the value of colname is more than mean + 2*std of the group\
    as defined in the "group" column.
    """
    df_filtered = df.copy()
    for i in range(3):
        group_mean = df_filtered.groupby("group")[colname].mean()
        group_std = df_filtered.groupby("group")[colname].std()
        outlier_threshold = group_mean + 2 * group_std
        outlier_threshold_df = df_filtered["group"].map(outlier_threshold)
        df_filtered = df_filtered[df_filtered[colname] <= outlier_threshold_df]
    return df_filtered

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
