import os

import geoutils as gu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xdem
from matplotlib.cm import ScalarMappable


def plot_dems(
    postproc_df: pd.DataFrame,
    output_directory: str | None = None,
    plot: bool = True,
    coregistered: bool = False,
    max_cols: int = 4,
) -> None:
    os.makedirs(output_directory, exist_ok=True)

    file_colname = "coregistered_dem_file" if coregistered else "raw_dem_file"
    df_without_void = postproc_df.dropna(subset=[file_colname]).loc[postproc_df["percent_nodata"] < 95]

    group_dict = df_without_void.groupby(["dataset", "site"])[file_colname].apply(list).to_dict()

    for (dataset, site), dem_files in group_dict.items():
        # compute the vmin and vmax with all DEMs
        min_list, max_list = [], []
        for dem_file in dem_files:
            dem = gu.Raster(dem_file)
            arr = dem.data
            min_list.append(np.nanmin(arr))
            max_list.append(np.nanmax(arr))
        vmin, vmax = np.median(min_list), np.median(max_list)

        # create the subplot with a good grid
        n = len(dem_files)
        ncols = min(max_cols, n)
        nrows = (n + max_cols - 1) // max_cols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), constrained_layout=True)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i, dem_file in enumerate(dem_files):
            dem = gu.Raster(dem_file)
            axes[i].imshow(dem.data, cmap="coolwarm", vmin=vmin, vmax=vmax)
            axes[i].axis("off")

            # get the code of the dem juste for the sub title
            code = postproc_df.loc[postproc_df[file_colname] == dem_file].index[0]
            axes[i].set_title(code)

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
        cbar.set_label("Altitude (m)")

        plt.suptitle(f"{dataset} - {site}", fontsize=16)

        # save or not and plot or not
        if output_directory:
            plt.savefig(os.path.join(output_directory, f"mosaic-{dataset}-{site}-DEMs.png"))
        if plot:
            plt.show()
        else:
            plt.close()


def plot_ddems(
    postproc_df: pd.DataFrame,
    output_directory: str | None = None,
    before_coreg: bool = True,
    plot: bool = True,
    vmin: int = -10,
    vmax: int = 10,
    max_cols: int = 4,
) -> None:
    os.makedirs(output_directory, exist_ok=True)
    file_colname = "ddem_before_file" if before_coreg else "ddem_after_file"
    df_without_void = postproc_df.dropna(subset=[file_colname])

    group_dict = df_without_void.groupby(["dataset", "site"])[file_colname].apply(list).to_dict()

    for (dataset, site), dem_files in group_dict.items():
        # create the subplot with a good grid
        n = len(dem_files)
        ncols = min(max_cols, n)
        nrows = (n + max_cols - 1) // max_cols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), constrained_layout=True)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i, dem_file in enumerate(dem_files):
            dem = gu.Raster(dem_file)
            axes[i].imshow(dem.data, cmap="coolwarm", vmin=vmin, vmax=vmax)
            axes[i].axis("off")

            # get the code of the dem juste for the sub title
            code = postproc_df.loc[postproc_df[file_colname] == dem_file].index[0]
            axes[i].set_title(code)

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

        plt.suptitle(f"{dataset} - {site}", fontsize=16)

        # save or not and plot or not
        if output_directory:
            plt.savefig(os.path.join(output_directory, f"mosaic-{dataset}-{site}-DDEMs.png"))
        if plot:
            plt.show()
        else:
            plt.close()


def plot_slopes(
    postproc_df: pd.DataFrame,
    file_column: str = "ddem_after_file",
    output_directory: str | None = None,
    plot: bool = True,
    vmin: int = 0,
    vmax: int = 15,
    max_cols: int = 4,
    verbose: bool = False,
) -> None:
    df_droped = postproc_df.dropna(subset=[file_column])
    for (dataset, site), group in df_droped.groupby(["dataset", "site"]):
        n = len(group)
        ncols = min(max_cols, n)
        nrows = (n + max_cols - 1) // max_cols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), constrained_layout=True)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i, (code, row) in enumerate(group.iterrows()):
            if verbose:
                print(f"Process {code}")
            dem = xdem.DEM(row[file_column])
            slope_dem = dem.slope()

            axes[i].imshow(slope_dem.data, cmap="coolwarm", vmin=vmin, vmax=vmax)
            axes[i].axis("off")
            axes[i].set_title(code)

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
        cbar.set_label("Slope (degree)")

        plt.suptitle(f"Slopes {dataset} - {site}", fontsize=16)

        # save or not and plot or not
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
            plt.savefig(os.path.join(output_directory, f"slope-{dataset}-{site}.png"))
        if plot:
            plt.show()
        else:
            plt.close()


def plot_hillshades(
    postproc_df: pd.DataFrame,
    file_column: str = "ddem_after_file",
    output_directory: str | None = None,
    plot: bool = True,
    vmin: int = 0,
    vmax: int = 255,
    max_cols: int = 4,
    verbose: bool = False,
) -> None:
    df_droped = postproc_df.dropna(subset=[file_column])
    for (dataset, site), group in df_droped.groupby(["dataset", "site"]):
        n = len(group)
        ncols = min(max_cols, n)
        nrows = (n + max_cols - 1) // max_cols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), constrained_layout=True)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i, (code, row) in enumerate(group.iterrows()):
            if verbose:
                print(f"Process {code}")
            dem = xdem.DEM(row[file_column])
            slope_dem = dem.hillshade()

            axes[i].imshow(slope_dem.data, cmap="gray", vmin=vmin, vmax=vmax)
            axes[i].axis("off")
            axes[i].set_title(code)

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

        plt.suptitle(f"Hillshade {dataset} - {site}", fontsize=16)

        # save or not and plot or not
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
            plt.savefig(os.path.join(output_directory, f"Hillshade-{dataset}-{site}.png"))
        if plot:
            plt.show()
        else:
            plt.close()


def barplot_nmad(postproc_df: pd.DataFrame, output_directory: str | None = None, plot: bool = True) -> None:
    df_sorted = postproc_df.sort_values("nmad_before_coreg", ascending=True)
    cols = ["nmad_before_coreg", "nmad_after_coreg"]

    # Supprimer les NaN
    df_sorted = df_sorted.dropna(subset=cols)

    # loop on each group dataset and site
    for (dataset, site), group in df_sorted.groupby(["dataset", "site"]):
        ax = group[cols].plot(kind="bar", figsize=(10, 5), width=0.8)

        # add label to show numeric values
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f")

        # styles
        ax.set_ylabel("NMAD")
        ax.set_title(f"NMAD before/after coregistration\nDataset: {dataset} | Site: {site}")
        plt.xticks(rotation=90)

        plt.tight_layout()

        # save the plot if output_directory is set
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
            filepath = os.path.join(output_directory, f"NMAD_before_vs_after_{dataset}_{site}.png")
            plt.savefig(filepath)

        # show or not the plot
        if plot:
            plt.show()
        else:
            plt.close()


def barplot_mean(postproc_df: pd.DataFrame, output_directory: str | None = None, plot: bool = True) -> None:
    df_sorted = postproc_df.sort_values("mean_before_coreg", ascending=True)
    cols = ["mean_before_coreg", "mean_after_coreg"]

    # Supprimer les NaN
    df_sorted = df_sorted.dropna(subset=cols)

    # loop on each group dataset and site
    for (dataset, site), group in df_sorted.groupby(["dataset", "site"]):
        ax = group[cols].plot(kind="bar", figsize=(10, 5), width=0.8)

        # add label to show numeric values
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f")

        # styles
        ax.set_ylabel("mean")
        ax.set_title(f"mean before/after coregistration\nDataset: {dataset} | Site: {site}")
        plt.xticks(rotation=90)

        plt.tight_layout()

        # save the plot if output_directory is set
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
            filepath = os.path.join(output_directory, f"mean_before_vs_after_{dataset}_{site}.png")
            plt.savefig(filepath)

        # show or not the plot
        if plot:
            plt.show()
        else:
            plt.close()


def barplot_median(postproc_df: pd.DataFrame, output_directory: str | None = None, plot: bool = True) -> None:
    df_sorted = postproc_df.sort_values("median_before_coreg", ascending=True)
    cols = ["median_before_coreg", "median_after_coreg"]

    # Supprimer les NaN
    df_sorted = df_sorted.dropna(subset=cols)

    # loop on each group dataset and site
    for (dataset, site), group in df_sorted.groupby(["dataset", "site"]):
        ax = group[cols].plot(kind="bar", figsize=(10, 5), width=0.8)

        # add label to show numeric values
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f")

        # styles
        ax.set_ylabel("median")
        ax.set_title(f"median before/after coregistration\nDataset: {dataset} | Site: {site}")
        plt.xticks(rotation=90)

        plt.tight_layout()

        # save the plot if output_directory is set
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
            filepath = os.path.join(output_directory, f"median_before_vs_after_{dataset}_{site}.png")
            plt.savefig(filepath)

        # show or not the plot
        if plot:
            plt.show()
        else:
            plt.close()
