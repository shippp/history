import os

import geoutils as gu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable


def plot_dems(
    postproc_df: pd.DataFrame, output_directory: str | None = None, plot: bool = True, max_cols: int = 4
) -> None:
    group_dict = (
        postproc_df.dropna(subset=["raw_dem_file"]).groupby(["dataset", "site"])["raw_dem_file"].apply(list).to_dict()
    )

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
            code = postproc_df.loc[postproc_df["raw_dem_file"] == dem_file].index[0]
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
            plt.savefig(os.path.join(output_directory, f"{dataset} - {site} DEMs"))
        if plot:
            plt.show()
        else:
            plt.close()
