"""
Standard-deviation DEM step: builds one STD DEM per (site, dataset) group from
coregistered DEMs, and generates the associated plots.
"""

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from matplotlib.figure import Figure
from rasterio.windows import Window

from history.config import Config
from history.postprocessing.io import is_output_up_to_date, parse_filename
from history.postprocessing.visualization import _read_raster_with_max_size

logger = logging.getLogger(__name__)


def is_existing_std_dem(dem_files: list[str | Path], output_path: str | Path, metadata_key: str = "dem_files") -> bool:
    """
    Check if an existing std DEM already matches the given input DEM list.

    Args:
        dem_files: List of input DEM paths used to compute the std DEM.
        output_path: Path to the supposed std DEM file.
        metadata_key: Metadata key used to store the original DEM file list.

    Returns:
        True if the file exists and its metadata matches the given DEM list, False otherwise.
    """
    output_path = Path(output_path)
    if not output_path.exists():
        return False

    try:
        with rasterio.open(output_path) as src:
            tags = src.tags(1)  # or src.tags() if not band-specific

        if metadata_key not in tags:
            return False

        # Normalize paths to absolute str for comparison
        founded_dem_files = [str(Path(p).resolve()) for p in json.loads(tags[metadata_key])]
        expected_dem_files = [str(Path(p).resolve()) for p in dem_files]

        return set(expected_dem_files) == set(founded_dem_files)

    except Exception as e:
        # Defensive: in case of malformed metadata or corrupted file
        logger.warning(f"Could not verify std_dem metadata ({output_path.name}): {e}")
        return False


def get_dem_files_from_std_dem(std_dem_file: str | Path, metadata_key: str = "dem_files") -> list[str]:
    """
    Retrieves the list of input DEM file paths stored in the metadata of a standard deviation DEM.

    Args:
        std_dem_file (str | Path): Path to the standard deviation DEM file.
        metadata_key (str, optional): Metadata tag name containing the input DEM file list.
            Defaults to "dem_files".

    Returns:
        list[str] | None: A list of DEM file paths if found in the metadata.
        Returns an empty list if the file does not exist, or None if the metadata key is missing.
    """
    std_dem_file = Path(std_dem_file)
    if not std_dem_file.exists():
        return []

    with rasterio.open(std_dem_file) as src:
        tags = src.tags(1)

        if metadata_key in tags:
            # Already present → return parsed JSON
            return json.loads(tags[metadata_key])
        else:
            return None


def create_std_dem(
    dem_files: list[str | Path],
    output_path: str | Path,
    overwrite: bool = False,
    block_size: int = 256,
    metadata_key: str = "dem_files",
) -> None:
    """
    Generates a standard deviation Digital Elevation Model (DEM) from a list of input DEM files.

    This function computes the pixel-wise standard deviation across multiple DEMs, processing
    the rasters block by block to efficiently handle large datasets.

    Args:
        dem_files (list[str | Path]): List of paths to input DEM raster files.
        output_path (str | Path): Path to the output standard deviation DEM file.
        block_size (int, optional): Size of the processing block in pixels. Defaults to 256.
        metadata_key (str, optional): Metadata tag name used to store the list of input DEM files
            in the output raster. Defaults to "dem_files".

    Returns:
        None: The function writes the resulting standard deviation DEM to the specified output path.
    """
    dem_files: list[Path] = [Path(f) for f in dem_files]
    output_path = Path(output_path)

    if len(dem_files) <= 1:
        logger.warning(f"Need at least 2 DEMs for computing the STD DEM: {output_path.name}.")
        return

    if not overwrite and is_output_up_to_date(dem_files, output_path) and is_existing_std_dem(dem_files, output_path):
        logger.info(f"Skip {output_path.name}: output is up to date.")
        return

    # first open the first raster of the list to have a reference profile
    with rasterio.open(dem_files[0]) as src_ref:
        profile = src_ref.profile.copy()
        width, height = src_ref.width, src_ref.height

    profile.update(dtype="float32", count=1)

    output_path.parent.mkdir(exist_ok=True, parents=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        # Loop through the raster by windows
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                win = Window(
                    col_off=x,
                    row_off=y,
                    width=min(block_size, width - x),
                    height=min(block_size, height - y),
                )

                # Read the corresponding window from each DEM
                block_stack = []
                for dem_path in dem_files:
                    with rasterio.open(dem_path) as src:
                        data = src.read(1, window=win, masked=True).filled(np.nan)
                        block_stack.append(data)

                # Compute std for this block
                block_stack = np.stack(block_stack, axis=0)

                # Avoid computing std on empty slices
                if np.all(np.isnan(block_stack)):
                    block_std = np.full(block_stack.shape[1:], np.nan, dtype="float32")
                else:
                    block_std = np.nanstd(block_stack, axis=0).astype("float32")

                # Write the result
                dst.write(block_std, 1, window=win)

        # Add metadata tags
        dem_files_str = [str(p) for p in dem_files]
        dst.update_tags(1, **{metadata_key: json.dumps(dem_files_str)})

    logger.info(f"STD DEM generated at {output_path}")


def create_std_dems(
    dem_files: Iterable[str | Path],
    output_dir: str | Path,
    overwrite: bool = False,
) -> None:
    """
    Group coregistered DEMs by (site, dataset) and compute one STD DEM per group.

    Parses each filename to extract site and dataset metadata, groups files
    accordingly, and calls ``create_std_dem`` for each group. Output filenames
    follow the pattern ``<site>_<dataset>_std_dem.tif`` inside ``output_dir``.
    Files whose names cannot be parsed are skipped with a warning.

    Parameters
    ----------
    dem_files : Iterable of str or Path
        Iterable of coregistered DEM file paths to process.
    output_dir : str or Path
        Directory where the STD DEM files will be written. Created if missing.
    overwrite : bool, optional
        If True, existing STD DEMs are recomputed. Default is False.
    """
    dem_files = [Path(f) for f in dem_files]
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    groups: dict[tuple[str, str], list[Path]] = {}
    for file in dem_files:
        try:
            _, metadatas = parse_filename(file)
            key = (metadatas["site"], metadatas["dataset"])
            groups.setdefault(key, []).append(file)
        except ValueError:
            logger.warning(f"Cannot parse filename for std_dem grouping: {file.name}")

    for (site, dataset), files in groups.items():
        logger.info(f"Creating std DEM for {site} / {dataset}")
        output_path = output_dir / f"{site}_{dataset}_std_dem.tif"
        create_std_dem(dem_files=files, output_path=output_path, overwrite=overwrite)


def generate_std_dem_plots(dem_path: str | Path, output_path: str | Path, overwrite: bool = False) -> None:
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
    dem_path = Path(dem_path)

    if not overwrite and is_output_up_to_date(dem_path, output_path):
        logger.debug(f"File {output_path} is up to date -> skipping.")
        return

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


def plot_std_dems(config: Config) -> None:
    """Generate plots for each STD DEM found in ``std_dems_dir``."""
    for file in config.proc_dir.std_dems_dir.glob("*.tif"):
        subdir = file.stem.replace("_std_dem", "")
        output_path = config.plot_dir / subdir / file.with_suffix(".png").name
        generate_std_dem_plots(file, output_path, overwrite=config.overwrite_plots)


def run_std_dem(config: Config) -> None:
    """Build one STD DEM per (site, dataset) group from coregistered DEMs, and plot them."""
    create_std_dems(
        dem_files=list(config.proc_dir.coreg_dems_dir.glob("*-DEM.tif")),
        output_dir=config.proc_dir.std_dems_dir,
        overwrite=config.overwrite,
    )

    if not config.no_plots:
        plot_std_dems(config)
