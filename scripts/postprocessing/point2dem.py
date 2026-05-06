"""
Convert dense point clouds to DEMs using PDAL.

This script reads all dense point cloud files from the symlinks directory,
converts each one to a DEM via PDAL, and writes the results to the raw DEMs
directory.  It requires the symlink tree to already exist (created by the
indexing step in the post-processing notebook).

Prerequisites
-------------
The symlink directory ``<base_dir>/processing/symlinks/dense_pointclouds/`` must
be populated before running this script.  This is done by the notebook step that
calls ``history.postprocessing.pipeline.index_submissions_and_link_files()``.

Configuration
-------------
Edit ``config.json`` (located next to this script) before running.  The relevant
fields are:

    {
        "base_dir": "/path/to/output",
        "references_data_mapping": {
            "<site>": {
                "<dataset>": {
                    "ref_dem":      "/path/to/reference_dem.tif",
                    "ref_dem_mask": "/path/to/reference_dem_mask.tif",
                    "landcover":    "/path/to/landcover.tif"
                }
            }
        }
    }

Valid site values   : "casa_grande", "iceland"
Valid dataset values: "aerial", "kh9mc", "kh9pc"

The script derives the following paths from ``base_dir``:

    <base_dir>/processing/
    ├── symlinks/dense_pointclouds/   ← input: symlinks to .laz / .las files
    └── raw_dems/                     ← output: one .tif per point cloud

Usage
-----
    python scripts/postprocessing/point2dem.py [--config PATH] [--overwrite]
                                               [--pdal-exec-path PATH]
                                               [--max-workers N] [--dry-run]

Options
-------
    --config PATH            Path to a JSON config file (required).
    --overwrite              Overwrite existing DEMs.
    --pdal-exec-path PATH    Path to the PDAL executable (default: "pdal").
    --max-workers N          Number of parallel conversion workers (default: 4).
    --dry-run                Parse inputs and print planned operations without
                             actually running PDAL.

Example
-------
    python scripts/postprocessing/point2dem.py --max-workers 8
    python scripts/postprocessing/point2dem.py --dry-run
    python scripts/postprocessing/point2dem.py --config config.local.json
    python scripts/postprocessing/point2dem.py --pdal-exec-path /opt/conda/bin/pdal --overwrite
"""

import argparse
import json
from pathlib import Path

import history
from history.postprocessing.io import ReferencesData


def main(
    config: str,
    overwrite: bool = False,
    pdal_exec_path: str = "pdal",
    max_workers: int = 4,
    dry_run: bool = False,
) -> None:
    """
    Convert all dense point clouds in the symlinks directory to DEMs.

    Reads ``<base_dir>/processing/symlinks/dense_pointclouds/`` and writes one
    GeoTIFF DEM per file into ``<base_dir>/processing/raw_dems/``.

    Parameters
    ----------
    config : str
        Path to a JSON config file.
    overwrite : bool
        If True, re-process point clouds whose DEM already exists.
    pdal_exec_path : str
        Path or name of the PDAL executable on the current system.
    max_workers : int
        Number of parallel PDAL workers.
    dry_run : bool
        If True, print planned operations without executing PDAL.
    """
    config_path = Path(config)
    config = json.load(open(config_path, "r", encoding="utf-8"))

    base_dir = Path(config["base_dir"])
    symlinks_dir = base_dir / "processing" / "symlinks"
    raw_dems_dir = base_dir / "processing" / "raw_dems"

    raw_mapping = config["references_data_mapping"]
    references_data = ReferencesData(
        {(site, dataset): paths for site, datasets in raw_mapping.items() for dataset, paths in datasets.items()}
    )

    pointcloud_files = list((symlinks_dir / "dense_pointclouds").iterdir())

    history.postprocessing.pipeline.process_pointclouds_to_dems(
        pointcloud_files,
        raw_dems_dir,
        references_data,
        pdal_exec_path,
        overwrite,
        dry_run,
        max_workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the post-processing workflow to convert point clouds into DEMs.")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a JSON config file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing DEMs if they already exist.",
    )

    parser.add_argument(
        "--pdal-exec-path",
        type=str,
        default="pdal",
        help="Path to the PDAL executable (default: 'pdal').",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel workers to use (default: 4).",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the workflow without executing PDAL (for testing).",
    )

    args = parser.parse_args()
    main(**vars(args))
