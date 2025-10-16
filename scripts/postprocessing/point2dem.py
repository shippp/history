import argparse
import json
from pathlib import Path

import history


def main(
    overwrite: bool = False,
    pdal_exec_path: str = "pdal",
    max_workers: int = 4,
    dry_run: bool = False,
) -> None:
    config_path = Path(__file__).parent / "config.json"
    config = json.load(open(config_path, "r", encoding="utf-8"))

    paths_manager = history.postprocessing.PathsManager(**config)

    postproc = history.postprocessing.PostProcessing(paths_manager)

    postproc.iter_convert_pointcloud_to_dem(overwrite, pdal_exec_path, max_workers, dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the post-processing workflow to convert point clouds into DEMs.")

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
