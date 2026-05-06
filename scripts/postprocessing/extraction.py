"""
Extract compressed submission archives into individual directories.

This script reads all archives found in ``<base_dir>/raw/`` and extracts each one
into a subdirectory of ``<base_dir>/extracted/``.  Supported formats: .zip, .7z,
.tgz, .tar.gz, .tar.bz2, .tar.xz.

Configuration
-------------
Edit ``config.json`` (located next to this script) before running:

    {
        "base_dir": "/path/to/output"
    }

The script derives the following paths from ``base_dir``:

    <base_dir>/
    ├── raw/            ← place compressed archives here before running
    └── extracted/      ← extracted submission folders are written here

Usage
-----
    python scripts/postprocessing/extraction.py [--config PATH] [--overwrite] [--max-workers N]

Options
-------
    --config PATH     Path to a JSON config file (required).
    --overwrite       Re-extract even if the output directory already exists.
    --max-workers N   Number of parallel extraction workers (default: CPU count).

Example
-------
    python scripts/postprocessing/extraction.py --max-workers 8
    python scripts/postprocessing/extraction.py --config config.local.json
"""

import argparse
import json
from pathlib import Path

import history


def main(config: str, overwrite: bool = False, max_workers: int | None = None) -> None:
    """
    Extract all submission archives from ``<base_dir>/raw/`` into ``<base_dir>/extracted/``.

    Parameters
    ----------
    config : str
        Path to a JSON config file.
    overwrite : bool
        If True, re-extract archives whose output directory already exists.
    max_workers : int or None
        Number of parallel extraction workers. Defaults to the system CPU count.
    """
    config_path = Path(config)
    config = json.load(open(config_path, "r", encoding="utf-8"))

    base_dir = Path(config["base_dir"])
    raw_dir = base_dir / "raw"
    extracted_dir = base_dir / "extracted"

    history.postprocessing.pipeline.uncompress_all_submissions(raw_dir, extracted_dir, overwrite, max_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uncompress all submission archives from raw_dir into extracted_dir.")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a JSON config file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted directories (default: False).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        dest="max_workers",
        help="Number of parallel extraction workers (default: CPU count).",
    )

    args = parser.parse_args()
    main(**vars(args))
