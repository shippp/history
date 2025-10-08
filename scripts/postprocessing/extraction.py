import argparse
import json
from pathlib import Path

import history


def main(overwrite: bool = False, dry_run: bool = False, verbose: bool = True) -> None:
    config_path = Path(__file__).parent / "config.json"
    config = json.load(open(config_path, "r", encoding="utf-8"))

    paths_manager = history.postprocessing.PathsManager(**config)

    postproc = history.postprocessing.PostProcessing(paths_manager)

    postproc.uncompress_all_submissions(overwrite, dry_run, verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the post-processing workflow for submissions.")

    # On crée des flags (booléens)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted data (default: False).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate actions without writing to disk (default: False).",
    )
    parser.add_argument(
        "--no-verbose",
        dest="verbose",
        action="store_false",
        help="Disable verbose output (default: True).",
    )

    args = parser.parse_args()
    main(**vars(args))
