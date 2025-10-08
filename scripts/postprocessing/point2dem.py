import argparse
import json
from pathlib import Path

import history


def main(
    overwrite: bool = False,
    dry_run: bool = False,
    asp_path: str | None = None,
    max_concurrent_commands: int = 1,
    max_threads_per_command: int = 4,
) -> None:
    config_path = Path(__file__).parent / "config.json"
    config = json.load(open(config_path, "r", encoding="utf-8"))

    paths_manager = history.postprocessing.PathsManager(**config)

    postproc = history.postprocessing.PostProcessing(paths_manager)

    postproc.iter_point2dem(overwrite, dry_run, asp_path, max_concurrent_commands, max_threads_per_command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the post-processing workflow for submissions.")

    # Boolean flags
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

    # String and integer arguments
    parser.add_argument(
        "--asp-path",
        type=str,
        default=None,
        help="Path to the ASP binaries or working directory (default: None).",
    )
    parser.add_argument(
        "--max-concurrent-commands",
        type=int,
        default=1,
        help="Maximum number of commands to run in parallel (default: 1).",
    )
    parser.add_argument(
        "--max-threads-per-command",
        type=int,
        default=4,
        help="Maximum number of threads per command (default: 4).",
    )

    args = parser.parse_args()
    main(**vars(args))
