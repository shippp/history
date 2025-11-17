import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

FILE_CODE_MAPPING: dict[str, dict[str, str]] = {
    "site": {"CG": "casa_grande", "IL": "iceland"},
    "dataset": {"AI": "aerial", "MC": "kh9mc", "PC": "kh9pc"},
    "images": {"RA": "raw", "PP": "preprocessed"},
    "use_of_camera_calibration": {"CY": "Yes", "CN": "No"},
    "use_of_gcps": {"GM": "Manual (provided)", "GA": "Automated approch", "GN": "No", "GY": "Yes"},
    "pointcloud_coregistration": {"PY": "Yes", "PN": "No"},
    "mtb_adjustment": {"MY": "Yes", "MN": "No"},
}

FILENAME_PATTERN = re.compile(
    r"""
    ^(?P<author>[^_]+)_
    (?P<site>[A-Z]{2})_
    (?P<dataset>[A-Z]{2})_
    (?P<images>[A-Z]{2})_
    (?P<camera_used>[A-Z]{2})_
    (?P<gcp_used>[A-Z]{2})_
    (?P<pointcloud_coregistration>[A-Z]{2})_
    (?P<mtp_adjustment>[A-Z]{2})
    (?:_(?P<version>V\d+))?
    .*$
    """,
    re.VERBOSE | re.IGNORECASE,
)

# ANSI colors
GREEN = "\033[92m"  # bright green
RED = "\033[91m"  # bright red
ORANGE = "\033[93m"  # yellow/orange
RESET = "\033[0m"

MANDATORY_PATTERNS = {
    "sparse_pointcloud": re.compile(r"sparse_pointcloud\.la[sz]$"),
    "dense_pointcloud": re.compile(r"dense_pointcloud\.la[sz]$"),
    "extrinsincs": re.compile(r"extrinsics\.csv"),
    "intrinsics": re.compile(r"intrinsics\.csv"),
}
OPTIONAL_SUFFIXS = ["dem.tif", "orthoimage.tif"]


def parse_filename(file: str | Path) -> tuple[str, dict[str, Any]]:
    """
    Parse a filename following the predefined code convention described in FILE_CODE_MAPPING_V1.

    This function extracts structured information from a filename built using a specific
    naming convention such as:
        AUTHOR_SITE_DATASET_IMAGES_CAMERAUSED_GCPUSED_POINTCLOUDCOREG_MTPADJ[_V1-DEM].tif

    Each short code (e.g., 'CG', 'AI', 'RA', 'CY') is validated against FILE_CODE_MAPPING_V1
    to ensure consistency and then mapped to its corresponding descriptive value.

    Args:
        file: Path or filename to parse.

    Returns:
        tuple[str, dict]:
            - code: normalized filename code (e.g., "ALICE_CG_AI_RA_CY_GY_PY_MY_V1")
            - metadatas: dictionary of parsed metadata fields mapped to their descriptive values.

    Raises:
        ValueError: If the filename does not respect the expected naming convention
                    or contains unknown codes not defined in FILE_CODE_MAPPING_V1.
    """
    match = FILENAME_PATTERN.match(Path(file).stem)

    if not match:
        raise ValueError(f"The filename {Path(file).stem} don't respect the code convention")

    match_dict = match.groupdict()
    metadatas = {"author": match_dict["author"]}

    for key, value in match_dict.items():
        if key in FILE_CODE_MAPPING:
            metadatas[key] = FILE_CODE_MAPPING[key].get(value)

    metadatas["version"] = match_dict.get("version")

    code = "_".join([v for v in match_dict.values() if v is not None])
    return code, metadatas


def main(input_dir: Path):
    """
    Inspect a submission directory and report the presence of required and optional files.

    This function recursively scans the provided directory and identifies submission files
    matching the expected suffix patterns. It then prints a color-coded summary showing
    which mandatory and optional files are present for each submission.

    Parameters
    ----------
    input_dir : pathlib.Path
        Path to the root directory containing the submission files.

    Raises
    ------
    FileNotFoundError
        If the provided directory does not exist.
    NotADirectoryError
        If the provided path is not a directory.
    """
    # Validate input_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory '{input_dir}' does not exist.")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"'{input_dir}' is not a directory.")

    files = [f for f in input_dir.rglob("*") if f.is_file()]
    print(f"Scanning directory: {input_dir}\n")
    print(f"Detected {len(files)} file(s):\n")

    files_found = defaultdict(list)

    # Traverse all files recursively
    for f in files:
        depth = len(f.parents) - len(input_dir.parents) - 1
        print(depth * "\t" + f"{f.name}", end="")

        file_code = next((key for key, pattern in MANDATORY_PATTERNS.items() if pattern.search(f.name)), None)

        if file_code:
            print(f" -> {file_code}", end="")
            try:
                code, _ = parse_filename(f)
                files_found[code].append(file_code)
                print(f", {GREEN}valid code{RESET}")
            except Exception as e:
                print(f", {RED}invalid code{RESET}: {e}")
        else:
            print("")

    print(f"\nDetected {len(files_found)} submission(s):")
    for key, file_codes in files_found.items():
        print(f" - {key}: ", end="")
        if len(file_codes) == len(MANDATORY_PATTERNS):
            print(f"{GREEN}all mandatory files present{RESET}")
        else:
            missing_files = set(MANDATORY_PATTERNS) - set(file_codes)
            print(f"{RED}missing mandatory file(s): {missing_files}{RESET}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check a submission directory for required and optional files.")
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Path to the input directory (default: current working directory).",
    )
    args = parser.parse_args()

    main(args.input_dir)
