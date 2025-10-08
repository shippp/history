"""
Submission Directory Validator
==============================

This script inspects a directory containing photogrammetric or remote sensing
submission data and checks for the presence of mandatory and optional files
according to expected naming conventions.

Each file must follow a structured naming pattern:
    <author>_<site>_<dataset>_<images>_<camera_used>_<gcp_used>_<pointcloud_coregistration>_<mtp_adjustment>_<suffix>

The script:
------------
- Parses submission codes to extract metadata (e.g. site, dataset type, processing options).
- Recursively scans the provided directory.
- Validates the presence of mandatory and optional output files.
- Prints a color-coded summary in the console:
    - ✅ Green: file found
    - 🔴 Red: missing mandatory file
    - 🟠 Orange: missing optional file

Example
-------
    $ python check_submissions.py /path/to/submissions
"""

import argparse
from collections import defaultdict
from pathlib import Path

# ANSI colors
GREEN = "\033[92m"  # bright green
RED = "\033[91m"  # bright red
ORANGE = "\033[93m"  # yellow/orange
RESET = "\033[0m"

MANDATORY_SUFFIXS = [
    "sparse_pointcloud.laz",
    "dense_pointcloud.laz",
    "camera_model_extrinsics.csv",
    "camera_model_intrinsics.csv",
]
OPTIONAL_SUFFIXS = ["dem.tif", "orthoimage.tif"]


def parse_code(code: str) -> dict[str, str | bool]:
    """
    Parse and interpret a submission code string into its metadata components.

    Each code follows a structured format composed of multiple segments separated by underscores.
    Each segment encodes a specific parameter of the dataset, such as site, dataset type,
    image preprocessing, and processing options.

    Example
    -------
    >>> parse_code("JD_CG_AI_RA_CY_GA_PY_MY")
    {
        'author': 'JD',
        'site': 'casa_grande',
        'dataset': 'aerial',
        'images': 'raw',
        'camera_used': True,
        'gcp_used': 'Automated approch',
        'pointcloud_coregistration': True,
        'mtp_adjustment': True
    }

    Parameters
    ----------
    code : str
        Submission code string to parse.

    Returns
    -------
    dict[str, str | bool]
        A dictionary mapping parameter names to their interpreted values.

    Raises
    ------
    ValueError
        If the code does not match the expected structure or contains unknown identifiers.
    """
    VALID_MAPPING = {
        "site": {"CG": "casa_grande", "IL": "iceland"},
        "dataset": {"AI": "aerial", "MC": "kh9mc", "PC": "kh9pc"},
        "images": {"RA": "raw", "PP": "preprocessed"},
        "camera_used": {"CY": True, "CN": False},
        "gcp_used": {"GM": "Manual (provided)", "GA": "Automated approch", "GN": "No"},
        "pointcloud_coregistration": {"PY": True, "PN": False},
        "mtp_adjustment": {"MY": True, "MN": False},
    }
    parts = code.split("_")
    # Check format length
    expected_parts = len(VALID_MAPPING) + 1  # author + mappings
    if len(parts) < expected_parts:
        raise ValueError(f"The code: {code} has unexpected format (expected ≥ {expected_parts} parts)")

    metadatas = {"author": parts[0]}

    # Normalize to uppercase for consistency
    parts = [p.upper() for p in parts]

    for i, (key, mapping) in enumerate(VALID_MAPPING.items()):
        value = parts[i + 1]
        if value not in mapping:
            raise ValueError(f"{value} is not a known code for {key}.")
        metadatas[key] = mapping[value]

    return metadatas


def main(input_dir: Path):
    """
    Inspect a submission directory and report the presence of required and optional files.

    This function recursively scans the provided directory and identifies submission files
    matching the expected suffix patterns. It then prints a color-coded summary of
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
    # Check input_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"The directory '{input_dir}' does not exist.")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"'{input_dir}' is not a directory.")

    print(f"Processing directory: {input_dir} : \n")

    files_found = defaultdict(list)

    # search all files in the directory recursivly
    for f in [f for f in input_dir.rglob("*") if f.is_file()]:
        for suffix in MANDATORY_SUFFIXS + OPTIONAL_SUFFIXS:
            if f.name.endswith(suffix):
                code = f.name.removesuffix("_" + suffix)
                files_found[code].append(suffix)

    # print all informations
    print(f"Found {len(files_found)} submission(s) : ")
    for code, suffixs in files_found.items():
        print(f"\n - {code} : ")
        for s in MANDATORY_SUFFIXS:
            status = f"{GREEN}True{RESET}" if s in suffixs else f"{RED}False{RESET}"
            print(f"\t{s} : {status}")
        for s in OPTIONAL_SUFFIXS:
            status = f"{GREEN}True{RESET}" if s in suffixs else f"{ORANGE}False{RESET}"
            print(f"\t{s} (Optional) : {status}")


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
