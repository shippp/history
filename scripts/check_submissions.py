import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any



# Defines the expected segments in the filename, separated by _
SEGMENT_SPECS_v2 = [
    ("author", r"[A-Za-z0-9]{4}", "exactly 4 alphanumeric characters"),
    ("site", r"(?:CG|IL)",     "either 'CG' or 'IL'"),
    ("dataset",   r"(?:AI|MC|PC)", "either AI, MC or PC"),
    ("images", r"(?:PP|RA)",     "either 'PP' or 'RA'"),
    ("calib_used",  r"C[YN]",   "either CY or CN"),
    ("gcp_used",   r"G[MAN]",   "either of 'GM', 'GA', 'GN'"),
    ("pointcloud_coregistration",  r"P[YN]",   "either of 'PY' or 'PN'"),
    ("mtp_adjustments",   r"M[YN]",   "either of 'MY' or 'MN'"),
]

# optional version segment
VERSION_REGEX = r"_v([0-9])"

# allowed suffixes and mandatory/optional files
ALLOWED_SUFFIXES = {"dense_pointcloud", "sparse_pointcloud", "extrinsics", "intrinsics"}
SUFFIX_REGEX = "(" + "|".join(re.escape(s) for s in ALLOWED_SUFFIXES) + ")"

MANDATORY_PATTERNS = {
    "sparse_pointcloud": re.compile(r"sparse_pointcloud\.la[sz]$"),
    "dense_pointcloud": re.compile(r"dense_pointcloud\.la[sz]$"),
    "extrinsincs": re.compile(r"extrinsics\.csv"),
    "intrinsics": re.compile(r"intrinsics\.csv"),
}
OPTIONAL_SUFFIXS = ["dem.tif", "orthoimage.tif"]

# ANSI colors, for printing
GREEN = "\033[92m"  # bright green
RED = "\033[91m"  # bright red
ORANGE = "\033[93m"  # yellow/orange
RESET = "\033[0m"


def parse_filename(fname: str | Path) -> tuple[str, dict[str, Any]]:
    """
    Parse a filename following the predefined code convention described in SEGMENT_SPECS_v2.

    This function extracts structured information from a filename built using a specific
    naming convention such as:
        AUTHOR_SITE_DATASET_IMAGES_CAMERAUSED_GCPUSED_POINTCLOUDCOREG_MTPADJ[_V1]_dense_pointcloud.laz

    Args:
        fname: Path or filename to parse.

    Returns:
        tuple[str, dict]:
            - code: normalized filename code (e.g., "TAG0_CG_AI_RA_CY_GY_PY_MY_v1")
            - metadatas: dictionary of parsed metadata fields mapped to their descriptive values.

    Raises:
        ValueError: If the filename does not respect the expected naming convention
                    or contains unknown codes not defined in FILE_CODE_MAPPING_V1.
    """
    # Get filename without extension and parent directories
    fname = Path(fname).stem

    # ---- Full regex ----
    base_pattern = "_".join(f"({regex})" for _, regex, _ in SEGMENT_SPECS_v2)
    full_pattern = rf"^{base_pattern}(?:{VERSION_REGEX})?_{SUFFIX_REGEX}$"
    full_re = re.compile(full_pattern)

    match = full_re.match(fname)
    if match:
        # identify each segment
        groups = match.groups()
        segment_values = groups[:len(SEGMENT_SPECS_v2)]
        version = groups[len(SEGMENT_SPECS_v2)]
        suffix = groups[len(SEGMENT_SPECS_v2) + 1]

        # build normalized code
        code = "_".join([v for v in segment_values if v is not None])
        
        # save all codes in dictionary
        metadatas = {name: value for (name, _, _), value in zip(SEGMENT_SPECS_v2, segment_values)}
        metadatas["version"] = version
        metadatas["suffix"] = suffix

        return code, metadatas

    # ---- Robust error explanation ----

    # 1. Validate suffix FIRST (robust to underscores)
    suffix_match = re.search(rf"_(?P<suffix>{SUFFIX_REGEX})$", fname)
    if not suffix_match:
        raise ValueError(
            f"Invalid or missing file suffix. "
            f"Expected one of {sorted(ALLOWED_SUFFIXES)}"
        )

    suffix = suffix_match.group("suffix")
    prefix = fname[: suffix_match.start()]

    # 2. Optional version
    version_match = re.search(r"_(?P<ver>{VERSION_REGEX})$", prefix)
    if version_match:
        prefix = prefix[: version_match.start()]
    elif "_v" in prefix:
        raise ValueError(
            "Invalid version segment: expected '_vX' where X is a digit"
        )

    # 3. Segment validation
    parts = prefix.split("_")
    if len(parts) != len(SEGMENT_SPECS_v2):
        raise ValueError(
            f"Expected {len(SEGMENT_SPECS_v2)} underscore-separated segments "
            f"before the optional version, got {len(parts)}"
        )

    for i, ((name, regex, description), value) in enumerate(
        zip(SEGMENT_SPECS_v2, parts), start=1
    ):
        if not re.fullmatch(regex, value):
            raise ValueError(
                f"Segment {i} ({name}) is invalid: '{value}' — "
                f"expected {description}"
            )

    raise ValueError("Filename does not match the expected pattern")

def test_parsing():
    """
    Quick check that parse_filename raises error when expected.
    """
    test_strings = [
        "TAG1_CG_MC_PP_CN_GN_PN_MN_dense_pointcloud.laz",  # correct
        "TAG1_IL_AI_RA_CN_GN_PN_MN_v2_dense_pointcloud.laz",  # correct
        "TAG1_CG_PC_PP_CY_GA_PY_MY_dense_pointcloud.las",  # correct despite las file
        "MyFullName_CG_MC_PP_CN_GN_PN_MN_dense_pointcloud.laz",  # incorrect author tag
        "TAG1_CA_MC_RA_CN_GN_PN_MN_dense_pointcloud.laz",  # incorrect site tag
        "TAG1_CG_M9_PP_CN_GN_PN_MN_dense_pointcloud.laz",   # incorrect dataset tag M9 
        "TAG1_CG_MC_PR_CN_GN_PN_MN_dense_pointcloud.laz",   # incorrect image tag PR
        "TAG1_CG_MC_PP_RN_GN_PN_MN_dense_pointcloud.laz",  # incorrect calib_used tag RN
        "TAG1_CG_MC_PP_CN_PY_PN_MN_dense_pointcloud.laz",  # incorrect gcp_used tag PY
        "TAG1_CG_MC_PP_CN_GN_PS_MN_dense_pointcloud.laz",  # incorrect pointcloud_coregistration tag PS
        "TAG1_CG_MC_PP_CN_GN_PY_MA_dense_pointcloud.laz",  # incorrect mtp_adjustments tag MA
        "TAG1_CG_MC_PP_CN_GN_PN_MN_dense_pc.laz",   # incorrect suffix
        "TAG1_CG_MC_PP_CN_GN_PN_MN_v10_dense_pointcloud.laz",  # incorrect version
        "TAG1_CG_MC_PP_CN_GN_PN_dense_pointcloud.laz",  # incorrect number of segments
        "my-submission_dense_pointcloud.laz",  # completely wrong name
    ]

    for fname in test_strings:
        try:
            parse_filename(fname)
            print(f"✅ OK: '{fname}'")
        except ValueError as e:
            print(f"❌ '{fname}'")
            print(f"   Reason: {e}")


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
        type=str,
        nargs="?",
        default=Path("."),
        help="Path to the input directory (default: current working directory).",
    )
    args = parser.parse_args()

    if args.input_dir == "test":
        test_parsing()
    else:
        main(Path(args.input_dir))