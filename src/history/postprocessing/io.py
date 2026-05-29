import re
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union
import logging

import pandas as pd
import py7zr
import gdown

logger = logging.getLogger(__name__)


FILE_CODE_MAPPING: dict[str, dict[str, str]] = {
    "site": {"CG": "casa_grande", "IL": "iceland"},
    "dataset": {"AI": "aerial", "MC": "kh9mc", "PC": "kh9pc"},
    "images": {"RA": "raw", "PP": "preprocessed"},
    "calib_used": {"CY": "Yes", "CN": "No"},
    "georef": {"GM": "Manual (provided)", "GA": "Automated approach", "GC": "Coregistration", "GN": "No/other"},
    "pointcloud_coregistration": {"PY": "Yes", "PN": "No"},
    "mtp_adjustments": {"MY": "Yes", "MN": "No"},
}

# Per-segment patterns — used for both parsing and error diagnosis.
_SEGMENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("author", re.compile(r"^[A-Za-z0-9]{3,6}$")),
    ("site", re.compile(r"^(CG|IL)$", re.IGNORECASE)),
    ("dataset", re.compile(r"^(AI|MC|PC)$", re.IGNORECASE)),
    ("images", re.compile(r"^(PP|RA)$", re.IGNORECASE)),
    ("calib_used", re.compile(r"^C[YN]$", re.IGNORECASE)),
    ("georef", re.compile(r"^G[MACN]$", re.IGNORECASE)),
    ("pointcloud_coregistration", re.compile(r"^P[YN]$", re.IGNORECASE)),
    ("mtp_adjustments", re.compile(r"^M[YN]$", re.IGNORECASE)),
]

_VERSION_PATTERN = re.compile(r"^V\d+$", re.IGNORECASE)


class FilenameParseError(ValueError):
    """Raised when a filename does not conform to the submission naming convention.

    Attributes
    ----------
    stem : str
        The filename stem that failed to parse.
    segment : str or None
        Name of the first segment that caused the failure, if identifiable.
    got : str or None
        The actual value found in the failing segment.
    """

    def __init__(self, stem: str, reason: str, segment: str | None = None, got: str | None = None):
        self.stem = stem
        self.segment = segment
        self.got = got
        msg = f"'{stem}': {reason}"
        if segment and got is not None:
            expected = list(FILE_CODE_MAPPING[segment]) if segment in FILE_CODE_MAPPING else None
            msg += f" — segment '{segment}': got '{got}'"
            if expected:
                msg += f", expected one of {expected}"
        super().__init__(msg)


def check_disk_space_and_estimate(archive_files: List[Path], output_dir: Path) -> Dict[str, float]:
    """
    Check available disk space and estimate space needed for extraction.

    Parameters
    ----------
    archive_files : List[Path]
        List of archive files to be extracted
    output_dir : Path
        Directory where files will be extracted

    Returns
    -------
    Dict[str, float]
        Dictionary with 'available_gb', 'archive_size_gb', 'estimated_extracted_gb'
    """
    # Get available disk space
    stat = shutil.disk_usage(output_dir.parent if output_dir.exists() else output_dir.parent)
    available_bytes = stat.free
    available_gb = available_bytes / (1024**3)

    # Calculate total archive size
    total_archive_size = sum(f.stat().st_size for f in archive_files)
    archive_size_gb = total_archive_size / (1024**3)

    # Estimate extracted size (rough estimate: 2-4x archive size for typical compression)
    # Use conservative estimate of 4x for safety
    estimated_extracted_gb = archive_size_gb * 4

    return {
        "available_gb": available_gb,
        "archive_size_gb": archive_size_gb,
        "estimated_extracted_gb": estimated_extracted_gb,
    }


def extract_single_archive(
    archive_file: Union[str, Path],
    output_dir: Path,
    overwrite: bool = False,
    dry_run: bool = True,
    verbose: bool = True,
) -> Union[str, None]:
    """
    Extract a single archive file to the specified output directory.

    Parameters
    ----------
    archive_file : str or Path
        Path to the archive file to be extracted
    output_dir : str or Path
        Directory to extract files to
    overwrite : bool, default False
        Whether to overwrite existing extracted directory
    dry_run : bool, default True
        If True, only print what would be done without actually extracting
    verbose : bool, default True
        Whether to print progress messages
    Returns
    -------
    str or None
        Path to the extraction directory if successful, None otherwise
    """
    # Define supported archive extensions
    archive_extensions = {
        ".zip": _extract_zip,
        ".7z": _extract_7z,
        ".tgz": _extract_tar,
        ".tar.gz": _extract_tar,
        ".tar.bz2": _extract_tar,
        ".tar.xz": _extract_tar,
    }

    # Convert str to Path
    archive_file = Path(archive_file)
    output_dir = Path(output_dir)

    # Determine extraction directory name (remove extension)
    if archive_file.suffix == ".gz" and archive_file.stem.endswith(".tar"):
        # Handle .tar.gz
        extract_name = archive_file.stem.replace(".tar", "")
    else:
        extract_name = archive_file.stem

    extract_dir = output_dir / extract_name

    # Check if already extracted and not overwriting
    if extract_dir.exists() and not overwrite:
        if verbose:
            print(f"Skipping {archive_file.name} - already extracted")
        result = str(extract_dir)
        return result

    if dry_run:
        if verbose:
            print(f"{archive_file.name} -> {extract_dir}")
        result = str(extract_dir)
        return result

    # Remove existing directory if overwriting
    if extract_dir.exists() and overwrite:
        if verbose:
            print(f"Removing existing directory: {extract_dir}")
        shutil.rmtree(extract_dir)

    # Determine extraction method
    extraction_func = None
    for ext, func in archive_extensions.items():
        if archive_file.name.endswith(ext):
            extraction_func = func
            break

    if extraction_func is None:
        if verbose:
            print(f"Unsupported archive format: {archive_file.name}")
        return None

    try:
        if verbose:
            print(f"Extracting {archive_file.name}...")

        # Create extraction directory
        extract_dir.mkdir(exist_ok=True)

        # Extract the archive
        extraction_func(archive_file, extract_dir)
        result = str(extract_dir)

        if verbose:
            print(f"Successfully extracted {archive_file.name}")

    except Exception as e:
        if verbose:
            print(f"Failed to extract {archive_file.name}: {e}")
        # Clean up partial extraction
        if extract_dir.exists():
            shutil.rmtree(extract_dir)

    return result


def _extract_zip(archive_path: Path, extract_dir: Path) -> None:
    """Extract ZIP archive."""
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


def _extract_7z(archive_path: Path, extract_dir: Path) -> None:
    """Extract 7Z archive."""
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        archive.extractall(extract_dir)


def _extract_tar(archive_path: Path, extract_dir: Path) -> None:
    """Extract TAR archive (including .tgz, .tar.gz, .tar.bz2, .tar.xz)."""
    with tarfile.open(archive_path, "r:*") as tar_ref:
        tar_ref.extractall(extract_dir)


def analyze_submissions(
    data_dir: Union[str, Path], verbose: bool = True
) -> tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
    """
    Analyze HISTORY experiment submissions and create a summary table.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing extracted submission files
    verbose : bool, default True
        Whether to print warnings about missing files and code inconsistencies

    Returns
    -------
    tuple[pd.DataFrame, Dict[str, Dict[str, str]]]
        Summary table with submission details and file compliance,
        and dictionary mapping adjusted experiment codes to file paths
    """
    data_dir = Path(data_dir)

    # Define code meanings
    codes = {
        "site": {"CG": "Casa Grande", "IL": "Iceland"},
        "dataset": {"AI": "Aerial", "MC": "KH-9 MC", "PC": "KH-9 PC"},
        "images": {"RA": "Raw", "PP": "Pre-processed"},
        "calibration": {"CY": "Yes", "CN": "No"},
        "gcp": {"GY": "Yes", "GN": "No"},
        "coregistration": {"PY": "Yes", "PN": "No"},
        "multi-temporal": {"MY": "Yes", "MN": "No"},
    }

    submissions = []
    files_dict = {}  # {adjusted_experiment_code: {file_types: str}}

    # Find all submission directories
    for submission_dir in data_dir.iterdir():
        if not submission_dir.is_dir():
            continue

        # Get all files in submission directory recursively
        all_files = list(submission_dir.rglob("*"))

        # Step 1: Find all files that end with mandatory file patterns
        mandatory_file_matches = {
            "sparse_pointcloud": [],
            "dense_pointcloud": [],
            "extrinsics": [],
            "intrinsics": [],
            "report": [],
        }

        for f in all_files:
            if not f.is_file():
                continue

            name_lower = f.name.lower()
            # Check for each mandatory file type separately
            if name_lower.endswith("sparse_pointcloud.laz") or name_lower.endswith("sparse_pointcloud.las"):
                mandatory_file_matches["sparse_pointcloud"].append(f)
            elif name_lower.endswith("dense_pointcloud.laz") or name_lower.endswith("dense_pointcloud.las"):
                mandatory_file_matches["dense_pointcloud"].append(f)
            elif name_lower.endswith("extrinsics.csv"):
                mandatory_file_matches["extrinsics"].append(f)
            elif name_lower.endswith("intrinsics.csv"):
                mandatory_file_matches["intrinsics"].append(f)
            elif name_lower.endswith((".pdf", ".docx", ".odt", ".txt")) and "report" in name_lower:
                mandatory_file_matches["report"].append(f)

        # Step 2: Parse codes and group files that belong together
        experiments = {}  # {author: {experiment_code: {file_types: bool}}}
        file_paths = {}  # {author: {experiment_code: {file_types: str}}}

        # Check for missing mandatory files in verbose mode
        if verbose:
            missing_file_types = [file_type for file_type, files in mandatory_file_matches.items() if not files]
            if missing_file_types:
                print(
                    f"Warning: {submission_dir.name} is missing mandatory file types: {', '.join(missing_file_types)}"
                )

        # Process all found files to extract experiment codes
        for file_type, files in mandatory_file_matches.items():
            for f in files:
                # Extract experiment prefix from filename (keep original case)
                filename = f.name

                # Remove the file type suffix to get experiment prefix
                if filename.lower().endswith("sparse_pointcloud.laz") or filename.lower().endswith(
                    "sparse_pointcloud.las"
                ):
                    prefix = (
                        filename.replace("sparse_pointcloud.laz", "")
                        .replace("sparse_pointcloud.las", "")
                        .replace("sparse_pointcloud.LAZ", "")
                        .replace("sparse_pointcloud.LAS", "")
                        .rstrip("_")
                    )
                elif filename.lower().endswith("dense_pointcloud.laz") or filename.lower().endswith(
                    "dense_pointcloud.las"
                ):
                    prefix = (
                        filename.replace("dense_pointcloud.laz", "")
                        .replace("dense_pointcloud.las", "")
                        .replace("dense_pointcloud.LAZ", "")
                        .replace("dense_pointcloud.LAS", "")
                        .rstrip("_")
                    )
                elif filename.lower().endswith("extrinsics.csv"):
                    # Handle camera_model_extrinsics.csv variation
                    prefix = filename.replace("extrinsics.csv", "").replace("extrinsics.CSV", "").rstrip("_")
                    if prefix.lower().endswith("camera_model"):
                        prefix = prefix[: -len("camera_model")].rstrip("_")
                elif filename.lower().endswith("intrinsics.csv"):
                    # Handle camera_model_intrinsics.csv variation
                    prefix = filename.replace("intrinsics.csv", "").replace("intrinsics.CSV", "").rstrip("_")
                    if prefix.lower().endswith("camera_model"):
                        prefix = prefix[: -len("camera_model")].rstrip("_")

                # Extract author (everything before first site code)
                author = prefix
                for site_code in codes["site"].keys():
                    if site_code in prefix:
                        pos = prefix.find(site_code)
                        if pos > 0:
                            author = prefix[:pos].rstrip("_")
                        break

                # Initialize nested structure
                if author not in experiments:
                    experiments[author] = {}
                    file_paths[author] = {}
                if prefix not in experiments[author]:
                    experiments[author][prefix] = {
                        "sparse_pointcloud": False,
                        "dense_pointcloud": False,
                        "extrinsics": False,
                        "intrinsics": False,
                        "report": False,
                    }
                    file_paths[author][prefix] = {
                        "sparse_pointcloud": "",
                        "dense_pointcloud": "",
                        "extrinsics": "",
                        "intrinsics": "",
                        "report": False,
                    }

                # Mark this file type as present and store file path
                experiments[author][prefix][file_type] = True
                file_paths[author][prefix][file_type] = str(f)

        # Check for code consistency within each experiment in verbose mode
        if verbose:
            # Check for inconsistencies within each experiment grouping
            for author, author_experiments in experiments.items():
                for experiment_code, file_presence in author_experiments.items():
                    # Get all file paths for this experiment that exist
                    experiment_file_paths = file_paths[author][experiment_code]
                    present_files = [path for path in experiment_file_paths.values() if path]

                    if len(present_files) > 1:  # Only check if we have multiple files
                        # Extract prefixes from all present files
                        prefixes = set()
                        for file_path in present_files:
                            filename = Path(file_path).name

                            # Extract prefix using same logic as above
                            if filename.lower().endswith("sparse_pointcloud.laz") or filename.lower().endswith(
                                "sparse_pointcloud.las"
                            ):
                                prefix = (
                                    filename.replace("sparse_pointcloud.laz", "")
                                    .replace("sparse_pointcloud.las", "")
                                    .replace("sparse_pointcloud.LAZ", "")
                                    .replace("sparse_pointcloud.LAS", "")
                                    .rstrip("_")
                                )
                            elif filename.lower().endswith("dense_pointcloud.laz") or filename.lower().endswith(
                                "dense_pointcloud.las"
                            ):
                                prefix = (
                                    filename.replace("dense_pointcloud.laz", "")
                                    .replace("dense_pointcloud.las", "")
                                    .replace("dense_pointcloud.LAZ", "")
                                    .replace("dense_pointcloud.LAS", "")
                                    .rstrip("_")
                                )
                            elif filename.lower().endswith("extrinsics.csv"):
                                prefix = (
                                    filename.replace("extrinsics.csv", "").replace("extrinsics.CSV", "").rstrip("_")
                                )
                                if prefix.lower().endswith("camera_model"):
                                    prefix = prefix[: -len("camera_model")].rstrip("_")
                            elif filename.lower().endswith("intrinsics.csv"):
                                prefix = (
                                    filename.replace("intrinsics.csv", "").replace("intrinsics.CSV", "").rstrip("_")
                                )
                                if prefix.lower().endswith("camera_model"):
                                    prefix = prefix[: -len("camera_model")].rstrip("_")

                            prefixes.add(prefix)

                        # Check if all prefixes are the same
                        if len(prefixes) > 1:
                            print(
                                f"Warning: {submission_dir.name} experiment {experiment_code} has files with different codes: {', '.join(sorted(prefixes))}"
                            )

        # Step 3: Create DataFrame from experiments
        for author, author_experiments in experiments.items():
            for experiment_code, file_presence in author_experiments.items():
                # Parse codes from experiment_code and build uppercase experiment code with XX for missing
                found_codes = {}
                experiment_parts = []

                # Keep author exactly as provided (preserve original case)
                experiment_parts.append(author)

                # Check each code category in order
                code_order = ["site", "dataset", "images", "calibration", "gcp", "coregistration", "multi-temporal"]

                for category in code_order:
                    code_dict = codes[category]
                    found_code = "XX"  # Default to XX if not found
                    found_meaning = "Unknown"

                    for code, meaning in code_dict.items():
                        if code in experiment_code:
                            found_code = code
                            found_meaning = meaning
                            break

                    if found_code == "XX":
                        if verbose:
                            print(f"Warning: experiment {experiment_code} missing or unrecognized code for {category}")

                    found_codes[category] = found_meaning
                    experiment_parts.append(found_code)

                # Build the formatted experiment code
                formatted_experiment_code = "_".join(experiment_parts)

                # Store file paths for this experiment
                files_dict[formatted_experiment_code] = file_paths[author][experiment_code]

                submission_summary = {
                    "submission_name": submission_dir.name,
                    "author": author,
                    "adjusted_experiment_code": formatted_experiment_code,
                    "site": found_codes.get("site", "Unknown"),
                    "dataset": found_codes.get("dataset", "Unknown"),
                    "images": found_codes.get("images", "Unknown"),
                    "calibration": found_codes.get("calibration", "Unknown"),
                    "gcp": found_codes.get("gcp", "Unknown"),
                    "coregistration": found_codes.get("coregistration", "Unknown"),
                    "multi-temporal": found_codes.get("multi-temporal", "Unknown"),
                    "sparse_pointcloud_file_found": file_presence["sparse_pointcloud"],
                    "dense_pointcloud_file_found": file_presence["dense_pointcloud"],
                    "extrinsics_file_found": file_presence["extrinsics"],
                    "intrinsics_file_found": file_presence["intrinsics"],
                    "report_file_found": file_presence["report"],
                }
                submissions.append(submission_summary)

    # Create DataFrame
    df = pd.DataFrame(submissions)
    if not df.empty:
        df = df.sort_values("submission_name").reset_index(drop=True)

    # Validation check: ensure all experiment codes have the same number of parts when split by '_'
    if files_dict:
        experiment_codes = list(files_dict.keys())
        code_lengths = [len(code.split("_")) for code in experiment_codes]

        if len(set(code_lengths)) > 1:
            print("Warning: Experiment codes have inconsistent number of parts:")
            for code in experiment_codes:
                parts = code.split("_")
                print(f"  {code}: {len(parts)} parts - {parts}")
        else:
            expected_length = 8  # author + 7 code parts
            if code_lengths and code_lengths[0] != expected_length:
                print(f"Warning: Expected {expected_length} parts in experiment codes, but found {code_lengths[0]}")

    return df, files_dict


def combine_intrinsics_files(files_dict: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """
    Combine all intrinsics CSV files from submissions into a single DataFrame.

    Parameters
    ----------
    files_dict : Dict[str, Dict[str, str]]
        Dictionary mapping experiment codes to file paths, as returned by analyze_submissions

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all intrinsics data, including split experiment code columns
    """
    combined_data = []

    for experiment_code, file_paths in files_dict.items():
        intrinsics_path = file_paths.get("intrinsics", "")

        try:
            # Read the CSV file
            df = pd.read_csv(intrinsics_path)

        except Exception as e:
            print(f"Warning: Could not read intrinsics file for {experiment_code}: {e}")
            # Create empty DataFrame with at least one row to preserve experiment metadata
            df = pd.DataFrame([{}])  # Single row with empty data

        # Add experiment code column
        df["experiment_code"] = experiment_code

        # Split experiment code into components
        code_parts = experiment_code.split("_")
        if len(code_parts) >= 8:  # Ensure we have all expected parts
            df["author"] = code_parts[0]
            df["site"] = code_parts[1]
            df["dataset"] = code_parts[2]
            df["images"] = code_parts[3]
            df["calibration"] = code_parts[4]
            df["gcp"] = code_parts[5]
            df["coregistration"] = code_parts[6]
            df["multi-temporal"] = code_parts[7]
        else:
            print(f"Warning: Experiment code {experiment_code} has insufficient parts ({len(code_parts)})")
            # Fill with empty strings for missing parts
            df["author"] = code_parts[0] if len(code_parts) > 0 else ""
            df["site"] = code_parts[1] if len(code_parts) > 1 else ""
            df["dataset"] = code_parts[2] if len(code_parts) > 2 else ""
            df["images"] = code_parts[3] if len(code_parts) > 3 else ""
            df["calibration"] = code_parts[4] if len(code_parts) > 4 else ""
            df["gcp"] = code_parts[5] if len(code_parts) > 5 else ""
            df["coregistration"] = code_parts[6] if len(code_parts) > 6 else ""
            df["multi-temporal"] = code_parts[7] if len(code_parts) > 7 else ""

        # Add the data to our combined list
        combined_data.append(df)

    # Combine all DataFrames
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        # Reorder columns to put experiment info first
        info_cols = [
            "experiment_code",
            "author",
            "site",
            "dataset",
            "images",
            "calibration",
            "gcp",
            "coregistration",
            "multi-temporal",
        ]
        other_cols = [col for col in combined_df.columns if col not in info_cols]
        combined_df = combined_df[info_cols + other_cols]
        return combined_df
    else:
        # Return empty DataFrame if no data found
        return pd.DataFrame()


def combine_extrinsics_files(files_dict: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """
    Combine all extrinsics CSV files from submissions into a single DataFrame.

    Parameters
    ----------
    files_dict : Dict[str, Dict[str, str]]
        Dictionary mapping experiment codes to file paths, as returned by analyze_submissions

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all extrinsics data, including split experiment code columns
    """
    combined_data = []

    for experiment_code, file_paths in files_dict.items():
        extrinsics_path = file_paths.get("extrinsics", "")

        try:
            # Read the CSV file
            df = pd.read_csv(extrinsics_path)

        except Exception as e:
            print(f"Warning: Could not find extrinsics file for {experiment_code}: {e}")
            # Create empty DataFrame with at least one row to preserve experiment metadata
            df = pd.DataFrame([{}])  # Single row with empty data

        # Add experiment code column
        df["experiment_code"] = experiment_code

        # Split experiment code into components
        code_parts = experiment_code.split("_")
        if len(code_parts) >= 8:  # Ensure we have all expected parts
            df["author"] = code_parts[0]
            df["site"] = code_parts[1]
            df["dataset"] = code_parts[2]
            df["images"] = code_parts[3]
            df["calibration"] = code_parts[4]
            df["gcp"] = code_parts[5]
            df["coregistration"] = code_parts[6]
            df["multi-temporal"] = code_parts[7]
        else:
            print(f"Warning: Experiment code {experiment_code} has insufficient parts ({len(code_parts)})")
            # Fill with empty strings for missing parts
            df["author"] = code_parts[0] if len(code_parts) > 0 else ""
            df["site"] = code_parts[1] if len(code_parts) > 1 else ""
            df["dataset"] = code_parts[2] if len(code_parts) > 2 else ""
            df["images"] = code_parts[3] if len(code_parts) > 3 else ""
            df["calibration"] = code_parts[4] if len(code_parts) > 4 else ""
            df["gcp"] = code_parts[5] if len(code_parts) > 5 else ""
            df["coregistration"] = code_parts[6] if len(code_parts) > 6 else ""
            df["multi-temporal"] = code_parts[7] if len(code_parts) > 7 else ""

        # Add the data to our combined list
        combined_data.append(df)

    # Combine all DataFrames
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        # Reorder columns to put experiment info first
        info_cols = [
            "experiment_code",
            "author",
            "site",
            "dataset",
            "images",
            "calibration",
            "gcp",
            "coregistration",
            "multi-temporal",
        ]
        other_cols = [col for col in combined_df.columns if col not in info_cols]
        combined_df = combined_df[info_cols + other_cols]
        return combined_df
    else:
        # Return empty DataFrame if no data found
        return pd.DataFrame()


def filter_experiment_data(
    df: pd.DataFrame,
    site: str = None,
    images: str = None,
    dataset: str = None,
    calibration: str = None,
    gcp: str = None,
    coregistration: str = None,
    multi_temporal: str = None,
) -> pd.DataFrame:
    """
    Filter experiment data by specified experimental conditions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with experiment data (from combine_intrinsics_files or combine_extrinsics_files)
    site : str, optional
        Site code to filter by (e.g., 'CG' for Casa Grande, 'IL' for Iceland)
    images : str, optional
        Images code to filter by (e.g., 'PP' for Pre-processed, 'RA' for Raw)
    dataset : str, optional
        Dataset code to filter by (e.g., 'AI' for Aerial, 'MC' for KH-9 MC, 'PC' for KH-9 PC)
    calibration : str, optional
        Calibration code to filter by (e.g., 'CY' for Yes, 'CN' for No)
    gcp : str, optional
        GCP code to filter by (e.g., 'GY' for Yes, 'GN' for No)
    coregistration : str, optional
        Coregistration code to filter by (e.g., 'PY' for Yes, 'PN' for No)
    multi_temporal : str, optional
        Multi-temporal code to filter by (e.g., 'MY' for Yes, 'MN' for No)

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    filtered_df = df.copy()

    # Apply filters only if parameters are specified
    if site is not None:
        filtered_df = filtered_df[filtered_df["site"] == site]
    if images is not None:
        filtered_df = filtered_df[filtered_df["images"] == images]
    if dataset is not None:
        filtered_df = filtered_df[filtered_df["dataset"] == dataset]
    if calibration is not None:
        filtered_df = filtered_df[filtered_df["calibration"] == calibration]
    if gcp is not None:
        filtered_df = filtered_df[filtered_df["gcp"] == gcp]
    if coregistration is not None:
        filtered_df = filtered_df[filtered_df["coregistration"] == coregistration]
    if multi_temporal is not None:
        filtered_df = filtered_df[filtered_df["multi-temporal"] == multi_temporal]

    return filtered_df


def mirror_as_symlinks(src_dir: str | Path, dst_dir: str | Path, overwrite: bool = False) -> None:
    """
    Create a mirrored directory structure where all files in the source directory
    are reproduced as symbolic links in the destination directory.

    The directory tree is preserved exactly, but every file becomes a symlink
    pointing to the original file in `src_dir`.

    Parameters
    ----------
    src_dir : str | Path
        Source directory containing real files.
    dst_dir : str | Path
        Destination directory where symlink copies will be created.
    overwrite : bool, optional
        If True, existing symlinks or files in the dst_dir will be replaced.
        Default is False.

    Returns
    -------
    None
        The function creates files/directories but returns nothing.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if not src_dir.is_dir():
        raise NotADirectoryError(f"Source directory does not exist: {src_dir}")

    # Create destination directory if needed
    dst_dir.mkdir(parents=True, exist_ok=True)

    for path in src_dir.rglob("*"):
        relative_path = path.relative_to(src_dir)
        target_path = dst_dir / relative_path

        if path.is_dir():
            # Recreate directory structure
            target_path.mkdir(exist_ok=True)
        else:
            # Create parent directories if missing
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Handle overwriting
            if target_path.exists():
                if overwrite:
                    if target_path.is_file() or target_path.is_symlink():
                        target_path.unlink()
                    else:
                        shutil.rmtree(target_path)
                else:
                    continue

            # Create symbolic link pointing to the source file
            target_path.symlink_to(path.resolve())


def parse_filename(file: str | Path) -> tuple[str, dict[str, Any]]:
    """
    Parse a filename following the predefined code convention described in FILE_CODE_MAPPING.

    This function extracts structured information from a filename built using a specific
    naming convention such as:
        AUTHOR_SITE_DATASET_IMAGES_CAMERAUSED_GCPUSED_POINTCLOUDCOREG_MTPADJ[_V1-DEM].tif

    Each short code (e.g., 'CG', 'AI', 'RA', 'CY') is validated against FILE_CODE_MAPPING
    to ensure consistency and then mapped to its corresponding descriptive value.

    Args:
        file: Path or filename to parse.

    Returns:
        tuple[str, dict]:
            - code: normalized filename code (e.g., "ALICE_CG_AI_RA_CY_GY_PY_MY_V1")
            - metadatas: dictionary of parsed metadata fields mapped to their descriptive values.

    Raises:
        ValueError: If the filename does not respect the expected naming convention
                    or contains unknown codes not defined in FILE_CODE_MAPPING.
    """
    stem = Path(file).stem
    parts = stem.split("_")

    raw: dict[str, str | None] = {}
    for i, (seg_name, seg_pattern) in enumerate(_SEGMENT_PATTERNS):
        if i >= len(parts):
            raise FilenameParseError(stem, f"filename too short, missing segment '{seg_name}'", seg_name, None)
        if not seg_pattern.match(parts[i]):
            raise FilenameParseError(stem, "invalid segment value", seg_name, parts[i])
        raw[seg_name] = parts[i]

    version_idx = len(_SEGMENT_PATTERNS)
    raw["version"] = parts[version_idx] if version_idx < len(parts) and _VERSION_PATTERN.match(parts[version_idx]) else None

    metadatas: dict[str, Any] = {"author": raw["author"]}
    for key, value in raw.items():
        if key in FILE_CODE_MAPPING and value is not None:
            metadatas[key] = FILE_CODE_MAPPING[key].get(value)
    metadatas["version"] = raw["version"]

    code = "_".join(v for v in raw.values() if v is not None)
    return code, metadatas


class ReferencesData:
    def __init__(self, references_data_mapping: dict[tuple[str, str], dict[str, str | Path]]):
        """
        Initialize a ReferencesData instance, which manages access to reference data
        for multiple sites and datasets, including DEMs, DEM masks, and landcover rasters.

        The class provides methods to retrieve the appropriate reference files for a
        given site and dataset, ensuring that all expected files exist.

        Parameters
        ----------
        references_data_mapping : dict
            A mapping from (site, dataset) tuples to dictionaries containing file paths for:
            - "ref_dem": reference DEM raster
            - "ref_dem_mask": corresponding DEM mask
            - "landcover": landcover raster

        Raises
        ------
        KeyError
            If expected (site, dataset) keys or sub-keys are missing.
        FileNotFoundError
            If any referenced file does not exist.
        """
        self.__check_keys_validity(list(references_data_mapping.keys()))
        self.__check_values_validity(references_data_mapping)
        self.__references_data_mapping = references_data_mapping

    def get_ref_dem(self, site: str, dataset: str) -> Path:
        return Path(self.__references_data_mapping[(site, dataset)]["ref_dem"])

    def get_ref_dem_mask(self, site: str, dataset: str) -> Path:
        return Path(self.__references_data_mapping[(site, dataset)]["ref_dem_mask"])

    def get_landcover(self, site: str, dataset: str) -> Path:
        return Path(self.__references_data_mapping[(site, dataset)]["landcover"])

    @staticmethod
    def __check_keys_validity(keys: list[tuple[str, str]]) -> None:
        expected_keys = {
            (site, dataset)
            for site in FILE_CODE_MAPPING["site"].values()
            for dataset in FILE_CODE_MAPPING["dataset"].values()
        }

        # Detect missing keys
        missing_keys = expected_keys - set(keys)
        if missing_keys:
            raise KeyError(
                f"The following (site, dataset) keys are missing in references_data_mapping: {sorted(missing_keys)}"
            )

    @staticmethod
    def __check_values_validity(references_data_mapping: dict[tuple[str, str], dict[str, str | Path]]) -> None:
        expected_sub_keys = set(["ref_dem", "ref_dem_mask", "landcover"])

        for (site, dataset), sub_dict in references_data_mapping.items():
            sub_keys = set(sub_dict.keys())

            # Detect missing keys
            missing_keys = expected_sub_keys - set(sub_keys)
            if missing_keys:
                raise KeyError(
                    f"The following ({site}, {dataset}) sub keys are missing in references_data_mapping: {sorted(missing_keys)}"
                )

            for file_type, file_path in sub_dict.items():
                fp = Path(file_path)
                if not fp.exists():
                    raise FileNotFoundError(
                        f"File '{fp}' for type '{file_type}' in ({site}, {dataset}) does not exist."
                    )


def get_filepaths_df(**kwargs: Iterable[str | Path]) -> pd.DataFrame:
    df = pd.DataFrame()
    df.index.name = "code"

    for key, files in kwargs.items():
        for f in files:
            try:
                code, metadatas = parse_filename(f)
                for k, v in metadatas.items():
                    df.at[code, k] = v
                df.at[code, key] = str(f)
            except ValueError:
                continue
    return df.sort_index()


def download_planned_submissions(outfile: str | Path) -> None:
    """
    Download the table of planned submissions from the shared Google sheet document.
    File is downloaded to `outfile` in CSV format and unused rows/columns are deleted.
    """
    # Convert POSIX path to str
    outfile= str(outfile)

    sheet_url="https://docs.google.com/spreadsheets/d/1jGuoQYSfSd-DqbKp_7ZpkTYFIgC47camspTsgqkT_gQ/edit?usp=sharing"
    verbose=(logger.getEffectiveLevel()<=20)
    gdown.download(url=sheet_url, output=outfile, format="csv", quiet=(not verbose))

    # Load as DataFrame and filter empty rows and Total row
    submissions_df = pd.read_csv(outfile, skiprows=2, delimiter=",", usecols=[0, 1, 2, 3, 4])
    submissions_df = submissions_df[~(submissions_df["Group name"] == "Total") & ~submissions_df["Group name"].isna()]

    # Save to file
    submissions_df.to_csv(outfile)

    logger.info(f"Found {len(submissions_df)} planned submissions.")
    logger.info(f"Saved CSV to file {outfile}.")

    return submissions_df