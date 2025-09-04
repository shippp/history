import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import py7zr

from .io import *


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


def uncompress_all_submissions(
    data_dir: Union[str, Path] = "/path/to/submissions/",
    output_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    dry_run: bool = True,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Uncompress all archive files in the submissions data directory.

    Supports multiple archive formats: .zip, .7z, .tgz, .tar.gz, .tar.bz2

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the compressed submission files
    output_dir : str or Path, optional
        Directory to extract files to. If None, creates 'extracted' subdirectory
    overwrite : bool, default False
        Whether to overwrite existing extracted directories
    dry_run : bool, default False
        If True, only print what would be done without actually extracting

    Returns
    -------
    Dict[str, str]
        Dictionary mapping archive filename to extraction directory path

    """
    if verbose and dry_run:
        print("Dry run:", dry_run)

    data_dir = Path(data_dir)
    if output_dir is None:
        output_dir = data_dir
    else:
        output_dir = Path(output_dir)

    # Create output directory
    if not dry_run:
        output_dir.mkdir(exist_ok=True)

    # Define supported archive extensions
    archive_extensions = {
        ".zip": _extract_zip,
        ".7z": _extract_7z,
        ".tgz": _extract_tar,
        ".tar.gz": _extract_tar,
        ".tar.bz2": _extract_tar,
        ".tar.xz": _extract_tar,
    }

    results = {}

    # Find all archive files
    archive_files = []
    # Sort extensions by length (longest first) to handle .tar.gz before .gz
    sorted_extensions = sorted(archive_extensions.keys(), key=len, reverse=True)
    for ext in sorted_extensions:
        archive_files.extend(data_dir.glob(f"*{ext}"))

    # Remove duplicates and macOS metadata files
    seen = set()
    unique_files = []
    for f in archive_files:
        if f not in seen and not f.name.startswith("._"):
            seen.add(f)
            unique_files.append(f)
    archive_files = unique_files

    if verbose:
        print(f"Found {len(archive_files)} archive files")
        [print(i) for i in sorted(archive_files)]

        # Check disk space and estimate requirements
        space_info = check_disk_space_and_estimate(archive_files, output_dir)
        print("\nDisk space analysis:")
        print(f"  Available space: {space_info['available_gb']:.1f} GB")
        print(f"  Archive size: {space_info['archive_size_gb']:.1f} GB")
        print(f"  Estimated extracted size: {space_info['estimated_extracted_gb']:.1f} GB")

        if space_info["estimated_extracted_gb"] > space_info["available_gb"]:
            print("WARNING: Estimated extracted size exceeds available space!")
        else:
            print("Sufficient disk space available")
        print()

    # Process each archive file
    for archive_file in sorted(archive_files):
        result = extract_single_archive(archive_file, output_dir, overwrite=overwrite, dry_run=dry_run, verbose=verbose)
        if result is not None:
            results[archive_file.name] = result

    # Check if any files are archives
    archive_files_l2 = []
    for ext in sorted_extensions:
        archive_files_l2.extend(output_dir.glob(f"**/*{ext}"))

    if verbose:
        print(f"\nFound {len(archive_files_l2)} encapsulated archive files")
        [print(i) for i in sorted(archive_files_l2)]

    if len(archive_files_l2) > 0:
        for archive_file in sorted(archive_files_l2):
            archive_dir = archive_file.parent
            result = extract_single_archive(
                archive_file, archive_dir, overwrite=overwrite, dry_run=dry_run, verbose=verbose
            )
            if result is not None:
                results[archive_file.name] = result
                archive_file.unlink()  # delete encapsulated archive

    if verbose:
        print(f"Extraction complete. Processed {len(results)} files.")

    return results


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

    # Define the expected mandatory files
    mandatory_files = [
        "sparse_pointcloud.laz",  # or .las
        "dense_pointcloud.laz",  # or .las
        "extrinsics.csv",
        "intrinsics.csv",
    ]

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
