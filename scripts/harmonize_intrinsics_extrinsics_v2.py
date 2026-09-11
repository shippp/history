"""Harmonize extrinsics and intrinsics CSV files for the v2 history postprocess run.

Not generic: it patches the specific naming/mandatory-column/row-count issues found in the
v2 submissions, nothing more. All paths are hardcoded to that run.

Must be run after the `symlinks` step. Each rule below is a small, independent function
that takes a DataFrame and returns a (possibly unchanged) one; a file a rule can't fix is
left as-is and simply fails downstream instead. Every fixed CSV replaces its symlink in
place, without ever touching the original file it pointed to (see replace_symlink_with_csv).
"""

import sys
from pathlib import Path

import pandas as pd
import rasterio
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from history.config import Config
from history.postprocessing.io import parse_filename

CONFIG_PATH = "/mnt/summer/USERS/DEHECQA/history/output/v2/config.toml"


def replace_symlink_with_csv(path: Path, df: pd.DataFrame) -> None:
    """Write df as CSV at path, unlinking a symlink there first so its target is never touched."""
    if path.is_symlink():
        path.unlink()
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Extrinsics rules
# ---------------------------------------------------------------------------


def rule_add_x_map_y_map(df: pd.DataFrame, crs) -> pd.DataFrame:
    """Fill missing x_map/y_map columns by reprojecting lon/lat into crs."""
    if {"x_map", "y_map"} <= set(df.columns) or not {"lon", "lat"} <= set(df.columns):
        return df

    valid = df["lon"].between(-180, 180) & df["lat"].between(-90, 90)
    if not valid.any():
        return df

    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    df.loc[valid, "x_map"], df.loc[valid, "y_map"] = transformer.transform(
        df.loc[valid, "lon"].to_numpy(), df.loc[valid, "lat"].to_numpy()
    )
    return df


def harmonize_extrinsics(config: Config) -> None:
    """Apply all extrinsics rules to every extrinsics symlink."""
    extrinsics_dir = config.proc_dir.symlinks_dir / "extrinsics"
    crs_cache: dict[tuple[str, str], rasterio.crs.CRS] = {}

    for file in sorted(extrinsics_dir.glob("*.csv")):
        _, metadata = parse_filename(file.name)
        site, dataset = metadata["site"], metadata["dataset"]

        if (site, dataset) not in crs_cache:
            with rasterio.open(config.references_data_mapping.get_ref_dem(site, dataset)) as src:
                crs_cache[(site, dataset)] = src.crs

        df = pd.read_csv(file)
        df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

        df = rule_add_x_map_y_map(df, crs_cache[(site, dataset)])

        replace_symlink_with_csv(file, df)


# ---------------------------------------------------------------------------
# Intrinsics rules
# ---------------------------------------------------------------------------

def rule_rename_principal_point_x(df: pd.DataFrame) -> pd.DataFrame:
    """Rename an obvious principal-point-x synonym (xp, x0, cx) to principal_point_x_mm."""
    if "principal_point_x_mm" in df.columns:
        return df
    for synonym in ["xp", "x0", "cx"]:
        if synonym in df.columns:
            return df.rename(columns={synonym: "principal_point_x_mm"})
    return df


def rule_rename_principal_point_y(df: pd.DataFrame) -> pd.DataFrame:
    """Rename an obvious principal-point-y synonym (yp, y0, cy) to principal_point_y_mm."""
    if "principal_point_y_mm" in df.columns:
        return df
    for synonym in ["yp", "y0", "cy"]:
        if synonym in df.columns:
            return df.rename(columns={synonym: "principal_point_y_mm"})
    return df


def rule_fix_focal_length_typo(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the 'focal_lenght' typo to 'focal_length'."""
    if "focal_length" not in df.columns and "focal_lenght" in df.columns:
        return df.rename(columns={"focal_lenght": "focal_length"})
    return df


def rule_fill_missing_focal_length(df: pd.DataFrame) -> pd.DataFrame:
    """Fill a still-missing focal_length column with NaN."""
    if "focal_length" not in df.columns:
        df["focal_length"] = float("nan")
    return df


def rule_fill_missing_pixel_pitch(df: pd.DataFrame) -> pd.DataFrame:
    """Fill a still-missing pixel_pitch column with NaN."""
    if "pixel_pitch" not in df.columns:
        df["pixel_pitch"] = float("nan")
    return df


def harmonize_intrinsics(config: Config) -> None:
    """Apply all intrinsics rules to every single-row intrinsics symlink.

    Multi-row files are left untouched -- they fail downstream on the row-count check instead.
    """
    intrinsics_dir = config.proc_dir.symlinks_dir / "intrinsics"

    for file in sorted(intrinsics_dir.glob("*.csv")):
        df = pd.read_csv(file)
        df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

        if len(df) != 1:
            continue

        df = rule_rename_principal_point_x(df)
        df = rule_rename_principal_point_y(df)
        df = rule_fix_focal_length_typo(df)
        df = rule_fill_missing_focal_length(df)
        df = rule_fill_missing_pixel_pitch(df)

        replace_symlink_with_csv(file, df)


if __name__ == "__main__":
    config = Config.from_toml_file(Path(CONFIG_PATH))
    harmonize_extrinsics(config)
    harmonize_intrinsics(config)
