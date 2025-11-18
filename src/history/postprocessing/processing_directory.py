from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from history.postprocessing.io import FILE_CODE_MAPPING, parse_filename
from history.postprocessing.visualization import plot_files_recap


class ProcessingDirectory:
    def __new__(cls, base_dir: str | Path | ProcessingDirectory):
        # If an instance of the same class is passed, return it directly
        if isinstance(base_dir, ProcessingDirectory):
            return base_dir

        # Otherwise, create a new instance normally
        return super().__new__(cls)

    def __init__(self, base_dir: str | Path | ProcessingDirectory):
        if isinstance(base_dir, ProcessingDirectory):
            return

        self.base_dir = Path(base_dir)

        self.sub_dirs = {
            (site, dataset): SubProcessingDirectory(self.base_dir / site / dataset)
            for site in FILE_CODE_MAPPING["site"].values()
            for dataset in FILE_CODE_MAPPING["dataset"].values()
        }

    def get_filepaths_df(self) -> pd.DataFrame:
        dfs = [sub_dir.get_filepaths_df() for sub_dir in self.sub_dirs.values()]
        dfs = [df for df in dfs if not df.empty]  # optionnel : filtrer les None / vides

        if not dfs:
            # Return an empty DataFrame with no error
            return pd.DataFrame()

        return pd.concat(dfs).sort_index(ascending=False)

    def get_statistics(self) -> pd.DataFrame:
        dfs = [sub_dir.get_statistics() for sub_dir in self.sub_dirs.values()]

        if not dfs:
            raise ValueError("No statistics founds")

        return pd.concat(dfs)

    def get_landcover_statistics(self) -> pd.DataFrame:
        dfs = [sub_dir.get_landcover_statistics() for sub_dir in self.sub_dirs.values()]

        if not dfs:
            raise ValueError("No landcover statistics founds")
        return pd.concat(dfs)

    def get_std_landcover_statistics(self) -> pd.DataFrame:
        dfs = [
            sub_dir.get_std_landcover_statistics()
            for sub_dir in self.sub_dirs.values()
            if sub_dir.std_landcover_statistics_file.exists()
        ]

        if not dfs:
            raise ValueError("No std landcover statistics founds.")
        return pd.concat(dfs)

    def plot(self) -> None:
        plot_files_recap(self.get_filepaths_df())

    def submissions_summary(self) -> None:
        """
        Print a high-level summary of all processed submissions.

        This method extracts the submission metadata into a DataFrame and reports:
        - the number of successfully processed submissions,
        - the list of submissions missing mandatory files,
        - the number of unique participants,
        - a cross-tabulated summary of submissions per site and dataset.

        Returns
        -------
        None
            This function prints a formatted summary to the console.
        """
        df = self.get_filepaths_df()
        if df.empty:
            print(f"{str(self.base_dir)}: Empty")
            return
        print("Submission Summary : ")
        print(f" - Total of submissions sucessfully processed: {len(df['coreg_dem_file'].dropna())}/{len(df)}")
        print(" - List of problematic submissions: ")
        for code in df.index[df["coreg_dem_file"].isna()]:
            print(f"\t{code}")

        print(f" - Number of participants : {len(df['author'].unique())}")
        print(" - Number of submission per site/dataset: ")
        ct = pd.crosstab(df["site"], df["dataset"])
        print("\t" + ct.to_string().replace("\n", "\n\t"))

    def __str__(self):
        df = self.get_filepaths_df()
        if df.empty:
            return f"{str(self.base_dir)}: Empty"
        list_str = [f"{c} : {len(df[c].dropna())}" for c in df.columns if c.endswith("_file")]
        return str(self.base_dir) + " :\n" + "\n".join(list_str)


class SubProcessingDirectory:
    def __new__(cls, base_dir: str | Path | SubProcessingDirectory):
        # If an instance of the same class is passed, return it directly
        if isinstance(base_dir, SubProcessingDirectory):
            return base_dir

        # Otherwise, create a new instance normally
        return super().__new__(cls)

    def __init__(self, base_dir: str | Path | SubProcessingDirectory):
        if isinstance(base_dir, SubProcessingDirectory):
            return

        self.base_dir = Path(base_dir)
        self.pointclouds_dir = self.base_dir / "pointclouds"
        self.symlinks_dir = SymlinksDirectory(self.base_dir / "symlinks")
        self.raw_dems_dir = self.base_dir / "raw_dems"
        self.coreg_dems_dir = self.base_dir / "coreg_dems"
        self.std_dems_dir = self.base_dir / "std_dems"
        self.ddems_before_dir = self.base_dir / "ddems" / "before_coregistration"
        self.ddems_after_dir = self.base_dir / "ddems" / "after_coregistration"
        self.aux_data = self.base_dir / "aux_data"
        self.statistics_file = self.base_dir / "statistics.csv"
        self.landcover_statistics_file = self.base_dir / "landcover_statistics.csv"
        self.std_landcover_statistics_file = self.base_dir / "std_landcover_statistics.csv"

        self.site = self.base_dir.parent.name
        self.dataset = self.base_dir.name

    def get_raw_dems(self) -> list[Path]:
        return list(self.raw_dems_dir.glob("*-DEM.tif"))

    def get_coreg_dems(self) -> list[Path]:
        return list(self.coreg_dems_dir.glob("*-DEM.tif"))

    def get_ddems_before(self) -> list[Path]:
        return list(self.ddems_before_dir.glob("*-DDEM.tif"))

    def get_ddems_after(self) -> list[Path]:
        return list(self.ddems_after_dir.glob("*-DDEM.tif"))

    def get_reference_dem(self) -> Path:
        return self._find_aux_file(r"ref_dem(?!.*mask)", "Reference DEM")

    def get_reference_dem_mask(self) -> Path:
        return self._find_aux_file(r"ref_dem.*mask", "Reference DEM mask")

    def get_reference_landcover(self) -> Path:
        return self._find_aux_file(r"landcover", "Landcover file")

    def get_filepaths_df(self) -> pd.DataFrame:
        mapping = {
            "dense_pointcloud_file": self.symlinks_dir.get_dense_pointclouds(),
            "raw_dem_file": self.get_raw_dems(),
            "coreg_dem_file": self.get_coreg_dems(),
            "ddem_before_file": self.get_ddems_before(),
            "ddem_after_file": self.get_ddems_after(),
        }
        df = pd.DataFrame(columns=mapping.keys())
        df.index.name = "code"
        for colname, files in mapping.items():
            for f in files:
                code, metadatas = parse_filename(f)
                if code not in df.index:
                    for k, v in metadatas.items():
                        df.at[code, k] = v
                df.at[code, colname] = f
        return df.sort_index(ascending=False)

    def get_statistics(self) -> pd.DataFrame:
        return pd.read_csv(self.statistics_file, index_col="code")

    def get_landcover_statistics(self) -> pd.DataFrame:
        return pd.read_csv(self.landcover_statistics_file)

    def get_std_landcover_statistics(self) -> pd.DataFrame:
        return pd.read_csv(self.std_landcover_statistics_file)

    def _find_aux_file(self, pattern: str, description: str) -> Path:
        """
        Search for a file in aux_data matching a regex pattern.

        Args:
            pattern: regex pattern to match against the file stem
            description: human-readable description for error message

        Returns:
            Path to the first matching file

        Raises:
            FileNotFoundError: if no file matches
        """
        for fp in Path(self.aux_data).rglob("*.tif"):
            if re.search(pattern, fp.stem, re.IGNORECASE):
                return fp

        raise FileNotFoundError(
            f"{description} not found.\n"
            f"Please ensure that the 'aux_data' directory exists and contains the required file.\n"
            f"Expected structure (example):\n"
            f"  {self.aux_data}/\n"
            "    ├── ref_dem.tif\n"
            "    ├── ref_dem_mask.tif\n"
            "    └── landcover.tif"
        )


class SymlinksDirectory:
    def __new__(cls, base_dir: str | Path | SymlinksDirectory):
        # If an instance of the same class is passed, return it directly
        if isinstance(base_dir, SymlinksDirectory):
            return base_dir

        # Otherwise, create a new instance normally
        return super().__new__(cls)

    def __init__(self, base_dir: str | Path | SymlinksDirectory):
        if isinstance(base_dir, SymlinksDirectory):
            return

        self.base_dir = Path(base_dir)
        self.dense_pointclouds_dir = self.base_dir / "dense_pointclouds"
        self.sparse_pointclouds_dir = self.base_dir / "sparse_pointclouds"
        self.extrinsics_dir = self.base_dir / "extrinsics"
        self.intrinsics_dir = self.base_dir / "intrinsics"
        self.raw_dems_dir = self.base_dir / "raw_dems"

    def get_dense_pointclouds(self) -> list[Path]:
        return list(self.dense_pointclouds_dir.glob("*.las")) + list(self.dense_pointclouds_dir.glob("*.laz"))
