import tomllib
from dataclasses import dataclass
from pathlib import Path

from history.postprocessing.io import ReferencesData



@dataclass(frozen=True)
class ProcConfig:
    base_dir: Path
    symlinks_dir: Path 
    raw_dems_dir: Path
    coreg_dems_dir: Path 
    before_coreg_ddems_dir: Path
    after_coreg_ddems_dir: Path
    std_dems_dir: Path

    @classmethod
    def from_base_dir(cls, base_dir: Path) -> 'ProcConfig':
        return cls(
            base_dir=base_dir,
            symlinks_dir=base_dir / "symlinks",
            raw_dems_dir=base_dir / "raw_dems",
            coreg_dems_dir=base_dir / "coregistered_dems",
            before_coreg_ddems_dir=base_dir / "ddems" / "before_coregistration",
            after_coreg_ddems_dir=base_dir / "ddems" / "after_coregistration",
            std_dems_dir=base_dir / "std_dems"
        )


@dataclass(frozen=True)
class Config:
    raw_dir: Path
    extracted_dir: Path
    proc_dir: ProcConfig
    plot_dir: Path

    references_data_mapping: ReferencesData

    overwrite: bool = False
    dry_run: bool = False
    no_plots: bool = False
    pdal_exec_path: str = "pdal"
    max_workers: int = 4

    @classmethod
    def from_toml_file(cls, path: Path) -> "Config":
        with open(path, "rb") as f:
            data = tomllib.load(f)

        references_data_mapping = {
            (site, dataset): paths
            for site, datasets in data["references_data_mapping"].items()
            for dataset, paths in datasets.items()
        }

        return cls(
            raw_dir=Path(data["raw_dir"]),
            extracted_dir=Path(data["extracted_dir"]),
            proc_dir=ProcConfig.from_base_dir(Path(data["proc_dir"])),
            plot_dir=Path(data["plot_dir"]),
            references_data_mapping=ReferencesData(references_data_mapping),
            overwrite=data.get("overwrite", False),
            dry_run=data.get("dry_run", False),
            pdal_exec_path=data.get("pdal_exec_path", "pdal"),
            max_workers=data.get("max_workers", 4),
        )





    