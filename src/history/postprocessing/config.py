import tomllib
from dataclasses import dataclass
from pathlib import Path

from history.postprocessing.io import ReferencesData


@dataclass(frozen=True)
class ProcConfig:
    """
    Directory layout for all intermediate processing outputs.

    All paths are derived from a single ``base_dir`` via ``from_base_dir``.

    Attributes
    ----------
    base_dir : Path
        Root of the processing tree.
    symlinks_dir : Path
        Typed symlink directories created by the ``symlinks`` step
        (``dense_pointclouds/``, ``sparse_pointclouds/``, ``extrinsics/``,
        ``intrinsics/``, ``dems/``).
    raw_dems_dir : Path
        DEMs produced by the ``point2dem`` step (``*-DEM.tif``).
    coreg_dems_dir : Path
        DEMs produced by the ``coregister`` step (``*-DEM.tif``).
    before_coreg_ddems_dir : Path
        Differential DEMs computed from raw DEMs (``*-DDEM.tif``).
    after_coreg_ddems_dir : Path
        Differential DEMs computed from coregistered DEMs (``*-DDEM.tif``).
    std_dems_dir : Path
        Standard-deviation DEMs, one per (site, dataset) group
        (``<site>_<dataset>_std_dem.tif``).
    """

    base_dir: Path
    symlinks_dir: Path
    raw_dems_dir: Path
    coreg_dems_dir: Path
    before_coreg_ddems_dir: Path
    after_coreg_ddems_dir: Path
    std_dems_dir: Path

    @classmethod
    def from_base_dir(cls, base_dir: Path) -> "ProcConfig":
        """Build a ``ProcConfig`` with all sub-paths rooted at ``base_dir``."""
        return cls(
            base_dir=base_dir,
            symlinks_dir=base_dir / "symlinks",
            raw_dems_dir=base_dir / "raw_dems",
            coreg_dems_dir=base_dir / "coregistered_dems",
            before_coreg_ddems_dir=base_dir / "ddems" / "before_coregistration",
            after_coreg_ddems_dir=base_dir / "ddems" / "after_coregistration",
            std_dems_dir=base_dir / "std_dems",
        )


@dataclass(frozen=True)
class Config:
    """
    Full runtime configuration for the post-processing pipeline.

    Loaded from a TOML file via ``from_toml_file``. CLI flags (``--overwrite``,
    ``--dry-run``, ``--no-plots``, ``--max-workers``) can override the values
    read from the file.

    Attributes
    ----------
    raw_dir : Path
        Directory containing compressed submission archives (input).
    extracted_dir : Path
        Directory where archives are extracted.
    proc_dir : ProcConfig
        Intermediate processing directory layout.
    plot_dir : Path
        Directory where all output plots are saved.
    references_data_mapping : ReferencesData
        Reference DEMs, masks, and landcover rasters for every
        (site, dataset) combination.
    overwrite : bool
        If True, existing outputs are recomputed. Default False.
    dry_run : bool
        If True, PDAL commands are prepared but not executed. Default False.
    no_plots : bool
        If True, plot generation is skipped for all steps. Default False.
    pdal_exec_path : str
        Path or name of the PDAL executable. Default ``"pdal"``.
    max_workers : int
        Number of parallel worker threads used by processing steps. Default 4.
    """

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
        """
        Load a ``Config`` instance from a TOML configuration file.

        Parameters
        ----------
        path : Path
            Path to the TOML file (see ``config.exemple.toml`` for the expected
            structure).

        Returns
        -------
        Config
            Fully initialised configuration object.
        """
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





    