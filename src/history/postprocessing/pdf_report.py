"""
Assemble pipeline output PNGs into a single PDF report.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)

# All (site, dataset) pairs in display order
_SITE_DATASET_PAIRS = [
    ("casa_grande", "aerial"),
    ("casa_grande", "kh9mc"),
    ("casa_grande", "kh9pc"),
    ("iceland", "aerial"),
    ("iceland", "kh9mc"),
    ("iceland", "kh9pc"),
]

_SITE_DISPLAY = {"casa_grande": "Casa Grande", "iceland": "Iceland"}
_DATASET_DISPLAY = {"aerial": "Aerial", "kh9mc": "KH-9 MC", "kh9pc": "KH-9 PC"}


# ──────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────────────────────────────────────


# Standard A4 portrait dimensions in inches
_PAGE_W = 8.27
_PAGE_H = 11.69


def _add_png_page(
    pdf: PdfPages,
    path: Path,
    orientation: Literal["auto", "portrait", "landscape"] = "auto",
    dpi: int = 150,
) -> None:
    """Insert an existing PNG as a new PDF page; warn and skip if absent.

    The image is centred and scaled to fill the available area while preserving
    its aspect ratio. Page orientation is determined by ``orientation``:
    ``"portrait"`` and ``"landscape"`` force A4 in that direction; ``"auto"``
    picks landscape when the image is wider than tall, portrait otherwise.
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"Missing figure (skipping): {path}")
        return

    img = mpimg.imread(str(path))
    h_px, w_px = img.shape[:2]

    if orientation == "landscape" or (orientation == "auto" and w_px > h_px):
        page_w, page_h = _PAGE_H, _PAGE_W  # A4 landscape
    else:
        page_w, page_h = _PAGE_W, _PAGE_H  # A4 portrait

    fig, ax = plt.subplots(figsize=(page_w, page_h))
    ax.imshow(img, aspect="equal")
    ax.axis("off")

    img_aspect = w_px / h_px
    page_aspect = page_w / page_h
    if img_aspect > page_aspect:
        ax_w = 1.0
        ax_h = page_aspect / img_aspect
    else:
        ax_h = 1.0
        ax_w = img_aspect / page_aspect

    ax.set_position([(1 - ax_w) / 2, (1 - ax_h) / 2, ax_w, ax_h])
    pdf.savefig(fig, dpi=dpi)
    plt.close(fig)


def _add_section_title_page(pdf: PdfPages, title: str) -> None:
    """Add a centred text page as a section separator (A4 portrait)."""
    fig, ax = plt.subplots(figsize=(_PAGE_W, _PAGE_H))
    ax.axis("off")
    ax.text(
        0.5, 0.5, title,
        ha="center", va="center",
        fontsize=28, fontweight="bold",
        transform=ax.transAxes,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _build_summary_page(pdf: PdfPages, df: pd.DataFrame) -> None:
    """Build the opening summary page: header text + site×dataset submission count table."""
    n_submissions = len(df)
    n_authors = df["author"].nunique() if "author" in df.columns else "?"

    # Build count pivot; rename to human-readable labels
    counts: pd.DataFrame = df.groupby(["site", "dataset"]).size().unstack(fill_value=0)
    counts.index = [_SITE_DISPLAY.get(s, s) for s in counts.index]
    counts.columns = [_DATASET_DISPLAY.get(d, d) for d in counts.columns]

    # Ensure every expected column is present (zero if no submissions)
    for col in _DATASET_DISPLAY.values():
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[[c for c in _DATASET_DISPLAY.values() if c in counts.columns]]

    fig, ax = plt.subplots(figsize=(_PAGE_W, _PAGE_H))
    ax.axis("off")

    today = datetime.now().strftime("%d %b %Y")
    ax.text(
        0.5, 0.88, "Submissions Summary",
        ha="center", va="top",
        fontsize=26, fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.78,
        f"As of {today} — {n_submissions} submissions from {n_authors} different authors",
        ha="center", va="top",
        fontsize=13, style="italic",
        transform=ax.transAxes,
    )

    # Use reset_index so the site name becomes a normal column with controllable width
    counts_display = counts.reset_index()
    counts_display.columns = [""] + list(counts.columns)

    table = ax.table(
        cellText=counts_display.values.tolist(),
        colLabels=counts_display.columns.tolist(),
        loc="center",
        cellLoc="center",
        bbox=[0.15, 0.52, 0.7, 0.18]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    pdf.savefig(fig)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def generate_pdf_report(
    extracted_dir: Path,
    plot_dir: Path,
    filename_renames: dict[str, str] | None = None,
    output_path: Path | None = None,
    orientation: Literal["auto", "portrait", "landscape"] = "auto",
    overwrite: bool = False,
) -> Path:
    """
    Assemble all pipeline output PNGs into a single PDF report.

    Parameters
    ----------
    extracted_dir :
        Directory containing extracted submission subdirectories; used to compute
        the summary statistics shown on the title page.
    plot_dir :
        Root of the plot directory produced by the pipeline steps.
    filename_renames :
        Optional rename rules forwarded to ``io.scan_submissions``.
    output_path :
        Destination path for the PDF. Defaults to ``plot_dir/report.pdf``.
    orientation :
        Page orientation for image pages. ``"portrait"`` forces A4 portrait,
        ``"landscape"`` forces A4 landscape, ``"auto"`` (default) picks
        landscape when the image is wider than tall and portrait otherwise.

    Returns
    -------
    Path
        Path to the generated PDF file.
    """
    from history.postprocessing import io
    from history.postprocessing.io import is_output_up_to_date

    plot_dir = Path(plot_dir)
    output_path = Path(output_path) if output_path else plot_dir / "report.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        png_files = list(plot_dir.rglob("*.png"))
        if png_files and is_output_up_to_date(png_files, output_path):
            logger.info(f"PDF report {output_path} is up to date -> skipping.")
            return output_path

    df = io.scan_submissions(extracted_dir, filename_renames=filename_renames)

    with PdfPages(output_path) as pdf:

        # 1. Summary page (generated programmatically from scan_submissions) ──
        _build_summary_page(pdf, df)

        # 2. Sankey diagram of planned submissions
        _add_png_page(pdf, plot_dir / "planned_submissions_sankey.png", orientation)

        # 3. Submissions presence map + file sizes
        _add_png_page(pdf, plot_dir / "files_presence_map.png", orientation)
        _add_png_page(pdf, plot_dir / "submissions_file_sizes.png", orientation)

        # 4. Global plots
        _add_png_page(pdf, plot_dir / "pointcloud_point_count.png", orientation)
        _add_png_page(pdf, plot_dir / "nmad_after_coregistration.png", orientation)

        # 5. Per-(site, dataset) sections
        for site, dataset in _SITE_DATASET_PAIRS:
            sub_dir = plot_dir / f"{site}_{dataset}"
            _add_section_title_page(pdf, f"{_SITE_DISPLAY[site]}  —  {_DATASET_DISPLAY[dataset]}")

            _add_png_page(pdf, sub_dir / "mosaic" / "mosaic_sparse_pointcloud_diff.png", orientation)
            _add_png_page(pdf, sub_dir / "mosaic" / "mosaic_raw_dem.png", orientation)
            _add_png_page(pdf, sub_dir / "mosaic" / "mosaic_ddem.png", orientation)
            _add_png_page(pdf, sub_dir / "mosaic" / "mosaic_hillshades.png", orientation)
            _add_png_page(pdf, sub_dir / "mosaic" / "mosaic_slopes.png", orientation)
            _add_png_page(pdf, sub_dir / "nmad_before_vs_after_coregistration.png", orientation)
            _add_png_page(pdf, sub_dir / "coregistration_shifts.png", orientation)

    logger.info(f"PDF report saved to {output_path}")
    return output_path
