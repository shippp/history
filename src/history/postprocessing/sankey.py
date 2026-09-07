"""Build Sankey diagrams from tabular flow data using Plotly."""
from __future__ import annotations

import itertools
import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from history.postprocessing import io

logger = logging.getLogger(__name__)

# Seaborn colorblind palette, grey removed so it stands out if used for edge columns.
DEFAULT_PALETTE = [
    "#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC",
    "#CA9161", "#FBAFE4", "#ECE133", "#56B4E9",
]


def hex_to_rgba(hex_color: str, alpha: float = 0.5) -> str:
    """Convert a HEX color string (e.g. "#FF5733") to a CSS rgba() string."""
    r, g, b = (int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def dataframe_to_sankey(
    df: pd.DataFrame,
    columns: list[str],
    columns_label: list[str] | None = None,
    palette: list[str] = DEFAULT_PALETTE,
    alpha: float = 0.5,
    title: str = "Sankey Diagram",
    font_size: int = 16,
) -> go.Figure:
    """Build a Plotly Sankey figure showing flows across a sequence of dataframe columns.

    Each column is one stage of the diagram, nodes are colored by column (cycling
    through `palette`), and node labels are annotated with their total flow.
    """
    df = df[list(columns)].dropna()

    # (column, value) keeps nodes unique even when the same value appears in several columns.
    nodes = [(col, val) for col in columns for val in sorted(df[col].unique())]
    node_index = {node: i for i, node in enumerate(nodes)}
    col_color = {col: hex_to_rgba(palette[i % len(palette)], alpha) for i, col in enumerate(columns)}
    colors = [col_color[col] for col, _ in nodes]

    source, target, value = [], [], []
    for left, right in itertools.pairwise(columns):
        for (lval, rval), count in df.groupby([left, right]).size().items():
            source.append(node_index[(left, lval)])
            target.append(node_index[(right, rval)])
            value.append(count)

    totals_in = [0] * len(nodes)
    totals_out = [0] * len(nodes)
    for s, t, v in zip(source, target, value):
        totals_out[s] += v
        totals_in[t] += v
    labels = [
        f"{val} ({total})" if total else str(val)
        for (_, val), total in zip(nodes, map(max, totals_in, totals_out))
    ]

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node={"label": labels, "color": colors, "pad": 20, "thickness": 25, "line": {"color": "black", "width": 0.5}},
            link={"source": source, "target": target, "value": value, "color": [colors[s] for s in source]},
        )
    )

    columns_label = list(columns_label or columns)
    n = len(columns_label)
    fig.update_layout(
        title_text=title,
        font_size=font_size,
        annotations=[
            {"x": i / (n - 1) if n > 1 else 0.5, "y": 1.05, "text": label, "showarrow": False, "font_size": font_size}
            for i, label in enumerate(columns_label)
        ],
    )

    return fig


def load_submissions_df(raw_csv: str | Path) -> pd.DataFrame:
    """Load a raw submissions CSV and expand its submission codes into metadata columns."""
    df = pd.read_csv(raw_csv)

    metadata_rows = []
    for code in df["Submission code"]:
        try:
            _, metadata = io.parse_filename(code)
        except io.FilenameParseError as e:
            logger.warning(f"Skipping unparsable submission code '{code}': {e}")
            metadata = {}
        metadata_rows.append(metadata)

    df = df.join(pd.DataFrame(metadata_rows, index=df.index))
    df["software"] = df["Stereo software"]
    return df


def save_sankey(raw_csv: str, output_png: str, columns: list[str] | None = None, **kwargs) -> go.Figure:
    """Build a Sankey diagram from a raw submissions CSV and save it to a PNG file.

    Uses every parsed metadata column, in order, as the diagram's stages unless `columns` is given.
    Extra keyword arguments are forwarded to `dataframe_to_sankey`.
    """
    df = load_submissions_df(raw_csv)
    fig = dataframe_to_sankey(df, columns=columns or list(df.columns), **kwargs)
    fig.write_image(output_png, width=1200, height=700, scale=2)
    return fig


if __name__ == "__main__":
    # block of code for test only
    fig = save_sankey(
        "/mnt/summer/USERS/DEHECQA/history/output/v2/tests/processing/planned_submissions.csv",
        "sankey_diagram3.png",
        columns=["dataset", "georef", "mtp_adjustments", "software"],
        title="Sankey diagram",
    )
    fig.show()
