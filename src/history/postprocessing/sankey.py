import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import numpy as np

from history.postprocessing import io

logger = logging.getLogger(__name__)


# ---------- Utilities ----------

def hex_to_rgba(hex_color: str, alpha: float = 0.5) -> str:
    """
    Convert HEX color to RGBA string, with default 50% transparency.

    :param hex_color: Color in HEX format (e.g. "#FF5733")
    :type hex_color: str
    :param alpha: Transparency value between 0 and 1
    :type alpha: float
    :return: RGBA color string
    :rtype: str
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"

# -- Set of palettes, mostly from Seaborn --

# Seaborn colorblind palette
colorblind=[
    "#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC",
    "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9"
    ]

# Same as previous, with gray removed, for edges only
colorblind_nogrey = [
    "#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC",
    "#CA9161", "#FBAFE4", "#ECE133", "#56B4E9"
]

# Seaborn deep palette
deep = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"
    ]

# Custom palette
palette_custom = [
    "#4C78A8", "#F58518", "#E45756", "#72B7B2",
    "#54A24B", "#EECA3B", "#B279A2", "#FF9DA6"
]
software_dataset_palette = [
    "#d62728","#ff9896","#9467bd","#c5b0d5","#17becf","#9edae5","#bcbd22","#dbdb8d",
    "#ff7f0e", "#1f77b4","#2ca02c"
]

light_grey = "#B0B0B0"

# ---------- Core builders ----------

def build_nodes(
    df: pd.DataFrame,
    columns: List[str],
    palette: List[str],
    edge_color: Tuple[str, None] = None,
    force_color: dict = {},
    alpha: float = 0.5
) -> Tuple[List[str], Dict[str, int], List[str]]:
    """
    Build Sankey nodes (labels, mapping, colors).

    First and last column nodes are colored grey.

    :param df: Input dataframe
    :type df: pd.DataFrame
    :param columns: Ordered list of columns defining the flow
    :type columns: List[str]
    :param palette: List of HEX colors for intermediate nodes
    :type palette: List[str]
    :param edge_color: HEX color for first and last columns
    :type edge_color: str
    :param alpha: Transparency for node colors
    :type alpha: float
    :return: Tuple of labels, label-to-index mapping, node colors
    :rtype: Tuple[List[str], Dict[str, int], List[str]]
    """
    #labels = pd.unique(df[columns].values.ravel())
    labels = []
    for col in columns:
        labels.extend(df[col].sort_values().unique())
    labels = np.array(labels)
    label_dict = {label: i for i, label in enumerate(labels)}

    # Map label → column index
    label_to_col: Dict[str, int] = {}
    for col_idx, col in enumerate(columns):
        for val in df[col].unique():
            label_to_col[val] = col_idx

    n_cols = len(columns)
    node_colors: List[str] = []
    for i, label in enumerate(labels):
        col_idx = label_to_col[label]

        if edge_color is not None:
            if col_idx == 0 or col_idx == n_cols - 1:
                color = hex_to_rgba(edge_color, alpha)
        else:
            if columns[col_idx] in force_color.keys():
                color = hex_to_rgba(force_color[columns[col_idx]], alpha)
            else:
                base_color = palette[i % len(palette)]
                color = hex_to_rgba(base_color, alpha)

        node_colors.append(color)

    return labels.tolist(), label_dict, node_colors


def build_links(
    df: pd.DataFrame,
    columns: List[str],
    label_dict: Dict[str, int]
) -> Tuple[List[int], List[int], List[int]]:
    """
    Build Sankey links between consecutive columns.

    :param df: Input dataframe
    :type df: pd.DataFrame
    :param columns: Ordered list of columns
    :type columns: List[str]
    :param label_dict: Mapping from label to node index
    :type label_dict: Dict[str, int]
    :return: source indices, target indices, values
    :rtype: Tuple[List[int], List[int], List[int]]
    """
    all_links = []

    for i in range(len(columns) - 1):
        src_col = columns[i]
        tgt_col = columns[i + 1]

        grouped = (
            df.groupby([src_col, tgt_col])
            .size()
            .reset_index(name="value")
        )
        grouped.columns = ["source", "target", "value"]
        all_links.append(grouped)

    all_links_df = pd.concat(all_links, ignore_index=True)

    source = all_links_df["source"].map(label_dict).astype(int).tolist()
    target = all_links_df["target"].map(label_dict).astype(int).astype(int).tolist()
    value = all_links_df["value"].astype(int).tolist()

    return source, target, value


def compute_nodes_totals(
    source: List[int],
    target: List[int],
    value: List[int],
    n_nodes: int
) -> List[int]:
    """
    Compute max in and out flow totals per node.

    :param source: Source node indices
    :type source: List[int]
    :param source: Target node indices
    :type source: List[int]
    :param value: Flow values
    :type value: List[int]
    :param n_nodes: Number of nodes
    :type n_nodes: int
    :return: Outgoing totals per node
    :rtype: List[int]
    """
    totals_in = [0] * n_nodes
    totals_out = [0] * n_nodes

    for s, t, v in zip(source, target, value):
        totals_out[s] += v
        totals_in[t] += v

    totals = np.maximum(totals_in, totals_out)

    return totals


def format_labels(
    labels: List[str],
    totals: List[int],
    hide_zero: bool = True
) -> List[str]:
    """
    Append totals to labels.

    :param labels: Node labels
    :type labels: List[str]
    :param totals: Outgoing totals per node
    :type totals: List[int]
    :param hide_zero: Hide nodes with zero outgoing flow
    :type hide_zero: bool
    :return: Formatted labels
    :rtype: List[str]
    """
    formatted = []
    for i, label in enumerate(labels):
        if hide_zero and totals[i] == 0:
            formatted.append(str(label))
        else:
            formatted.append(f"{label} ({totals[i]})")
    return formatted


def build_sankey_figure(
    labels: List[str],
    node_colors: List[str],
    source: List[int],
    target: List[int],
    value: List[int],
    column_titles: List[str],
    title: str = "Sankey Diagram",
    font_size: int = 20,
) -> go.Figure:
    """
    Create Plotly Sankey figure.

    :param labels: Node labels
    :type labels: List[str]
    :param node_colors: Node colors
    :type node_colors: List[str]
    :param source: Source indices
    :type source: List[int]
    :param target: Target indices
    :type target: List[int]
    :param value: Flow values
    :type value: List[int]
    :param column_titles: Column names for annotations
    :type column_titles: List[str]
    :param title: Figure title
    :type title: str
    :return: Plotly figure
    :rtype: go.Figure
    """
    link_colors = [node_colors[s] for s in source]

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors
        )
    )])

    # Column annotations
    annotations = []
    n = len(column_titles)
    for i, col in enumerate(column_titles):
        annotations.append(
            dict(
                x=i / (n - 1) if n > 1 else 0.5,
                y=1.05,
                text=col,
                showarrow=False,
                font=dict(size=font_size)
            )
        )

    fig.update_layout(
        title_text=title,
        font=dict(size=font_size),
        annotations=annotations
    )

    return fig


def show_palette(palette, width=0.5):
    """
    A small utility function to display a HEX palette with plotly
    """
    fig = go.Figure()

    for i, color in enumerate(palette):
        fig.add_shape(
            type="rect",
            x0=i*width, x1=(i+1)*width,
            y0=0, y1=1,
            fillcolor=color,
            line=dict(width=0)
        )

        fig.add_annotation(
            x=(i+0.5)*width,
            y=-0.2,
            text=color,
            showarrow=False,
            font=dict(size=12)
        )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    fig.update_layout(
        title="Color Palette",
        height=200,
        margin=dict(t=40, b=40, l=20, r=20)
    )

    fig.show()


# ---------- Main API ----------

def dataframe_to_sankey(
    df: pd.DataFrame,
    columns: List[str],
    output_file: str = "sankey.png",
    columns_label: Optional[List[str]] = None,
    palette: Optional[List[str]] = None,
    edge_color: str = None,
    force_color: dict = {},
    alpha: float = 0.5,
    title: str = "Sankey Diagram",
    font_size: int = 20,
) -> go.Figure:
    """
    Convert a pandas DataFrame into a Sankey diagram and save as PNG.

    :param df: Input dataframe
    :type df: pd.DataFrame
    :param columns: Ordered list of columns defining the flow
    :type columns: List[str]
    :param output_file: Output PNG file path
    :type output_file: str
    :param columns_label: Column labels to be used for the plot
    :type columns_label: List[str]
    :param palette: List of HEX colors for intermediate nodes
    :type palette: Optional[List[str]]
    :param edge_color: HEX color for first and last column nodes
    :type edge_color: str
    :param alpha: Transparency for nodes and links
    :type alpha: float
    :param title: Figure title
    :type title: str
    :return: Plotly Sankey figure
    :rtype: go.Figure
    """
    if palette is None:
        # Defaults to Seaborn colorblind palette, with gray removed, for edges only
        palette = software_dataset_palette

    df_clean = df[columns].dropna()
    labels, label_dict, node_colors = build_nodes(
        df_clean,
        columns,
        palette,
        edge_color=edge_color,
        force_color=force_color,
        alpha=alpha,
    )

    source, target, value = build_links(df_clean, columns, label_dict)

    totals = compute_nodes_totals(source, target, value, len(labels))
    formatted_labels = format_labels(labels, totals)

    if columns_label is None:
        columns_label = columns

    fig = build_sankey_figure(
        formatted_labels,
        node_colors,
        source,
        target,
        value,
        columns_label,
        title=title,
        font_size=font_size,
    )

    # Save PNG (requires kaleido)
    fig.write_image(output_file, width=1200, height=700, scale=2)

    return fig


# ---------- Pipeline integration ----------

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


def save_sankey(raw_csv: str | Path, output_png: str | Path, columns: List[str], **kwargs) -> go.Figure:
    """
    Build a Sankey diagram from a raw submissions CSV and save it to a PNG file.

    Loads `raw_csv`, expands submission codes into metadata columns, then delegates
    to `dataframe_to_sankey`. Extra keyword arguments are forwarded to it.
    """
    df = load_submissions_df(raw_csv)
    return dataframe_to_sankey(df, columns=columns, output_file=output_png, **kwargs)


if __name__ == "__main__":

    fig = save_sankey(
        "test_file.csv",
        "sankey_diagram3.png",
        columns=["dataset", "georef", "mtp_adjustment", "software"],
        title="Sankey diagram"
    )

    fig.show()
