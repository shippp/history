from check_submissions import parse_filename
import pandas as pd
import sankey


# Load CSV and remove empty lines
experiments_df = pd.read_csv("planned_experiments2.csv", delimiter=";")
experiments_df = experiments_df[~experiments_df["Submission code"].isna()]

# Extract experiment codes from experiment names
fname = experiments_df["Submission code"][0] + "_intrinsics"
code, metadata = parse_filename(fname)

df_columns = list(metadata.keys()) + ["software"]
exp_metadata = pd.DataFrame(columns=df_columns)

for i, row in experiments_df.iterrows():
    exp_name = row["Submission code"]
    fname = exp_name + "_intrinsics.csv"  # Need to add suffix for parse_filename to work
    try:
        code, metadata = parse_filename(fname)
    except ValueError as e:
        print(exp_name, e)

    metadata["software"] = row["Stereo software"]
    exp_metadata.loc[len(exp_metadata)] = metadata

# fig = dataframe_to_sankey(
#     exp_metadata,
#     # columns=["dataset", "georef", "mtp_adjustments"],
#     columns=["dataset", "georef", "mtp_adjustments", "software"],
#     output_file="experiment_sankey.png",
#     title="Sankey diagram",
#     edge_color=None,
# )
# fig.show()

custom_palette = [
    "#DE8F05",
    "#0173B2",
    "#029E73",
    "#D55E00", "#CC78BC",
    "#CA9161", "#FBAFE4", 
    "#ECE133", "#56B4E9"
]

# Matplotlib tab20 palette
tab20_palette = [
"#1f77b4","#aec7e8","#ff7f0e","#ffbb78","#2ca02c","#98df8a","#d62728","#ff9896","#9467bd","#c5b0d5",
"#8c564b","#c49c94","#e377c2","#f7b6d2","#7f7f7f","#c7c7c7","#bcbd22","#dbdb8d","#17becf","#9edae5",
]

# Matplotlib tab20 palette customized
software_dataset_palette = [
"#d62728","#ff9896","#9467bd","#c5b0d5","#8c564b","#c49c94","#e377c2","#f7b6d2",
"#ff7f0e", "#1f77b4","#2ca02c"
]

software_dataset_palette = [
"#d62728","#ff9896","#9467bd","#c5b0d5","#17becf","#9edae5","#bcbd22","#dbdb8d",
"#ff7f0e", "#1f77b4","#2ca02c"
]

# fig = sankey.dataframe_to_sankey(
#     exp_metadata,
#     # columns=["dataset", "georef", "mtp_adjustments"],
#     columns=["dataset", "georef", "mtp_adjustments", "software"],
#     columns_label=["Dataset", "Georef. strategy", "Multitemp BBA", "Software"],
#     output_file="experiment_sankey.png",
#     palette=custom_palette,
#     title="",
#     force_color={"software": "#B0B0B0"},
#     font_size=22,
# )
# fig.show()

fig = sankey.dataframe_to_sankey(
    exp_metadata,
    columns=["software", "dataset", "georef"],
    output_file="experiment_sankey2.png",
    title="",
    force_color={"georef": "#B0B0B0"},
    columns_label=["Software", "Dataset", "Georef. strategy"],
    font_size=24,
    palette=software_dataset_palette,
    alpha=0.6,
)
fig.show()
