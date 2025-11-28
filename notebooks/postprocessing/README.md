# Post-Processing the Results

The goal of the Post-Processing workflow is to compare, evaluate, and analyse the outcomes of the stereo reconstruction submissions.  
It takes all user submissions as input, validates and organizes them, and then processes the data to generate a comprehensive set of analytical figures, statistical summaries, and cross-comparisons between methods and configurations.


## Submission

Each submission need to be composed of 4 mandatory files and optional files:
Each file provided need to follow a filenaming with your author tag (composed with only letter or number) and with codes

Example of a valid submission for the author tag "toto":
- my_submisison
    - toto_CG_AI_PP_CY_GM_PN_MN_dense_pointcloud.laz
    - toto_CG_AI_PP_CY_GM_PN_MN_sparse_pointcloud.laz
    - toto_CG_AI_PP_CY_GM_PN_MN_intrinsics.csv
    - toto_CG_AI_PP_CY_GM_PN_MN_extrinsics.csv
    - toto_CG_AI_PP_CY_GM_PN_MN_dem.tif
    - toto_CG_AI_PP_CY_GM_PN_MN_orthoimage.tif

You can also add a version if you want to provid 2 differents submission with the same code example : "toto_CG_AI_PP_CY_GM_PN_MN_V1_dense_pointcloud.laz", Important the version need to appear before the file suffix and every other file of your submission need to have the same version.

| Identifier | Option 1           | Code 1 | Option 2             | Code 2 | Option 3           | Code 3 |
|-----------|---------------------|--------|------------------------|--------|---------------------|--------|
| **Site**  | Casa Grande         | CG     | Iceland                | IL     |                     |        |
| **Dataset** | Aerial            | AI     | KH-9 MC                | MC     | KH-9 PC             | PC     |
| **Images**  | Raw               | RA     | Pre-processed          | PP     |                     |        |
| **Use of initial Camera Calibration Information** | Yes | CY | No | CN | | |
| **Use of Ground Control Points** | Manual (provided) | GM | Automated approach | GA | No | GN |
| **Point Cloud Coregistration** | Yes | PY | No | PN | | |
| **Multi-temporal Bundle Adjustment** | Yes | MY | No | MN | | |

## Post-Processing Steps

The complete Post-Processing workflow is divided into **8 main steps**, starting from an archived directory containing all raw submissions and ending with the generation of all plots, figures, and summary statistics (see this notebook : [post_process_workflow.ipynb](post_process_workflow.ipynb)). 

 
1. **Extract archives** : Recursively extract all archive files found in the input directory.
2. **Analyse submissions and create symlinks**: Inspect all extracted files, detect all **valid** submissions, and create a clean, structured symlink hierarchy.
3. **Convert point clouds to DEMs** : Convert each point cloud into a DEM using [pdal](https://pdal.io/en/2.9.2/). The generated DEM is spatially aligned with the corresponding reference DEM. If the point cloud lacks a CRS, try EPSG:4326 and then the CRS of the reference DEM.
4. **Coregister DEMs** : Apply DEM coregistration consisting of:  
    - a horizontal alignment using the **Nuth–Kaab** method  
    - a vertical shift correction using the corresponding reference DEM  
    Coregistration ignores masked areas and pixels with low slopes.
5. **Generate Differential DEMs (DDEMs)** : For both raw and coregistered DEMs, compute the difference relative to the corresponding reference DEM.
6. **Compute landcover-based statistics** : Using the reference landcover, compute landcover-stratified statistics for **coregistered** DDEMs.
7. **Generate STD DEMs** : Compute standard deviation DEMs for both raw and coregistered DEMs.
10. **Compute landcover-based statistics on STD DEMs** : Using the reference landcover, compute landcover-stratified statistics for all standard deviation DEMs.

**Note** : Some plots and visualization will be generated at each processing steps. 





 
