"""
CCR hazard mapping on a local computer: six classifiers with three seeds, site maps, Site A metal HQ maps,
feature maps, metrics tables and feature-group importance.

Inputs (exported from Earth Engine; already in FEATURE_DIR):
  <stem>_Features.tif   feature stack (QA60 dropped)
  <stem>_Label.tif      reference label: class 1 = HI_norm >= 25th percentile (higher HI), class 0 = the rest
  <stem>_HI.tif         optional; without it HI is kriged here from the well composite with the GEE variogram
Well HI/HQ composites, raw sample tables (well gradient) and ccr-units.shp are read from WELL_DIR.

Outputs (OUT_DIR)
  Fig1_All_sites_RandomForest.png    site basemap | HI normalised | risk map | reference label | RF classification |
                                     RF probability | error map | false alarms and missed contamination (Sites A-D)
  Fig2_<site>_metal_HQ_RandomForest.png   per site: HQ class | normalised HQ | RF classification | RF probability |
                                     error map, for every exceeded metal (HQ_METALS, HQ_RF_SITES)
  Fig3_All_sites_exceeded_metal_HQ.png    kriged HQ classes of every exceeded metal at every site
  Fig4_Site_HQ_and_exceeding_wells.png    maximum well HQ and wells with HQ >= 1, per metal and site
  Fig5_Well_HI_maps(_kriged).png     wells coloured by HI on a basemap (optionally over the kriged HI)
  Fig_<site>_site_HI_risk_label_RF_error.png, Fig_confusion_matrices.png, Fig_model_comparison.png,
  Fig_feature_groups_importance.png
  feature_maps/Features_<site>_<group>.png       every band of a feature group, one site
  feature_groups/FeatureGroup_<group>_all_sites.png   every band of a feature group, Sites A-D
  Table1_models_test.tex, Table2_confusion_matrices.tex, Table3_metal_HQ_models.tex, Tables_CCR_models.docx, CSVs
  checkpoints/   results of every step (restored on a rerun, CHECKPOINT = "auto"); cache/   downloaded basemaps

Every map shows upgradient (triangle up) and downgradient (triangle down) wells and the CCR units.

Requirements: numpy, pandas, scipy, scikit-learn, matplotlib, tifffile (or rasterio), pillow; python-docx for Word.
Run:  python hazard_ml.py      (edit the settings block first)
"""
import functools
import io
import itertools
import re
import struct
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ==== settings: EDIT here =============================================================================================
_REPO = Path.cwd() if (Path.cwd() / "data").is_dir() else Path.cwd().parent   # repository root (run from root or notebooks/)
ROOT = _REPO / "data"                                   # input folder; or an absolute path
FEATURE_DIR = ROOT / "features_labels"                  # <stem>_Features.tif, <stem>_Label.tif
WELL_DIR = ROOT / "well_maps"                           # well composites, raw sample CSVs, ccr-units.shp
OUT_DIR = _REPO / "results"                             # all figures, tables and checkpoints
SEARCH_DIRS = [FEATURE_DIR, WELL_DIR]                   # input files are looked up in this order
SITES = {   # label, file stem, well HI composite, raw samples (well gradient), GEE kriging variogram
    "Glen Lyn": dict(label="Site A", stem="glen_lyn", wells="GlenLyn_wells_HI_composite_named.csv",
                     samples="Glenlyn_samples.csv", kriging=dict(range=1500, sill=50, nugget=3, maxDistance=2500)),
    "Bremo": dict(label="Site B", stem="bremo", wells="Bremo_wells_HI_composite_named.csv",
                  samples="Bremo_samples.csv", kriging=dict(range=2000, sill=75, nugget=5, maxDistance=3000)),
    "Chesterfield": dict(label="Site C", stem="chesterfield", wells="Chesterfield_wells_HI_composite_named.csv",
                         samples="Chesterfield_samples.csv",
                         kriging=dict(range=2500, sill=200, nugget=10, maxDistance=4000)),
    "Possum Point": dict(label="Site D", stem="possum_point", wells="PossumPoint_wells_HI_composite_named.csv",
                         samples="PossumPoints.csv", kriging=dict(range=1800, sill=30, nugget=2, maxDistance=2800)),
}
RUN_SITES = list(SITES)
CCR_SHP = "ccr-units.shp"           # CCR units (with .dbf, .prj); None = maps without units
CCR_NAME_FIELD = "Impoundmen"
CHECKPOINT = "auto"                 # 'auto': each step saves its results in CHECKPOINT_DIR and, on a rerun, restores
                                    # them when the step's settings and input files are unchanged (else it reruns);
                                    # 'refresh': rerun every step and overwrite the checkpoints; 'off': no checkpoints
CHECKPOINT_DIR = None               # None = OUT_DIR / "checkpoints"
FEATURE_MAPS = True                 # maps of every feature band: per site and group, and per group for all sites

# ---- metal HQ (Fig. 2 per site, Fig. 3 all sites) ----
HQ_METALS = "exceeded"              # 'exceeded' = every metal with HQ >= 1 at one or more wells of the site; or a list,
                                    # e.g. ["arsenic", "cobalt", "manganese", "molybdenum", "lead"]; or a dict per site,
                                    # e.g. {"Glen Lyn": [...], "Bremo": "exceeded"}
HQ_RF_SITES = list(SITES)           # sites where an RF is trained on each metal's HQ label (Fig. 2, Table 3);
                                    # e.g. ["Glen Lyn"] for Site A only (about 1-2 min per metal and seed on a laptop)
HQ_FROM_SAMPLES = ["zinc"]          # metals missing from the HI composites: HQ per well = mean sample concentration /
                                    # screening level (as the composites); added to the HQ maps and Fig. 4 when the raw
                                    # sample table has them (HI is not changed)
SCREENING_MG_L = {}                 # screening levels (mg/L) to use instead of the samples' standard_threshold,
                                    # e.g. {"zinc": 6.0} (EPA tap-water RSL); empty = standard_threshold of the data
NON_METALS = ["sulfate", "total_dissolved_solids", "chloride", "fluoride", "ph", "cyanide"]
HQ_CLASSES = [0.5, 1, 1.5, 2]       # HQ classes: < 0.5, 0.5-1, 1-1.5, 1.5-2, >= 2
HQ_NUGGET_FRAC = 0.0                # HQ kriging: exponential, site range and maxDistance, sill = variance of well HQ,
                                    # nugget = HQ_NUGGET_FRAC x sill (0 = exact at the wells)
HQ_MODELS = ["RandomForest"]        # models trained on each metal's HQ label

# ---- vegetation index rows (Fig. 6a-c: index level, CSTV, rate of change; one row per site) ----
INDEX_ROWS = True
INDEX_ROW_ORDER = ["CCC", "LCC", "LAI", "CIRE", "MTCI", "NDRE1", "NDRE2", "NDVI", "SAVI"]
INDEX_ROW_GROUPS = {"Biophysical parameter indices": ["CCC", "LCC", "LAI"],
                    "Vegetation indices": ["CIRE", "MTCI", "NDRE1", "NDRE2", "NDVI", "SAVI"]}
INDEX_ROW_PANEL_IN = 1.9            # panel width, inches
INDEX_ROW_WELLS = True              # wells and CCR units on every panel

# ---- SHAP (Figs 7-8) ----
SHAP = True                         # SHAP values of the Random Forest (seed MAP_SEED) on held-out test pixels
SHAP_METHOD = "tree"                # 'tree': exact TreeSHAP (pip install shap); 'saabas': tree contributions (no
                                    # extra package; used automatically when shap is missing)
SHAP_N = 1000                       # test pixels explained per site (TreeSHAP on 100 deep trees: minutes per site)
SHAP_TOP = 15                       # features shown per site
SHAP_POINT = 4                      # point size in the beeswarm

# ---- leave-one-site-out (Figs 9-10, Table 9) ----
LOSO = True                         # train on three sites, test on the fourth
LOSO_MODELS = None                  # None = MODELS; e.g. ["RandomForest"]
LOSO_MAX_PER_SITE = 20000           # training pixels drawn per training site (stratified)
LOSO_MAX_TEST = 50000               # held-out pixels scored (stratified sample; the map uses all pixels)
LOSO_STANDARDISE = "per_site"       # 'per_site': each site z-scored with its own mean and SD (no labels used);
                                    # 'train': the training sites' mean and SD applied to the held-out site

# ---- threshold tuning and skill relative to the majority class (Table 1b) ----
TUNE_THRESHOLD = True               # also score the test blocks at the threshold that maximises balanced accuracy on
                                    # the validation blocks (partition 'test_tuned'); the 0.5 results are kept
# ---- feature-set ablation and location baselines (Table 10, Fig. 11) ----
ABLATION = True
ABLATION_MODELS = ["RandomForest"]
FEATURE_SETS = {                    # name: feature groups (as in band_groups), 'all', 'distance' or 'coords'
    "All features": "all",
    "Vegetation only": ["Sentinel-2", "Vegetation indices", "CSTV", "Rate of change"],
    "Environment only": ["Topographic", "Soil", "Climate", "Hydrologic", "Land cover"],
    "Distance to CCR unit": "distance",          # distance (m) of each pixel to the nearest CCR unit
    "Pixel coordinates": "coords",               # easting and northing (m) of each pixel
}
ABL_COLORS = {"All features": "#4D4D4D", "Vegetation only": "#1A7A1A", "Environment only": "#D95F0E",
              "Distance to CCR unit": "#7B3294", "Pixel coordinates": "#2166AC"}
ABL_FIG_SETS = ["All features", "Vegetation only", "Environment only"]   # predictor sets shown in Fig. 11 (all
                                                                         # sets stay in Table 10)
ABL_FIG_METRIC = ("OA", "Accuracy (test)")     # Fig. 11 metric: column of Table 10 and axis label
ABL_FIG_BASELINE = True                        # Fig. 11: mark the majority-class accuracy of each site
ABL_FIG_STYLE = dict(width_cm=None, height_cm=7.5, bar_w=0.8, ylim=(0.5, 1.0), err_lw=0.8, capsize=2.5,
                     baseline_color="black", baseline_lw=1.4,                # width None = PAGE["width_cm"]
                     hatch={"All features": "", "Vegetation only": "///", "Environment only": "..."},
                     alpha={"All features": 1.0, "Vegetation only": 0.55, "Environment only": 0.3},
                     edge_lw=0.8, hatch_lw=0.8,    # bars: site colour; predictor set = fill pattern and shade
                     tick_dir="in",                # tick marks inside the axes ("in", "out" or "inout")
                     box=True, box_color="black", box_lw=1.0)      # black outline on all four sides
SITE_COLORS = {"Site A": "#156082", "Site B": "#E97132", "Site C": "#196B24", "Site D": "#0F9ED5"}   # all figures
# ---- label sensitivity (Table 11) ----
SENSITIVITY = True
SENS_MODELS = ["RandomForest"]
SENS_ALPHAS = [25, 33, 50]          # class 0 = lowest ALPHA % of HI_norm (rule labels for every variant)
SENS_MAX_WELL_DIST_M = 250          # extra run at the 25th percentile without pixels farther than this from a well
                                    # (kriging least certain); None = skip

# ---- well HI maps (Fig. 5) ----
WELLMAP = True                      # map of the wells coloured by HI on a basemap, one panel per site
WELLMAP_BASEMAP = None              # Fig. 5 basemap; None = same as BASEMAP
WELLMAP_PAD = 0.0                   # map extent beyond the site ROI, fraction of the ROI size on each side
WELLMAP_ASPECT = None               # same panel box for every site: "max" (shape of the tallest site, wider sites get
                                    # more map above/below), "min" (shape of the widest site), a number (height/width,
                                    # 1 = square; 0.72 = landscape, taller sites get more map left/right) or None
WELLMAP_SCALE_UNITS = "mi"          # scale bar of Fig. 5: "mi" (miles) or "km"
WELLMAP_ROI_BOX = False             # dashed outline of the site ROI (useful only when the map extends beyond it)
WELLMAP_TITLE_BOLD = False          # "(a)  Site A" titles in bold
WELLMAP_INTERP = "both"             # True: kriged HI under the wells; False: wells only; 'both': one figure of each
WELLMAP_INTERP_ALPHA = 0.6          # opacity of the kriged HI
WELLMAP_LABELS = True               # well names next to the markers
WELLMAP_LABEL_ITERS = 300          # iterations of the label placement (moves overlapping well names apart)
WELLMAP_LABEL_MAX_PT = 28           # farthest a well name may be moved from its well, points (leader line drawn)
WELLMAP_COLOC_M = 15                # wells closer than this (m) are drawn apart ...
WELLMAP_OFFSET_M = 30               # ... on a circle of this radius (m), with a line to the true position
WELLMAP_COLS = 2                    # panels per row
WELLMAP_HI_LOG = True               # logarithmic HI colour scale
WELLMAP_PANEL_IN = 4.6              # panel width, inches

# ---- well HI against nearby vegetation ----
HIVI_ANALYSIS = True                # Step 8: well HI against the vegetation indices around the wells
INDICES = ["NDVI", "SAVI", "CIRE", "NDRE1", "NDRE2", "MTCI", "LAI", "LCC", "CCC"]   # indices (summer median level)
ALPHA = 0.05                        # significance level (solid trend line where p < ALPHA)
# Fig. 12: one row per index, one column per site
HIVI_INDICES = ["CIRE", "CCC"]      # indices of Fig. 12, e.g. ["CIRE"] for a single row of four panels
HIVI_LOG_X = True                   # HI on a log axis (well HI spans about two orders of magnitude)
HIVI_TREND = "theil-sen"            # trend line: "theil-sen" (robust), "ols" or None; fitted on log10(HI) if log axis
HIVI_SHOW_PARTIAL = True            # also print the partial rho controlling for distance to the nearest CCR unit
HIVI_STYLE = dict(marker_size=48, up="^", down="v", unknown="o", face="black", edge="black", unknown_face="0.6",
                  trend="#C0392B", trend_lw=1.8, panel_w=3.3, panel_h=2.9)
HIVI_BUFFER_M = 60                  # buffer radius (m) of every HI-vegetation analysis (Fig. 12, land cover, all indices)
HIVI_MASK = None                    # None = all pixels; "vegetated" = NDVI >= HIVI_MIN_NDVI and no water/built pixels
HIVI_MIN_NDVI = 0.3
HIVI_ADJUST = None                  # None = index as measured; "landcover" = index minus the value expected from the
                                    # Dynamic World land-cover composition of the pixel (OLS over the site)
HIVI_LANDCOVER = True               # land-cover diagnostic at the same buffer (Table H5, H6, FigS_HI_vs_vegetation_landcover)
HIVI_ALL = True                     # HI against all nine indices (INDICES) at the same buffer (Table H7, FigS_HI_vs_all_indices)
HIVI_ALL_EXCLUDE = []               # indices left out of the all-indices figure and table, e.g. ["NDRE2", "MTCI"]
HIVI_ALL_STYLE = dict(panel_w=3.0, panel_h=2.4)   # panel size of the all-indices figure, inches (rows = indices)
HIVI_LC = ["DW_Water", "DW_Trees", "DW_Grass", "DW_FloodedVeg", "DW_Crops", "DW_Shrubs", "DW_Built", "DW_Bare"]
HIVI_MIN_WELLS_CLASS = 6            # wells needed for rho within one land-cover class (Table H6)
HIVI_LC_STYLE = dict(
                       lc_colors={"Trees": "#1B7837", "Grass/crops/shrubs": "#A6D96A", "Water/flooded": "#4393C3",
                                  "Built/bare": "#8C510A"},
                       lc_marker_size=46, lc_edge="black")

# ---- basemaps (Fig. 1 panels a and h, panel a of the single-site figures) ----
BASEMAP = "topo"                    # "none", "topo", "satellite" or "streets" (any key of BASEMAPS; tiles need internet;
                                    # contextily is used when installed, else a direct download)
BASEMAP_H = None                    # panel (h) of Fig. 1, drawn faded; None = same as BASEMAP
BASEMAP_RETRIES = 4                 # attempts per tile (with pauses) before giving up
BASEMAP_TILE_DELAY = 0.15           # pause between tile requests, seconds (some servers drop fast repeated requests)
BASEMAP_ZOOM = "auto"               # "auto" (as contextily.add_basemap: zoom from the map extent, few tiles) or a
                                    # number: 16 = about 2 m pixels, 17 = about 1 m (four times the tiles)
CTX_PROVIDERS = {"satellite": "Esri.WorldImagery", "topo": "Esri.WorldTopoMap",   # contextily provider objects used
                 "streets": "OpenStreetMap.Mapnik"}                               # for these keys (as your map code)
FADE = 0.45                         # opacity of the basemap in panel (h)
BASEMAPS = {   # key: (tile URL, maximum zoom, credit); tiles need internet and are cached in OUT_DIR/cache
    "satellite": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                  19, "Esri, Maxar, Earthstar Geographics"),
    "topo": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
             19, "Esri, USGS, NOAA"),
    "streets": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
                19, "Esri, HERE, Garmin"),
    "hillshade": ("https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/"
                  "{z}/{y}/{x}", 16, "Esri, USGS"),
    "natgeo": ("https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}",
               16, "Esri, National Geographic"),
    "gray": ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/"
             "{z}/{y}/{x}", 16, "Esri, HERE, Garmin"),
    "usgs_topo": ("https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}", 16,
                  "USGS The National Map"),
    "usgs_imagery": ("https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/"
                     "{z}/{y}/{x}", 16, "USGS The National Map"),
    "carto_light": ("https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png", 19, "CARTO, OpenStreetMap"),
    "opentopo": ("https://a.tile.opentopomap.org/{z}/{x}/{y}.png", 17, "OpenTopoMap (CC-BY-SA), OpenStreetMap"),
    "osm": ("https://tile.openstreetmap.org/{z}/{x}/{y}.png", 19, "OpenStreetMap contributors"),
    "s2": (None, None, "Sentinel-2 true colour"),            # from the feature stack, no internet
    "none": (None, None, ""),                                 # plain white background
}
BASEMAP_FALLBACKS = {               # tried in turn when a tile server cannot be reached (e.g. a firewall closes the
    "topo": ["usgs_topo", "opentopo", "osm"],                 # connection: WinError 10054)
    "satellite": ["usgs_imagery"],
    "streets": ["osm", "carto_light"],
}
BASEMAP_FILE = "{stem}_basemap.tif"  # your own basemap, used first when present in SEARCH_DIRS (no internet needed):
                                     # an RGB GeoTIFF in EPSG:4326 (lon/lat) covering the site. Easiest: run
                                     # hazard_ml_gee.py with BASEMAP_EXPORT = "naip" and BASEMAP_ONLY = True (NAIP
                                     # aerial photo, Earth Engine), put the <stem>_basemap.tif files in FEATURE_DIR.
                                     # Used for every basemap panel; None = do not look for one

# ---- hazard and label (as the Code Editor script, LABEL_RULE 'percentile') ----
HI_SOURCE = "auto"                  # 'file' (<stem>_HI.tif), 'krige' (here, GEE variogram) or 'auto' (file if present)
LABEL_SOURCE = "auto"               # 'file' (<stem>_Label.tif), 'rule' (ALPHA_PERCENTILE) or 'auto'
ALPHA_PERCENTILE = 25               # class 0 = lowest 25 % of HI_norm
MIN_HQ_IN_HI = 5                    # composite wells whose HI sums fewer constituents are left out
RISK_BREAKS = [1, 5, 10, 20]        # risk classes 0-4: HI < 1, 1-5, 5-10, 10-20, >= 20
RISK_NAMES = ["HI < 1", "1–5", "5–10", "10–20", "≥ 20"]

# ---- spatial split, preprocessing (HazardFuse-matched E2 design) ----
BLOCK_PX, SPLIT_FRAC, BUFFER_PX, N_SPLIT_TRIALS = 32, (0.70, 0.15, 0.15), 8, 500   # SPLIT_FRAC: train, validation, test;
                                    # 8-px (160 m) buffer; (0.70, 0.0, 0.30) drops the validation set (no threshold
                                    # tuning then)
SPLIT_SEED = 42
VAL_TO_TRAIN = False                # True: keep the test blocks of SPLIT_FRAC and train on the validation blocks too
SEEDS = [42, 43, 44]
SEED_SCOPE = "models"               # 'models': one split, the seeds change the model randomness (as the GEE script);
                                    # 'split': a new block split per seed as well
DROP_BANDS = ["QA60"]
NO_CATEGORICAL = ["DW_LandCover"]   # class codes are not distances: left out of KNN and SVM
STANDARDISE = True
CLASS_WEIGHTS = True                # 'balanced'
MAX_TRAIN_SVM = 10000               # stratified training subsample of the SVM (as GEE / HazardFuse)
NB_BINS = 64                        # Naive Bayes: z-scores clipped to +/-4 and binned to 0..NB_BINS (as GEE)
MODELS = ["RandomForest", "CART", "SVM", "NaiveBayes", "GradientBoost", "KNN"]
MODEL_PARAMS = dict(
    RandomForest=dict(n_estimators=100, max_features="sqrt", n_jobs=-1),          # smileRandomForest(100)
    CART=dict(min_samples_leaf=1),                                                  # smileCart(minLeafPopulation 1)
    SVM=dict(kernel="rbf", C=1.0, gamma="1/n_features"),                            # libsvm RBF, gamma 1/n
    NaiveBayes=dict(),                                                              # smileNaiveBayes (categorical)
    GradientBoost=dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=256),       # smileGradientTreeBoost
    KNN=dict(n_neighbors=15, n_jobs=-1),                                            # smileKNN(15)
)
THRESHOLD = 0.5
MAP_MODEL, MAP_SEED = "RandomForest", 42     # model and seed of the classification and error maps
MAP_NAME = "RF"                              # how the map model is named on the figures

# ---- figures ----
FONT_FAMILY = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
FONT = dict(          # font sizes in points; all figures
    base=12,          # default text
    title=14,         # axes titles
    label=13,         # axis labels
    tick=11,          # tick labels
    annot=11,         # notes inside or beside panels (well HQ range, colour-bar labels)
    legend=12,        # legends
    suptitle=16,      # figure titles
    panel=17,         # panel letters (a)-(h) and metal names (Figs 1-2)
    row=16,           # row labels: Site A-D (Fig. 1), HQ / RF rows (Fig. 2)
    caption=13,       # column captions under Fig. 1, test scores under Fig. 2
    key=12,           # colour-key tick labels (Figs 1-2)
    mapnote=10,       # north arrow, scale bar, imagery credit
    cell=8,           # numbers inside heatmap cells (HQ-vegetation figures)
    wellname=7,       # well names on the well HI maps (Fig. 5)
)
# Per-figure font sizes, points. Each figure is drawn with FONT, then FONT_ALL, then its own entry below;
# afterwards FONT is restored. The entries start at the FONT values, so edit only what should change.
# Keys as in FONT (title, label, tick and legend also set the matplotlib defaults used for unstyled text).
FONT_ALL = dict()                   # applied to every figure, e.g. dict(tick=12, legend=12)
FIG_FONTS = {
    "feature_maps": dict(label=13, legend=12, suptitle=16, tick=11, title=14),
        # ^ Features_<site>_<group> (feature maps per site)
    "feature_groups_all_sites": dict(annot=11, key=12, label=13, legend=12, row=16, suptitle=16, tick=11, title=14),
        # ^ FeatureGroup_<group>_all_sites
    "site_maps": dict(annot=11, label=13, legend=12, mapnote=10, suptitle=16, tick=11, title=14),
        # ^ Fig_<site>_site_HI_risk_label_RF_error
    "confusion": dict(annot=11, label=13, legend=12, suptitle=16, tick=11, title=14),
        # ^ Fig_confusion_matrices
    "comparison": dict(label=13, legend=12, suptitle=16, tick=11, title=14),
        # ^ Fig_model_comparison
    "groups": dict(annot=11, label=13, legend=12, tick=11, title=14),
        # ^ Fig_feature_groups_importance
    "fig1": dict(caption=13, key=12, label=13, legend=12, mapnote=10, panel=17, row=16, tick=11, title=14),
        # ^ Fig1_All_sites (all sites: basemap, HI, classes, RF map, errors)
    "fig2": dict(annot=11, key=12, label=13, legend=12, panel=17, row=16, tick=11, title=14, print_pt=12),
        # ^ Fig2_<site>_metal_HQ; vertical layout (FIG2_LAYOUT): all text at print_pt
    "fig3": dict(annot=11, key=12, label=13, legend=12, mapnote=10, panel=17, row=16, tick=11, title=14),
        # ^ Fig3_All_sites_exceeded_metal_HQ; metal names = panel - 1; note in each HQ map = annot - 2
    "fig4": dict(annot=11, label=13, legend=12, tick=11, title=14),
        # ^ Fig4_Site_HQ_and_exceeding_wells
    "fig5": dict(label=13, legend=12, mapnote=10, tick=11, title=14, wellname=7),
        # ^ Fig5_Well_HI_maps
    "fig6": dict(key=12, label=13, legend=12, mapnote=10, row=16, suptitle=16, tick=11, title=14),
        # ^ Fig6a-c_Index_rows (level, CSTV, rate)
    "fig7": dict(key=12, label=13, legend=12, tick=11, title=14),
        # ^ Fig7_SHAP_beeswarm
    "fig8": dict(label=13, legend=12, tick=11, title=14),
        # ^ Fig8_SHAP_feature_groups; site bars and legend
    "fig9": dict(annot=11, key=12, label=13, legend=12, tick=11, title=14),
        # ^ Fig9_LOSO_scores
    "fig10": dict(annot=11, label=13, legend=12, row=16, tick=11, title=14),
        # ^ Fig10_LOSO_maps
    "fig11": dict(label=13, legend=12, tick=11, title=14),
        # ^ Fig11_feature_set_ablation
    "fig12": dict(annot=11, label=13, legend=12, tick=11, title=14),
        # ^ hi_vegetation/Fig12_HI_vs_vegetation
    "figS_hivi_all": dict(annot=10, label=12, legend=12, tick=10, title=12),
        # ^ hi_vegetation/FigS_HI_vs_all_indices_60m
    "figS8": dict(label=13, legend=12, tick=11, title=14),
        # ^ hi_vegetation/FigS_HI_vs_vegetation_landcover
}
PAGE = dict(width_cm=15.9)            # text width of the page (A4 with 2.54 cm margins = 15.9 cm); Fig. 11 width
PANEL_IN = dict(fig1=2.2, fig2=2.5, fig3=2.3, groups=2.2)   # width of one map panel, inches (smaller = larger text)
METALS_PER_FIG = dict(fig2=8)          # Fig. 2 (horizontal): most metals per figure; more are split into equal parts
FIG2_LAYOUT = "vertical"              # Fig. 2: "vertical" (one row per metal, one column per map) or "horizontal"
FIG2_SPLIT = False                    # vertical Fig. 2: False = all metals of a site in one figure; True = split
                                      # into parts that fit PRINT["max_h_cm"]
PRINT = dict(width_cm=18.0,           # vertical figures are drawn at their printed size: insert them at this width
             max_h_cm=24.0,           # (A4 text width) and the text is font_pt; taller figures are split into parts
             font_pt=12,              # max_h_cm = None: one tall figure
             label_in=0.45,           # width of the row-label column (metal name), inches
             gap_in=0.06, dpi=600)
HQ_STYLE = dict(fig2="classes", fig3="continuous")   # HQ maps: 'classes' (HQ_CLASSES) or 'continuous' (same colours
                                                     # at the class edges, saturating at the last edge)
HQ_FIG3_ORDER = "sites"             # Fig. 3 metal order: 'sites' (exceeded at more sites first, then A-Z) or
                                    # 'alphabetical'
GROUP_BOXES = False                 # Fig. 3: dashed boxes around the HI column and the HQ panels
FIG3_EXCLUDE = ["Be", "Cr", "Tl"]   # Fig. 3: metals left out of the figure (symbols as in the panel titles)
FIG3_HQ_KEY = False                 # Fig. 3: HQ class legend in a column on the right
FIG3_MAP_KEY = False                # Fig. 3: legend row of wells and CCR units under the maps
FIG3_HI_KEY = dict(title="Hazard Index",   # title above the HI colour bar (e.g. "Hazard Index\n(normalised)")
                   span=2,          # 1 = colour bar above the HI column only; 2 = above the basemap and HI columns
                   width=0.9,       # colour-bar length, fraction of the columns it spans
                   bar_h=0.8,       # colour-bar thickness, as a multiple of the title font size
                   header_h=0.30)   # height of the header row (title + colour bar), fraction of a map panel
LAYOUT_PAD = 0.02                   # space around and between panels, inches (constrained layout)
LAYOUT_GAP = 0.005                  # extra gap between panels, fraction of the panel size
GROUP_MAX_COLS = 12                 # most bands drawn per feature-group figure (all sites)
DPI = 300
RAMP = ["#1A7A1A", "#3EAA2E", "#96D626", "#E8E824", "#FAC829", "#F57B12", "#E8220A"]   # green -> red
COLORS = dict(
    hi=RAMP,                                                             # HI / HQ normalised, 0 -> 1
    prob=RAMP,                                                           # RF probability of class 1, 0 -> 1
    risk=["#1A7A1A", "#96D626", "#E8E824", "#F57B12", "#E8220A"],        # risk classes 0-4, low -> critical
    hq=["#1A7A1A", "#7FBF4D", "#FDE38A", "#F4865A", "#D7301F"],          # HQ classes (HQ_CLASSES)
    box_hi="#D7301F", box_hq="#2171B5",                                  # Fig. 3 group boxes
    na="#D9D9D9",                                                        # Fig. 4: HQ < 1 / not measured
    wellmap_hi=["#1A7A1A", "#96D626", "#E8E824", "#FAC829", "#F57B12", "#E8220A"],   # Fig. 5 HI colours (low-high)
    wellmap_ccr="black", wellmap_roi="black", wellmap_edge="black",
    label=["#1A7A1A", "#E8220A"],                                        # class 0 lower HI, class 1 higher HI
    error={"TN": "#4878C0", "FP": "#F5A65B", "FN": "#D7301F", "TP": "#2E9B3C"},
    shap=["#2166AC", "#B2ABD2", "#B2182B"],                              # SHAP beeswarm: low -> high feature value
    ccr="#5A1E8C", wells="black", test_outline="black",
    models={"RandomForest": "#1A7A1A", "CART": "#96D626", "SVM": "#2166AC", "NaiveBayes": "#9E9AC8",
            "GradientBoost": "#F57B12", "KNN": "#7F7F7F"},
    groups={"Vegetation indices": "#1A7A1A", "Sentinel-2": "#3B528B", "CSTV": "#CB181D", "Rate of change": "#F57B12",
            "Topographic": "#996633", "Climate": "#A50026", "Soil": "#D95F0E", "Hydrologic": "#2171B5",
            "Land cover": "#88B053", "Seasonal": "#9E9AC8"},
)
WELL_SIZE = dict(single=30, multi=14, wellmap=70, print=9)   # well marker area on the single-site maps and on Figs 1-2
CCR_LW = dict(single=1.0, multi=0.6, print=0.5)
PAL = {   # feature maps (as the Code Editor visualisation)
    "vegetation": ["#08519C", "#3182BD", "#9ECAE1", "#C7E9C0", "#41AB5D", "#006D2C"],   # blue (low) -> green (high)
    "reflectance": ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
    "cstv": ["#008000", "#FFFF00", "#FF0000"],                 # CSTV: green (0) - yellow (1) - red (2)
    "dem": ["#006600", "#33cc33", "#ffff66", "#cc9933", "#996633", "#f5f5f5"],
    "slope": ["#ffffff", "#fdd49e", "#fc8d59", "#d7301f", "#7f0000"],
    "aspect": ["#ff0000", "#ffff00", "#00ff00", "#00ffff", "#0000ff", "#ff00ff", "#ff0000"],
    "diverging": ["#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#b2182b"],
    "temperature": ["#313695", "#74add1", "#ffffbf", "#f46d43", "#a50026"],
    "humidity": ["#f7fbff", "#9ecae1", "#4292c6", "#08519c", "#08306b"],
    "moisture": ["#8c510a", "#d8b365", "#f6e8c3", "#5ab4ac", "#01665e"],
    "ph": ["#d73027", "#fee08b", "#1a9850", "#4575b4", "#313695"],
    "sand": ["#fffbe6", "#fee391", "#fec44f", "#d95f0e", "#993404"],
    "silt": ["#fff5eb", "#fdd0a2", "#fd8d3c", "#d94801", "#7f2704"],
    "clay": ["#fcfbfd", "#dadaeb", "#9e9ac8", "#6a51a3", "#3f007d"],
    "precip": ["#ffffd9", "#c7e9b4", "#7fcdbb", "#1d91c0", "#0c2c84"],
    "upstream": ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
    "hand": ["#08306b", "#4292c6", "#c6dbef", "#fee391", "#cc4c02"],
    "twi": ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"],
    "rate": ["#d73027", "#fc8d59", "#f7f7f7", "#91cf60", "#1a9850"],
    "dw": ["#ffffff", "#397D49"],
    "dwclass": ["#419BDF", "#397D49", "#88B053", "#7A87C6", "#E49635", "#DFC35A", "#C4281B", "#A59B8F", "#B39FE1"],
}
STRETCH = (2, 98)                   # percentiles of the feature-map colour range
CSTV_RANGE = (0, 2)                 # fixed CSTV colour range (min 0, 1 at the middle, max 2)
CSTV_STYLE = "continuous"           # 'continuous' (green-yellow-red ramp) or 'classes' (three blocks: 0-2/3, 2/3-4/3, 4/3-2)
VI_RANGE = {"NDVI": (0, 1), "SAVI": (0, 1), "NDRE1": (0, 1), "NDRE2": (0, 1),     # fixed colour ranges of vegetation
            "NDVI_spring": (0, 1), "NDVI_autumn": (0, 1)}                         # indices (blue 0 to green 1); other
                                                                                  # indices use STRETCH
# display scaling of SoilGrids layers on the feature maps (stored as pH x 10 and g/kg); zeros are SoilGrids no-data
# (water, built-up) and are left blank and out of the colour range. Display only: the model features are unchanged.
SOIL_DISPLAY_SCALE = {"pH": 0.1, "Sand_pct_0_5cm": 0.1, "Silt_pct_0_5cm": 0.1, "Clay_pct_0_5cm": 0.1}
SOIL_ZERO_AS_NODATA = True
EDGE_PX = 3                         # border pixels left out of the stretch (edge artefacts of curvature, TWI)
MAPS_PER_ROW = 4
# ======================================================================================================================

import matplotlib as mpl
mpl.use("Agg") if "ipykernel" not in sys.modules else None
import matplotlib.pyplot as plt
if tuple(int(v) for v in re.findall(r"\d+", mpl.__version__)[:2]) < (3, 5):
    raise ImportError(f"matplotlib {mpl.__version__} found; 3.5 or newer is needed (pip install -U matplotlib)")
import matplotlib.patheffects
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from scipy.ndimage import distance_transform_edt
from scipy.stats import binomtest, mannwhitneyu, rankdata, spearmanr
from scipy.stats import t as t_dist
from matplotlib.patches import Rectangle
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

plt.rcParams.update({"font.family": "serif", "font.serif": FONT_FAMILY, "font.size": FONT["base"],
                     "mathtext.fontset": "stix", "axes.titlesize": FONT["title"], "axes.labelsize": FONT["label"],
                     "xtick.labelsize": FONT["tick"], "ytick.labelsize": FONT["tick"], "legend.fontsize": FONT["legend"]})
VI_NAMES = ["NDVI", "MTCI", "SAVI", "CIRE", "NDRE1", "NDRE2", "LAI", "LCC", "CCC"]
S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
GRADIENT = {"upgradient": "Upgradient", "background": "Upgradient", "downgradient": "Downgradient",
            "compliance": "Downgradient"}


# ---------------------------------------------------------------- files
def find(name):
    """First match of a file name in SEARCH_DIRS (exact name, then names ending with it)."""
    for d in SEARCH_DIRS:
        p = Path(d) / name
        if p.is_file():
            return p
    for d in SEARCH_DIRS:
        hits = sorted(Path(d).glob(f"*{name}")) if Path(d).is_dir() else []
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name} not found in {', '.join(map(str, SEARCH_DIRS))}")


def metres(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------- readers and table writers
def read_raster(path):
    """(bands, rows, cols) float array, geotransform (x0, y0, dx, dy) and band names. rasterio if installed, else
    tifffile with a pure-Python LZW decoder (EPSG:4326 GeoTIFFs from Earth Engine)."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            a = src.read(masked=True).astype("float64").filled(np.nan)
            t = src.transform
            if src.crs is not None and src.crs.to_epsg() not in (4326, None):
                raise ValueError(f"{path.name}: CRS {src.crs} - this script expects EPSG:4326 (as Earth Engine export)")
            return a, (t.c, t.f, t.a, -t.e), list(src.descriptions)
    except ImportError:
        pass
    import tifffile
    with tifffile.TiffFile(path) as tf:
        p = tf.pages[0]
        try:
            a = p.asarray()
        except ValueError:                                  # LZW without imagecodecs
            a = _read_lzw(tf, p)
        tags = {tg.name: tg.value for tg in p.tags.values()}
    a = np.asarray(a, dtype="float64")
    a = a[None] if a.ndim == 2 else np.moveaxis(a, -1, 0)
    sx, sy = tags["ModelPixelScaleTag"][:2]
    x0, y0 = tags["ModelTiepointTag"][3:5]
    names = re.findall(r'role="description">([^<]*)<', str(tags.get("GDAL_METADATA", "")))
    nod = tags.get("GDAL_NODATA")
    if nod is not None:
        a[a == float(str(nod).strip("\x00"))] = np.nan
    return a, (x0, y0, sx, sy), names or [f"band_{i + 1}" for i in range(a.shape[0])]


def _lzw(data):
    out, table = bytearray(), [bytes([i]) for i in range(256)] + [b"", b""]
    nbits, prev, buf, cnt, pos = 9, None, 0, 0, 0
    while True:
        while cnt < nbits and pos < len(data):
            buf = (buf << 8) | data[pos]; pos += 1; cnt += 8
        if cnt < nbits:
            break
        cnt -= nbits
        code = (buf >> cnt) & ((1 << nbits) - 1)
        buf &= (1 << cnt) - 1
        if code == 256:
            table, nbits, prev = table[:258], 9, None
            continue
        if code == 257:
            break
        if prev is None:
            entry = table[code]
        elif code < len(table):
            entry = table[code]; table.append(prev + entry[:1])
        else:
            entry = prev + prev[:1]; table.append(entry)
        out += entry
        prev = entry
        if len(table) + 1 >= (1 << nbits) and nbits < 12:
            nbits += 1
    return bytes(out)


def _read_lzw(tf, p):
    spp, h, w = p.samplesperpixel, p.imagelength, p.imagewidth
    dt = p.dtype.newbyteorder(tf.byteorder)
    out = np.full((h, w, spp), np.nan)
    tw, tl = (p.tilewidth, p.tilelength) if p.is_tiled else (w, p.rowsperstrip)
    ncol = -(-w // tw)
    for k, (off, n) in enumerate(zip(p.dataoffsets, p.databytecounts)):
        tf.filehandle.seek(off)
        raw = tf.filehandle.read(n)
        b = _lzw(raw) if p.compression == 5 else raw
        r0, c0 = ((k // ncol) * tl, (k % ncol) * tw) if p.is_tiled else (k * tl, 0)
        nr = tl if p.is_tiled else min(tl, h - r0)
        tile = np.frombuffer(b[:nr * tw * spp * dt.itemsize], dt).reshape(nr, tw, spp)
        rr, cc = min(nr, h - r0), min(tw, w - c0)
        out[r0:r0 + rr, c0:c0 + cc] = tile[:rr, :cc]
    return out


def read_dbf(path):
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[4:8])[0]
    hl, rl = struct.unpack("<HH", b[8:12])
    fields, o = [], 32
    while b[o] != 0x0D:
        fields.append((b[o:o + 11].split(b"\0")[0].decode(), chr(b[o + 11]), b[o + 16])); o += 32
    rows = []
    for i in range(n):
        r, rec, p_ = b[hl + i * rl + 1:hl + (i + 1) * rl], {}, 0
        for name, t, ln in fields:
            v = r[p_:p_ + ln].decode("utf-8", "replace").strip(); p_ += ln
            rec[name] = float(v) if t in "FN" and v else v
        rows.append(rec)
    return rows


def read_shp(path):
    b, o, shapes = open(path, "rb").read(), 100, []
    while o < len(b):
        _, cl = struct.unpack(">ii", b[o:o + 8]); rec = b[o + 8:o + 8 + 2 * cl]; o += 8 + 2 * cl
        if struct.unpack("<i", rec[:4])[0] not in (5, 15, 25):
            shapes.append([]); continue
        npart, npt = struct.unpack("<ii", rec[36:44])
        parts = list(struct.unpack(f"<{npart}i", rec[44:44 + 4 * npart]))
        pts = np.frombuffer(rec[44 + 4 * npart:44 + 4 * npart + 16 * npt], "<f8").reshape(-1, 2)
        shapes.append([pts[a:b_] for a, b_ in zip(parts, parts[1:] + [npt])])
    return shapes


def albers_inverse(x, y):
    """USA Contiguous Albers Equal Area Conic (USGS, NAD83) to longitude, latitude."""
    a, f = 6378137.0, 1 / 298.257222101
    e2 = 2 * f - f * f; e = np.sqrt(e2)
    lat0, lon0, p1, p2 = np.radians([23.0, -96.0, 29.5, 45.5])
    m_ = lambda p: np.cos(p) / np.sqrt(1 - e2 * np.sin(p) ** 2)
    q_ = lambda p: (1 - e2) * (np.sin(p) / (1 - e2 * np.sin(p) ** 2)
                               - np.log((1 - e * np.sin(p)) / (1 + e * np.sin(p))) / (2 * e))
    n = (m_(p1) ** 2 - m_(p2) ** 2) / (q_(p2) - q_(p1)); C = m_(p1) ** 2 + n * q_(p1); r0 = a * np.sqrt(C - n * q_(lat0)) / n
    rho, th = np.hypot(x, r0 - y), np.arctan2(x, r0 - y)
    qq = (C - (rho * n / a) ** 2) / n
    phi = np.arcsin(np.clip(qq / 2, -1, 1))
    for _ in range(10):
        s_ = np.sin(phi)
        phi = phi + (1 - e2 * s_ * s_) ** 2 / (2 * np.cos(phi)) * (qq / (1 - e2) - s_ / (1 - e2 * s_ * s_)
                                                                   + np.log((1 - e * s_) / (1 + e * s_)) / (2 * e))
    return np.degrees(lon0 + th / n), np.degrees(phi)


def tex_table(df, caption, label, path, colfmt=None, group_header=None):
    esc = lambda v: str(v).replace("_", "\\_").replace("<", "$<$").replace(">", "$>$")
    df = df.map(esc) if hasattr(df, "map") else df.applymap(esc)
    df.columns = [esc(c) for c in df.columns]
    caption = esc(caption)
    cols = list(df.columns)
    colfmt = colfmt or ("l" * 2 + "c" * (len(cols) - 2))
    caption = re.sub(r"^Table \w+\.\s*", "", caption)        # LaTeX numbers the table itself
    L = ["\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\small",
         f"\\begin{{tabular}}{{{colfmt}}}", "\\toprule"]
    if group_header:
        L.append(group_header)
    L.append(" & ".join(cols) + " \\\\")
    L.append("\\midrule")
    for _, r in df.iterrows():
        L.append(" & ".join(str(v) for v in r.values) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tex = "\n".join(L).replace("%", "\\%").replace("\u00b1", "$\\pm$").replace("\u03c1", "$\\rho$") \
        .replace("\u0394", "$\\Delta$").replace("\u00b2", "$^2$").replace("\u03b4", "$\\delta$").replace("\u2265", "$\\geq$") \
        .replace("log10(HI)", "$\\log_{10}$(HI)").replace("\u2013", "--").replace("\u03ba", "$\\kappa$").replace("\u2212", "$-$")
    Path(path).write_text(tex, encoding="utf-8")
    print("saved", Path(path).name)


def docx_tables(tables, path):
    try:
        from docx import Document
        from docx.shared import Pt
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
    except ImportError:
        print("python-docx not installed (pip install python-docx): Word tables skipped")
        return

    def rule(cell, edge, sz):
        tcPr = cell._tc.get_or_add_tcPr()
        b = tcPr.find(qn("w:tcBorders"))
        if b is None:
            b = OxmlElement("w:tcBorders"); tcPr.append(b)
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(sz)); e.set(qn("w:color"), "000000")
        b.append(e)

    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(10)
    for caption, df in tables:
        p = doc.add_paragraph(); r = p.add_run(caption.split(".")[0] + "."); r.bold = True
        p.add_run(caption[len(caption.split(".")[0]) + 1:])
        t = doc.add_table(rows=len(df) + 1, cols=len(df.columns))
        for j, c in enumerate(df.columns):
            t.cell(0, j).text = str(c)
        for i, (_, row) in enumerate(df.iterrows(), 1):
            for j, v in enumerate(row.values):
                t.cell(i, j).text = str(v)
        for j in range(len(df.columns)):                     # three-line table
            rule(t.cell(0, j), "top", 12); rule(t.cell(0, j), "bottom", 6); rule(t.cell(len(df), j), "bottom", 12)
        for row in t.rows:
            for c in row.cells:
                for pp in c.paragraphs:
                    for rr in pp.runs:
                        rr.font.size = Pt(9)
        doc.add_paragraph()
    doc.save(path)
    print("saved", Path(path).name)


_FONT_BASE = []                     # FONT, matplotlib font defaults and PRINT font outside the figure being drawn
_RC_FONT = {"font.size": "base", "axes.titlesize": "title", "axes.labelsize": "label", "xtick.labelsize": "tick",
            "ytick.labelsize": "tick", "legend.fontsize": "legend"}


def _set_fonts(key):
    """FONT = FONT outside the figure + FONT_ALL + FIG_FONTS[key]; the matplotlib defaults follow."""
    base, _, pp = _FONT_BASE[0]
    cfg = FIG_FONTS.get(key, {})
    FONT.clear(); FONT.update(base); FONT.update(FONT_ALL)
    FONT.update({k: v for k, v in cfg.items() if k in FONT})
    PRINT["font_pt"] = cfg.get("print_pt", pp)
    plt.rcParams.update({rc: FONT[k] for rc, k in _RC_FONT.items()})


def fonts_for(key):
    """Decorator: draw the figure(s) of this function with the font sizes of FIG_FONTS[key]."""
    def deco(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            _FONT_BASE.append((dict(FONT), {rc: plt.rcParams[rc] for rc in _RC_FONT}, PRINT["font_pt"]))
            try:
                _set_fonts(key)
                return f(*args, **kwargs)
            finally:
                font, rc, pp = _FONT_BASE.pop()
                FONT.clear(); FONT.update(font); plt.rcParams.update(rc); PRINT["font_pt"] = pp
        return wrapper
    return deco


def site_color(site, j=0):
    """Colour of a site on every figure (SITE_COLORS); other labels fall back to the tab10 colours."""
    return SITE_COLORS.get(site, plt.get_cmap("tab10")(j))


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    constrained = (fig.get_layout_engine().__class__.__name__ == "ConstrainedLayoutEngine"
                   if hasattr(fig, "get_layout_engine") else fig.get_constrained_layout())
    if constrained:                                     # matplotlib >= 3.6 / older versions
        if hasattr(fig, "get_layout_engine"):
            fig.get_layout_engine().set(w_pad=LAYOUT_PAD, h_pad=LAYOUT_PAD, wspace=LAYOUT_GAP, hspace=LAYOUT_GAP)
        else:
            fig.set_constrained_layout_pads(w_pad=LAYOUT_PAD, h_pad=LAYOUT_PAD, wspace=LAYOUT_GAP, hspace=LAYOUT_GAP)
    fig.savefig(out / f"{name}.png", dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("  saved", name)


# ---------------------------------------------------------------- site data
def nice(b):
    """Band name as shown on figures: CSTV_NDVI_Diff -> CSTV_NDVI."""
    return re.sub(r"_Diff$", "", b)


# earlier group names, renamed in results restored from older checkpoints
OLD_GROUP_NAMES = {"Vegetation index": "Vegetation indices", "Terrain": "Topographic", "Hydrology": "Hydrologic"}


def band_groups(names):
    g = {"Vegetation indices": [b for b in names if b in VI_NAMES],
         "Sentinel-2": [b for b in names if b in S2_BANDS],
         "CSTV": [b for b in names if b.startswith("CSTV_")],
         "Rate of change": [b for b in names if b.endswith("_Rate")],
         "Topographic": [b for b in names if b in ("DEM", "Slope_deg", "Aspect_deg", "Aspect_deg_sin", "Aspect_deg_cos",
                                                "Curvature")],
         "Climate": [b for b in names if b in ("Temperature", "Humidity", "Moisture", "PrecipJJA_mm_mean")],
         "Soil": [b for b in names if b in ("pH", "Sand_pct_0_5cm", "Silt_pct_0_5cm", "Clay_pct_0_5cm")],
         "Hydrologic": [b for b in names if b in ("UpstreamArea_km2", "HAND_m", "TWI")],
         "Land cover": [b for b in names if b.startswith("DW_")],
         "Seasonal": [b for b in names if b.startswith(("NDVI_spring", "NDVI_autumn", "NDVI_harm"))]}
    used = {b for v in g.values() for b in v}
    other = [b for b in names if b not in used]
    if other:
        g["Other"] = other
    return {k: v for k, v in g.items() if v}


def wells_of(cfg, x0, y0, dx, dy, rows, cols):
    w = pd.read_csv(find(cfg["wells"]))
    if "n_HQ_in_HI" in w:
        w = w[w["n_HQ_in_HI"] >= MIN_HQ_IN_HI]
    w = w[w["HI"] > 0].reset_index(drop=True)
    if "gradient" not in w or "well_name" not in w:
        try:
            g = pd.read_csv(find(cfg["samples"]), usecols=["well_name", "well_designation", "well_latitude",
                                                           "well_longitude"])
            g = g.groupby("well_name").agg(des=("well_designation", "first"), lat=("well_latitude", "first"),
                                           lon=("well_longitude", "first")).reset_index()
            d = metres(w["latitude"].to_numpy()[:, None], w["longitude"].to_numpy()[:, None],
                       g["lat"].to_numpy()[None], g["lon"].to_numpy()[None])
            j = d.argmin(1)
            near = d[np.arange(len(w)), j] <= 30
            if "well_name" not in w:
                w["well_name"] = np.where(near, g["well_name"].to_numpy()[j], [f"W{i + 1}" for i in range(len(w))])
            if "gradient" not in w:
                des = g["des"].astype(str).str.split(",").str[0].str.strip().str.lower().map(GRADIENT)
                w["gradient"] = np.where(near, des.to_numpy()[j], "Unknown")
        except (FileNotFoundError, ValueError, KeyError):
            w["gradient"] = w.get("gradient", "Unknown")
            w["well_name"] = w.get("well_name", [f"W{i + 1}" for i in range(len(w))])
    w["gradient"] = w["gradient"].fillna("Unknown")
    w = hq_from_samples(w, cfg)
    w["row"] = np.floor((y0 - w["latitude"]) / dy).astype(int)
    w["col"] = np.floor((w["longitude"] - x0) / dx).astype(int)
    return w[w["row"].between(0, rows - 1) & w["col"].between(0, cols - 1)].reset_index(drop=True)


def hq_from_samples(w, cfg):
    """HQ of the HQ_FROM_SAMPLES metals from the raw sample table: mean concentration per well (mg/L) divided by the
    screening level (SCREENING_MG_L, else the table's standard_threshold); wells matched by name or within 30 m."""
    need = [m for m in HQ_FROM_SAMPLES if f"HQ_{m}" not in w]
    if not need:
        return w
    try:
        g = pd.read_csv(find(cfg["samples"]), low_memory=False)
    except (FileNotFoundError, ValueError):
        return w
    if not {"contaminant", "concentration", "well_name"} <= set(g.columns):
        return w
    g["metal"] = g["contaminant"].astype(str).str.lower().str.split(" ").str[0]
    for m in need:
        q = g[g["metal"] == m].copy()
        if q.empty:
            continue
        unit = q.get("measurement_unit", pd.Series("mg/L", index=q.index)).astype(str).str.lower()
        q["c"] = pd.to_numeric(q["concentration"], errors="coerce") * np.where(unit.str.contains("ug|\u00b5g"), 1e-3, 1.0)
        thr = SCREENING_MG_L.get(m)
        if thr is None and "standard_threshold" in q:
            su = q.get("standard_unit", pd.Series("mg/L", index=q.index)).astype(str).str.lower()
            t = pd.to_numeric(q["standard_threshold"], errors="coerce") * np.where(su.str.contains("ug|\u00b5g"), 1e-3, 1.0)
            thr = float(t.dropna().mode().iloc[0]) if t.notna().any() else None
        if not thr:
            print(f"  {m}: no screening level (set SCREENING_MG_L); not added")
            continue
        a = q.groupby("well_name").agg(c=("c", "mean"), lat=("well_latitude", "first"), lon=("well_longitude", "first"))
        conc = np.full(len(w), np.nan)
        for i, r in w.iterrows():
            if r.get("well_name") in a.index:
                conc[i] = a.loc[r["well_name"], "c"]
            elif {"lat", "lon"} <= set(a.columns) and a["lat"].notna().any():
                d = metres(r["latitude"], r["longitude"], a["lat"].to_numpy(), a["lon"].to_numpy())
                if np.nanmin(d) <= 30:
                    conc[i] = a["c"].to_numpy()[np.nanargmin(d)]
        if np.isfinite(conc).any():
            w[m] = conc
            w[f"HQ_{m}"] = conc / thr
            print(f"  {m}: HQ from {len(q)} samples, screening level {thr:g} mg/L, max HQ {np.nanmax(conc / thr):.3f}")
    return w


def krige_gee(w, x0, y0, dx, dy, rows, cols, k, col="HI"):
    """Ordinary kriging of HI with the fixed exponential variogram of the GEE script (range, sill, nugget, metres);
    pixels farther than maxDistance from every well are masked (as ee kriging)."""
    lat_c = y0 - dy * rows / 2
    kx, ky = 111320 * np.cos(np.radians(lat_c)), 110540
    x = (w["longitude"].to_numpy() - x0) * kx
    y = (y0 - w["latitude"].to_numpy()) * ky
    z = w[col].to_numpy(float)
    nug, sill, rng = k["nugget"], k["sill"], k["range"]
    gam = lambda h: np.where(h > 0, nug + (sill - nug) * (1 - np.exp(-3 * h / rng)), 0.0)
    n = len(z)
    G = np.zeros((n + 1, n + 1)); G[:n, n] = G[n, :n] = 1
    G[:n, :n] = gam(np.hypot(x[:, None] - x[None], y[:, None] - y[None]))
    Gi = np.linalg.pinv(G)
    rr, cc = np.indices((rows, cols))
    gx, gy = ((cc + 0.5) * dx * kx).ravel(), ((rr + 0.5) * dy * ky).ravel()
    out = np.empty(gx.size); dmin = np.empty(gx.size)
    for i in range(0, gx.size, 20000):
        s = slice(i, i + 20000)
        h = np.hypot(x[:, None] - gx[s][None], y[:, None] - gy[s][None])
        g = np.ones((n + 1, h.shape[1])); g[:n] = gam(h)
        out[s] = (Gi @ g)[:n].T @ z
        dmin[s] = h.min(0)
    out[dmin > k["maxDistance"]] = np.nan
    return out.reshape(rows, cols)


def load_site(name):
    cfg = SITES[name]
    A, (x0, y0, dx, dy), names = read_raster(find(f"{cfg['stem']}_Features.tif"))
    X = np.moveaxis(A, 0, -1)
    rows, cols = X.shape[:2]
    keep = [i for i, b in enumerate(names) if b not in DROP_BANDS]
    X, names = X[..., keep].astype(np.float32), [names[i] for i in keep]
    w = wells_of(cfg, x0, y0, dx, dy, rows, cols)
    HI = None
    if HI_SOURCE in ("file", "auto"):
        try:
            HI = read_raster(find(f"{cfg['stem']}_HI.tif"))[0][0]
            src = "file"
        except FileNotFoundError:
            if HI_SOURCE == "file":
                raise
    if HI is None:
        HI = krige_gee(w, x0, y0, dx, dy, rows, cols, cfg["kriging"])
        src = "kriged here (GEE variogram)"
    valid_hi = np.isfinite(HI)
    hmin, hmax = np.nanmin(HI), np.nanmax(HI)
    HIn = (HI - hmin) / max(hmax - hmin, 1e-12)
    alpha = np.nanpercentile(HIn[valid_hi], ALPHA_PERCENTILE)
    L_rule = np.where(valid_hi, (HIn >= alpha).astype(float), np.nan)
    L = None
    if LABEL_SOURCE in ("file", "auto"):
        try:
            L = read_raster(find(f"{cfg['stem']}_Label.tif"))[0][0]
            lsrc = "file"
        except FileNotFoundError:
            if LABEL_SOURCE == "file":
                raise
    if L is None:
        L, lsrc = L_rule, f"rule ({ALPHA_PERCENTILE}th percentile of HI_norm)"
    else:
        ok = np.isfinite(L) & np.isfinite(L_rule)
        print(f"  label file; the {ALPHA_PERCENTILE}th-percentile rule on this HI agrees on "
              f"{(L[ok] == L_rule[ok]).mean():.1%} of the pixels")
    risk = np.digitize(HI, RISK_BREAKS).astype(float)
    risk[~valid_hi] = np.nan
    valid = np.isin(L, [0, 1]) & np.isfinite(X).any(-1)
    print(f"{name} ({cfg['label']}): grid {rows} x {cols}, {len(names)} bands, HI {hmin:.2f}-{hmax:.2f} ({src}), "
          f"label: {lsrc}, class-1 share {np.nanmean(L[valid]):.3f}, {len(w)} wells")
    return dict(name=name, label=cfg["label"], stem=cfg["stem"], X=X, bands=names,
                Y=np.where(valid, L, 0).astype(int), valid=valid, fvalid=np.isfinite(X).any(-1), HI=HI, HIn=HIn,
                risk=risk, alpha=alpha, wells=w, geo=(x0, y0, dx, dy), target="HI")


def ccr_units(S):
    if not CCR_SHP:
        return []
    try:
        shp, dbf = find(CCR_SHP), find(Path(CCR_SHP).with_suffix(".dbf").name)
    except FileNotFoundError:
        return []
    try:
        geographic = find(Path(CCR_SHP).with_suffix(".prj").name).read_text().startswith("GEOGCS")
    except FileNotFoundError:
        geographic = False
    x0, y0, dx, dy = S["geo"]
    rows, cols = S["valid"].shape
    units = []
    for rec, parts in zip(read_dbf(dbf), read_shp(shp)):
        rings = [np.column_stack((p[:, 0], p[:, 1]) if geographic else albers_inverse(p[:, 0], p[:, 1]))
                 for p in parts]
        if rings:
            v = np.vstack(rings)
            if ((v[:, 0] >= x0) & (v[:, 0] <= x0 + cols * dx) & (v[:, 1] <= y0) & (v[:, 1] >= y0 - rows * dy)).any():
                units.append((str(rec.get(CCR_NAME_FIELD, "")), str(rec.get("WMU_type", "")), rings))
    return units


# ---------------------------------------------------------------- map drawing
def extent(S):
    x0, y0, dx, dy = S["geo"]
    rows, cols = S["valid"].shape
    return (x0, x0 + cols * dx, y0 - rows * dy, y0)


def aspect(S):
    """Map height / width as drawn (lon scaled by cos(lat))."""
    e = extent(S)
    return (e[3] - e[2]) / ((e[1] - e[0]) * np.cos(np.radians((e[2] + e[3]) / 2)))


def fit_rows(fig, A, gap_in=0.12, passes=2):
    """Shrink the figure height until consecutive map rows are gap_in inches apart (equal-aspect maps are limited
    by the column width, which leaves blank bands between rows)."""
    for _ in range(passes):
        fig.canvas.draw()
        H = fig.get_figheight()
        y = [(min(a.get_position().y0 for a in row), max(a.get_position().y1 for a in row)) for row in A]
        extra = sum(max(0.0, (y[i][0] - y[i + 1][1]) * H - gap_in) for i in range(len(y) - 1))
        if extra < 0.02:
            break
        fig.set_figheight(H - extra)


def frame(ax, S):
    e = extent(S)
    ax.set_xlim(e[0], e[1]); ax.set_ylim(e[2], e[3])
    ax.set_aspect(1 / np.cos(np.radians((e[2] + e[3]) / 2)))
    ax.set_xticks([]); ax.set_yticks([])


def overlay(ax, S, units, wells=True, mode="single"):
    for _, kind, rings in units:
        for r in rings:
            ax.add_patch(Polygon(r, closed=True, fill=False, ec=COLORS["ccr"], lw=CCR_LW[mode],
                                 hatch="////" if kind == "LF" else None, zorder=4))
    if wells:
        w = S["wells"]
        for g, mk in (("Upgradient", "^"), ("Downgradient", "v"), ("Unknown", "o")):
            s = w["gradient"].eq(g)
            ax.scatter(w.loc[s, "longitude"], w.loc[s, "latitude"], marker=mk, s=WELL_SIZE[mode], c=COLORS["wells"],
                       edgecolors=COLORS["wells"], lw=0.4, zorder=5)


def show(ax, S, Z, cmap, norm, units, title, wells=True, mode="single"):
    cmap = cmap.copy(); cmap.set_bad("#ffffff", 0)
    im = ax.imshow(Z, extent=extent(S), cmap=cmap, norm=norm, interpolation="nearest", origin="upper")
    frame(ax, S); overlay(ax, S, units, wells, mode)
    if title:
        ax.set_title(title, fontweight="bold")
    return im


def hbar(fig, ax, im, label, ticks=None, ticklabels=None):
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03)
    if ticks is not None:
        cb.set_ticks(ticks)
        if ticklabels is not None:
            cb.set_ticklabels(ticklabels)
    cb.ax.tick_params(labelsize=FONT["tick"]); cb.set_label(label, fontsize=FONT["annot"])
    cb.outline.set_linewidth(0.5)
    return cb


def map_handles(units, extra=(), unknown=False):
    h = [Line2D([], [], marker="^", ls="", color=COLORS["wells"], ms=6, label="Upgradient well"),
         Line2D([], [], marker="v", ls="", color=COLORS["wells"], ms=6, label="Downgradient well")]
    if unknown:
        h.append(Line2D([], [], marker="o", ls="", color=COLORS["wells"], ms=5, label="Well, gradient not recorded"))
    if units:
        if any(k != "LF" for _, k, _ in units):
            h.append(Patch(fc="none", ec=COLORS["ccr"], label="CCR impoundment"))
        if any(k == "LF" for _, k, _ in units):
            h.append(Patch(fc="none", ec=COLORS["ccr"], hatch="////", label="CCR landfill"))
    h += list(extra)
    return h


def map_legend(fig, units, extra=(), y=0.0, loc=None, unknown=False):
    h = map_handles(units, extra, unknown)
    if loc:
        fig.legend(handles=h, loc=loc, ncol=min(len(h), 6), frameon=False)
    else:
        fig.legend(handles=h, loc="lower center", ncol=min(len(h), 6), frameon=False, bbox_to_anchor=(0.5, y))


def true_colour(S):
    rgb = []
    for b in ("B4", "B3", "B2"):
        if b not in S["bands"]:
            return None
        v = S["X"][..., S["bands"].index(b)]
        lo, hi = np.nanpercentile(v[S["valid"]], (2, 98))
        rgb.append(np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1))
    return np.nan_to_num(np.dstack(rgb) ** 0.8, nan=1.0)


# ---------------------------------------------------------------- feature maps
def palette_of(b):
    if b.endswith("_Rate") or "minus_summer" in b:
        return PAL["rate"], "zero"
    if b in VI_NAMES or b in ("NDVI_spring", "NDVI_autumn", "NDVI_harm_amplitude"):
        return PAL["vegetation"], VI_RANGE.get(b, "stretch")
    if b in S2_BANDS:
        return PAL["reflectance"], "stretch"
    if b == "CSTV_valid":
        return ["#ffffff", "#000000"], (0, 1)
    if b.startswith("CSTV_"):
        return PAL["cstv"], CSTV_RANGE
    if b == "DW_LandCover":
        return PAL["dwclass"], (-0.5, 8.5)
    if b.startswith("DW_"):
        return PAL["dw"], (0, 1)
    fixed = {"DEM": ("dem", "stretch"), "Slope_deg": ("slope", "stretch"), "Aspect_deg": ("aspect", (0, 360)),
             "Aspect_deg_sin": ("diverging", (-1, 1)), "Aspect_deg_cos": ("diverging", (-1, 1)),
             "Curvature": ("diverging", "zero"), "Temperature": ("temperature", "stretch"),
             "Humidity": ("humidity", "stretch"), "Moisture": ("moisture", "stretch"), "pH": ("ph", "stretch"),
             "Sand_pct_0_5cm": ("sand", "stretch"), "Silt_pct_0_5cm": ("silt", "stretch"),
             "Clay_pct_0_5cm": ("clay", "stretch"), "PrecipJJA_mm_mean": ("precip", "stretch"),
             "UpstreamArea_km2": ("upstream", "log"), "HAND_m": ("hand", "stretch"), "TWI": ("twi", "stretch"),
             "NDVI_harm_phase": ("aspect", (-np.pi, np.pi))}
    if b in fixed:
        k, mode = fixed[b]
        return PAL[k], mode
    return PAL["reflectance"], "stretch"


def band_style(S, b):
    """Values, colour map, normalisation, stretch mode and 'constant' flag of one feature band."""
    v = S["X"][..., S["bands"].index(b)].astype(float)
    if b in SOIL_DISPLAY_SCALE:                               # pH x 10 -> pH, g/kg -> %
        if SOIL_ZERO_AS_NODATA:
            v = np.where(v == 0, np.nan, v)
        v = v * SOIL_DISPLAY_SCALE[b]
    pal, mode = palette_of(b)
    discrete = b == "DW_LandCover" or (b.startswith("CSTV_") and b != "CSTV_valid" and CSTV_STYLE == "classes")
    cm = ListedColormap(pal) if discrete else LinearSegmentedColormap.from_list(b, pal)
    inner = np.zeros_like(S["valid"]); e = EDGE_PX
    inner[e:-e or None, e:-e or None] = True                  # border pixels left out of the stretch
    vv = v[S["valid"] & np.isfinite(v) & inner]
    if vv.size == 0:
        return np.full_like(v, np.nan), cm, Normalize(0, 1), None, True, np.nan
    flat = np.nanmax(vv) - np.nanmin(vv) <= 1e-6 * max(1.0, abs(np.nanmean(vv)))
    if flat:
        nm, mode = Normalize(np.nanmean(vv) - 1, np.nanmean(vv) + 1), None
    elif isinstance(mode, tuple):
        nm = Normalize(*mode)
    elif mode == "zero":
        m = np.nanmax(np.abs(np.nanpercentile(vv, STRETCH))) or 1
        nm = Normalize(-m, m)
    elif mode == "log":
        pos = vv[vv > 0]
        nm = mpl.colors.LogNorm(np.nanpercentile(pos, STRETCH[0]), np.nanpercentile(pos, 99.5))
        v = np.where(v > 0, v, np.nan)
    else:
        lo, hi = np.nanpercentile(vv, STRETCH)
        nm = Normalize(lo, hi if hi > lo else lo + 1e-6)
    return v, cm, nm, mode, flat, float(np.nanmean(vv))


def band_bar(fig, ax, im, mode, size=None):
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=size or FONT["tick"]); cb.outline.set_linewidth(0.5)
    if mode != "log":
        cb.formatter = mpl.ticker.ScalarFormatter(useOffset=False); cb.update_ticks()
    cb.ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3) if mode != "log" else mpl.ticker.LogLocator())
    if mode == CSTV_RANGE and CSTV_RANGE == (0, 2):              # CSTV key as specified: min 0, 1, max 2
        cb.set_ticks([0, 1, 2]); cb.set_ticklabels(["min 0", "1", "max 2"])
    return cb


@fonts_for("feature_maps")
def fig_feature_groups(S, units, out):
    """One figure per feature group for one site: every band of the group."""
    for g, bands in band_groups(S["bands"]).items():
        n = len(bands)
        nc = min(MAPS_PER_ROW, n)
        nr = -(-n // nc)
        fig = plt.figure(figsize=(3.0 * nc, (3.0 * aspect(S) + 0.75) * nr + 0.9), constrained_layout=True)
        gs = fig.add_gridspec(nr + 1, nc, height_ratios=[aspect(S) + 0.25] * nr + [0.12])
        axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(nc)] for i in range(nr)])
        for ax, b in zip(axs.ravel(), bands):
            v, cm, nm, mode, flat, mean = band_style(S, b)
            im = show(ax, S, v, cm, nm, units, f"{nice(b)} (constant {mean:.4g})" if flat else nice(b), wells=True)
            cb = band_bar(fig, ax, im, mode)
            if flat:
                cb.ax.set_visible(False)
        for ax in axs.ravel()[n:]:
            ax.axis("off")
        fig.suptitle(f"{S['label']}: {g} ({n} bands)", fontweight="bold", fontsize=FONT["suptitle"])
        lg = fig.add_subplot(gs[-1, :]); lg.axis("off")            # map legend below the maps
        h = map_handles(units, unknown=S["wells"]["gradient"].eq("Unknown").any())
        lg.legend(handles=h, loc="center", ncol=min(len(h), 3 if nc < 3 else len(h)), frameon=False,
                  fontsize=FONT["legend"])
        save(fig, out / "feature_maps", f"Features_{S['label'].replace(' ', '')}_{re.sub('[^A-Za-z0-9]', '', g)}")


@fonts_for("feature_groups_all_sites")
def fig_feature_groups_all_sites(FS, units_by, out):
    """One figure per feature group: rows = sites, columns = the group's bands (each panel stretched to its site)."""
    order = []
    for S in FS:
        for g, bands in band_groups(S["bands"]).items():
            if g not in [o[0] for o in order]:
                order.append((g, list(bands)))
            else:
                k = [o[0] for o in order].index(g)
                order[k][1].extend(b for b in bands if b not in order[k][1])
    P = PANEL_IN["groups"]
    for g, bands in order:
        bands = bands[:GROUP_MAX_COLS]
        n, m = len(FS), len(bands)
        hr = [aspect(S) + 0.22 for S in FS]                    # rows sized to each site's map and colour bar
        fig = plt.figure(figsize=(P * m + 0.6, P * (sum(hr) + 0.12) + 0.9), constrained_layout=True)
        gs = fig.add_gridspec(n + 1, m, height_ratios=hr + [0.12])
        A = np.array([[fig.add_subplot(gs[i, j]) for j in range(m)] for i in range(n)])
        for i, S in enumerate(FS):
            for j, b in enumerate(bands):
                ax = A[i, j]
                if b not in S["bands"]:
                    ax.axis("off"); ax.text(0.5, 0.5, "not in stack", ha="center", va="center",
                                            transform=ax.transAxes, fontsize=FONT["annot"])
                    continue
                v, cm, nm, mode, flat, mean = band_style(S, b)
                im = show(ax, S, v, cm, nm, units_by[S["label"]], None, mode="multi")
                cb = band_bar(fig, ax, im, mode, size=FONT["key"] - 1)
                if flat:
                    cb.ax.set_visible(False)
                    ax.text(0.5, 0.5, f"constant\n{mean:.4g}", ha="center", va="center", transform=ax.transAxes,
                            fontsize=FONT["annot"])
                if i == 0:
                    ax.set_title(nice(b), fontweight="bold", fontsize=FONT["title"])
            A[i, 0].set_ylabel(S["label"], fontweight="bold", fontsize=FONT["row"])
        lg = fig.add_subplot(gs[-1, :]); lg.axis("off")            # group name and map legend below the maps
        h = map_handles([u for S in FS for u in units_by[S["label"]]],
                        unknown=any(S["wells"]["gradient"].eq("Unknown").any() for S in FS))
        lg.legend(handles=h, loc="center right", ncol=len(h), frameon=False, fontsize=FONT["legend"])
        lg.text(0.0, 0.5, f"{g}: all sites", transform=lg.transAxes, ha="left", va="center", fontweight="bold",
                fontsize=FONT["suptitle"])
        save(fig, out / "feature_groups", f"FeatureGroup_{re.sub('[^A-Za-z0-9]', '', g)}_all_sites")


# ---------------------------------------------------------------- split and models
def block_split(S, seed):
    """32-px blocks; the assignment (test, validation, train by cumulative share) closest to the target shares and
    class-1 fraction in N_SPLIT_TRIALS random orders; buffer of BUFFER_PX pixels. Codes 0 train, 1 buffer, 2 val,
    3 test (HazardFuse codes)."""
    valid, Y = S["valid"], S["Y"]
    R, C = valid.shape
    nbc = -(-C // BLOCK_PX)
    rr, cc = np.indices((R, C))
    bid = (rr // BLOCK_PX) * nbc + (cc // BLOCK_PX)
    nb = bid.max() + 1
    bn = np.bincount(bid[valid], minlength=nb).astype(float)
    bp = np.bincount(bid[valid], weights=Y[valid], minlength=nb)
    cand = np.flatnonzero(bn > 0)
    nv, pall = valid.sum(), Y[valid].mean()
    ftr, fva, fte = SPLIT_FRAC
    rng = np.random.default_rng(seed)
    best, best_s = None, np.inf
    for _ in range(N_SPLIT_TRIALS):
        order = rng.permutation(cand)
        frac = bn[order] / nv
        mid = np.cumsum(frac) - frac / 2
        a = np.full(nb, -1)
        a[order] = np.where(mid < fte, 2, np.where(mid < fte + fva, 1, 0))
        s, ok = 0.0, True
        for code, target in ((2, fte), (1, fva), (0, ftr)):
            if target == 0:                                   # e.g. SPLIT_FRAC = (0.70, 0.0, 0.30): no validation set
                continue
            sel = a == code
            n, pos = bn[sel].sum(), bp[sel].sum()
            if n == 0 or pos == 0 or pos == n:
                ok = False; break
            s += abs(n / nv - target) + abs(pos / n - pall)
        if ok and s < best_s:
            best, best_s = a, s
    sp = best[bid]
    if VAL_TO_TRAIN:                                          # same test blocks, validation blocks used for training
        sp = np.where(sp == 1, 0, sp)
    d_eval = distance_transform_edt(~np.isin(sp, [1, 2]))
    d_test = distance_transform_edt(sp != 2)
    code = np.full((R, C), 1, dtype=np.int8)
    code[valid & (sp == 0) & (d_eval > BUFFER_PX)] = 0
    code[valid & (sp == 1) & (d_test > BUFFER_PX)] = 2
    code[valid & (sp == 2)] = 3
    code[~valid] = -1
    return code


def make_model(name, seed, n_feat):
    p = dict(MODEL_PARAMS.get(name, {}))
    cw = "balanced" if CLASS_WEIGHTS else None
    if name == "RandomForest":
        return RandomForestClassifier(random_state=seed, class_weight=cw, **p)
    if name == "CART":
        return DecisionTreeClassifier(random_state=seed, class_weight=cw, **p)
    if name == "SVM":
        if p.get("gamma") == "1/n_features":
            p["gamma"] = 1.0 / n_feat
        return SVC(random_state=seed, class_weight=cw, **p)
    if name == "NaiveBayes":
        return CategoricalNB(min_categories=NB_BINS + 1, **p)
    if name == "GradientBoost":
        import inspect
        if "class_weight" in inspect.signature(HistGradientBoostingClassifier).parameters:
            return HistGradientBoostingClassifier(random_state=seed, class_weight=cw, **p)
        m = HistGradientBoostingClassifier(random_state=seed, **p)   # scikit-learn < 1.2: weights passed to fit()
        m.needs_sample_weight = CLASS_WEIGHTS
        return m
    if name == "KNN":
        return KNeighborsClassifier(**p)
    raise ValueError(name)


def scores_of(y, pred, score):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    out = dict(OA=accuracy_score(y, pred), kappa=cohen_kappa_score(y, pred),
               macro_F1=f1_score(y, pred, average="macro", zero_division=0),
               balanced_acc=0.5 * (recall_score(y, pred, pos_label=0, zero_division=0)
                                   + recall_score(y, pred, pos_label=1, zero_division=0)),
               MCC=matthews_corrcoef(y, pred),
               AUC=roc_auc_score(y, score) if score is not None and len(np.unique(y)) == 2 else np.nan,
               TN=int(cm[0, 0]), FP=int(cm[0, 1]), FN=int(cm[1, 0]), TP=int(cm[1, 1]))
    for c in (0, 1):
        out[f"prec_{c}"] = precision_score(y, pred, pos_label=c, zero_division=0)
        out[f"rec_{c}"] = recall_score(y, pred, pos_label=c, zero_division=0)
        out[f"f1_{c}"] = f1_score(y, pred, pos_label=c, zero_division=0)
    return out


def tune_threshold(y, s, default):
    """Threshold on the class-1 score that maximises balanced accuracy (mean of the two class recalls) on the
    validation pixels; ties go to the threshold closest to the default."""
    if len(np.unique(y)) < 2:
        return default
    cand = np.unique(np.quantile(s, np.linspace(0.005, 0.995, 199)))
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    best, best_t = -1.0, default
    for t in cand:
        p = s >= t
        ba = 0.5 * ((p & (y == 1)).sum() / n1 + (~p & (y == 0)).sum() / n0)
        if ba > best + 1e-12 or (abs(ba - best) <= 1e-12 and abs(t - default) < abs(best_t - default)):
            best, best_t = ba, float(t)
    return best_t


def class_1_score(m, X):
    if hasattr(m, "predict_proba") and not isinstance(m, SVC):
        return m.predict_proba(X)[:, 1]
    return m.decision_function(X)


def prepare(S, code):
    """Standardised matrix of all pixels (training mean and SD; missing -> 0), binned matrix for Naive Bayes."""
    F = S["X"].reshape(-1, S["X"].shape[-1]).astype(float)
    tr = (code.ravel() == 0)
    mu = np.nanmean(F[tr], 0)
    sd = np.nanstd(F[tr], 0)
    sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
    mu[~np.isfinite(mu)] = 0.0
    Z = (F - mu) / sd if STANDARDISE else F.copy()
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    Q = np.rint((np.clip(Z, -4, 4) + 4) / 8 * NB_BINS).astype(int)
    return Z, Q


def run_models(S, models=None, importance=True, code=None):
    """All models and seeds; returns per-seed metrics, importances and the map model's prediction. `code`: a fixed
    partition (block_split codes) to use instead of a new one, e.g. the HI partition for label variants."""
    models = models or MODELS
    rows, imp_rows, maps = [], [], {}
    code0 = block_split(S, SPLIT_SEED) if code is None else np.where(S["valid"], code, -1)
    Yf = S["Y"].ravel()
    cat_idx = [i for i, b in enumerate(S["bands"]) if b in NO_CATEGORICAL]
    for seed in SEEDS:
        code = code0 if (SEED_SCOPE == "models" or code is not None) else block_split(S, seed)
        Z, Q = prepare(S, code)
        cf = code.ravel()
        part = {p: np.flatnonzero(cf == c) for p, c in (("train", 0), ("val", 2), ("test", 3))}
        print(f"  seed {seed}: train {len(part['train'])}, val {len(part['val'])}, test {len(part['test'])} pixels; "
              f"test class-1 share {Yf[part['test']].mean():.3f}")
        rng = np.random.default_rng(seed)
        for name in models:
            cols = [i for i in range(Z.shape[1]) if not (name in ("KNN", "SVM") and i in cat_idx)]
            D = Q if name == "NaiveBayes" else Z
            tr = part["train"]
            if name == "SVM" and MAX_TRAIN_SVM and len(tr) > MAX_TRAIN_SVM:      # stratified subsample
                ytr = Yf[tr]
                tr = np.sort(np.concatenate([rng.choice(tr[ytr == c], int(round(MAX_TRAIN_SVM * (ytr == c).mean())),
                                                        replace=False) for c in (0, 1)]))
            m = make_model(name, seed, len(cols))
            fit_kw = {}
            if CLASS_WEIGHTS and (name == "NaiveBayes" or getattr(m, "needs_sample_weight", False)):
                cnt = np.bincount(Yf[tr], minlength=2)
                fit_kw["sample_weight"] = (len(tr) / (2 * cnt))[Yf[tr]]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(D[tr][:, cols], Yf[tr], **fit_kw)
            thr0 = 0.0 if isinstance(m, SVC) else THRESHOLD
            sc = {}
            for p, idx in part.items():
                if not len(idx):                             # no validation set when SPLIT_FRAC[1] == 0
                    continue
                Xp = D[idx][:, cols]
                s = sc[p] = class_1_score(m, Xp)
                pred = (s >= thr0).astype(int)
                rows.append(dict(site=S["label"], target=S["target"], model=name, seed=seed, partition=p, n=len(idx),
                                 threshold=thr0, **scores_of(Yf[idx], pred, s)))
            if TUNE_THRESHOLD and len(part["val"]):                # threshold from the validation blocks
                thr = tune_threshold(Yf[part["val"]], sc["val"], thr0)
                idx = part["test"]
                rows.append(dict(site=S["label"], target=S["target"], model=name, seed=seed, partition="test_tuned",
                                 n=len(idx), threshold=thr,
                                 **scores_of(Yf[idx], (sc["test"] >= thr).astype(int), sc["test"])))
            if importance and hasattr(m, "feature_importances_"):
                for i, c in enumerate(cols):
                    imp_rows.append(dict(site=S["label"], model=name, seed=seed, feature=S["bands"][c],
                                         importance=m.feature_importances_[i]))
            if importance and name == "RandomForest":        # group permutation importance on the test pixels
                Xt = D[part["test"]][:, cols]
                yt = Yf[part["test"]]
                base = accuracy_score(yt, (class_1_score(m, Xt) >= THRESHOLD).astype(int))
                for g, bands in band_groups(S["bands"]).items():
                    gi = [cols.index(S["bands"].index(b)) for b in bands if S["bands"].index(b) in cols]
                    drops = []
                    for _ in range(3):
                        Xp = Xt.copy()
                        Xp[:, gi] = Xt[rng.permutation(len(Xt))][:, gi]
                        drops.append(base - accuracy_score(yt, (class_1_score(m, Xp) >= THRESHOLD).astype(int)))
                    imp_rows.append(dict(site=S["label"], model="RF permutation accuracy", seed=seed, feature=g,
                                         importance=float(np.mean(drops))))
            if name == MAP_MODEL and seed == MAP_SEED:
                valid_idx = np.flatnonzero(S["valid"].ravel())
                s = class_1_score(m, D[valid_idx][:, cols])
                pred = np.full(S["valid"].size, np.nan); prob = np.full(S["valid"].size, np.nan)
                pred[valid_idx] = (s >= THRESHOLD).astype(int); prob[valid_idx] = s
                maps = dict(pred=pred.reshape(S["valid"].shape), prob=prob.reshape(S["valid"].shape), code=code)
            r = [x for x in rows if x["model"] == name and x["seed"] == seed and x["partition"] == "test"][0]
            print(f"    {S['target']:<10} {name:<13} test Accuracy {r['OA']:.3f}  kappa {r['kappa']:.3f}  macro F1 {r['macro_F1']:.3f}  "
                  f"AUC {r['AUC']:.3f}")
    return pd.DataFrame(rows), pd.DataFrame(imp_rows), maps


# ---------------------------------------------------------------- result figures
@fonts_for("site_maps")
def fig_site_maps(S, units, maps, out):
    """Site | HI normalised | risk | reference label | RF classification | error."""
    fig = plt.figure(figsize=(11.0, 2 * 3.5 * (aspect(S) + 0.3) + 1.3), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[aspect(S) + 0.3] * 2 + [0.1])
    A = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    img, ext, src = S["basemap"]
    A[0].imshow(img, extent=ext, origin="upper", interpolation="bilinear")
    frame(A[0], S); overlay(A[0], S, units)
    north_scale(A[0], S, src)
    A[0].set_title("(a) Site", fontweight="bold")
    cax = fig.colorbar(mpl.cm.ScalarMappable(cmap=ListedColormap(["#ffffff"])), ax=A[0], orientation="horizontal",
                       fraction=0.05, pad=0.03)
    cax.ax.set_visible(False)
    im = show(A[1], S, np.where(S["valid"], S["HIn"], np.nan), LinearSegmentedColormap.from_list("hi", COLORS["hi"]),
              Normalize(0, 1), units, "(b) HI normalised")
    hbar(fig, A[1], im, f"HI_norm (HI {np.nanmin(S['HI']):.2f}–{np.nanmax(S['HI']):.2f})")
    im = show(A[2], S, np.where(S["valid"], S["risk"], np.nan), ListedColormap(COLORS["risk"]), BoundaryNorm(np.arange(-0.5, 5), 5), units,
              "(c) Risk map")
    hbar(fig, A[2], im, "Kriged HI class", ticks=range(5), ticklabels=RISK_NAMES)
    Lm = np.where(S["valid"], S["Y"], np.nan).astype(float)
    im = show(A[3], S, Lm, ListedColormap(COLORS["label"]), BoundaryNorm([-0.5, 0.5, 1.5], 2), units,
              f"(d) Reference label")
    hbar(fig, A[3], im, f"Class 0 = lowest {ALPHA_PERCENTILE}% of HI_norm", ticks=[0, 1],
         ticklabels=["0 lower HI", "1 higher HI"])
    sc = maps.get("scores", {})
    im = show(A[4], S, maps["pred"], ListedColormap(COLORS["label"]), BoundaryNorm([-0.5, 0.5, 1.5], 2), units,
              f"(e) {MAP_NAME} classification")
    hbar(fig, A[4], im, f"Test Accuracy {sc.get('OA', np.nan):.3f} (seed {MAP_SEED})", ticks=[0, 1],
         ticklabels=["Class 0", "Class 1"])
    e = np.where(np.isfinite(maps["pred"]), S["Y"] * 2 + np.nan_to_num(maps["pred"]), np.nan)   # 0 TN 1 FP 2 FN 3 TP
    cols = [COLORS["error"][k] for k in ("TN", "FP", "FN", "TP")]
    im = show(A[5], S, e, ListedColormap(cols), BoundaryNorm(np.arange(-0.5, 4), 4), units, "(f) Error map")
    test = (maps["code"] == 3).astype(float)
    x0, y0, dx, dy = S["geo"]
    rows_, cols_ = test.shape
    A[5].contour(x0 + (np.arange(cols_) + 0.5) * dx, y0 - (np.arange(rows_) + 0.5) * dy, test, levels=[0.5],
                 colors=COLORS["test_outline"], linewidths=0.8, linestyles="--")
    hbar(fig, A[5], im, "Reference × 2 + prediction; dashed: test blocks", ticks=range(4),
         ticklabels=["TN", "FP", "FN", "TP"])
    fig.suptitle(S["label"], fontweight="bold", fontsize=FONT["suptitle"])
    lg = fig.add_subplot(gs[-1, :]); lg.axis("off")                # legend row below the colour bars
    h = map_handles(units, unknown=S["wells"]["gradient"].eq("Unknown").any())
    lg.legend(handles=h, loc="center", ncol=len(h), frameon=False, fontsize=FONT["legend"])
    save(fig, out, f"Fig_{S['label'].replace(' ', '')}_site_HI_risk_label_RF_error")


@fonts_for("confusion")
def fig_confusion(R, out):
    T = R[R.partition == "test"]
    sites = list(dict.fromkeys(T.site))
    fig, axs = plt.subplots(len(sites), len(MODELS), figsize=(1.85 * len(MODELS) + 0.6, 1.95 * len(sites) + 0.5),
                            squeeze=False)
    for i, s in enumerate(sites):
        for j, m in enumerate(MODELS):
            ax = axs[i, j]
            g = T[(T.site == s) & (T.model == m)]
            cm = np.array([[g.TN.sum(), g.FP.sum()], [g.FN.sum(), g.TP.sum()]], float)
            pct = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
            ax.imshow(pct, cmap="Greens", vmin=0, vmax=100)
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, f"{pct[a, b]:.0f}%\n{int(cm[a, b] / len(SEEDS)):,}", ha="center", va="center",
                            fontsize=FONT["annot"] - 0.5, color="white" if pct[a, b] > 60 else "black")
            ax.set_xticks([0, 1], ["0", "1"]); ax.set_yticks([0, 1], ["0", "1"])
            if i == 0:
                ax.set_title(m, fontweight="bold", fontsize=FONT["label"])
            if j == 0:
                ax.set_ylabel(f"{s}\nReference")
            if i == len(sites) - 1:
                ax.set_xlabel("Predicted")
    fig.suptitle(f"Test confusion matrices (row %, mean count over seeds {', '.join(map(str, SEEDS))})",
                 fontweight="bold", fontsize=FONT["suptitle"])
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out, "Fig_confusion_matrices")


@fonts_for("comparison")
def fig_comparison(R, out):
    T = R[R.partition == "test"]
    sites = list(dict.fromkeys(T.site))
    mets = [("OA", "Accuracy"), ("kappa", "Kappa"), ("macro_F1", "Macro F1"), ("f1_0", "F1 class 0"),
            ("f1_1", "F1 class 1")]
    fig, axs = plt.subplots(1, len(mets), figsize=(2.4 * len(mets), 0.3 * len(MODELS) * len(sites) / 2 + 2.4),
                            sharey=True)
    for k, (c, lab) in enumerate(mets):
        ax = axs[k]
        for j, s in enumerate(sites):
            g = T[T.site == s].groupby("model")[c].agg(["mean", "std"]).reindex(MODELS)
            y = np.arange(len(MODELS)) + (j - (len(sites) - 1) / 2) * 0.7 / len(sites)
            ax.errorbar(g["mean"], y, xerr=g["std"], fmt=["o", "s", "^", "D", "v"][j % 5], ms=4, lw=0.8,
                        capsize=1.5, color=site_color(s, j), label=s)
        ax.set_title(lab, fontweight="bold"); ax.grid(axis="x", color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)
        for b in np.arange(len(MODELS) - 1) + 0.5:
            ax.axhline(b, color="#bdbdbd", lw=0.5)
    axs[0].set_yticks(range(len(MODELS)), MODELS)
    axs[0].set_ylim(len(MODELS) - 0.5, -0.5)               # shared y: first model on top (set once)
    h, l = axs[0].get_legend_handles_labels()
    fig.suptitle(f"Test metrics, mean ± SD over seeds {', '.join(map(str, SEEDS))}", fontweight="bold",
                 fontsize=FONT["suptitle"])
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.legend(h, l, loc="lower center", ncol=len(sites), frameon=False, bbox_to_anchor=(0.5, 0.0))
    save(fig, out, "Fig_model_comparison")


@fonts_for("groups")
def fig_feature_group_importance(I, bands_by_site, out):
    """RF impurity importance summed per feature group, and drop in test accuracy when a group is permuted."""
    sites = list(dict.fromkeys(I.site))
    gi = I[I.model == "RandomForest"].copy()
    rows = []
    for s in sites:
        groups = band_groups(bands_by_site[s])
        b2g = {b: g for g, bs in groups.items() for b in bs}
        d = gi[gi.site == s].assign(group=lambda t: t.feature.map(b2g))
        g = d.groupby(["seed", "group"]).importance.sum().groupby("group").agg(["mean", "std"])
        p = I[(I.site == s) & (I.model == "RF permutation accuracy")].groupby("feature").importance.agg(["mean", "std"])
        for grp in groups:
            rows.append(dict(site=s, group=grp, n_bands=len(groups[grp]),
                             impurity=g.loc[grp, "mean"] if grp in g.index else np.nan,
                             impurity_sd=g.loc[grp, "std"] if grp in g.index else np.nan,
                             perm_dAcc=p.loc[grp, "mean"] if grp in p.index else np.nan,
                             perm_sd=p.loc[grp, "std"] if grp in p.index else np.nan))
    G = pd.DataFrame(rows)
    G.to_csv(out / "feature_group_importance.csv", index=False, float_format="%.4f")
    order = G.groupby("group").impurity.mean().sort_values(ascending=False).index.tolist()
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 0.38 * len(order) + 1.8), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    for k, (c, sd, lab) in enumerate((("impurity", "impurity_sd", "Impurity importance (sum over the group)"),
                                      ("perm_dAcc", "perm_sd", "Drop in test accuracy when the group is permuted"))):
        ax = axs[k]
        for j, s in enumerate(sites):
            d = G[G.site == s].set_index("group").reindex(order)
            y = np.arange(len(order)) + (j - (len(sites) - 1) / 2) * 0.8 / len(sites)
            ax.barh(y, d[c], 0.8 / len(sites), xerr=d[sd], color=site_color(s, j), label=s,
                    error_kw=dict(lw=0.6, capsize=1.5))
        ax.set_yticks(range(len(order)), order); ax.invert_yaxis()
        ax.axvline(0, color="black", lw=0.6); ax.set_xlabel(lab)
        ax.grid(axis="x", color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)
    axs[0].legend(frameon=False, fontsize=FONT["annot"])
    axs[0].set_title("(a) Feature groups, RandomForest", fontweight="bold")
    axs[1].set_title("(b) Group permutation (test pixels)", fontweight="bold")
    # (c) top 15 single features, mean over sites and seeds
    top = gi.groupby(["site", "feature"]).importance.mean().groupby("feature").mean().sort_values(ascending=False)[:15]
    allb = sorted({b for v in bands_by_site.values() for b in v})
    b2g = {b: g for g, bs in band_groups(allb).items() for b in bs}
    ax = axs[2]
    ax.barh(range(len(top)), top.values, color=[COLORS["groups"].get(b2g.get(f), "#7f7f7f") for f in top.index])
    ax.set_yticks(range(len(top)), [nice(f) for f in top.index]); ax.invert_yaxis()
    ax.set_xlabel("Impurity importance (mean over sites and seeds)")
    ax.set_title("(c) Top 15 features", fontweight="bold")
    ax.grid(axis="x", color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)
    used = list(dict.fromkeys(b2g.get(f) for f in top.index))
    ax.legend(handles=[Patch(color=COLORS["groups"].get(g, "#7f7f7f"), label=g) for g in used], frameon=False,
              fontsize=FONT["annot"], loc="lower right")
    fig.tight_layout()
    save(fig, out, "Fig_feature_groups_importance")
    return G


# ---------------------------------------------------------------- basemap, north arrow, scale bar
def _tile_xy(lon, lat, z):
    n = 2 ** z
    return (lon + 180) / 360 * n, (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n


def _fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                                             "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
                                               "Referer": "https://www.arcgis.com/"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def padded_extent(S, pad=0.0):
    lon0, lon1, lat0, lat1 = extent(S)
    dx, dy = (lon1 - lon0) * pad, (lat1 - lat0) * pad
    return (lon0 - dx, lon1 + dx, lat0 - dy, lat1 + dy)


def tile_basemap(S, url, z, ext=None):
    """Web-map tiles at zoom z, mosaicked and resampled to a lon/lat grid (the site's extent unless ext is given)."""
    from PIL import Image
    lon0, lon1, lat0, lat1 = ext or extent(S)
    fx0, fy0 = _tile_xy(lon0, lat1, z)          # upper-left
    fx1, fy1 = _tile_xy(lon1, lat0, z)          # lower-right
    tx = range(int(fx0), int(fx1) + 1)
    ty = range(int(fy0), int(fy1) + 1)
    print(f"  basemap: {len(tx) * len(ty)} tiles at zoom {z}")
    M = np.full((256 * len(ty), 256 * len(tx), 3), 255, np.uint8)
    failed = 0
    for j, y in enumerate(ty):
        for i, x in enumerate(tx):
            for attempt in range(BASEMAP_RETRIES):              # slow down and retry when the server drops a request
                try:
                    t = Image.open(io.BytesIO(_fetch(url.format(z=z, x=x, y=y)))).convert("RGB")
                    M[256 * j:256 * (j + 1), 256 * i:256 * (i + 1)] = np.asarray(t)
                    break
                except Exception:
                    if i == j == 0 and attempt == BASEMAP_RETRIES - 1:
                        raise                                   # first tile never arrives: server unreachable
                    time.sleep(1.0 + 2.0 * attempt)
            else:
                failed += 1
            time.sleep(BASEMAP_TILE_DELAY)
    if failed:
        raise IOError(f"{failed} of {len(tx) * len(ty)} tiles failed")
    w = int(round((fx1 - fx0) * 256)); h = int(round((fy1 - fy0) * 256))
    lons = np.linspace(lon0, lon1, w, endpoint=False) + (lon1 - lon0) / w / 2
    lats = np.linspace(lat1, lat0, h, endpoint=False) - (lat1 - lat0) / h / 2
    px = ((_tile_xy(lons, lats[0], z)[0] - tx[0]) * 256).astype(int).clip(0, M.shape[1] - 1)
    py = ((_tile_xy(lons[0], lats, z)[1] - ty[0]) * 256).astype(int).clip(0, M.shape[0] - 1)
    return M[py][:, px]


def auto_zoom(ext, zmax=19):
    """Zoom level contextily would choose for this extent."""
    lon0, lon1, lat0, lat1 = ext
    z = min(np.ceil(np.log2(360 * 2.0 / max(lon1 - lon0, 1e-9))), np.ceil(np.log2(360 * 2.0 / max(lat1 - lat0, 1e-9))))
    return int(min(max(z, 1), zmax))


def ctx_basemap(url, z, ext, key=None):
    """Tiles through contextily (its provider object for key when listed in CTX_PROVIDERS, else the URL), resampled
    from Web Mercator to the lon/lat grid of ext."""
    import contextily as ctx
    lon0, lon1, lat0, lat1 = ext
    src = url
    if key in CTX_PROVIDERS:
        src = ctx.providers
        for part in CTX_PROVIDERS[key].split("."):
            src = src[part]
    img, (mx0, mx1, my0, my1) = ctx.bounds2img(lon0, lat0, lon1, lat1, zoom=z, source=src, ll=True)
    H, W = img.shape[:2]
    R_ = 6378137.0
    lons = np.linspace(lon0, lon1, W, endpoint=False) + (lon1 - lon0) / W / 2
    lats = np.linspace(lat1, lat0, H, endpoint=False) - (lat1 - lat0) / H / 2
    X = R_ * np.radians(lons); Y = R_ * np.log(np.tan(np.pi / 4 + np.radians(lats) / 2))
    px = ((X - mx0) / (mx1 - mx0) * W).astype(int).clip(0, W - 1)
    py = ((my1 - Y) / (my1 - my0) * H).astype(int).clip(0, H - 1)
    return np.asarray(img)[py][:, px][..., :3]


def file_basemap(S, ext):
    """Your own basemap GeoTIFF (BASEMAP_FILE, EPSG:4326, RGB); None when there is none."""
    if not BASEMAP_FILE:
        return None
    try:
        path = find(BASEMAP_FILE.format(stem=S["stem"]))
    except FileNotFoundError:
        return None
    A, (x0, y0, dx, dy), _ = read_raster(path)
    img = np.moveaxis(A[:3] if A.shape[0] >= 3 else np.repeat(A[:1], 3, 0), 0, -1)
    img = np.nan_to_num(img, nan=255.0)
    img = img / 255.0 if np.nanmax(img) > 1.5 else img
    fext = (x0, x0 + A.shape[2] * dx, y0 - A.shape[1] * dy, y0)
    print(f"  basemap from {path.name}")
    return np.clip(img, 0, 1), fext, "Basemap: " + path.stem


def get_basemap(S, key=None, pad=0.0, ext=None):
    """(image, extent, credit) for a BASEMAPS key, extent widened by pad (fraction of the site extent on each side):
    your own BASEMAP_FILE first, then cached tiles (OUT_DIR/cache), then the tile servers of the key and its
    BASEMAP_FALLBACKS (contextily when installed, else a direct download); Sentinel-2 true colour if none works."""
    key = key or BASEMAP
    if key not in BASEMAPS:
        raise ValueError(f"BASEMAP '{key}' unknown; use one of {', '.join(BASEMAPS)}")
    tag = (f"_ext{abs(hash(tuple(round(v, 5) for v in ext))) % 10**8}" if ext is not None
           else (f"_pad{pad:g}" if pad else ""))
    ext = ext if ext is not None else padded_extent(S, pad)
    if key == "none":
        return np.ones((2, 2, 3)), ext, BASEMAPS["none"][2]
    if key != "s2":
        own = file_basemap(S, ext)
        if own is not None:
            return own
    errors = []
    for k in [key] + BASEMAP_FALLBACKS.get(key, []):
        url, zmax, credit = BASEMAPS[k]
        if not url:
            continue
        z = auto_zoom(ext, zmax) if BASEMAP_ZOOM == "auto" else min(int(BASEMAP_ZOOM), zmax)
        cache = Path(OUT_DIR) / "cache" / f"basemap_{S['stem']}_{k}_z{z}{tag}.npy"
        if cache.is_file():
            return np.load(cache), ext, credit
        for how in ("contextily", "direct"):
            try:
                img = (ctx_basemap(url, "auto" if BASEMAP_ZOOM == "auto" else z, ext, k) if how == "contextily"
                       else tile_basemap(S, url, z, ext))
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, img)
                if k != key:
                    print(f"  basemap '{key}' not reachable; '{k}' used instead")
                return img, ext, credit
            except ImportError:
                pass
            except Exception as ex:
                errors.append(f"{k} ({how}): {type(ex).__name__}")
    if errors:
        print(f"  basemap '{key}' not available ({'; '.join(errors)}); Sentinel-2 true colour used. "
              "Run basemap_check() to test the tile servers from this computer.")
    rgb = true_colour(S)
    if rgb is None:
        rgb = np.ones(S["valid"].shape + (3,))
    return rgb, extent(S), BASEMAPS["s2"][2]


def basemap_check(lon=-78.28, lat=37.71, z=14):
    """Fetch one tile from every tile server and report which ones this computer can reach."""
    x, y = (int(v) for v in _tile_xy(lon, lat, z))
    for k, (url, zmax, _) in BASEMAPS.items():
        if not url:
            continue
        u = url.format(z=min(z, zmax), x=x, y=y)
        try:
            n = len(_fetch(u))
            print(f"  OK    {k:<13} {n:>7} bytes")
        except Exception as ex:
            print(f"  FAIL  {k:<13} {type(ex).__name__}: {str(ex)[:90]}")
    try:
        import contextily
        print(f"  contextily {contextily.__version__} installed")
    except ImportError:
        print("  contextily not installed (pip install contextily)")


def north_scale(ax, S, source=None, small=False, ext=None, credit=None, arrow_xy=(0.08, 0.95), units="km",
                color=None):
    """North arrow (upper left), scale bar (lower left) and the imagery credit."""
    credit = credit if credit is not None else S.get("basemap", (None, None, ""))[2]
    dark = credit in (BASEMAPS["satellite"][2], BASEMAPS["s2"][2])
    col = color or ("white" if dark else "black")             # readable on imagery and on light maps
    lon0, lon1, lat0, lat1 = ext or extent(S)
    fs = FONT["mapnote"]
    ax.annotate("N", xy=arrow_xy, xytext=(arrow_xy[0], arrow_xy[1] - 0.18), xycoords="axes fraction", ha="center",
                va="center",
                fontsize=fs, fontweight="bold", color=col,
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2), zorder=7)
    width_m = (lon1 - lon0) * 111320 * np.cos(np.radians((lat0 + lat1) / 2))
    if units == "mi":
        nice = [0.1, 0.25, 0.5, 1, 2, 5]
        Lmi = max([v for v in nice if v * 1609.344 <= width_m / 3.5] or [nice[0]])
        L, text = Lmi * 1609.344, f"{Lmi:g} mi"
    else:
        nice = [100, 200, 250, 500, 1000, 2000, 5000]
        L = max([v for v in nice if v <= width_m / 3.5] or [nice[0]])
        text = f"{L / 1000:g} km" if L >= 1000 else f"{L} m"
    dlon = L / (111320 * np.cos(np.radians((lat0 + lat1) / 2)))
    x0 = lon0 + 0.05 * (lon1 - lon0); y0 = lat0 + 0.06 * (lat1 - lat0)
    ax.plot([x0, x0 + dlon], [y0, y0], color=col, lw=2.2 if not small else 1.6, solid_capstyle="butt", zorder=7)
    ax.text(x0 + dlon / 2, y0 + 0.025 * (lat1 - lat0), text, ha="center",
            va="bottom", fontsize=fs, color=col, fontweight="bold", zorder=7)
    if source:
        ax.text(0.98, 0.015, source, transform=ax.transAxes, ha="right", va="bottom", fontsize=fs - 2,
                color=col, zorder=7)


# ---------------------------------------------------------------- site dictionaries
def light(S):
    """Site dictionary without the feature stack (for the multi-site figure)."""
    return {k: v for k, v in S.items() if k != "X"}


def bare(S):
    """Site dictionary without the feature stack and basemaps (what a checkpoint keeps)."""
    return {k: v for k, v in S.items() if k not in ("X", "basemap", "basemap_h", "basemap_wm")}


# ---------------------------------------------------------------- Fig. 1: all sites
def header(fig, ax, letter, im=None, ticks=None, ticklabels=None):
    """Panel letter and a horizontal colour key above a figure column."""
    ax.axis("off")
    ax.text(0.5, 1.0, letter, transform=ax.transAxes, ha="center", va="top", fontweight="bold",
            fontsize=FONT["panel"])
    if im is None:
        return
    cax = ax.inset_axes([0.08, 0.05, 0.84, 0.26])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.4)
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.tick_params(labelsize=FONT["key"], length=0, pad=2)
    if ticks is not None:
        cb.set_ticks(ticks); cb.set_ticklabels(ticklabels)


@fonts_for("fig1")
def fig_all_sites(SS, units_by, maps_by, out):
    cols = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    captions = ["Site basemap", "Hazard Index\n(normalised)", "Contamination\nrisk map",
                f"Reference label\n(class 0: lowest {ALPHA_PERCENTILE}% HI)", f"{MAP_NAME}\nclassification",
                f"{MAP_NAME}\nprobability", "Error map", "False alarms (FP) and\nmissed contamination (FN)"]
    n = len(SS)
    P = PANEL_IN["fig1"]
    hr = [aspect(S) for S in SS]                               # rows sized to each site's map: no blank bands
    fig = plt.figure(figsize=(P * 8 + 0.6, P * (sum(hr) + 0.42) + 1.2), constrained_layout=True)
    gs = fig.add_gridspec(n + 2, 8, height_ratios=[0.30] + hr + [0.12])
    H = [fig.add_subplot(gs[0, j]) for j in range(8)]
    A = np.array([[fig.add_subplot(gs[i + 1, j]) for j in range(8)] for i in range(n)])
    cm_hi = LinearSegmentedColormap.from_list("hi", COLORS["hi"])
    cm_p = LinearSegmentedColormap.from_list("p", COLORS["prob"])
    cm_lab = ListedColormap(COLORS["label"]); nb2 = BoundaryNorm([-0.5, 0.5, 1.5], 2)
    cm_err = ListedColormap([COLORS["error"][k] for k in ("TN", "FP", "FN", "TP")])
    cm_fe = ListedColormap([COLORS["error"]["FP"], COLORS["error"]["FN"]])
    ims = {}
    for i, S in enumerate(SS):
        units, maps = units_by[S["label"]], maps_by[S["label"]]
        v = S["valid"]
        img, ext, src = S["basemap"]
        ax = A[i, 0]
        ax.imshow(img, extent=ext, origin="upper", interpolation="bilinear")
        frame(ax, S); overlay(ax, S, units, mode="multi"); north_scale(ax, S, src if i == n - 1 else None, small=True)
        ims["b"] = show(A[i, 1], S, np.where(v, S["HIn"], np.nan), cm_hi, Normalize(0, 1), units, None, mode="multi")
        ims["c"] = show(A[i, 2], S, np.where(v, S["risk"], np.nan), ListedColormap(COLORS["risk"]),
                        BoundaryNorm(np.arange(-0.5, 5), 5), units, None, mode="multi")
        ims["d"] = show(A[i, 3], S, np.where(v, S["Y"], np.nan).astype(float), cm_lab, nb2, units, None, mode="multi")
        ims["e"] = show(A[i, 4], S, maps["pred"], cm_lab, nb2, units, None, mode="multi")
        ims["f"] = show(A[i, 5], S, maps["prob"], cm_p, Normalize(0, 1), units, None, mode="multi")
        e = np.where(np.isfinite(maps["pred"]), S["Y"] * 2 + np.nan_to_num(maps["pred"]), np.nan)
        ims["g"] = show(A[i, 6], S, e, cm_err, BoundaryNorm(np.arange(-0.5, 4), 4), units, None, mode="multi")
        ax = A[i, 7]
        img_h, ext_h, _ = S.get("basemap_h", S["basemap"])
        ax.imshow(img_h, extent=ext_h, origin="upper", interpolation="bilinear", alpha=FADE)
        fe = np.where(e == 1, 0.0, np.where(e == 2, 1.0, np.nan))
        ims["h"] = ax.imshow(fe, extent=extent(S), cmap=cm_fe, norm=nb2, interpolation="nearest", origin="upper")
        frame(ax, S); overlay(ax, S, units, mode="multi")
        A[i, 0].set_ylabel(S["label"], fontweight="bold", fontsize=FONT["row"])
    for j in range(8):
        A[-1, j].set_xlabel(captions[j], fontsize=FONT["caption"])
    header(fig, H[0], cols[0])
    header(fig, H[1], cols[1], ims["b"], [0, 1], ["0", "1"])
    header(fig, H[2], cols[2], ims["c"], range(5), ["Low", "", "", "", "Critical"])
    header(fig, H[3], cols[3], ims["d"], [0, 1], ["Class 0", "Class 1"])
    header(fig, H[4], cols[4], ims["e"], [0, 1], ["Lower HI", "Higher HI"])
    header(fig, H[5], cols[5], ims["f"], [0, 1], ["0", "1"])
    header(fig, H[6], cols[6], ims["g"], range(4), ["TN", "FP", "FN", "TP"])
    header(fig, H[7], cols[7], ims["h"], [0, 1], ["FP", "FN"])
    h = map_handles([u for S in SS for u in units_by[S["label"]]],
                    unknown=any(S["wells"]["gradient"].eq("Unknown").any() for S in SS))
    lg = fig.add_subplot(gs[-1, :]); lg.axis("off")                # legend row below the column captions
    lg.legend(handles=h, loc="center", ncol=len(h), frameon=False, fontsize=FONT["legend"])
    save(fig, out, f"Fig1_All_sites_{MAP_MODEL}")


# ---------------------------------------------------------------- Fig. 2-4: metal HQ
SYMBOL = dict(zinc="Zn", arsenic="As", cobalt="Co", manganese="Mn", molybdenum="Mo", lead="Pb", boron="B", strontium="Sr",
              lithium="Li", radium="Ra", thallium="Tl", beryllium="Be", cadmium="Cd", chromium="Cr", mercury="Hg",
              nickel="Ni", antimony="Sb", barium="Ba", selenium="Se", silver="Ag", copper="Cu", uranium="U")
SYM = {**SYMBOL, "hexavalent_chromium": "Cr(VI)"}


def hq_target(S, metal):
    """Kriged HQ of one metal, HQ_norm, label (class 1 = HQ_norm >= ALPHA_PERCENTILE) as a site dictionary."""
    col = f"HQ_{metal}"
    w = S["wells"].dropna(subset=[col])
    x0, y0, dx, dy = S["geo"]
    rows, cols = S["valid"].shape
    k = dict(SITES[S["name"]]["kriging"])
    var = float(np.var(w[col]))
    if var <= 0:
        raise ValueError(f"{metal}: HQ is constant at the wells")
    k.update(sill=var, nugget=HQ_NUGGET_FRAC * var)
    HQ = krige_gee(w, x0, y0, dx, dy, rows, cols, k, col=col)
    HQ = np.clip(HQ, w[col].min(), w[col].max())                 # no values beyond the well range
    valid = S["fvalid"] & np.isfinite(HQ)
    lo, hi = np.nanmin(HQ[valid]), np.nanmax(HQ[valid])
    HQn = (HQ - lo) / max(hi - lo, 1e-12)
    alpha = np.nanpercentile(HQn[valid], ALPHA_PERCENTILE)
    Y = np.where(valid, HQn >= alpha, 0).astype(int)
    sym = SYMBOL.get(metal, metal)
    print(f"  {sym}: well HQ {w[col].min():.2f}-{w[col].max():.2f}, {int((w[col] >= 1).sum())} wells >= 1; "
          f"class-1 share {Y[valid].mean():.3f}")
    T = dict(S, Y=Y, valid=valid, HQ=HQ, HQn=HQn, target=sym, metal=metal,
             well_hq=(float(w[col].min()), float(w[col].max()), int((w[col] >= 1).sum()), len(w)))
    T["stem"] = S["stem"]
    return T


@fonts_for("fig2")
def fig_hq(TT, units, maps_by, R, out, suffix=""):
    rows = ["Hazard\nquotient", "Normalised\nHQ", f"{MAP_NAME}\nclassification", f"{MAP_NAME}\nprobability",
            f"{MAP_NAME}\nerror map"]
    m = len(TT)
    P = PANEL_IN["fig2"]
    a = aspect(TT[0])
    fig = plt.figure(figsize=(P * m + 2.4, P * a * 5 + 1.6), constrained_layout=True)
    gs = fig.add_gridspec(6, m, height_ratios=[a] * 5 + [0.1])
    A = np.array([[fig.add_subplot(gs[i, j]) for j in range(m)] for i in range(5)])
    cm_hq, nb_hq = hq_cmap(HQ_STYLE["fig2"])
    cm_hi = LinearSegmentedColormap.from_list("hi", COLORS["hi"])
    cm_p = LinearSegmentedColormap.from_list("p", COLORS["prob"])
    cm_lab = ListedColormap(COLORS["label"]); nb2 = BoundaryNorm([-0.5, 0.5, 1.5], 2)
    cm_err = ListedColormap([COLORS["error"][k] for k in ("TN", "FP", "FN", "TP")])
    for j, T in enumerate(TT):
        mp = maps_by[T["target"]]
        v = T["valid"]
        show(A[0, j], T, np.where(v, T["HQ"], np.nan), cm_hq, nb_hq, units, None, mode="multi")
        im_n = show(A[1, j], T, np.where(v, T["HQn"], np.nan), cm_hi, Normalize(0, 1), units, None, mode="multi")
        im_c = show(A[2, j], T, mp["pred"], cm_lab, nb2, units, None, mode="multi")
        im_p = show(A[3, j], T, mp["prob"], cm_p, Normalize(0, 1), units, None, mode="multi")
        e = np.where(np.isfinite(mp["pred"]), T["Y"] * 2 + np.nan_to_num(mp["pred"]), np.nan)
        im_e = show(A[4, j], T, e, cm_err, BoundaryNorm(np.arange(-0.5, 4), 4), units, None, mode="multi")
        r = R[(R.site == T["label"]) & (R.target == T["target"]) & (R.partition == "test") & (R.model == MAP_MODEL)]
        lo, hi, nex, nw = T["well_hq"]
        A[0, j].set_title(T["target"], fontweight="bold", fontsize=FONT["panel"], pad=4)
    for i, lab in enumerate(rows):
        A[i, 0].set_ylabel(f"({'abcde'[i]})\n{lab}", fontweight="bold", fontsize=FONT["row"])
    names = [f"HQ < {HQ_CLASSES[0]:g}"] + [f"{a:g} ≤ HQ < {b:g}" for a, b in zip(HQ_CLASSES[:-1], HQ_CLASSES[1:])] \
        + [f"HQ ≥ {HQ_CLASSES[-1]:g}"]
    A[0, -1].legend(handles=[Patch(color=c, label=t) for c, t in zip(COLORS["hq"], names)], loc="center left",
                    bbox_to_anchor=(1.03, 0.5), frameon=False, fontsize=FONT["legend"])
    for row, im in ((1, im_n), (3, im_p)):                       # keys beside the last column, as the legends
        cax = A[row, -1].inset_axes([1.06, 0.12, 0.055, 0.76])
        cb = fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1])
        cb.ax.invert_yaxis(); cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=FONT["key"])
        cb.set_label("0 lowest, 1 highest HQ" if row == 1 else "P(class 1)", fontsize=FONT["annot"])
    A[2, -1].legend(handles=[Patch(color=COLORS["label"][1], label="Higher HQ (class 1)"),
                             Patch(color=COLORS["label"][0], label="Lower HQ (class 0)")], loc="center left",
                    bbox_to_anchor=(1.03, 0.5), frameon=False, fontsize=FONT["legend"])
    A[4, -1].legend(handles=[Patch(color=COLORS["error"][k], label=k) for k in ("TN", "FP", "FN", "TP")],
                    loc="center left", bbox_to_anchor=(1.03, 0.5), frameon=False, fontsize=FONT["legend"])
    lg = fig.add_subplot(gs[-1, :]); lg.axis("off")                # legend row below the test scores
    h = map_handles(units, unknown=TT[0]["wells"]["gradient"].eq("Unknown").any())
    lg.legend(handles=h, loc="center", ncol=len(h), frameon=False, fontsize=FONT["legend"])
    fit_rows(fig, A)
    save(fig, out, f"Fig2_{TT[0]['label'].replace(' ', '')}_metal_HQ_{MAP_MODEL}{suffix}")


@fonts_for("fig2")
def fig_hq_vertical(TT, units, maps_by, R, out):
    """Fig. 2, vertical: one row per metal (name on the left), one column per map (HQ, normalised
    HQ, classification, probability, error map), colour keys under the columns. Drawn at the printed size
    (PRINT['width_cm'], text PRINT['font_pt']); all metals of the site in one figure, or (FIG2_SPLIT) parts with the
    same number of rows that fit PRINT['max_h_cm']."""
    fs = PRINT["font_pt"]; pt = 1 / 72; g = PRINT["gap_in"]; M = 0.03
    W = PRINT["width_cm"] / 2.54
    heads = ["Hazard\nquotient (HQ)", "Normalised\nHQ", f"{MAP_NAME}\nclassification", f"{MAP_NAME}\nprobability",
             f"{MAP_NAME}\nerror map"]
    nc = len(heads)
    lw = PRINT["label_in"]
    cw = (W - 2 * M - lw - nc * g) / nc
    a = aspect(TT[0]); ph = cw * a
    top_h = fs * 1.45 * pt + 0.04                                  # site line
    head_h = 2 * fs * 1.25 * pt + 0.06                             # column headers (two lines)
    row_h = ph + g
    key_h = 0.08 + 0.11 + fs * 1.4 * pt + 0.02                     # colour bars / legends under the columns
    key_h = max(key_h, 0.06 + 2 * fs * 1.4 * pt)
    leg_h = 0.06 + fs * 1.5 * pt + 0.03
    fixed = 2 * M + top_h + head_h + key_h + leg_h
    cap = PRINT["max_h_cm"] / 2.54 if (PRINT["max_h_cm"] and FIG2_SPLIT) else np.inf
    per = max(1, int((cap - fixed) // row_h)) if np.isfinite(cap) else len(TT)
    n = len(TT)
    parts = [list(c) for c in np.array_split(np.arange(n), -(-n // per))] if n > per else [list(range(n))]
    cm_hq, nb_hq = hq_cmap(HQ_STYLE["fig2"])
    cm_hi = LinearSegmentedColormap.from_list("hi", COLORS["hi"])
    cm_p = LinearSegmentedColormap.from_list("p", COLORS["prob"])
    cm_lab = ListedColormap(COLORS["label"]); nb2 = BoundaryNorm([-0.5, 0.5, 1.5], 2)
    cm_err = ListedColormap([COLORS["error"][k] for k in ("TN", "FP", "FN", "TP")])
    for k, idx in enumerate(parts):
        H = fixed + len(idx) * row_h
        fig = plt.figure(figsize=(W, H))

        def box(x, y, w, h):                                       # inches from the top-left corner
            return [x / W, 1 - (y + h) / H, w / W, h / H]

        def xcol(j):
            return M + lw + g + j * (cw + g)
        fig.text(M / W, 1 - (M + top_h / 2) / H, TT[0]["label"] + ("" if k == 0 else " (cont.)"), ha="left",
                 va="center", fontsize=fs, fontweight="bold")
        for j, h in enumerate(heads):
            fig.text((xcol(j) + cw / 2) / W, 1 - (M + top_h + head_h - 0.03) / H, h, ha="center", va="bottom",
                     fontsize=fs, fontweight="bold", linespacing=1.05)
        y = M + top_h + head_h
        for i in idx:
            T = TT[i]; mp = maps_by[T["target"]]; v = T["valid"]
            e = np.where(np.isfinite(mp["pred"]), T["Y"] * 2 + np.nan_to_num(mp["pred"]), np.nan)
            layers = [(np.where(v, T["HQ"], np.nan), cm_hq, nb_hq), (np.where(v, T["HQn"], np.nan), cm_hi,
                      Normalize(0, 1)), (mp["pred"], cm_lab, nb2), (mp["prob"], cm_p, Normalize(0, 1)),
                      (e, cm_err, BoundaryNorm(np.arange(-0.5, 4), 4))]
            for j, (Z, cmap, norm) in enumerate(layers):
                ax = fig.add_axes(box(xcol(j), y, cw, ph))
                show(ax, T, Z, cmap, norm, units, None, mode="print")
            fig.text((M + lw / 2) / W, 1 - (y + ph / 2) / H, f"{T['target']}", ha="center", va="center",
                     fontsize=fs, fontweight="bold")
            y += row_h
        ky = y + 0.06                                              # keys under the columns
        bh = 0.11

        def bar(j, cmap, norm, ticks, labels, extend="neither"):
            cax = fig.add_axes(box(xcol(j) + 0.04, ky, cw - 0.08, bh))
            cb = fig.colorbar(mpl.cm.ScalarMappable(norm, cmap), cax=cax, orientation="horizontal", ticks=ticks,
                              extend=extend, extendfrac=0.08)
            cb.ax.set_xticklabels(labels); cb.outline.set_linewidth(0.4)
            cb.ax.tick_params(labelsize=fs, length=2, pad=1)
        if HQ_STYLE["fig2"] == "classes":                          # discrete HQ classes: equal-width boxes
            nb = BoundaryNorm(np.arange(len(COLORS["hq"]) + 1), len(COLORS["hq"]))
            bar(0, ListedColormap(COLORS["hq"]), nb, list(range(1, len(HQ_CLASSES) + 1)),
                [f"{c:g}" for c in HQ_CLASSES])
        else:
            bar(0, cm_hq, nb_hq, [0] + HQ_CLASSES, [f"{c:g}" for c in [0] + HQ_CLASSES[:-1]]
                + [f"≥ {HQ_CLASSES[-1]:g}"], "max")
        bar(1, cm_hi, Normalize(0, 1), [0, 0.5, 1], ["0", "0.5", "1"])
        bar(3, cm_p, Normalize(0, 1), [0, 0.5, 1], ["0", "0.5", "1"])
        for j, hs, ncol in ((2, [Patch(color=COLORS["label"][0], label="Class 0"),
                                 Patch(color=COLORS["label"][1], label="Class 1")], 1),
                            (4, [Patch(color=COLORS["error"][q], label=q) for q in ("TN", "FP", "FN", "TP")], 2)):
            fig.legend(handles=hs, loc="upper center", bbox_to_anchor=((xcol(j) + cw / 2) / W, 1 - (ky - 0.03) / H),
                       ncol=ncol, frameon=False, fontsize=fs, handlelength=1.0, handletextpad=0.35,
                       columnspacing=0.6, labelspacing=0.15, borderaxespad=0, borderpad=0)
        h = map_handles(units, unknown=TT[0]["wells"]["gradient"].eq("Unknown").any())
        fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1 - (ky + key_h - 0.02) / H), ncol=len(h),
                   frameon=False, fontsize=fs, handlelength=1.2, handletextpad=0.4, columnspacing=0.9,
                   borderaxespad=0, borderpad=0)
        name = f"Fig2_{TT[0]['label'].replace(' ', '')}_metal_HQ_{MAP_MODEL}" + (f"_part{k + 1}" if len(parts) > 1
                                                                                else "")
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{name}.png", dpi=PRINT["dpi"])        # no tight bbox: the printed size stays exact
        plt.close(fig)
        print(f"  saved {name}  ({PRINT['width_cm']:g} x {H * 2.54:.1f} cm)")


def table_hq(R, hq_all, out):
    """Table 3: test performance with each metal's HQ label, every site where the HQ models were trained."""
    T = R[R.partition == "test"]
    cols = ["OA", "prec_0", "rec_0", "f1_0", "prec_1", "rec_1", "f1_1", "kappa", "macro_F1", "AUC"]
    rows = []
    for label, TT in hq_all.items():
        for t in TT:
            lo, hi, nex, nw = t["well_hq"]
            for mdl, g in T[(T.site == label) & (T.target == t["target"])].groupby("model", sort=False):
                rows.append([label.replace("Site ", ""), t["target"], f"{lo:.2f}\u2013{hi:.2f}", f"{nex}/{nw}", mdl]
                            + [f"{g[c].mean():.3f} \u00b1 {g[c].std():.3f}" for c in cols])
    if not rows:
        return []
    TB = pd.DataFrame(rows, columns=["Site", "Metal", "Well HQ", "Wells HQ \u2265 1", "Model", "Accuracy", "Prec. 0",
                                     "Rec. 0", "F1 0", "Prec. 1", "Rec. 1", "F1 1", "\u03ba", "Macro F1", "ROC AUC"])
    TB.to_csv(out / "Table3_metal_HQ_models.csv", index=False, encoding="utf-8-sig")
    cap = (f"Table 3. Test performance when the label is each exceeded metal's kriged HQ (class 0 = lowest "
           f"{ALPHA_PERCENTILE} % of the normalised HQ); mean \u00b1 SD over seeds {', '.join(map(str, SEEDS))}, "
           "same spatial split design as Table 1. Wells HQ \u2265 1: wells exceeding / wells with the metal.")
    tex_table(TB, cap, "tab:hq_models", out / "Table3_metal_HQ_models.tex", colfmt="lllll" + "c" * 10,
              group_header="\\multicolumn{5}{c}{} & & \\multicolumn{3}{c}{Class 0} & \\multicolumn{3}{c}{Class 1} "
                           "& & & \\\\\n\\cmidrule(lr){7-9}\\cmidrule(lr){10-12}")
    return [(cap, TB)]


def hq_metals_of(S):
    want = HQ_METALS.get(S["name"], "exceeded") if isinstance(HQ_METALS, dict) else HQ_METALS
    w = S["wells"]
    cols = [c for c in w if c.startswith("HQ_") and c[3:] not in NON_METALS]
    if want == "exceeded":
        out = [c[3:] for c in cols if w[c].max() >= 1 and w[c].nunique() >= 3]
        return sorted(out, key=lambda m: -w[f"HQ_{m}"].ge(1).sum())
    return [m for m in want if f"HQ_{m}" in w]


def hq_cmap(style):
    """HQ colours: discrete classes, or continuous with the class colours at the class edges."""
    edges = [0] + HQ_CLASSES
    if style == "classes":
        return ListedColormap(COLORS["hq"]), BoundaryNorm([-1e9] + HQ_CLASSES + [1e9], len(HQ_CLASSES) + 1)
    top = HQ_CLASSES[-1]
    pos = [e / top for e in edges]
    cm = LinearSegmentedColormap.from_list("hq", list(zip(pos, COLORS["hq"][:len(pos)])))
    cm.set_over(COLORS["hq"][-1])
    return cm, Normalize(0, top, clip=True)


def hq_class_names():
    tags = ["Low", "Moderate", "Exceeds threshold", "High", "Very high"]
    names = [f"HQ < {HQ_CLASSES[0]:g}"] + [f"{a:g} \u2264 HQ < {b:g}" for a, b in zip(HQ_CLASSES[:-1], HQ_CLASSES[1:])] \
        + [f"HQ \u2265 {HQ_CLASSES[-1]:g}"]
    return [f"{n} ({t})" for n, t in zip(names, tags)] if len(names) == len(tags) else names


@fonts_for("fig3")
def fig_hq_all_sites(hq_all, SS, units_by, out):
    """Fig. 3, per site: basemap | HI normalised | kriged HQ of the metals exceeded at the site (HQ >= 1 at one or
    more wells), each named on top. Each row is packed left (no empty cells); metals follow one order at every
    site (HQ_FIG3_ORDER), so metals exceeded at all sites line up in the first columns."""
    rows = []
    for S in SS:
        rows.append((S, {t["target"]: t for t in hq_all.get(S["label"], [])
                         if t["well_hq"][2] > 0 and t["target"] not in FIG3_EXCLUDE}))
    score = {}
    for _, TT in rows:
        for sym, t in TT.items():
            a, b = score.get(sym, (0, 0))
            score[sym] = (a + 1, b + t["well_hq"][2])
    if HQ_FIG3_ORDER == "alphabetical":
        metals = sorted(score)
    else:                                                      # metals exceeded at more sites first, then A-Z
        metals = sorted(score, key=lambda m: (-score[m][0], m))
    rows = [(S, TT, [m for m in metals if m in TT]) for S, TT in rows]   # each row packed left, no gaps
    k = max(len(ms) for _, _, ms in rows)
    if k == 0:
        return
    P = PANEL_IN["fig3"]
    hr = [FIG3_HI_KEY["header_h"]] + [aspect(S) + 0.16 for S, _, _ in rows]   # header row, then the site rows
    wr = [1, 1, 0.12] + [1] * k + ([1.35] if FIG3_HQ_KEY else [])   # basemap, HI, gap, HQ panels, (HQ key)
    hr_all = hr + ([0.12] if FIG3_MAP_KEY else [])
    fig = plt.figure(figsize=(P * sum(wr) + 0.4, P * sum(hr_all) + 0.7), constrained_layout=True)
    gs = fig.add_gridspec(len(hr_all), len(wr), height_ratios=hr_all, width_ratios=wr)
    cm_hi = LinearSegmentedColormap.from_list("hi", COLORS["hi"])
    cm_hq, nm_hq = hq_cmap(HQ_STYLE["fig3"])
    hi_axes, hq_axes = [], []
    for i, (S, TT, ms) in enumerate(rows):
        units = units_by[S["label"]]
        ax = fig.add_subplot(gs[i + 1, 0])
        img, ext, src = S["basemap"]
        ax.imshow(img, extent=ext, origin="upper", interpolation="bilinear")
        frame(ax, S); overlay(ax, S, units, mode="multi"); north_scale(ax, S, None, small=True)
        ax.set_ylabel(S["label"], fontweight="bold", fontsize=FONT["row"])
        ax = fig.add_subplot(gs[i + 1, 1]); hi_axes.append(ax)
        im_hi = show(ax, S, np.where(S["valid"], S["HIn"], np.nan), cm_hi, Normalize(0, 1), units, None, mode="multi")
        for j, sym in enumerate(ms):
            ax = fig.add_subplot(gs[i + 1, 3 + j])
            ax.set_title(sym, fontweight="bold", fontsize=FONT["panel"] - 1, pad=3)
            hq_axes.append(ax)
            t = TT[sym]
            show(ax, S, np.where(t["valid"], t["HQ"], np.nan), cm_hq, nm_hq, units, None, mode="multi")
            lo, hi, nex, nw = t["well_hq"]
            ax.text(0.03, 0.03, f"max {hi:.1f}; {nex}/{nw} \u2265 1", transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=FONT["annot"] - 2, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.0))
    hk = FIG3_HI_KEY                                           # HI colour bar: its own cell of the layout, the
    c0 = 2 - max(1, min(2, hk["span"]))                         # title above it, so the two cannot overlap
    side = (1 - hk["width"]) / 2
    sub = gs[0, c0:2].subgridspec(1, 3, width_ratios=[side, hk["width"], side], wspace=0)
    cax = fig.add_subplot(sub[0, 1])
    cb = fig.colorbar(im_hi, cax=cax, orientation="horizontal", ticks=[0, 1])
    cax.set_box_aspect(hk["bar_h"] * FONT["label"] / 72 / (P * hk["span"] * hk["width"]))
    cax.set_title(hk["title"], fontweight="bold", fontsize=FONT["label"], color=COLORS["box_hi"], pad=3)
    cb.ax.tick_params(labelsize=FONT["key"], length=0, pad=1.5); cb.outline.set_linewidth(0.4)
    hd = fig.add_subplot(gs[0, 3:(-1 if FIG3_HQ_KEY else None)]); hd.axis("off")
    hd.text(0.0, 0.5, "Hazard quotient (HQ) of the metals exceeded at each site", transform=hd.transAxes,
            ha="left", va="center", fontweight="bold", fontsize=FONT["label"] + 1, color=COLORS["box_hq"])
    if FIG3_HQ_KEY:
        kx = fig.add_subplot(gs[1:len(rows) + 1, -1]); kx.axis("off")   # HQ key in its own column
        leg = kx.legend(handles=[Patch(color=c, label=t) for c, t in zip(COLORS["hq"], hq_class_names())],
                        loc="center left", frameon=False, fontsize=FONT["legend"], title="Hazard quotient (HQ)",
                        title_fontsize=FONT["legend"])
        leg.get_title().set_fontweight("bold")
    if FIG3_MAP_KEY:
        lg = fig.add_subplot(gs[-1, :]); lg.axis("off")        # map legend row
        h = map_handles([u for S, _, _ in rows for u in units_by[S["label"]]],
                        unknown=any(S["wells"]["gradient"].eq("Unknown").any() for S, _, _ in rows))
        lg.legend(handles=h, loc="center", ncol=len(h), frameon=False, fontsize=FONT["legend"])
    if GROUP_BOXES:                                            # dashed boxes, drawn once the layout is fixed
        fig.canvas.draw()
        if hasattr(fig, "set_layout_engine"):
            fig.set_layout_engine("none")
        else:
            fig.set_constrained_layout(False)
        for axes, col in ((hi_axes, COLORS["box_hi"]), (hq_axes, COLORS["box_hq"])):
            bb = [a.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()) for a in axes]
            x0 = min(b.x0 for b in bb) - 0.004; x1 = max(b.x1 for b in bb) + 0.004
            y0 = min(b.y0 for b in bb) - 0.006; y1 = max(b.y1 for b in bb) + 0.006
            if axes is hi_axes and FIG3_HI_KEY["span"] == 1:   # colour bar above the HI column only: include it
                y1 = max(y1, cax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).y1
                         + 0.04)
            fig.add_artist(mpl.patches.Rectangle((x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure, fill=False,
                                                 ec=col, lw=1.4, ls="--", zorder=10))
    save(fig, out, "Fig3_All_sites_exceeded_metal_HQ")


RA = {"radium": "Ra-226+228", "hexavalent_chromium": "Cr(VI)"}


@fonts_for("fig4")
def fig_hq_summary(SS, out):
    """Fig. 4: (a) maximum well HQ per metal and site, (b) wells with HQ >= 1 / wells sampled; also as Table 4."""
    metals = sorted({c[3:] for S in SS for c in S["wells"] if c.startswith("HQ_") and c[3:] not in NON_METALS})
    M = pd.DataFrame(index=metals, columns=[S["label"] for S in SS], dtype=float)
    N = M.copy(); E = M.copy()
    for S in SS:
        w = S["wells"]
        for m in metals:
            c = f"HQ_{m}"
            if c in w and w[c].notna().any():
                M.loc[m, S["label"]] = w[c].max(); E.loc[m, S["label"]] = (w[c] >= 1).sum()
                N.loc[m, S["label"]] = w[c].notna().sum()
    order = M.max(1).sort_values(ascending=False).index
    M, N, E = M.loc[order], N.loc[order], E.loc[order]
    names = [RA.get(m, SYMBOL.get(m, m)) for m in order]
    T = pd.DataFrame({"Metal": names})
    for c in M.columns:
        T[f"{c} max HQ"] = [("n.m." if not np.isfinite(v) else f"{v:.2f}") for v in M[c]]
        T[f"{c} wells \u2265 1"] = [("n.m." if not np.isfinite(n) else f"{int(e)}/{int(n)}") for e, n in zip(E[c], N[c])]
    T.to_csv(out / "Table4_site_HQ_exceedance.csv", index=False, encoding="utf-8-sig")
    vmax = max(1.01, np.nanmax(M.values))
    fig, A = plt.subplots(1, 2, figsize=(4.2 + 0.85 * 2 * len(M.columns), 0.36 * len(M) + 1.4), constrained_layout=True)
    cm1 = LinearSegmentedColormap.from_list("hq", RAMP); nm1 = mpl.colors.LogNorm(1, vmax)
    bins = [0, 1e-9, 10, 25, 50, 75, 100.01]
    cm2 = ListedColormap(["#1A7A1A", "#3EAA2E", "#96D626", "#E8E824", "#F57B12", "#E8220A"])
    nm2 = BoundaryNorm(bins, cm2.N)
    for ax, Z, cm, nm, fmt, title in (
            (A[0], M.values, cm1, nm1, "hq", "(a) Maximum well HQ"),
            (A[1], np.where(N.values > 0, 100 * E.values / np.where(N.values > 0, N.values, 1), np.nan), cm2, nm2,
             "share", "(b) Wells with HQ \u2265 1")):
        R_, C_ = Z.shape
        for r in range(R_):
            for c in range(C_):
                v = Z[r, c]
                if not np.isfinite(v):
                    col, txt, tc = "#FFFFFF", "n.m.", "#7F7F7F"
                elif fmt == "hq":
                    col = cm(nm(v)) if v >= 1 else COLORS["na"]
                    txt = f"{v:.1f}" if v >= 0.05 else "<0.1"; tc = "black"
                else:
                    col = cm(nm(v)); txt = f"{int(E.values[r, c])}/{int(N.values[r, c])}"
                    tc = "white" if v >= 50 or v == 0 else "black"
                ax.add_patch(mpl.patches.Rectangle((c, r), 1, 1, fc=col, ec="white", lw=1.2))
                ax.text(c + 0.5, r + 0.5, txt, ha="center", va="center", fontsize=FONT["annot"], color=tc)
        ax.set_xlim(0, C_); ax.set_ylim(R_, 0)
        ax.set_xticks(np.arange(C_) + 0.5, M.columns); ax.xaxis.tick_top()
        ax.set_yticks(np.arange(R_) + 0.5, names if ax is A[0] else [""] * R_)
        ax.tick_params(length=0, labelsize=FONT["tick"] + 1)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontweight="bold", loc="left", fontsize=FONT["title"])
    cb = fig.colorbar(mpl.cm.ScalarMappable(nm1, cm1), ax=A[0], fraction=0.06, pad=0.02)
    t = [v for v in (1, 2, 5, 10, 20, 50, 100) if v <= vmax] + [round(vmax, 1)]
    cb.set_ticks(t); cb.set_ticklabels([f"{v:g}" for v in t]); cb.ax.minorticks_off()
    cb.set_label("Maximum well HQ (HQ \u2265 1; grey < 1)", fontsize=FONT["annot"])
    cb = fig.colorbar(mpl.cm.ScalarMappable(nm2, cm2), ax=A[1], fraction=0.06, pad=0.02,
                      ticks=[0.5e-9, 5, 17.5, 37.5, 62.5, 88])
    cb.set_ticklabels(["0", "<10", "10\u201325", "25\u201350", "50\u201375", "\u226575"])
    cb.set_label("Share of wells exceeding (%)", fontsize=FONT["annot"])
    save(fig, out, "Fig4_Site_HQ_and_exceeding_wells")
    return T


# ---------------------------------------------------------------- HI against vegetation (well buffers)
def bh(p):
    p = np.asarray(p, float); out = np.full_like(p, np.nan); ok = np.isfinite(p)
    q, m = p[ok], ok.sum()
    if m == 0:
        return out
    o = np.argsort(q)
    adj = np.minimum.accumulate((q[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    r = np.empty(m); r[o] = np.minimum(adj, 1); out[ok] = r
    return out


def partial_spearman(a, b, z):
    ra, rb, rz = rankdata(a), rankdata(b), rankdata(z)
    res = lambda y: y - np.polyval(np.polyfit(rz, y, 1), rz)
    r_ = np.corrcoef(res(ra), res(rb))[0, 1]
    n = len(ra)
    t_ = r_ * np.sqrt((n - 3) / max(1 - r_ ** 2, 1e-12))
    return r_, 2 * t_dist.sf(abs(t_), n - 3)


def veg_out():
    return Path(OUT_DIR) / "hi_vegetation"


def dist_to_units(w, units):
    """Distance (m) from each well to the nearest CCR unit edge; 0 inside a unit; NaN without units."""
    from matplotlib.path import Path as MPath
    if not units:
        return np.full(len(w), np.nan)
    lat0 = float(w["latitude"].mean()); kx, ky = 111320 * np.cos(np.radians(lat0)), 110540
    P = np.column_stack(((w["longitude"] - w["longitude"].mean()) * kx, (w["latitude"] - lat0) * ky))
    d = np.full(len(w), np.inf)
    for _, _, rings in units:
        for r in rings:
            V = np.column_stack(((r[:, 0] - w["longitude"].mean()) * kx, (r[:, 1] - lat0) * ky))
            inside = MPath(V).contains_points(P)
            a, b = V[:-1], V[1:]
            ab = b - a; L2 = np.maximum((ab ** 2).sum(1), 1e-12)
            t = np.clip(((P[:, None, :] - a[None]) * ab[None]).sum(2) / L2[None], 0, 1)
            q = a[None] + t[..., None] * ab[None]
            dd = np.sqrt(((P[:, None, :] - q) ** 2).sum(2)).min(1)
            d = np.minimum(d, np.where(inside, 0.0, dd))
    return d


def colocated_offsets(w):
    """Display positions of the wells: wells closer than WELLMAP_COLOC_M are spread on a circle of
    WELLMAP_OFFSET_M around their centroid. Returns display lon, lat and a flag 'moved'."""
    lat0 = float(w["latitude"].mean()); kx, ky = 111320 * np.cos(np.radians(lat0)), 110540
    x = w["longitude"].to_numpy() * kx; y = w["latitude"].to_numpy() * ky
    n = len(w); grp = np.arange(n)
    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(x[i] - x[j], y[i] - y[j]) < WELLMAP_COLOC_M:
                grp[grp == grp[j]] = grp[i]
    lon, lat, moved = w["longitude"].to_numpy().copy(), w["latitude"].to_numpy().copy(), np.zeros(n, bool)
    for g in np.unique(grp):
        idx = np.flatnonzero(grp == g)
        if len(idx) < 2:
            continue
        cx, cy = x[idx].mean(), y[idx].mean()
        for k, i in enumerate(idx):
            a = np.pi / 2 + 2 * np.pi * k / len(idx)
            lon[i] = (cx + WELLMAP_OFFSET_M * np.cos(a)) / kx
            lat[i] = (cy + WELLMAP_OFFSET_M * np.sin(a)) / ky
            moved[i] = True
    return lon, lat, moved


@fonts_for("fig6")
def fig_index_rows(FS, units_by, out):
    """Fig. 6a-c: one row per site (basemap, then each vegetation index), for the index level, CSTV and rate of
    change. CSTV has one shared key (CSTV_RANGE); level and rate have one key per index under the last row, with
    the same colour range at every site."""
    fams = [("level", "{}", "Biophysical parameter and vegetation indices (summer median)"),
            ("cstv", "CSTV_{}_Diff", "CSTV of biophysical parameter and vegetation indices"),
            ("rate", "{}_Rate", "Rate of change (Sen's slope) of biophysical parameter and vegetation indices")]
    P = INDEX_ROW_PANEL_IN
    n = len(FS)
    for code, pat, title in fams:
        cols = [ix for ix in INDEX_ROW_ORDER if any(pat.format(ix) in S["bands"] for S in FS)]
        if not cols:
            continue
        m = len(cols)
        hr = [0.16] + [aspect(S) for S in FS] + [0.13, 0.14]       # key/title row, sites, keys, group labels
        fig = plt.figure(figsize=(P * (m + 1) + 0.5, P * sum(hr) + 0.3), constrained_layout=True)
        gs = fig.add_gridspec(n + 3, m + 1, height_ratios=hr)
        norms, cmaps = {}, {}
        for ix in cols:                                       # one colour range per index, pooled over the sites
            b = pat.format(ix)
            vals = np.concatenate([S["X"][..., S["bands"].index(b)][S["valid"]] for S in FS if b in S["bands"]])
            vals = vals[np.isfinite(vals)]
            if code == "cstv":
                norms[ix] = Normalize(*CSTV_RANGE)
                cmaps[ix] = (ListedColormap(PAL["cstv"]) if CSTV_STYLE == "classes"
                             else LinearSegmentedColormap.from_list("cstv", PAL["cstv"]))
            elif code == "rate":
                mx = np.nanpercentile(np.abs(vals), STRETCH[1]) or 1
                norms[ix] = Normalize(-mx, mx)
                cmaps[ix] = LinearSegmentedColormap.from_list("rate", PAL["rate"])
            else:
                lo, hi = VI_RANGE.get(ix, tuple(np.nanpercentile(vals, STRETCH)))
                norms[ix] = Normalize(lo, hi if hi > lo else lo + 1e-6)
                cmaps[ix] = LinearSegmentedColormap.from_list("veg", PAL["vegetation"])
        ims = {}
        for i, S in enumerate(FS):
            units = units_by[S["label"]] if INDEX_ROW_WELLS else []
            ax = fig.add_subplot(gs[i + 1, 0])
            img, ext, _ = S["basemap"]
            ax.imshow(img, extent=ext, origin="upper", interpolation="bilinear")
            frame(ax, S); overlay(ax, S, units_by[S["label"]], mode="multi")
            north_scale(ax, S, None, small=True)
            ax.set_ylabel(S["label"], fontweight="bold", fontsize=FONT["row"])
            for j, ix in enumerate(cols):
                ax = fig.add_subplot(gs[i + 1, j + 1])
                b = pat.format(ix)
                if b not in S["bands"]:
                    frame(ax, S); ax.axis("off")
                    continue
                v = S["X"][..., S["bands"].index(b)].astype(float)
                ims[ix] = show(ax, S, np.where(S["valid"], v, np.nan), cmaps[ix], norms[ix], units, None,
                               wells=INDEX_ROW_WELLS, mode="multi")
                if i == 0:
                    ax.set_title(ix, fontweight="bold", fontsize=FONT["title"], pad=3)
        top = fig.add_subplot(gs[0, 1:]); top.axis("off")
        top.text(0.0, 0.5, title, transform=top.transAxes, ha="left", va="center", fontweight="bold",
                 fontsize=FONT["suptitle"])
        if code == "cstv":                                    # one key, top right (as the CSTV palette)
            cax = top.inset_axes([0.68, 0.25, 0.32, 0.4])
            cb = fig.colorbar(ims[cols[0]], cax=cax, orientation="horizontal", ticks=[0, 1, 2])
            cb.set_ticklabels(["min 0", "1", "max 2"]); cb.ax.tick_params(labelsize=FONT["key"], length=0)
            cb.outline.set_linewidth(0.4)
        else:                                                 # one key per index under the last row
            for j, ix in enumerate(cols):
                kx = fig.add_subplot(gs[n + 1, j + 1]); kx.axis("off")
                cax = kx.inset_axes([0.06, 0.45, 0.88, 0.35])
                cb = fig.colorbar(mpl.cm.ScalarMappable(norms[ix], cmaps[ix]), cax=cax, orientation="horizontal")
                cb.ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
                cb.ax.tick_params(labelsize=FONT["key"] - 2, length=2, pad=1); cb.outline.set_linewidth(0.4)
        gl = fig.add_subplot(gs[n + 2, 1:]); gl.axis("off")   # bracket and name of each index group
        for name, members in INDEX_ROW_GROUPS.items():
            idx = [cols.index(x) for x in members if x in cols]
            if not idx:
                continue
            x0, x1 = (min(idx) + 0.05) / m, (max(idx) + 0.95) / m
            gl.annotate("", xy=(x0, 0.85), xytext=(x1, 0.85), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="<->", color=COLORS["box_hq"], lw=1.2))
            gl.text((x0 + x1) / 2, 0.35, name, transform=gl.transAxes, ha="center", va="center",
                    color=COLORS["box_hq"], fontsize=FONT["label"] + 1)
        save(fig, out, f"Fig6{'abc'[[f[0] for f in fams].index(code)]}_Index_rows_{code}")


def declutter(ax, texts, pts, iters=None, pad_pt=1.0, marker_pt=4.5, max_pt=None, leader_pt=6.0):
    """Move overlapping well names apart in screen space: labels repel each other and the well markers, a weak
    spring holds each near its well, offsets stay within max_pt points and labels stay inside the map. A thin
    leader line joins a label that moved more than leader_pt points."""
    iters = iters or WELLMAP_LABEL_ITERS
    max_pt = max_pt or WELLMAP_LABEL_MAX_PT
    fig = ax.figure
    r = fig.canvas.get_renderer()
    pt2px = fig.dpi / 72.0
    P = ax.transData.transform(pts)                           # wells, pixels
    o0 = np.array([4.0, 3.0]) * pt2px                         # default offset of the label's lower-left corner
    WH = np.array([mpl.text.Text.get_window_extent(t, r).size for t in texts]) + 2 * pad_pt * pt2px
    O = np.tile(o0, (len(texts), 1))
    bb = ax.get_window_extent(r)
    mk, rmax = marker_pt * pt2px, max_pt * pt2px
    n = len(texts)
    for _ in range(iters):
        L = P + O                                             # lower-left corners
        D = 0.04 * (o0 - O)                                   # spring back toward the default position
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                ox = min(L[i, 0] + WH[i, 0], L[j, 0] + WH[j, 0]) - max(L[i, 0], L[j, 0])
                oy = min(L[i, 1] + WH[i, 1], L[j, 1] + WH[j, 1]) - max(L[i, 1], L[j, 1])
                if ox > 0 and oy > 0:
                    v = (L[i] + WH[i] / 2) - (L[j] + WH[j] / 2)
                    if ox < oy:
                        d = np.array([(1 if v[0] >= 0 else -1) * (ox / 2 + 0.3), 0.0])
                    else:
                        d = np.array([0.0, (1 if v[1] >= 0 else -1) * (oy / 2 + 0.3)])
                    D[i] += d; D[j] -= d; moved = True
            for k in range(len(P)):
                ox = min(L[i, 0] + WH[i, 0], P[k, 0] + mk) - max(L[i, 0], P[k, 0] - mk)
                oy = min(L[i, 1] + WH[i, 1], P[k, 1] + mk) - max(L[i, 1], P[k, 1] - mk)
                if ox > 0 and oy > 0:
                    v = (L[i] + WH[i] / 2) - P[k]
                    if ox < oy:
                        D[i, 0] += (1 if v[0] >= 0 else -1) * (ox + 0.3)
                    else:
                        D[i, 1] += (1 if v[1] >= 0 else -1) * (oy + 0.3)
                    moved = True
        O = O + np.clip(D, -3 * pt2px, 3 * pt2px)
        nrm = np.hypot(*O.T)                                  # stay near the well
        far = nrm > rmax
        O[far] *= (rmax / nrm[far])[:, None]
        L = P + O                                             # stay inside the map
        O[:, 0] += np.clip(bb.x0 - L[:, 0], 0, None) - np.clip(L[:, 0] + WH[:, 0] - bb.x1, 0, None)
        O[:, 1] += np.clip(bb.y0 - L[:, 1], 0, None) - np.clip(L[:, 1] + WH[:, 1] - bb.y1, 0, None)
        if not moved:
            break
    for t, o in zip(texts, O):
        t.xyann = tuple(o / pt2px)
        if t.arrow_patch is not None:
            t.arrow_patch.set_visible(np.hypot(*(o - o0)) / pt2px > leader_pt)


def wellmap_views(SS):
    """Map extents for Fig. 5: the site ROI plus WELLMAP_PAD, widened in one direction so that every panel has the
    same shape (WELLMAP_ASPECT) and the panels fill equal boxes."""
    E = [padded_extent(S, WELLMAP_PAD) for S in SS]

    def asp(e):
        return (e[3] - e[2]) / ((e[1] - e[0]) * np.cos(np.radians((e[2] + e[3]) / 2)))
    if WELLMAP_ASPECT is None or not E:
        return E
    A = (max(map(asp, E)) if WELLMAP_ASPECT == "max" else min(map(asp, E)) if WELLMAP_ASPECT == "min"
         else float(WELLMAP_ASPECT))
    out = []
    for e in E:
        c = np.cos(np.radians((e[2] + e[3]) / 2))
        if asp(e) < A:                                   # wider than the target: more map above and below
            h, m = A * (e[1] - e[0]) * c, (e[2] + e[3]) / 2
            e = (e[0], e[1], m - h / 2, m + h / 2)
        else:                                            # taller: more map left and right
            w, m = (e[3] - e[2]) / (A * c), (e[0] + e[1]) / 2
            e = (m - w / 2, m + w / 2, e[2], e[3])
        out.append(e)
    return out


@fonts_for("fig5")
def fig_hi_wells(SS, units_by, out, interp=False):
    """Fig. 5: wells coloured by HI (triangle up upgradient, down downgradient) on a basemap, CCR units, site ROI,
    co-located wells offset; with interp=True the kriged HI is drawn under the wells."""
    n = len(SS); nc = min(WELLMAP_COLS, n); nr = -(-n // nc)
    P = WELLMAP_PANEL_IN
    label_sets = []
    asp = [((e[3] - e[2]) / ((e[1] - e[0]) * np.cos(np.radians((e[2] + e[3]) / 2)))) for e in (S["view_wm"] for S in SS)]
    hr = [max(asp[r * nc:(r + 1) * nc]) for r in range(nr)]
    fig = plt.figure(figsize=(P * nc + 1.2 * nc, P * sum(hr) + 0.9), constrained_layout=True)
    gs = fig.add_gridspec(nr + 1, nc, height_ratios=hr + [0.09])
    cm = LinearSegmentedColormap.from_list("hiw", COLORS["wellmap_hi"])
    for k, S in enumerate(SS):
        ax = fig.add_subplot(gs[k // nc, k % nc])
        img, iext, credit = S["basemap_wm"]
        ext = S["view_wm"]
        ax.imshow(img, extent=iext, origin="upper", interpolation="bilinear", zorder=0)
        w = S["wells"]
        hi = w["HI"].to_numpy(float)
        vmin, vmax = max(np.nanmin(hi), 1e-3), np.nanmax(hi)
        if interp:
            vmin, vmax = min(vmin, np.nanmin(S["HI"])), max(vmax, np.nanmax(S["HI"]))
        nm = mpl.colors.LogNorm(vmin, vmax) if WELLMAP_HI_LOG else Normalize(vmin, vmax)
        if interp:
            ax.imshow(np.where(np.isfinite(S["HI"]), S["HI"], np.nan), extent=extent(S), origin="upper", cmap=cm,
                      norm=nm, alpha=WELLMAP_INTERP_ALPHA, interpolation="bilinear", zorder=1)
        e = extent(S)
        if WELLMAP_ROI_BOX:
            ax.add_patch(mpl.patches.Rectangle((e[0], e[2]), e[1] - e[0], e[3] - e[2], fill=False,
                                               ec=COLORS["wellmap_roi"], ls="--", lw=1.0, zorder=3))
        for _, kind, rings in units_by[S["label"]]:
            for r in rings:
                ax.add_patch(Polygon(r, closed=True, fill=False, ec=COLORS["wellmap_ccr"], lw=0.9,
                                     hatch="////" if kind == "LF" else None, zorder=4))
        lon, lat, moved = colocated_offsets(w)
        for i in np.flatnonzero(moved):
            ax.plot([lon[i], w["longitude"].iloc[i]], [lat[i], w["latitude"].iloc[i]], color="black", lw=0.5,
                    zorder=5)
            ax.plot(w["longitude"].iloc[i], w["latitude"].iloc[i], "o", ms=2.2, color="black", zorder=6)
        sc = None
        for g, mk in (("Upgradient", "^"), ("Downgradient", "v"), ("Unknown", "o")):
            sel = w["gradient"].eq(g).to_numpy()
            if sel.any():
                sc = ax.scatter(lon[sel], lat[sel], c=hi[sel], cmap=cm, norm=nm, marker=mk, s=WELL_SIZE["wellmap"],
                                edgecolors=COLORS["wellmap_edge"], linewidths=0.6, zorder=7)
        if WELLMAP_LABELS and "well_name" in w:
            texts = []
            for i, name in enumerate(w["well_name"].astype(str)):
                t = ax.annotate(name, (lon[i], lat[i]), xytext=(4, 3), textcoords="offset points",
                                fontsize=FONT["wellname"], zorder=8, annotation_clip=False,
                                arrowprops=dict(arrowstyle="-", lw=0.45, color="0.25", shrinkA=0, shrinkB=3),
                                path_effects=[mpl.patheffects.withStroke(linewidth=1.6, foreground="white")])
                t.set_in_layout(False)
                texts.append(t)
            label_sets.append((ax, texts, np.column_stack((lon, lat))))
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_aspect(1 / np.cos(np.radians((ext[2] + ext[3]) / 2)))
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5)); ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))
        ax.tick_params(labelsize=FONT["tick"])
        ax.set_ylabel("Latitude (\u00b0)", fontsize=FONT["label"])
        ax.set_xlabel("Longitude (\u00b0)", fontsize=FONT["label"])
        ax.set_title(f"({'abcdefgh'[k]})  {S['label']}", loc="left", fontsize=FONT["title"] + 2,
                     fontweight="bold" if WELLMAP_TITLE_BOLD else "normal")
        covers = iext[0] <= ext[0] and iext[1] >= ext[1] and iext[2] <= ext[2] and iext[3] >= ext[3]
        north_scale(ax, S, credit if k == n - 1 else None, ext=ext, credit=credit, arrow_xy=(0.95, 0.42),
                    units=WELLMAP_SCALE_UNITS, color=None if covers else "black")   # black on a blank margin
        cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        if WELLMAP_HI_LOG:
            t = [vmin] + [v for v in (2, 5, 10, 20, 50, 100, 200) if vmin * 1.15 < v < vmax / 1.15] + [vmax]
            cb.set_ticks(t); cb.set_ticklabels([f"{v:.2f}" if v < 10 else f"{v:.1f}" if v in (vmin, vmax) else f"{v:g}"
                                                for v in t])
            cb.ax.minorticks_off()
        cb.set_label("Hazard index (HI)" + (", kriged and wells" if interp else ""), fontsize=FONT["label"])
        cb.ax.tick_params(labelsize=FONT["tick"])
    lg = fig.add_subplot(gs[-1, :]); lg.axis("off")
    h = [Line2D([], [], marker="^", ls="", mfc="white", mec="black", ms=8, label="Upgradient"),
         Line2D([], [], marker="v", ls="", mfc="white", mec="black", ms=8, label="Downgradient"),
         Line2D([], [], ls="-", lw=0.6, color="black", marker="o", ms=3, markevery=[1],
                label="True position (co-located wells offset)"),
         Patch(fc="none", ec=COLORS["wellmap_ccr"], label="CCR impoundment"),
         Patch(fc="none", ec=COLORS["wellmap_ccr"], hatch="////", label="CCR landfill")]
    if WELLMAP_ROI_BOX:
        h.insert(2, Line2D([], [], ls="--", color=COLORS["wellmap_roi"], label="Site ROI"))
    lg.legend(handles=h, loc="center", ncol=3, frameon=False, fontsize=FONT["legend"])
    if label_sets:                                         # move overlapping well names apart (after the layout)
        fig.canvas.draw()
        for ax, texts, pts in label_sets:
            declutter(ax, texts, pts)
    save(fig, out, "Fig5_Well_HI_maps" + ("_kriged" if interp else ""))


# ---------------------------------------------------------------- SHAP (Random Forest, held-out test pixels)
def saabas(m, X):
    """Tree contributions (Saabas): for every split on a sample's path, the change in the class-1 probability is
    credited to the split feature; averaged over the trees. An approximation of TreeSHAP used when the shap package
    is not installed (same output shape; base value = mean root probability)."""
    out = np.zeros(X.shape, float)
    base = 0.0
    for est in m.estimators_:
        t = est.tree_
        v = t.value[:, 0, :]
        p = v[:, 1] / np.maximum(v.sum(1), 1e-12)             # class-1 probability at each node
        parent = np.full(t.node_count, -1)
        for nd in range(t.node_count):
            for ch in (t.children_left[nd], t.children_right[nd]):
                if ch >= 0:
                    parent[ch] = nd
        child = np.flatnonzero(parent >= 0)
        from scipy.sparse import csr_matrix
        M = csr_matrix((p[child] - p[parent[child]], (np.arange(len(child)), t.feature[parent[child]])),
                       shape=(len(child), X.shape[1]))
        path = est.decision_path(X.astype(np.float32))         # samples x nodes (0/1)
        out += np.asarray((path[:, child] @ M).todense())
        base += p[0]
    return out / len(m.estimators_), base / len(m.estimators_)


def shap_site(S):
    """SHAP values of the map model (Random Forest, seed MAP_SEED, same split and standardisation as Table 1) for
    SHAP_N test pixels (stratified). Returns values, standardised feature values, bands,
    method."""
    code = block_split(S, SPLIT_SEED)
    Z, _ = prepare(S, code)
    Yf = S["Y"].ravel(); cf = code.ravel()
    tr, te = np.flatnonzero(cf == 0), np.flatnonzero(cf == 3)
    m = make_model("RandomForest", MAP_SEED, Z.shape[1])
    m.fit(Z[tr], Yf[tr])
    rng = np.random.default_rng(MAP_SEED)
    k = min(SHAP_N, len(te))
    pick = np.concatenate([rng.choice(te[Yf[te] == c], max(1, int(round(k * (Yf[te] == c).mean()))), replace=False)
                           for c in (0, 1) if (Yf[te] == c).any()])
    Xs = Z[pick]
    method = SHAP_METHOD
    vals = None
    if method == "tree":
        try:
            import shap
            ex = shap.TreeExplainer(m)
            sv = ex.shap_values(Xs, check_additivity=False)
            sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if np.ndim(sv) == 3 else sv)
            vals = np.asarray(sv, float)
            print(f"  SHAP (TreeExplainer, shap {shap.__version__}): {len(Xs)} test pixels")
        except ImportError:
            print("  shap not installed (pip install shap): Saabas tree contributions used instead")
            method = "saabas"
    if vals is None:
        vals, _ = saabas(m, Xs)
        method = "saabas"
        print(f"  Saabas tree contributions: {len(Xs)} test pixels")
    return vals, Xs, list(S["bands"]), method


@fonts_for("fig7")
def fig_shap(SH, out):
    """Fig. 7: SHAP beeswarm per site (top SHAP_TOP features); Fig. 8: mean |SHAP| per feature group; CSV."""
    labels = list(SH)
    method = {v[3] for v in SH.values()}
    mname = "SHAP" if method == {"tree"} else "Tree contribution (Saabas)"
    n = len(labels)
    fig, A = plt.subplots(1, n, figsize=(4.2 * n + 1.0, 0.36 * SHAP_TOP + 1.4), constrained_layout=True,
                          squeeze=False)
    cm = LinearSegmentedColormap.from_list("shap", COLORS["shap"])
    rows = []
    for j, lab in enumerate(labels):
        vals, Z, bands, _ = SH[lab]
        mean_abs = np.abs(vals).mean(0)
        order = np.argsort(mean_abs)[::-1][:SHAP_TOP]
        ax = A[0, j]
        rng = np.random.default_rng(0)
        for r, f in enumerate(order):
            v = vals[:, f]
            z = Z[:, f]
            col = (pd.Series(z).rank(pct=True)).to_numpy()   # colour by the feature's value rank
            h, edges = np.histogram(v, bins=60)              # jitter width follows the point density
            dens = h[np.clip(np.digitize(v, edges) - 1, 0, len(h) - 1)] / max(h.max(), 1)
            y = r + rng.uniform(-0.38, 0.38, len(v)) * dens
            ax.scatter(v, y, c=col, cmap=cm, vmin=0, vmax=1, s=SHAP_POINT, lw=0, alpha=0.85, rasterized=True)
        ax.axvline(0, color="0.4", lw=0.7)
        ax.set_yticks(range(len(order)), [nice(bands[f]) for f in order])
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.tick_params(labelsize=FONT["tick"])
        ax.set_xlabel(f"{mname} value (class 1)", fontsize=FONT["label"])
        ax.set_title(lab, fontweight="bold", fontsize=FONT["title"])
        ax.grid(axis="x", color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)
        b2g = {b: g for g, bs in band_groups(bands).items() for b in bs}
        for f in range(len(bands)):
            rows.append(dict(site=lab, feature=bands[f], group=b2g.get(bands[f], "Other"),
                             mean_abs_shap=float(mean_abs[f]), method=SH[lab][3]))
    cb = fig.colorbar(mpl.cm.ScalarMappable(Normalize(0, 1), cm), ax=A[0, -1], fraction=0.05, pad=0.02, ticks=[0, 1])
    cb.set_ticklabels(["Low", "High"]); cb.set_label("Feature value", fontsize=FONT["label"])
    cb.ax.tick_params(labelsize=FONT["key"])
    save(fig, out, "Fig7_SHAP_beeswarm")
    _set_fonts("fig8")                                     # Fig. 8 has its own font sizes
    D = pd.DataFrame(rows)
    D.to_csv(out / "shap_mean_abs_by_feature.csv", index=False, float_format="%.5f")
    G = D.groupby(["site", "group"]).mean_abs_shap.sum().unstack(0)
    G = G.loc[G.mean(1).sort_values(ascending=False).index]
    G.to_csv(out / "shap_mean_abs_by_group.csv", float_format="%.5f")
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(G) + 1.4), constrained_layout=True)
    for j, lab in enumerate(labels):
        y = np.arange(len(G)) + (j - (n - 1) / 2) * 0.8 / n
        ax.barh(y, G[lab], 0.8 / n, color=site_color(lab, j), label=lab)
    ax.set_yticks(range(len(G)), G.index); ax.invert_yaxis()
    ax.set_xlabel(f"Mean |{mname}| summed over the group", fontsize=FONT["label"])
    ax.tick_params(labelsize=FONT["tick"])
    ax.grid(axis="x", color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=FONT["legend"], loc="lower right")
    save(fig, out, "Fig8_SHAP_feature_groups")


# ---------------------------------------------------------------- leave-one-site-out
def loso_compute(FS):
    """Train on all sites but one, test on the held-out site; every model and seed. Features common to all sites,
    standardised per site (LOSO_STANDARDISE='per_site') or with the training sites' mean and SD ('train')."""
    bands = [b for b in FS[0]["bands"] if all(b in S["bands"] for S in FS)]
    cat_idx = [i for i, b in enumerate(bands) if b in NO_CATEGORICAL]
    rng0 = np.random.default_rng(SPLIT_SEED)
    D = {}
    for S in FS:                                           # valid pixels, common bands
        F = S["X"][..., [S["bands"].index(b) for b in bands]].reshape(-1, len(bands)).astype(float)
        idx = np.flatnonzero(S["valid"].ravel())
        y = S["Y"].ravel()[idx]
        D[S["label"]] = dict(F=F[idx], y=y, idx=idx)
    def standardise(F, mu, sd):
        Z = np.nan_to_num((F - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
        return Z, np.rint((np.clip(Z, -4, 4) + 4) / 8 * NB_BINS).astype(int)
    def stats(F):
        mu, sd = np.nanmean(F, 0), np.nanstd(F, 0)
        sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0; mu[~np.isfinite(mu)] = 0.0
        return mu, sd
    def sample(y, k, rng):
        if len(y) <= k:
            return np.arange(len(y))
        return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), int(round(k * (y == c).mean())),
                                                  replace=False) for c in (0, 1)]))
    rows, maps = [], {}
    for test in [S["label"] for S in FS]:
        train = [l for l in D if l != test]
        tr_pick = {l: sample(D[l]["y"], LOSO_MAX_PER_SITE, rng0) for l in train}
        te_pick = sample(D[test]["y"], LOSO_MAX_TEST, rng0)
        if LOSO_STANDARDISE == "per_site":
            Ztr = [standardise(D[l]["F"][tr_pick[l]], *stats(D[l]["F"])) for l in train]
            st_te = stats(D[test]["F"])
        else:
            mu, sd = stats(np.vstack([D[l]["F"][tr_pick[l]] for l in train]))
            Ztr = [standardise(D[l]["F"][tr_pick[l]], mu, sd) for l in train]
            st_te = (mu, sd)
        Xtr = np.vstack([z for z, _ in Ztr]); Qtr = np.vstack([q for _, q in Ztr])
        ytr = np.concatenate([D[l]["y"][tr_pick[l]] for l in train])
        Zte_all, Qte_all = standardise(D[test]["F"], *st_te)
        yte = D[test]["y"][te_pick]
        print(f"  LOSO test {test}: train {len(ytr)} pixels from {', '.join(train)}; test {len(yte)} pixels")
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            for name in LOSO_MODELS or MODELS:
                cols = [i for i in range(len(bands)) if not (name in ("KNN", "SVM") and i in cat_idx)]
                Dtr, Dte = (Qtr, Qte_all) if name == "NaiveBayes" else (Xtr, Zte_all)
                tri = np.arange(len(ytr))
                if name == "SVM" and MAX_TRAIN_SVM and len(tri) > MAX_TRAIN_SVM:
                    tri = sample(ytr, MAX_TRAIN_SVM, rng)
                m = make_model(name, seed, len(cols))
                kw = {}
                if CLASS_WEIGHTS and (name == "NaiveBayes" or getattr(m, "needs_sample_weight", False)):
                    cnt = np.bincount(ytr[tri], minlength=2)
                    kw["sample_weight"] = (len(tri) / (2 * cnt))[ytr[tri]]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(Dtr[tri][:, cols], ytr[tri], **kw)
                s = class_1_score(m, Dte[te_pick][:, cols])
                pred = (s >= (0 if isinstance(m, SVC) else THRESHOLD)).astype(int)
                rows.append(dict(site=test, model=name, seed=seed, n_train=len(tri), n=len(yte),
                                 **scores_of(yte, pred, s)))
                if name == MAP_MODEL and seed == MAP_SEED:      # map of the held-out site
                    S = next(x for x in FS if x["label"] == test)
                    pm = np.full(S["valid"].size, np.nan)
                    pm[D[test]["idx"]] = (class_1_score(m, Dte[:, cols]) >= THRESHOLD).astype(int)
                    maps[test] = pm.reshape(S["valid"].shape)
                r = rows[-1]
                print(f"    {name:<13} seed {seed}: Accuracy {r['OA']:.3f}  macro F1 {r['macro_F1']:.3f}")
    return pd.DataFrame(rows), maps


def loso_report(FS, L, maps, R_within, units_by, out):
    """Table 9, Figs 9-10 from the leave-one-site-out results."""
    L.to_csv(out / "loso_metrics_per_seed.csv", index=False, float_format="%.4f")
    # table
    cols = ["OA", "prec_0", "rec_0", "f1_0", "prec_1", "rec_1", "f1_1", "kappa", "macro_F1", "AUC"]
    tab = []
    for (s_, mdl), g in L.groupby(["site", "model"], sort=False):
        tab.append([s_, mdl] + [f"{g[c].mean():.3f} \u00b1 {g[c].std():.3f}" for c in cols])
    TB = pd.DataFrame(tab, columns=["Held-out site", "Model", "Accuracy", "Prec. 0", "Rec. 0", "F1 0", "Prec. 1",
                                    "Rec. 1", "F1 1", "\u03ba", "Macro F1", "ROC AUC"])
    TB.to_csv(out / "Table9_LOSO.csv", index=False, encoding="utf-8-sig")
    cap = (f"Table 9. Leave-one-site-out test: models trained on the other sites ({LOSO_MAX_PER_SITE:,} pixels per "
           f"site, features standardised {'per site' if LOSO_STANDARDISE == 'per_site' else 'with the training sites'}"
           f") and tested on every valid pixel of the held-out site (up to {LOSO_MAX_TEST:,}); mean \u00b1 SD over "
           f"seeds {', '.join(map(str, SEEDS))}.")
    tex_table(TB, cap, "tab:loso", out / "Table9_LOSO.tex", colfmt="ll" + "c" * 10,
              group_header="\\multicolumn{2}{c}{} & & \\multicolumn{3}{c}{Class 0} & \\multicolumn{3}{c}{Class 1} "
                           "& & & \\\\\n\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}")
    fig_loso_scores(L, R_within, out)
    fig_loso_maps(FS, maps, L, units_by, out)
    return [(cap, TB)]


@fonts_for("fig9")
def fig_loso_scores(L, R_within, out):
    """Fig. 9: held-out-site Accuracy and macro F1 per model (within-site test value in brackets)."""
    T = R_within[(R_within.partition == "test") & (R_within.target == "HI")]
    sites = list(dict.fromkeys(L.site)); models = list(dict.fromkeys(L.model))
    cm = LinearSegmentedColormap.from_list("acc", RAMP[::-1])
    fig, A = plt.subplots(1, 2, figsize=(1.35 * len(sites) * 2 + 3.2, 0.5 * len(models) + 1.6),
                          constrained_layout=True)
    for ax, (c, lab) in zip(A, (("OA", "Accuracy"), ("macro_F1", "Macro F1"))):
        M = L.groupby(["model", "site"])[c].mean().unstack().reindex(index=models, columns=sites)
        W = T.groupby(["model", "site"])[c].mean().unstack().reindex(index=models, columns=sites)
        im = ax.imshow(M.values, cmap=cm, vmin=0.3, vmax=1.0, aspect="auto")
        for i in range(len(models)):
            for j in range(len(sites)):
                v, w = M.values[i, j], W.values[i, j]
                ax.text(j, i, f"{v:.2f}" + (f"\n({w:.2f})" if np.isfinite(w) else ""), ha="center", va="center",
                        fontsize=FONT["annot"], color="white" if v < 0.5 or v > 0.9 else "black")
        ax.set_xticks(range(len(sites)), sites); ax.set_yticks(range(len(models)), models)
        ax.tick_params(labelsize=FONT["tick"], length=0)
        ax.set_title(f"{lab}, held-out site (within-site test)", fontweight="bold", fontsize=FONT["title"])
    cb = fig.colorbar(im, ax=A, fraction=0.03, pad=0.02); cb.ax.tick_params(labelsize=FONT["key"])
    save(fig, out, "Fig9_LOSO_scores")


@fonts_for("fig10")
def fig_loso_maps(FS, maps, L, units_by, out):
    """Fig. 10: per held-out site: reference label | LOSO classification (MAP_MODEL, seed MAP_SEED) | error map."""
    n = len(FS)
    P = PANEL_IN["fig1"] * 1.2
    hr = [aspect(S) for S in FS] + [0.14]
    fig = plt.figure(figsize=(P * 3 + 1.2, P * sum(hr) + 1.0), constrained_layout=True)
    gs = fig.add_gridspec(n + 1, 3, height_ratios=hr)
    cm_lab = ListedColormap(COLORS["label"]); nb2 = BoundaryNorm([-0.5, 0.5, 1.5], 2)
    cm_err = ListedColormap([COLORS["error"][k] for k in ("TN", "FP", "FN", "TP")])
    for i, S in enumerate(FS):
        units = units_by[S["label"]]
        pr = maps.get(S["label"])
        ax = fig.add_subplot(gs[i, 0])
        show(ax, S, np.where(S["valid"], S["Y"], np.nan).astype(float), cm_lab, nb2, units, None, mode="multi")
        ax.set_ylabel(S["label"], fontweight="bold", fontsize=FONT["row"])
        if pr is None:
            continue
        ax = fig.add_subplot(gs[i, 1])
        show(ax, S, pr, cm_lab, nb2, units, None, mode="multi")
        r = L[(L.site == S["label"]) & (L.model == MAP_MODEL)]
        ax.text(0.02, 0.02, f"Accuracy {r.OA.mean():.3f}", transform=ax.transAxes, fontsize=FONT["annot"],
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        ax = fig.add_subplot(gs[i, 2])
        e = np.where(np.isfinite(pr), S["Y"] * 2 + np.nan_to_num(pr), np.nan)
        show(ax, S, e, cm_err, BoundaryNorm(np.arange(-0.5, 4), 4), units, None, mode="multi")
        if i == 0:
            for j, t in enumerate(("Reference label", f"{MAP_NAME} trained on the other sites", "Error map")):
                fig.axes[-3 + j].set_title(t, fontweight="bold", fontsize=FONT["title"])
    lg = fig.add_subplot(gs[-1, :]); lg.axis("off")
    h = [Patch(color=COLORS["label"][0], label="Class 0 (lower HI)"), Patch(color=COLORS["label"][1],
         label="Class 1 (higher HI)")] + [Patch(color=COLORS["error"][k], label=k) for k in ("TN", "FP", "FN", "TP")]
    h += map_handles([u for S in FS for u in units_by[S["label"]]])
    lg.legend(handles=h, loc="center", ncol=5, frameon=False, fontsize=FONT["legend"])
    save(fig, out, "Fig10_LOSO_maps")


# ---------------------------------------------------------------- skill, feature-set ablation, label sensitivity
SKILL_COLS = ["OA", "balanced_acc", "MCC", "kappa", "prec_0", "rec_0", "f1_0", "prec_1", "rec_1", "f1_1"]
SKILL_HEAD = ["Accuracy", "Bal. acc.", "MCC", "κ", "Prec. 0", "Rec. 0", "F1 0", "Prec. 1", "Rec. 1", "F1 1"]


def baseline_row(T):
    """'Everything is class 1' on the same test pixels."""
    n1 = (T.TP + T.FN).iloc[0]; n = T.n.iloc[0]; s1 = n1 / n
    return dict(OA=s1, balanced_acc=0.5, MCC=0.0, kappa=0.0, prec_0=0.0, rec_0=0.0, f1_0=0.0, prec_1=s1, rec_1=1.0,
                f1_1=2 * s1 / (1 + s1), FN=0.0)


def table_skill(R, out):
    """Table 1b: skill at the 0.5 threshold and at the validation-tuned threshold, with the majority-class baseline
    (mean of the seeds)."""
    R = R[R.target == "HI"]
    rows = []
    for site, g in R.groupby("site", sort=False):
        T = g[g.partition == "test"]
        b = baseline_row(T)
        rows.append(dict(Site=site, Model="All class 1 (baseline)", **{h: b[c] for c, h in zip(SKILL_COLS, SKILL_HEAD)},
                         FN=0, **{"Bal. acc. (tuned)": np.nan, "MCC (tuned)": np.nan, "Rec. 0 (tuned)": np.nan,
                                  "Rec. 1 (tuned)": np.nan, "FN (tuned)": np.nan, "Threshold (tuned)": np.nan}))
        for model, h in T.groupby("model", sort=False):
            m = h[SKILL_COLS + ["FN"]].mean()
            tt = g[(g.partition == "test_tuned") & (g.model == model)]
            tm = tt[["balanced_acc", "MCC", "rec_0", "rec_1", "FN", "threshold"]].mean() if len(tt) else None
            rows.append(dict(Site=site, Model=model, **{hh: m[c] for c, hh in zip(SKILL_COLS, SKILL_HEAD)},
                             FN=m.FN, **({"Bal. acc. (tuned)": tm.balanced_acc, "MCC (tuned)": tm.MCC,
                                         "Rec. 0 (tuned)": tm.rec_0, "Rec. 1 (tuned)": tm.rec_1,
                                         "FN (tuned)": tm.FN, "Threshold (tuned)": tm.threshold} if tm is not None
                                        else {})))
    TB = pd.DataFrame(rows)
    tuned = (R.partition == "test_tuned").any()
    if not tuned:                                             # no validation set: default threshold only
        TB = TB[[c for c in TB.columns if "(tuned)" not in c]]
    TB.to_csv(out / "Table1b_skill_baseline_tuned.csv", index=False, float_format="%.4f", encoding="utf-8-sig")
    show = TB.copy()
    for c in show.columns[2:]:
        show[c] = [("" if not np.isfinite(v) else f"{v:,.0f}" if c.startswith("FN") else f"{v:.3f}")
                   for v in show[c]]
    cap = ("Table 1b. Test skill of the six classifiers relative to the majority-class baseline (all pixels predicted "
           "as class 1): accuracy, balanced accuracy, Matthews correlation coefficient (MCC), Cohen's κ, class-wise "
           "precision, recall and F1 and false negatives (FN) at the default threshold"
           + (", and balanced accuracy, MCC, class recalls and FN at the threshold that maximises balanced accuracy on "
              "the validation blocks (tuned)" if tuned else "") + f". Mean of seeds {', '.join(map(str, SEEDS))}.")
    tex_table(show, cap, "tab:skill", out / "Table1b_skill_baseline_tuned.tex", colfmt="ll" + "c" * (len(show.columns) - 2))
    return [(cap, show)]


def pixel_xy(S):
    """Easting and northing (m) of the pixel centres relative to the grid origin."""
    x0, y0, dx, dy = S["geo"]
    R_, C_ = S["valid"].shape
    lat_c = y0 - dy * R_ / 2
    rr, cc = np.indices((R_, C_))
    return (cc + 0.5) * dx * 111320 * np.cos(np.radians(lat_c)), -(rr + 0.5) * dy * 110540


def pixel_lonlat(S):
    x0, y0, dx, dy = S["geo"]
    rr, cc = np.indices(S["valid"].shape)
    return x0 + (cc + 0.5) * dx, y0 - (rr + 0.5) * dy


def pixel_distance_to_units(S, units, chunk=20000):
    """Distance (m) of every pixel centre to the nearest CCR unit edge (0 inside a unit)."""
    lon, lat = pixel_lonlat(S)
    out = np.full(lon.size, np.nan)
    if not units:
        return out.reshape(lon.shape)
    flat = pd.DataFrame({"longitude": lon.ravel(), "latitude": lat.ravel()})
    for a in range(0, len(flat), chunk):
        out[a:a + chunk] = dist_to_units(flat.iloc[a:a + chunk], units)
    return out.reshape(lon.shape)


def pixel_distance_to_wells(S):
    xm, ym = pixel_xy(S)
    x0, y0, dx, dy = S["geo"]
    lat_c = y0 - dy * S["valid"].shape[0] / 2
    wx = (S["wells"]["longitude"].to_numpy() - x0) * 111320 * np.cos(np.radians(lat_c))
    wy = (S["wells"]["latitude"].to_numpy() - y0) * 110540
    d = np.full(xm.shape, np.inf)
    for a, b in zip(wx, wy):
        d = np.minimum(d, np.hypot(xm - a, ym - b))
    return d


def feature_set(S, spec, units):
    if spec == "all":
        return S["X"], S["bands"]
    if spec == "distance":
        d = pixel_distance_to_units(S, units)
        if not np.isfinite(d).any():
            return None, None
        return d[..., None].astype(np.float32), ["dist_ccr_m"]
    if spec == "coords":
        xm, ym = pixel_xy(S)
        return np.stack([xm, ym], -1).astype(np.float32), ["x_m", "y_m"]
    groups = band_groups(S["bands"])
    bands = [b for g in spec for b in groups.get(g, [])]
    if not bands:
        return None, None
    return S["X"][..., [S["bands"].index(b) for b in bands]], bands


def summarise_runs(R, key, keys_order):
    """Mean ± SD over seeds of the test scores (0.5 and tuned threshold) per site and run."""
    rows = []
    for (site, k), g in R.groupby(["site", key], sort=False):
        t = g[g.partition == "test"]; tt = g[g.partition == "test_tuned"]
        r = dict(Site=site, **{key: k}, n_test=int(t.n.iloc[0]), class1_share=float(((t.TP + t.FN) / t.n).iloc[0]))
        for c in ["OA", "balanced_acc", "MCC", "kappa", "rec_0", "rec_1", "AUC"]:
            r[c] = t[c].mean(); r[c + "_sd"] = t[c].std()
        for c in ["balanced_acc", "MCC", "rec_0", "rec_1"]:
            r[c + "_tuned"] = tt[c].mean() if len(tt) else np.nan
        rows.append(r)
    out = pd.DataFrame(rows)
    out[key] = pd.Categorical(out[key], [k for k in keys_order if k in set(out[key])], ordered=True)
    return out.sort_values(["Site", key]).reset_index(drop=True)


def runs_table(Sm, key, cap, label, path_tex):
    show = pd.DataFrame({"Site": Sm.Site, key.replace("_", " ").capitalize(): Sm[key].astype(str),
                         "Accuracy": [f"{a:.3f}" for a in Sm.OA],
                         "Baseline": [f"{a:.3f}" for a in Sm.class1_share],
                         "Bal. acc.": [f"{a:.3f}" for a in Sm.balanced_acc], "MCC": [f"{a:.3f}" for a in Sm.MCC],
                         "κ": [f"{a:.3f}" for a in Sm.kappa], "Rec. 0": [f"{a:.3f}" for a in Sm.rec_0],
                         "Rec. 1": [f"{a:.3f}" for a in Sm.rec_1], "AUC": [f"{a:.3f}" for a in Sm.AUC],
                         "Bal. acc. (tuned)": [f"{a:.3f}" for a in Sm.balanced_acc_tuned],
                         "MCC (tuned)": [f"{a:.3f}" for a in Sm.MCC_tuned]})
    if not np.isfinite(Sm.balanced_acc_tuned).any():          # no validation set: no tuned columns
        show = show.drop(columns=["Bal. acc. (tuned)", "MCC (tuned)"])
        cap = cap.split(" Tuned:")[0]
    tex_table(show, cap, label, path_tex, colfmt="ll" + "c" * (len(show.columns) - 2))
    return show


@fonts_for("fig11")
def fig_ablation(Sm, out):
    """Fig. 11: test accuracy of RF per site (x axis) with all predictors, vegetation-only and environment-only
    predictors (bars, mean ± SD over seeds). Bars take the site colour (SITE_COLORS); the predictor set is shown by
    fill pattern and shade (ABL_FIG_STYLE). Black line: majority-class accuracy of the site."""
    st = ABL_FIG_STYLE
    c, lab = ABL_FIG_METRIC
    sets = [f for f in ABL_FIG_SETS if f in set(Sm.feature_set.astype(str))]
    sites_ = list(dict.fromkeys(Sm.Site))
    W = (st["width_cm"] or PAGE["width_cm"]) / 2.54
    fig, ax = plt.subplots(figsize=(W, st["height_cm"] / 2.54), constrained_layout=True)
    w = st["bar_w"] / len(sets)
    mpl.rcParams["hatch.linewidth"] = st["hatch_lw"]
    for j, site in enumerate(sites_):
        d = Sm[Sm.Site == site].assign(fs=lambda x: x.feature_set.astype(str)).set_index("fs").reindex(sets)
        col = SITE_COLORS.get(site, "#888888")
        for k, fs in enumerate(sets):
            x = j + (k - (len(sets) - 1) / 2) * w
            fc = mpl.colors.to_rgba(col, st["alpha"].get(fs, 1.0))
            ax.bar(x, d.loc[fs, c], w * 0.92, yerr=d.loc[fs, c + "_sd"], color=fc, edgecolor=col,
                   lw=st["edge_lw"], hatch=st["hatch"].get(fs, ""), error_kw=dict(lw=st["err_lw"],
                   capsize=st["capsize"]), zorder=2)
        if ABL_FIG_BASELINE:
            ax.hlines(d["class1_share"].iloc[0], j - st["bar_w"] / 2, j + st["bar_w"] / 2,
                      color=st["baseline_color"], lw=st["baseline_lw"], zorder=3)
    ax.set_xticks(range(len(sites_)), sites_)
    ax.set_ylabel(lab, fontsize=FONT["label"])
    ax.set_ylim(*st["ylim"])
    ax.tick_params(labelsize=FONT["tick"], direction=st["tick_dir"], length=4, width=st["box_lw"],
                   color=st["box_color"])
    ax.tick_params(axis="x", length=0)                       # categories: no tick marks under the bar groups
    ax.grid(axis="y", color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(st["box"] or sp.spine_type in ("left", "bottom"))
        sp.set_color(st["box_color"]); sp.set_linewidth(st["box_lw"])
    grey = "#555555"
    h = [Patch(facecolor=mpl.colors.to_rgba(grey, st["alpha"].get(fs, 1.0)), edgecolor=grey,
               hatch=st["hatch"].get(fs, ""), label=fs) for fs in sets]
    if ABL_FIG_BASELINE:
        h.append(mpl.lines.Line2D([], [], color=st["baseline_color"], lw=st["baseline_lw"], label="Majority class"))
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(h), frameon=False,
               fontsize=FONT["legend"], handlelength=1.6, columnspacing=1.0)
    save(fig, out, "Fig11_feature_set_ablation")


def step_ablation(D):
    """Step 3b. RF with feature subsets and location-only baselines (same blocks and seeds): does the within-site
    accuracy come from vegetation, from the environment, or from location alone? (Table 10, Fig. 11)"""
    out = Path(OUT_DIR)
    res = []
    for S in sites(D):
        units = D["units"][S["label"]]
        sig = signature(S["sig"], model_settings(), FEATURE_SETS, ABLATION_MODELS, bool(units))

        def compute(S=S, units=units):
            rr = []
            for name, spec in FEATURE_SETS.items():
                X, bands = feature_set(S, spec, units)
                if X is None:
                    print(f"  {S['label']} {name}: not available, skipped")
                    continue
                print(f"  {S['label']} feature set '{name}' ({len(bands)} features)")
                R, _, _ = run_models(dict(S, X=X, bands=bands, target="HI"), ABLATION_MODELS, False)
                rr.append(R.assign(feature_set=name, n_features=len(bands)))
            return pd.concat(rr, ignore_index=True)
        res.append(checkpoint(f"ablation_{S['stem']}", sig, compute))
    R = pd.concat(res, ignore_index=True)
    R.to_csv(out / "ablation_metrics_per_seed.csv", index=False, float_format="%.4f")
    Sm = summarise_runs(R, "feature_set", list(FEATURE_SETS))
    Sm.to_csv(out / "Table10_feature_set_ablation.csv", index=False, float_format="%.4f")
    cap = ("Table 10. Random Forest test performance with different feature sets and location-only baselines (same "
           "spatial blocks and seeds). Baseline: accuracy of predicting every pixel as class 1. Balanced accuracy and "
           "MCC are 0.5 and 0 for the baseline. Tuned: threshold maximising balanced accuracy on the validation "
           "blocks.")
    show = runs_table(Sm, "feature_set", cap, "tab:ablation", out / "Table10_feature_set_ablation.tex")
    if "Bal. acc. (tuned)" not in show.columns:
        cap = cap.split(" Tuned:")[0]
    D["docs"][10] = [(cap, show)]
    fig_ablation(Sm, out)
    print(Sm[["Site", "feature_set", "OA", "balanced_acc", "MCC", "kappa"]].round(3).to_string(index=False))


def step_sensitivity(D):
    """Step 3c. Label sensitivity: RF with the class-0 percentile at SENS_ALPHAS and without pixels far from any
    well (Table 11)."""
    out = Path(OUT_DIR)
    res = []
    for S in sites(D):
        sig = signature(S["sig"], model_settings(), SENS_ALPHAS, SENS_MAX_WELL_DIST_M, SENS_MODELS, "same blocks, main labels")

        def compute(S=S):
            rr = []
            base_valid = S["valid"] & np.isfinite(S["HIn"])
            ordn = lambda n: f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"
            variants = [(f"{ordn(a)} percentile" + (" (main labels)" if a == SENS_ALPHAS[0] else ""), a, None)
                        for a in SENS_ALPHAS]
            if SENS_MAX_WELL_DIST_M:
                variants.append((f"{ordn(SENS_ALPHAS[0])} percentile, ≤ {SENS_MAX_WELL_DIST_M:g} m from a well",
                                 SENS_ALPHAS[0], SENS_MAX_WELL_DIST_M))
            dw = pixel_distance_to_wells(S) if SENS_MAX_WELL_DIST_M else None
            code_hi = block_split(S, SPLIT_SEED)              # every variant is tested on the same blocks
            for name, a, dmax in variants:
                valid = base_valid & (dw <= dmax) if dmax else base_valid.copy()
                if a == SENS_ALPHAS[0]:                       # reference: the labels of the main HI models
                    Y = np.where(valid, S["Y"], 0).astype(int)
                else:
                    thr = np.nanpercentile(S["HIn"][base_valid], a)
                    Y = np.where(valid, (S["HIn"] >= thr), 0).astype(int)
                print(f"  {S['label']} labels '{name}': {valid.sum()} pixels, class-1 share {Y[valid].mean():.3f}")
                R, _, _ = run_models(dict(S, Y=Y, valid=valid, target="HI"), SENS_MODELS, False, code=code_hi)
                rr.append(R.assign(label_variant=name, n_pixels=int(valid.sum())))
            return pd.concat(rr, ignore_index=True)
        res.append(checkpoint(f"sensitivity_{S['stem']}", sig, compute))
    R = pd.concat(res, ignore_index=True)
    R.to_csv(out / "label_sensitivity_per_seed.csv", index=False, float_format="%.4f")
    order = list(dict.fromkeys(R.label_variant))
    Sm = summarise_runs(R, "label_variant", order)
    Sm.to_csv(out / "Table11_label_sensitivity.csv", index=False, float_format="%.4f")
    cap = ("Table 11. Sensitivity of the Random Forest test performance to the label definition: class 0 = lowest 25, "
           "33 or 50 % of the normalised HI (rule labels), and the 25 % labels without pixels farther than "
           f"{SENS_MAX_WELL_DIST_M} m from a monitoring well. Baseline: accuracy of predicting every pixel as class 1.")
    show = runs_table(Sm, "label_variant", cap, "tab:sensitivity", out / "Table11_label_sensitivity.tex")
    if "Bal. acc. (tuned)" not in show.columns:
        cap = cap.split(" Tuned:")[0]
    D["docs"][11] = [(cap, show)]
    print(Sm[["Site", "label_variant", "OA", "class1_share", "balanced_acc", "MCC"]].round(3).to_string(index=False))


def panel_letter(k):
    """a, b, ..., z, aa, ab, ... for panel k (0-based)."""
    a = "abcdefghijklmnopqrstuvwxyz"
    return a[k] if k < 26 else a[k // 26 - 1] + a[k % 26]


@fonts_for("fig12")
def fig_hi_vs_vi(W, out_docs, radius=None, note=""):
    """Fig. 12: well HI against CIRE and CCC (HIVI_INDICES) within HIVI_BUFFER_M of the well."""
    return _hi_vs_vi(W, out_docs, radius, note, HIVI_INDICES, "Fig12_HI_vs_vegetation", "Table_HI_vs_vegetation",
                     "H3", HIVI_STYLE)


@fonts_for("figS_hivi_all")
def fig_hi_vs_all(W, out_docs, radius=None, note=""):
    """Supplementary figure: well HI against all nine indices (INDICES) within HIVI_BUFFER_M of the well."""
    r = HIVI_BUFFER_M if radius is None else radius
    ixs = [i for i in INDICES if i not in HIVI_ALL_EXCLUDE]
    return _hi_vs_vi(W, out_docs, radius, note, ixs, f"FigS_HI_vs_all_indices_{r:g}m",
                     f"Table_HI_vs_all_indices_{r:g}m", "H7", {**HIVI_STYLE, **HIVI_ALL_STYLE})


def _hi_vs_vi(W, out_docs, radius, note, indices, fig_name, table_name, tab_label, st):
    """Well HI (x) against the vegetation index level in the buffer around the well (y), one row per index and one
    column per site; each point is a monitoring well (upgradient / downgradient markers), Spearman rho, p and n, the
    partial rho on the distance to the CCR units and a robust trend line (solid where p < ALPHA)."""
    from scipy.stats import theilslopes
    vo = veg_out(); vo.mkdir(parents=True, exist_ok=True)
    sites_ = sorted(W.site.unique())
    ixs = [i for i in indices if i in W]
    if not ixs:
        return
    fig, A = plt.subplots(len(ixs), len(sites_), figsize=(st["panel_w"] * len(sites_), st["panel_h"] * len(ixs)),
                          constrained_layout=True, squeeze=False)
    rows = []
    for i, ix in enumerate(ixs):
        col = ix
        for j, site in enumerate(sites_):
            ax = A[i, j]
            d = W[W.site == site][["HI", col, "gradient", "dist_ccr_m"]].dropna(subset=["HI", col])
            d = d[d.HI > 0] if HIVI_LOG_X else d
            for g, mk, fc in (("Upgradient", st["up"], st["face"]), ("Downgradient", st["down"], st["face"])):
                e = d[d.gradient == g]
                ax.scatter(e.HI, e[col], marker=mk, s=st["marker_size"], c=fc, edgecolors=st["edge"], lw=0.6,
                           zorder=3, label=f"{g} well")
            e = d[~d.gradient.isin(["Upgradient", "Downgradient"])]
            if len(e):
                ax.scatter(e.HI, e[col], marker=st["unknown"], s=st["marker_size"] * 0.7, c=st["unknown_face"],
                           edgecolors=st["edge"], lw=0.6, zorder=3, label="Gradient unknown")
            rho = p = rp = pp = slope = np.nan
            if len(d) >= 4 and d.HI.nunique() >= 3:
                rho, p = spearmanr(d.HI, d[col])
                dd = d.dropna(subset=["dist_ccr_m"])
                if HIVI_SHOW_PARTIAL and len(dd) >= 5:
                    rp, pp = partial_spearman(dd.HI, dd[col], dd.dist_ccr_m)
                x = np.log10(d.HI) if HIVI_LOG_X else d.HI
                if HIVI_TREND == "theil-sen":
                    slope, icpt, _, _ = theilslopes(d[col], x)
                elif HIVI_TREND == "ols":
                    slope, icpt = np.polyfit(x, d[col], 1)
                if HIVI_TREND:
                    xs = np.linspace(x.min(), x.max(), 50)
                    ax.plot(10 ** xs if HIVI_LOG_X else xs, icpt + slope * xs, color=st["trend"], lw=st["trend_lw"],
                            ls="-" if p < ALPHA else "--", zorder=2)
            pt = lambda v: "p < 0.001" if v < 0.001 else f"p = {v:.3f}"
            txt = f"ρ = {rho:+.2f}, {pt(p)}, n = {len(d)}"
            if HIVI_SHOW_PARTIAL and np.isfinite(rp):
                txt += f"\npartial ρ = {rp:+.2f}, {pt(pp)}"
            ax.set_title(txt.replace("-", "\u2212"), loc="right", fontsize=FONT["annot"], linespacing=1.1)
            if HIVI_LOG_X:
                ax.set_xscale("log")
                lo, hi_ = d.HI.min(), d.HI.max()
                ticks = [t for t in (0.5, 1, 2, 5, 10, 20, 50, 100, 200) if lo / 1.3 <= t <= hi_ * 1.3]
                ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
                ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
                ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            ax.grid(color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)
            ax.tick_params(labelsize=FONT["tick"])
            ax.set_title(f"({panel_letter(i * len(sites_) + j)}) {site}", loc="left", fontsize=FONT["title"],
                         va="bottom")
            if j == 0:
                ax.set_ylabel(f"{ix}{note} ({hivi_radius_label(radius)})", fontsize=FONT["label"])
            if i == len(ixs) - 1:
                ax.set_xlabel("Hazard Index (HI) at the well", fontsize=FONT["label"])
            rows.append(dict(site=site, index=ix, n=len(d), rho=rho, p=p, rho_partial=rp, p_partial=pp,
                             trend_slope_per_log10_HI=slope))
    h, l = A[0, 0].get_legend_handles_labels()
    for ax in A.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(l), frameon=False,
               fontsize=FONT["legend"])
    save(fig, vo, fig_name)
    T = pd.DataFrame(rows)
    T.to_csv(vo / f"{table_name}.csv", index=False, float_format="%.4f")
    show = T.assign(rho=T.rho.map(lambda v: f"{v:+.2f}".replace("-", "\u2212")), p=T.p.map(lambda v: f"{v:.3f}"),
                    rho_partial=T.rho_partial.map(lambda v: "" if not np.isfinite(v) else f"{v:+.2f}".replace("-", "\u2212")),
                    p_partial=T.p_partial.map(lambda v: "" if not np.isfinite(v) else f"{v:.3f}"))
    show = show[["site", "index", "n", "rho", "p", "rho_partial", "p_partial"]]
    show.columns = ["Site", "Index", "Wells", "ρ", "p", "Partial ρ", "p (partial)"]
    out_docs.append((f"Table {tab_label}. Spearman's ρ between the well HI and {', '.join(ixs)}{note} "
                     f"({hivi_radius_label(radius)}); partial ρ controls for the distance to the nearest CCR unit.",
                     show))
    print(show.to_string(index=False))
    return T


def tables(R, out):
    R = R[R.target == "HI"]
    T = R[R.partition == "test"]
    cols = ["OA", "prec_0", "rec_0", "f1_0", "prec_1", "rec_1", "f1_1", "kappa", "macro_F1", "AUC"]
    summ = T.groupby(["site", "model"], sort=False)[cols].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(out / "metrics_test_mean_sd.csv", index=False, float_format="%.4f")
    cms = T.groupby(["site", "model", "seed"], sort=False)[["TN", "FP", "FN", "TP"]].sum().reset_index()
    cms.to_csv(out / "confusion_matrices_test.csv", index=False)
    head = ["Site", "Model", "Accuracy", "Prec. 0", "Rec. 0", "F1 0", "Prec. 1", "Rec. 1", "F1 1", "κ",
            "Macro F1", "ROC AUC"]
    rows = []
    for _, r in summ.iterrows():
        rows.append([r.site, r.model] + [f"{r[c + '_mean']:.3f} ± {r[c + '_std']:.3f}" for c in cols])
    TB = pd.DataFrame(rows, columns=head)
    cap = (f"Table 1. Test performance of the six classifiers (mean ± SD over seeds {', '.join(map(str, SEEDS))}; "
           f"{BLOCK_PX}-pixel spatial blocks, {int(SPLIT_FRAC[0] * 100)}/{int(SPLIT_FRAC[1] * 100)}/"
           f"{int(SPLIT_FRAC[2] * 100)} % split, {BUFFER_PX}-pixel buffer). Precision, recall and F1 per class; "
           f"class 0 = lowest {ALPHA_PERCENTILE} % of the normalised HI.")
    tex_table(TB, cap, "tab:ccr_models", out / "Table1_models_test.tex", colfmt="ll" + "c" * 10,
              group_header="\\multicolumn{2}{c}{} & & \\multicolumn{3}{c}{Class 0} & \\multicolumn{3}{c}{Class 1} "
                           "& & & \\\\\n\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}")
    rows = []
    for (s, m, seed), g in cms.groupby(["site", "model", "seed"], sort=False):
        rows.append([s, m, seed, int(g.TN.iloc[0]), int(g.FP.iloc[0]), int(g.FN.iloc[0]), int(g.TP.iloc[0])])
    TC = pd.DataFrame(rows, columns=["Site", "Model", "Seed", "TN", "FP", "FN", "TP"])
    cap2 = ("Table 2. Test confusion matrices per seed (reference 0/1 × predicted 0/1): TN = class 0 predicted 0, "
            "FP = class 0 predicted 1, FN = class 1 predicted 0, TP = class 1 predicted 1.")
    tex_table(TC, cap2, "tab:ccr_cm", out / "Table2_confusion_matrices.tex", colfmt="llccccc")
    return summ, [(cap, TB)] + table_skill(R, out) + [(cap2, TC)]


# ---------------------------------------------------------------- checkpoints
import hashlib
import pickle


def ck_dir():
    return Path(CHECKPOINT_DIR) if CHECKPOINT_DIR else Path(OUT_DIR) / "checkpoints"


def stamp(*names):
    """Size and modification time of input files (missing files count too), so a checkpoint is redone when an
    input file changes."""
    out = []
    for n in names:
        if not n:
            continue
        try:
            f = find(n)
            st = f.stat()
            out.append((f.name, st.st_size, int(st.st_mtime)))
        except FileNotFoundError:
            out.append((n, None))
    return out


def site_stamp(name):
    cfg = SITES[name]
    st = cfg["stem"]
    shp = [CCR_SHP, str(Path(CCR_SHP).with_suffix(".dbf"))] if CCR_SHP else []
    return stamp(f"{st}_Features.tif", f"{st}_Label.tif", f"{st}_HI.tif", cfg["wells"], cfg["samples"], *shp)


def signature(*parts):
    return hashlib.md5(repr(parts).encode()).hexdigest()


def checkpoint(name, sig, compute):
    """Result of compute(), restored from CHECKPOINT_DIR/<name>.pkl when it was saved with the same signature
    (settings and input files); otherwise computed and saved."""
    p = ck_dir() / f"{name}.pkl"
    if CHECKPOINT == "auto" and p.is_file():
        try:
            with open(p, "rb") as f:
                d = pickle.load(f)
            if d.get("sig") == sig:
                print(f"  restored checkpoint {name}")
                return d["value"]
            print(f"  checkpoint {name}: settings or inputs changed, recomputing")
        except Exception as ex:
            print(f"  checkpoint {name} unreadable ({type(ex).__name__}), recomputing")
    value = compute()
    if CHECKPOINT != "off":
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(dict(sig=sig, value=value), f, protocol=4)
    return value


def model_settings():
    return (MODEL_PARAMS, SEEDS, SEED_SCOPE, BLOCK_PX, SPLIT_FRAC, BUFFER_PX, N_SPLIT_TRIALS, SPLIT_SEED,
            STANDARDISE, CLASS_WEIGHTS, MAX_TRAIN_SVM, NB_BINS, NO_CATEGORICAL, THRESHOLD, MAP_MODEL, MAP_SEED,
            TUNE_THRESHOLD)


def label_settings():
    return (HI_SOURCE, LABEL_SOURCE, ALPHA_PERCENTILE, MIN_HQ_IN_HI, RISK_BREAKS, DROP_BANDS, HQ_FROM_SAMPLES,
            SCREENING_MG_L, NON_METALS)


# ---------------------------------------------------------------- run, step by step
# Each step adds its results to the dictionary D and writes its figures and tables; in the notebook run the cells in
# order (a step needs the steps above it). With CHECKPOINT = "auto" a rerun restores finished steps from disk.
def step_load():
    """Step 1. Sites: features, kriged HI, labels, risk classes, wells, CCR units, basemaps."""
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    D = dict(S={}, units={}, order=[], R={}, I={}, maps={}, hq={}, shap={}, docs={})
    for name in RUN_SITES:
        sig = signature(site_stamp(name), label_settings(), SITES[name])
        S = checkpoint(f"site_{SITES[name]['stem']}", sig, lambda: load_site(name))
        S["sig"] = sig
        units = ccr_units(S)
        S["basemap"] = get_basemap(S, BASEMAP)
        S["basemap_h"] = S["basemap"] if BASEMAP_H in (None, BASEMAP) else get_basemap(S, BASEMAP_H)
        D["S"][S["label"]] = S; D["units"][S["label"]] = units; D["order"].append(S["label"])
    return D


def sites(D):
    return [D["S"][l] for l in D["order"]]


def step_feature_maps(D):
    """Step 2. Maps of every feature band: per site and group, per group for all sites."""
    out = Path(OUT_DIR)
    for S in sites(D):
        fig_feature_groups(S, D["units"][S["label"]], out)
    fig_feature_groups_all_sites(sites(D), D["units"], out)


def step_models(D):
    """Step 3. Six classifiers x seeds on the HI label (spatial blocks); site figures (map model)."""
    out = Path(OUT_DIR)
    for S in sites(D):
        sig = signature(S["sig"], model_settings(), MODELS)
        R, I, maps = checkpoint(f"models_{S['stem']}_HI", sig, lambda: run_models(S, MODELS, True))
        if len(I):
            I = I.assign(feature=I.feature.replace(OLD_GROUP_NAMES))
        t = R[(R.model == MAP_MODEL) & (R.seed == MAP_SEED) & (R.partition == "test")].iloc[0]
        maps["scores"] = dict(OA=t.OA, kappa=t.kappa, AUC=t.AUC)
        D["R"][S["label"]], D["I"][S["label"]], D["maps"][S["label"]] = R, I, maps
        fig_site_maps(S, D["units"][S["label"]], maps, out)
    pd.concat(D["R"].values()).to_csv(out / "metrics_HI_per_seed_all_partitions.csv", index=False,
                                      float_format="%.4f")


HQ_KEYS = ("Y", "valid", "HQ", "HQn", "target", "metal", "well_hq")


def step_metal_hq(D):
    """Step 4. Kriged HQ of every exceeded metal; RF on each metal's HQ label (HQ_RF_SITES); Figs 2-4, Tables 3-4."""
    out = Path(OUT_DIR)
    for S in sites(D):
        rf = S["name"] in HQ_RF_SITES
        sig = signature(S["sig"], HQ_METALS, HQ_NUGGET_FRAC, rf and (model_settings(), HQ_MODELS))

        def compute(S=S, rf=rf):
            res = []
            for metal in hq_metals_of(S):
                try:
                    T = hq_target(S, metal)
                except (KeyError, ValueError) as ex:
                    print(f"  {metal} skipped: {ex}")
                    continue
                RT, mp = (run_models(T, HQ_MODELS, False)[::2] if rf else (None, None))
                slim = {k: T[k] for k in HQ_KEYS}
                slim["HQ"] = slim["HQ"].astype(np.float32); slim["HQn"] = slim["HQn"].astype(np.float32)
                res.append((slim, RT, mp))
            return res
        res = checkpoint(f"hq_{S['stem']}", sig, compute)
        D["hq"][S["label"]] = [(dict(bare(S), **t), r, mp) for t, r, mp in res]   # site data + this metal
    hq_all = {l: [t for t, _, _ in v] for l, v in D["hq"].items()}
    fig_hq_all_sites(hq_all, sites(D), D["units"], out)
    T4 = fig_hq_summary(sites(D), out)
    cap4 = ("Table 4. Maximum well HQ and wells with HQ ≥ 1 (exceeding / sampled) per metal and site; "
            "n.m. = not measured.")
    tex_table(T4, cap4, "tab:hq_sites", out / "Table4_site_HQ_exceedance.tex", colfmt="l" + "c" * (len(T4.columns) - 1))
    D["docs"][4] = [(cap4, T4)]
    RT = [r for v in D["hq"].values() for _, r, _ in v if r is not None]
    if RT:
        RT = pd.concat(RT, ignore_index=True)
        RT.to_csv(out / "metrics_metal_HQ_per_seed.csv", index=False, float_format="%.4f")
        hq_rf = {}
        for l, v in D["hq"].items():
            TT = [t for t, r, _ in v if r is not None]
            if TT:
                hq_rf[l] = TT
                maps = {t["target"]: mp for t, r, mp in v if r is not None}
                if FIG2_LAYOUT == "vertical":
                    fig_hq_vertical(TT, D["units"][l], maps, RT, out)
                    continue
                parts = -(-len(TT) // METALS_PER_FIG["fig2"])       # equal parts of at most METALS_PER_FIG['fig2']
                k = -(-len(TT) // parts)
                for q in range(0, len(TT), k):
                    fig_hq(TT[q:q + k], D["units"][l], maps, RT, out, f"_part{q // k + 1}" if parts > 1 else "")
        D["docs"][3] = table_hq(RT, hq_rf, out)


def step_well_maps(D):
    """Step 5. Wells coloured by HI on a basemap, with and/or without the kriged HI (Fig. 5)."""
    for S, view in zip(sites(D), wellmap_views(sites(D))):   # same panel shape for every site (WELLMAP_ASPECT)
        S["view_wm"] = view
        S["basemap_wm"] = get_basemap(S, WELLMAP_BASEMAP or BASEMAP, ext=view)
    for interp in ((False, True) if WELLMAP_INTERP == "both" else (bool(WELLMAP_INTERP),)):
        fig_hi_wells(sites(D), D["units"], Path(OUT_DIR), interp)


def step_index_rows(D):
    """Step 6. Vegetation index level, CSTV and rate of change, one row per site (Figs 6a-c)."""
    fig_index_rows(sites(D), D["units"], Path(OUT_DIR))


def step_shap(D):
    """Step 7. SHAP of the Random Forest on held-out test pixels (Figs 7-8)."""
    for S in sites(D):
        sig = signature(S["sig"], model_settings(), SHAP_METHOD, SHAP_N)
        D["shap"][S["label"]] = checkpoint(f"shap_{S['stem']}", sig, lambda S=S: shap_site(S))
    fig_shap(D["shap"], Path(OUT_DIR))


# ---------------------------------------------------------------- well HI against vegetation: settings, sensitivity
def hivi_radius_label(radius):
    r = HIVI_BUFFER_M if radius is None else radius
    return "pixel at the well" if r == 0 else f"{r:g} m buffer"


def lc_adjusted(S, ix):
    """Index minus the value expected from the Dynamic World land-cover probabilities of the pixel (OLS fitted over
    all valid pixels of the site): the index relative to land cover of the same composition."""
    B, X = S["bands"], S["X"]
    lc = [c for c in HIVI_LC if c in B]
    y = X[..., B.index(ix)].astype(float)
    Z = np.stack([X[..., B.index(c)] for c in lc], -1).astype(float)
    ok = S["valid"] & np.isfinite(y) & np.isfinite(Z).all(-1)
    A = np.column_stack([np.ones(ok.sum()), Z[ok]])
    coef = np.linalg.lstsq(A, y[ok], rcond=None)[0]
    return y - (coef[0] + (Z * coef[1:]).sum(-1))


def hivi_mask(S, kind):
    B, X = S["bands"], S["X"]
    m = S["valid"].copy()
    if kind == "vegetated":
        m &= X[..., B.index("NDVI")] >= HIVI_MIN_NDVI
        for c, mx in (("DW_Water", 0.5), ("DW_Built", 0.5)):
            if c in B:
                m &= X[..., B.index(c)] < mx
    return m


def buffer_values(S, layers, radius, mask):
    """Mean of each layer over the pixels of `mask` whose centres lie within `radius` m of each well (radius 0: the
    pixel nearest to the well). Returns (wells x layers, pixel counts)."""
    w = S["wells"]
    x0, y0, dx, dy = S["geo"]
    R_, C_ = S["valid"].shape
    lat_c = y0 - dy * R_ / 2
    rr, cc = np.indices((R_, C_))
    out = np.full((len(w), len(layers)), np.nan)
    npx = np.zeros(len(w), int)
    for k, (lat, lon) in enumerate(zip(w["latitude"], w["longitude"])):
        yy = (y0 - (rr + 0.5) * dy - lat) * 110540
        xx = (x0 + (cc + 0.5) * dx - lon) * 111320 * np.cos(np.radians(lat_c))
        d2 = xx ** 2 + yy ** 2
        m = (d2 <= radius ** 2) if radius > 0 else (d2 == d2.min())
        m &= mask
        npx[k] = m.sum()
        if npx[k]:
            out[k] = [np.nanmean(L[m]) for L in layers]
    return out, npx


def hivi_wells(S, units, radius, mask, adjust, indices):
    """One row per well: HI, gradient, distance to the CCR units and each index under the given buffer settings."""
    B = S["bands"]
    layers = [lc_adjusted(S, ix) if adjust == "landcover" else S["X"][..., B.index(ix)].astype(float)
              for ix in indices]
    vals, npx = buffer_values(S, layers, HIVI_BUFFER_M if radius is None else radius, hivi_mask(S, mask))
    w = S["wells"][["well_name", "HI", "gradient"]].copy()
    for j, ix in enumerate(indices):
        w[ix] = vals[:, j]
    w["n_pixels_buffer"] = npx
    w["dist_ccr_m"] = dist_to_units(S["wells"], units)
    w["site"] = S["label"]
    return w


@fonts_for("figS8")
def hivi_landcover(SS, units_by, out_docs):
    """Land-cover diagnostic: land-cover composition within HIVI_BUFFER_M of each well, its Spearman correlation
    with the well HI and with the indices (Table H5), and HI against each index coloured by the dominant land cover
    of the buffer (Fig. S-HIVI-LC)."""
    vo = veg_out()
    st = HIVI_LC_STYLE
    groups = {"Trees": ["DW_Trees"], "Grass/crops/shrubs": ["DW_Grass", "DW_Crops", "DW_Shrubs"],
              "Water/flooded": ["DW_Water", "DW_FloodedVeg"], "Built/bare": ["DW_Built", "DW_Bare"]}
    rows, frames = [], []
    for S in SS:
        B = S["bands"]
        extra = [c for c in HIVI_LC + ["NDVI", "HAND_m", "DEM"] if c in B]
        w = hivi_wells(S, units_by[S["label"]], HIVI_BUFFER_M, None, None, HIVI_INDICES)
        v, _ = buffer_values(S, [S["X"][..., B.index(c)].astype(float) for c in extra], HIVI_BUFFER_M,
                             S["valid"])
        for j, c in enumerate(extra):
            w[c] = v[:, j]
        G = pd.DataFrame({g: w[[c for c in cs if c in w]].sum(1) for g, cs in groups.items()})
        w["dominant_cover"] = G.idxmax(1)
        frames.append(w)
        for c in extra + ["dist_ccr_m"]:
            d = w[["HI", c] + HIVI_INDICES].dropna()
            if d[c].nunique() < 3:
                continue
            r = dict(site=S["label"], covariate=c, rho_HI=spearmanr(d.HI, d[c])[0])
            for ix in HIVI_INDICES:
                r[f"rho_{ix}"] = spearmanr(d[ix], d[c])[0]
            rows.append(r)
    W = pd.concat(frames, ignore_index=True)
    W.to_csv(vo / f"wells_HI_vegetation_landcover_{HIVI_BUFFER_M:g}m.csv", index=False, float_format="%.4g")
    T = pd.DataFrame(rows)
    T.to_csv(vo / "Table_HI_vegetation_landcover_drivers.csv", index=False, float_format="%.3f")
    show = T.pivot(index="covariate", columns="site", values="rho_HI").round(2)
    for ix in HIVI_INDICES[:1]:
        show = show.join(T.pivot(index="covariate", columns="site", values=f"rho_{ix}").round(2),
                         rsuffix=f" ({ix})")
    show = show.reset_index()
    fmt = lambda v: ("" if not np.isfinite(v) else f"{v:+.2f}".replace("-", "\u2212")) if isinstance(v, float) else v
    show = show.apply(lambda col: col.map(fmt))
    out_docs.append((f"Table H5. Spearman's ρ of the land-cover fractions, NDVI, HAND and elevation within "
                     f"{HIVI_BUFFER_M:g} m of the wells, and of the distance to the CCR units, with the well HI "
                     f"(first four columns) and with {HIVI_INDICES[0]} (last four columns).", show))
    sites_ = sorted(W.site.unique())
    fig, A = plt.subplots(len(HIVI_INDICES), len(sites_), figsize=(HIVI_STYLE["panel_w"] * len(sites_),
                          HIVI_STYLE["panel_h"] * len(HIVI_INDICES)), constrained_layout=True, squeeze=False)
    for i, ix in enumerate(HIVI_INDICES):
        for j, site in enumerate(sites_):
            ax = A[i, j]
            d = W[(W.site == site) & (W.HI > 0)].dropna(subset=[ix])
            for g, col in st["lc_colors"].items():
                e = d[d.dominant_cover == g]
                for grad, mk in (("Upgradient", HIVI_STYLE["up"]), ("Downgradient", HIVI_STYLE["down"])):
                    f = e[e.gradient == grad]
                    ax.scatter(f.HI, f[ix], marker=mk, s=st["lc_marker_size"], c=col, edgecolors=st["lc_edge"],
                               lw=0.5, zorder=3)
                f = e[~e.gradient.isin(["Upgradient", "Downgradient"])]
                ax.scatter(f.HI, f[ix], marker="o", s=st["lc_marker_size"] * 0.7, c=col, edgecolors=st["lc_edge"],
                           lw=0.5, zorder=3)
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            ax.grid(color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)
            ax.tick_params(labelsize=FONT["tick"])
            ax.set_title(f"({panel_letter(i * len(sites_) + j)}) {site}", loc="left", fontsize=FONT["title"])
            if j == 0:
                ax.set_ylabel(f"{ix} ({HIVI_BUFFER_M:g} m buffer)", fontsize=FONT["label"])
            if i == len(HIVI_INDICES) - 1:
                ax.set_xlabel("Hazard Index (HI) at the well", fontsize=FONT["label"])
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="s", ls="", markersize=9, markerfacecolor=c, markeredgecolor="black", label=g)
         for g, c in st["lc_colors"].items()]
    h += [Line2D([], [], marker=HIVI_STYLE["up"], ls="", markersize=8, color="0.3", label="Upgradient well"),
          Line2D([], [], marker=HIVI_STYLE["down"], ls="", markersize=8, color="0.3", label="Downgradient well")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(h), frameon=False,
               fontsize=FONT["legend"], title=f"Dominant land cover within {HIVI_BUFFER_M:g} m",
               title_fontsize=FONT["legend"])
    save(fig, vo, "FigS_HI_vs_vegetation_landcover")
    # HI against the indices among wells of the same dominant land cover (like compared with like)
    rows = []
    for (site, cov), e in W.groupby(["site", "dominant_cover"]):
        for ix in HIVI_INDICES:
            d = e.dropna(subset=["HI", ix])
            if len(d) >= HIVI_MIN_WELLS_CLASS and d.HI.nunique() >= 3:
                r, p_ = spearmanr(d.HI, d[ix])
                rows.append(dict(site=site, dominant_cover=cov, index=ix, n=len(d), rho=r, p=p_))
    C = pd.DataFrame(rows)
    C.to_csv(vo / "Table_HI_vs_vegetation_within_landcover.csv", index=False, float_format="%.4f")
    if len(C):
        show = C.assign(rho=C.rho.map(lambda v: f"{v:+.2f}".replace("-", "\u2212")), p=C.p.map(lambda v: f"{v:.3f}"))
        show.columns = ["Site", "Dominant land cover", "Index", "Wells", "ρ", "p"]
        out_docs.append((f"Table H6. Spearman's ρ between the well HI and the index among wells with the same dominant "
                         f"land cover within {HIVI_BUFFER_M:g} m (classes with at least {HIVI_MIN_WELLS_CLASS} "
                         "wells).", show))
        print(show.to_string(index=False))
    return W, T


def step_hivi(D):
    """Step 8. Well HI against the vegetation index level within HIVI_BUFFER_M (60 m) of the well: CIRE and CCC
    (Fig. 12, Table H3), all nine indices (FigS_HI_vs_all_indices, Table H7) and the land-cover diagnostic
    (Table H5, H6, FigS_HI_vs_vegetation_landcover)."""
    SS = sites(D)
    docs = []
    ixs = list(dict.fromkeys(HIVI_INDICES + (INDICES if HIVI_ALL else [])))
    W = pd.concat([hivi_wells(S, D["units"][S["label"]], HIVI_BUFFER_M, HIVI_MASK, HIVI_ADJUST,
                              [i for i in ixs if i in S["bands"]]) for S in SS], ignore_index=True)
    note = ", land-cover adjusted" if HIVI_ADJUST == "landcover" else ""
    if HIVI_MASK == "vegetated":
        note += ", vegetated pixels"
    fig_hi_vs_vi(W, docs, radius=HIVI_BUFFER_M, note=note)
    if HIVI_ALL:
        fig_hi_vs_all(W, docs, radius=HIVI_BUFFER_M, note=note)
    if HIVI_LANDCOVER:
        hivi_landcover(SS, D["units"], docs)
    W.to_csv(veg_out() / f"wells_HI_vegetation_{HIVI_BUFFER_M:g}m.csv", index=False, float_format="%.5g")
    docx_tables(docs, veg_out() / "Tables_HI_vs_vegetation.docx")


def step_loso(D):
    """Step 9. Leave-one-site-out: train on the other sites, test on the held-out site (Table 9, Figs 9-10)."""
    FS = sites(D)
    sig = signature([S["sig"] for S in FS], model_settings(), LOSO_MODELS or MODELS, LOSO_MAX_PER_SITE, LOSO_MAX_TEST,
                    LOSO_STANDARDISE)
    L, maps = checkpoint("loso", sig, lambda: loso_compute(FS))
    R = pd.concat(D["R"].values(), ignore_index=True) if D["R"] else pd.DataFrame(columns=["partition", "target"])
    D["docs"][9] = loso_report(FS, L, maps, R, D["units"], Path(OUT_DIR))


def step_summary(D):
    """Step 10. Fig. 1 (all sites), Tables 1-2, confusion matrices, model comparison, feature-group importance, and
    the Word file with every table made so far."""
    out = Path(OUT_DIR)
    R = pd.concat(D["R"].values(), ignore_index=True)
    I = pd.concat([i for i in D["I"].values() if len(i)], ignore_index=True)
    I.to_csv(out / "feature_importance_per_seed.csv", index=False, float_format="%.5f")
    summ, docs = tables(R, out)
    D["docs"][1] = docs
    fig_all_sites(sites(D), D["units"], D["maps"], out)
    fig_confusion(R, out)
    fig_comparison(R, out)
    if len(I):
        fig_feature_group_importance(I, {l: D["S"][l]["bands"] for l in D["order"]}, out)
    docx_tables([t for k in sorted(D["docs"]) for t in D["docs"][k]], out / "Tables_CCR_models.docx")
    print(summ[["site", "model", "OA_mean", "kappa_mean", "macro_F1_mean", "AUC_mean"]].round(3).to_string(index=False))


def main():
    D = step_load()
    if FEATURE_MAPS:
        step_feature_maps(D)
    step_models(D)
    if ABLATION:
        step_ablation(D)
    if SENSITIVITY:
        step_sensitivity(D)
    step_metal_hq(D)
    if WELLMAP:
        step_well_maps(D)
    if INDEX_ROWS:
        step_index_rows(D)
    if SHAP:
        step_shap(D)
    if HIVI_ANALYSIS:
        step_hivi(D)
    if LOSO and len(D["order"]) > 1:
        step_loso(D)
    step_summary(D)
    return D


if __name__ == "__main__":
    main()
