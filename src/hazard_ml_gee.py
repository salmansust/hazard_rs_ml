"""
CCR site features, hazard labels and risk classes with the Earth Engine Python API.

Python port of the Code Editor script (features_v2 / multi-site CCR classification, LABEL_RULE 'percentile'):
  - Sentinel-2 summer (June-August) median 2019-2025 with Cloud Score+ masking: 12 bands, 9 vegetation indices
  - CSTV (standardised absolute change first -> last summer), rates of change (Sen's slope per year)
  - terrain (SRTM), climate (MODIS LST, NLDAS humidity, GLDAS soil moisture, CHIRPS JJA precipitation, or Landsat
    LST), soil (SoilGrids 0-5 cm), hydrology (MERIT upstream area, HAND, TWI), Dynamic World probabilities
  - optional seasonal and harmonic NDVI features
  - kriged HI of the well composite (ee kriging, exponential), HI_norm, binary label (class 0 = lowest
    ALPHA_PERCENTILE % of HI_norm), 5-tier risk class (HI < 1, 1-5, 5-10, 10-20, >= 20)
  - exports (Google Drive, or local GeoTIFF with geemap): <site>_Features, <site>_Label, <site>_HI,
    <site>_HI_norm, <site>_risk_class, and optionally the yearly stacks
  - optional interactive map of every band (geemap), coloured as the Code Editor script

Run:  pip install earthengine-api geemap
      python hazard_ml_gee.py            (or run the cells in Colab / Jupyter)
The rasters it writes are the inputs of hazard_ml.py (place them in data/features_labels/).
"""
import math
from pathlib import Path

import ee

# ==== settings: EDIT here =============================================================================================
EE_PROJECT = "your-ee-project"           # Google Cloud project registered for Earth Engine (edit)
SITES_TO_RUN = ["Glen Lyn", "Bremo", "Chesterfield", "Possum Point"]
SITES = {   # roi = [lon_min, lat_min, lon_max, lat_max]; asset = well HI composite (FeatureCollection, property HI)
    "Glen Lyn": dict(roi=[-80.88602532062332, 37.36408912138987, -80.86027611408035, 37.38305127717347],
                     asset="projects/your-ee-project/assets/GlenLyn_wells_HI_composite",
                     kriging=dict(range=1500, sill=50, nugget=3, maxDistance=2500), splitHI=5.0),
    "Bremo": dict(roi=[-78.30039054301348, 37.69660048434903, -78.27352553752031, 37.71500299536265],
                  asset="projects/your-ee-project/assets/Bremo_wells_HI_composite",
                  kriging=dict(range=2000, sill=75, nugget=5, maxDistance=3000), splitHI=5.0),
    "Chesterfield": dict(roi=[-77.41016043242651, 37.36006733962347, -77.3585761886521, 37.39553411886849],
                         asset="projects/your-ee-project/assets/Chesterfield_wells_HI_composite",
                         kriging=dict(range=2500, sill=200, nugget=10, maxDistance=4000), splitHI=5.0),
    "Possum Point": dict(roi=[-77.30699481987425, 38.53014484806756, -77.26262035393187, 38.565386642500634],
                         asset="projects/your-ee-project/assets/PossumPoint_wells_HI_composite",
                         kriging=dict(range=1800, sill=30, nugget=2, maxDistance=2800), splitHI=5.0),
}
WELLS_FROM_CSV = {}     # optional: {"Glen Lyn": "GlenLyn_wells_HI_composite.csv"} (columns latitude, longitude, HI)
                        # used instead of the asset when given

# ---- label (as LABEL_RULE 'percentile' of the Code Editor script) ----
LABEL_RULE = "percentile"     # 'percentile': class 0 = lowest ALPHA_PERCENTILE % of HI_norm; 'HI': class 1 = HI >= splitHI
ALPHA_PERCENTILE = 25
RISK_BREAKS = [1, 5, 10, 20]  # risk class 0..4: HI < 1, 1-5, 5-10, 10-20, >= 20

# ---- features ----
START_YEAR, END_YEAR = 2019, 2025
YEARS = list(range(START_YEAR, END_YEAR + 1))
SUMMER, SPRING, AUTUMN = (6, 8), (4, 5), (9, 10)
CLOUD_MASK = "csplus"         # 'csplus' (Cloud Score+) or 'qa60'
CS_THRESHOLD = 0.6
SCENE_CLOUD_MAX = 30
INDEX_CLAMP = {"MTCI": (-2, 15), "CIRE": (-1, 20), "LCC": (0, 5), "CCC": (-2, 15)}
RATE_METHOD = "sens"          # 'sens' or 'ols'
CSTV_NDVI_MIN = 0.2
CSTV_FILL = 0
ADD_CSTV_MASK = True
STATIC_CLIMATE = "coarse"     # 'coarse' (MODIS/NLDAS/GLDAS/CHIRPS), 'landsat' (Landsat LST only) or 'none'
YEARLY_FORCING = True
SEASONAL_FEATURES = "both"    # 'none', 'seasonal', 'harmonic' or 'both'
S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
VI_NAMES = ["NDVI", "MTCI", "SAVI", "CIRE", "NDRE1", "NDRE2", "LAI", "LCC", "CCC"]
DW_SRC = ["water", "trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"]
DW_NAMES = ["DW_Water", "DW_Trees", "DW_Grass", "DW_FloodedVeg", "DW_Crops", "DW_Shrubs", "DW_Built", "DW_Bare",
            "DW_Snow"]

# ---- exports ----
EXPORT_TARGET = "drive"       # 'drive' (Export.image.toDrive tasks), 'local' (geemap download) or 'none'
EXPORT_FOLDER = "CCR_Yearly_Feature_Stacks"         # Drive folder
LOCAL_DIR = Path("CCR_Feature_Stacks")              # local folder with EXPORT_TARGET = 'local'
EXPORT_CRS, EXPORT_SCALE = "EPSG:4326", 10
EXPORT_MAIN = True            # <site>_Features, <site>_Label, <site>_HI, <site>_HI_norm, <site>_risk_class
EXPORT_YEARLY = False         # <site>_FeatureStack_<year>, <site>_Label_<year>
START_TASKS = True            # start the Drive tasks (False: create them only)
BASEMAP_EXPORT = "naip"       # '<site>_basemap' for the map figures when web tiles cannot be downloaded:
                              # 'naip' (USDA NAIP aerial photo, about 1 m, true colour), 's2' (Sentinel-2 summer true
                              # colour, 10 m) or None
BASEMAP_SCALE = 2             # metres per pixel of the basemap (NAIP 1-2; Sentinel-2 10)
BASEMAP_PAD = 0.15            # extent beyond the ROI, fraction of the ROI size on each side (covers the well maps)
BASEMAP_ONLY = False          # True: export the basemaps only (skip features and labels)

# ---- visualisation (geemap, interactive) ----
SHOW_MAP = False              # True in a notebook: every band as a layer, coloured as the Code Editor script
BAND_STRETCH = (2, 98)
PAL = {
    "vegetation": ["a50026", "f46d43", "fee08b", "d9ef8b", "66bd63", "006837"],
    "reflectance": ["440154", "3b528b", "21918c", "5ec962", "fde725"],
    "cstv": ["ffffff", "fee5d9", "fcae91", "fb6a4a", "cb181d", "67000d"],
    "dem": ["006600", "33cc33", "ffff66", "cc9933", "996633", "f5f5f5"],
    "slope": ["ffffff", "fdd49e", "fc8d59", "d7301f", "7f0000"],
    "aspect": ["ff0000", "ffff00", "00ff00", "00ffff", "0000ff", "ff00ff", "ff0000"],
    "diverging": ["2166ac", "92c5de", "f7f7f7", "f4a582", "b2182b"],
    "temperature": ["313695", "74add1", "ffffbf", "f46d43", "a50026"],
    "humidity": ["f7fbff", "9ecae1", "4292c6", "08519c", "08306b"],
    "moisture": ["8c510a", "d8b365", "f6e8c3", "5ab4ac", "01665e"],
    "ph": ["d73027", "fee08b", "1a9850", "4575b4", "313695"],
    "sand": ["fffbe6", "fee391", "fec44f", "d95f0e", "993404"],
    "silt": ["fff5eb", "fdd0a2", "fd8d3c", "d94801", "7f2704"],
    "clay": ["fcfbfd", "dadaeb", "9e9ac8", "6a51a3", "3f007d"],
    "precip": ["ffffd9", "c7e9b4", "7fcdbb", "1d91c0", "0c2c84"],
    "upstream": ["f7fbff", "c6dbef", "6baed6", "2171b5", "08306b"],
    "hand": ["08306b", "4292c6", "c6dbef", "fee391", "cc4c02"],
    "twi": ["ffffcc", "a1dab4", "41b6c4", "2c7fb8", "253494"],
    "rate": ["d73027", "fc8d59", "f7f7f7", "91cf60", "1a9850"],
    "hi": ["1a9641", "a6d96a", "ffffbf", "fdae61", "d73027"],
    "risk": ["2b83ba", "66bd63", "fee08b", "f46d43", "d73027"],
    "label": ["2b83ba", "d73027"],
}
# ======================================================================================================================


def init():
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT)


def roi_of(cfg):
    return ee.Geometry.Rectangle(cfg["roi"], proj="EPSG:4326", geodesic=False)


# ---------------------------------------------------------------- features (port of makeFeatures)
def s2_collection(region):
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
           .filterDate(ee.Date.fromYMD(START_YEAR, 1, 1), ee.Date.fromYMD(END_YEAR + 1, 1, 1))
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", SCENE_CLOUD_MAX)))
    if CLOUD_MASK == "csplus":
        cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        return col.linkCollection(cs, ["cs_cdf"]).map(
            lambda img: img.select(S2_BANDS).updateMask(img.select("cs_cdf").gte(CS_THRESHOLD)).divide(10000)
            .copyProperties(img, ["system:time_start"]))

    def qa(img):
        q = img.select("QA60")
        ok = q.bitwiseAnd(1 << 10).eq(0).And(q.bitwiseAnd(1 << 11).eq(0))
        return img.select(S2_BANDS).updateMask(ok).divide(10000).copyProperties(img, ["system:time_start"])
    return col.map(qa)


def add_indices(image):
    b = image.select
    idx = [image.normalizedDifference(["B8", "B4"]).rename("NDVI"),
           b("B6").subtract(b("B5")).divide(b("B5").subtract(b("B4"))).rename("MTCI"),
           b("B8").subtract(b("B4")).divide(b("B8").add(b("B4")).add(0.5)).multiply(1.5).rename("SAVI"),
           b("B7").divide(b("B5")).subtract(1).rename("CIRE"),
           image.normalizedDifference(["B6", "B5"]).rename("NDRE1"),
           image.normalizedDifference(["B7", "B5"]).rename("NDRE2"),
           b("B8").subtract(b("B4").multiply(1.35)).multiply(11.728).rename("LAI"),
           b("B8").divide(b("B3")).divide(b("B3").divide(b("B4"))).multiply(0.1134).add(0.14).rename("LCC"),
           b("B8").divide(b("B3")).subtract(1).multiply(0.8364).subtract(1.3231).rename("CCC")]
    out = ee.Image.cat(idx)
    for k, (lo, hi) in INDEX_CLAMP.items():               # ratio indices explode where the denominator ~ 0
        out = out.addBands(out.select(k).max(lo).min(hi), None, True)
    return image.addBands(out).copyProperties(image, ["system:time_start"])


def landsat_lst(region):
    col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
           .filterBounds(region)
           .filterDate(ee.Date.fromYMD(START_YEAR, 1, 1), ee.Date.fromYMD(END_YEAR + 1, 1, 1))
           .filter(ee.Filter.calendarRange(SUMMER[0], SUMMER[1], "month")))

    def one(img):
        clear = img.select("QA_PIXEL").bitwiseAnd(int("11111", 2)).eq(0)
        return (img.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15).rename("Temperature")
                .updateMask(clear).copyProperties(img, ["system:time_start"]))
    return col.map(one)


def seasonal_composite(col, year, months):
    c = col.filter(ee.Filter.calendarRange(months[0], months[1], "month"))
    if year is not None:
        c = c.filter(ee.Filter.calendarRange(year, year, "year"))
    return c.median()


def make_features(region):
    s2 = s2_collection(region).map(add_indices)
    summer = seasonal_composite(s2, None, SUMMER).clip(region)
    per_year = [seasonal_composite(s2, y, SUMMER).clip(region) for y in YEARS]

    # rate of change of each index over the yearly summer medians
    rates = []
    for v in VI_NAMES:
        col = ee.ImageCollection.fromImages([ee.Image.constant(y - START_YEAR).float().rename("t")
                                             .addBands(per_year[i].select(v).rename("v"))
                                             for i, y in enumerate(YEARS)])
        fit = (col.reduce(ee.Reducer.sensSlope()).select("slope") if RATE_METHOD == "sens"
               else col.reduce(ee.Reducer.linearFit()).select("scale"))
        rates.append(fit.rename(v + "_Rate"))
    rates = ee.Image.cat(rates)

    # CSTV: standardised absolute change END_YEAR - START_YEAR
    first, last = per_year[0], per_year[-1]
    cstv_ok = first.select("NDVI").gte(CSTV_NDVI_MIN)
    cstv = []
    for v in VI_NAMES:
        d = last.select(v).subtract(first.select(v)).rename("d")
        st = d.reduceRegion(reducer=ee.Reducer.mean().combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True),
                            geometry=region, scale=20, maxPixels=1e9, bestEffort=True)
        cstv.append(d.subtract(ee.Number(st.get("d_mean"))).abs()
                    .divide(ee.Number(st.get("d_stdDev")).max(1e-12))
                    .updateMask(cstv_ok).unmask(CSTV_FILL).clip(region).rename("CSTV_" + v + "_Diff"))
    cstv = ee.Image.cat(cstv)
    if ADD_CSTV_MASK:
        cstv = cstv.addBands(cstv_ok.unmask(0).toFloat().clip(region).rename("CSTV_valid"))

    # static: terrain, soil, hydrology
    dem = ee.Image("USGS/SRTMGL1_003").clip(region).rename("DEM")
    terr = ee.Algorithms.Terrain(dem)
    slope = terr.select("slope").rename("Slope_deg")
    curv = dem.convolve(ee.Kernel.fixed(width=3, height=3, weights=[[1, 1, 1], [1, -8, 1], [1, 1, 1]],
                                        normalize=False)).rename("Curvature")
    terrain = ee.Image.cat(dem, slope, terr.select("aspect").rename("Aspect_deg"), curv)
    soil = ee.Image.cat(
        ee.Image("projects/soilgrids-isric/phh2o_mean").select("phh2o_0-5cm_mean").rename("pH"),
        ee.Image("projects/soilgrids-isric/sand_mean").select("sand_0-5cm_mean").rename("Sand_pct_0_5cm"),
        ee.Image("projects/soilgrids-isric/silt_mean").select("silt_0-5cm_mean").rename("Silt_pct_0_5cm"),
        ee.Image("projects/soilgrids-isric/clay_mean").select("clay_0-5cm_mean").rename("Clay_pct_0_5cm")
    ).clip(region)
    merit = ee.Image("MERIT/Hydro/v1_0_1").clip(region)
    upa = merit.select("upa").rename("UpstreamArea_km2")
    twi = (upa.multiply(1e6).divide(ee.Image.pixelArea().sqrt())
           .divide(slope.multiply(math.pi / 180).tan().add(1e-6)).max(1e-6).log().rename("TWI"))
    hydro = ee.Image.cat(upa, merit.select("hnd").rename("HAND_m"), twi)
    static_bands = ee.Image.cat(terrain, soil, hydro)

    # climate
    lst = landsat_lst(region)
    lst_all = lst.median().clip(region)

    def coarse_climate(y):                                    # y = None: all years, all seasons
        d0 = ee.Date.fromYMD(START_YEAR if y is None else y, 1 if y is None else SUMMER[0], 1)
        d1 = ee.Date.fromYMD(END_YEAR + 1, 1, 1) if y is None else ee.Date.fromYMD(y, SUMMER[1] + 1, 1)
        temp = (ee.ImageCollection("MODIS/061/MOD11A1").filterDate(d0, d1).select("LST_Day_1km").mean()
                .multiply(0.02).subtract(273.15).rename("Temperature"))
        hum = (ee.ImageCollection("NASA/NLDAS/FORA0125_H002").filterDate(d0, d1).select("specific_humidity")
               .mean().rename("Humidity"))
        moist = (ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H").filterDate(d0, d1)
                 .select("SoilMoi0_10cm_inst").mean().rename("Moisture"))
        yrs = YEARS if y is None else [y]
        precip = ee.ImageCollection([ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                                     .filterDate(ee.Date.fromYMD(yr, SUMMER[0], 1), ee.Date.fromYMD(yr, SUMMER[1] + 1, 1))
                                     .select("precipitation").sum() for yr in yrs]).mean().rename("PrecipJJA_mm_mean")
        return dict(temp=temp, hum=hum, moist=moist, precip=precip)

    climate = None
    if STATIC_CLIMATE == "landsat":
        climate = lst_all
    elif STATIC_CLIMATE == "coarse":
        cc = coarse_climate(None)
        climate = ee.Image.cat(cc["temp"], cc["hum"], cc["moist"], cc["precip"]).clip(region)

    def dw(y):
        c = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(region)
             .filterDate(ee.Date.fromYMD(START_YEAR, 1, 1), ee.Date.fromYMD(END_YEAR + 1, 1, 1))
             .filter(ee.Filter.calendarRange(SUMMER[0], SUMMER[1], "month")))
        if y is not None:
            c = c.filter(ee.Filter.calendarRange(y, y, "year"))
        prob = c.select(DW_SRC).mean().rename(DW_NAMES)
        return ee.Image.cat(prob, c.select("label").mode().rename("DW_LandCover"),
                            prob.select(["DW_Trees", "DW_Grass", "DW_FloodedVeg", "DW_Crops", "DW_Shrubs"])
                            .reduce(ee.Reducer.sum()).rename("DW_Vegetation")).clip(region)

    seasonal = None
    if SEASONAL_FEATURES in ("seasonal", "both"):
        spring = seasonal_composite(s2, None, SPRING).select("NDVI")
        autumn = seasonal_composite(s2, None, AUTUMN).select("NDVI")
        seasonal = ee.Image.cat(spring.rename("NDVI_spring"), autumn.rename("NDVI_autumn"),
                                spring.subtract(summer.select("NDVI")).rename("NDVI_spring_minus_summer"),
                                autumn.subtract(summer.select("NDVI")).rename("NDVI_autumn_minus_summer")
                                ).clip(region)
    if SEASONAL_FEATURES in ("harmonic", "both"):
        t0 = ee.Date.fromYMD(START_YEAR, 1, 1)

        def harm_img(img):
            t = ee.Date(img.get("system:time_start")).difference(t0, "year")
            w = t.multiply(2 * math.pi)
            return ee.Image.cat(ee.Image.constant(1), ee.Image.constant(t), ee.Image.constant(w.cos()),
                                ee.Image.constant(w.sin()), img).float()
        harm = (s2.select("NDVI").map(harm_img).reduce(ee.Reducer.linearRegression(4, 1)).select("coefficients")
                .arrayProject([0]).arrayFlatten([["c0", "c_t", "c_cos", "c_sin"]]))
        hb = ee.Image.cat(harm.select("c_cos").hypot(harm.select("c_sin")).rename("NDVI_harm_amplitude"),
                          harm.select("c_sin").atan2(harm.select("c_cos")).rename("NDVI_harm_phase")).clip(region)
        seasonal = hb if seasonal is None else seasonal.addBands(hb)

    stack = summer.select(VI_NAMES).addBands(summer.select(S2_BANDS)).addBands(cstv).addBands(terrain)
    if climate is not None:
        stack = stack.addBands(climate)
    stack = stack.addBands(soil).addBands(hydro).addBands(rates).addBands(dw(None))
    if seasonal is not None:
        stack = stack.addBands(seasonal)

    yearly = []
    for i, y in enumerate(YEARS):
        img = per_year[i].select(VI_NAMES).addBands(per_year[i].select(S2_BANDS)).addBands(dw(y))
        if YEARLY_FORCING:
            cy = coarse_climate(y)
            lst_y = lst.filter(ee.Filter.calendarRange(y, y, "year")).median().unmask(lst_all)
            img = (img.addBands(cy["temp"] if STATIC_CLIMATE == "coarse" else lst_y.rename("Temperature"))
                   .addBands(cy["hum"]).addBands(cy["precip"]))
        yearly.append(img.addBands(static_bands).clip(region).toFloat().set("year", y))
    return dict(features=stack.toFloat(), yearly=yearly, summer=summer, rates=rates)


# ---------------------------------------------------------------- hazard: kriged HI, HI_norm, label, risk class
def wells_of(name, cfg):
    csv = WELLS_FROM_CSV.get(name)
    if csv:
        import pandas as pd
        d = pd.read_csv(csv)
        return ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.longitude), float(r.latitude)]),
                                                {"HI": float(r.HI)}) for r in d.itertuples() if r.HI > 0])
    return ee.FeatureCollection(cfg["asset"])


def make_hazard(name, cfg, region):
    k = cfg["kriging"]
    hi = (wells_of(name, cfg).kriging(propertyName="HI", shape="exponential", range=k["range"], sill=k["sill"],
                                      nugget=k["nugget"], maxDistance=k["maxDistance"], reducer=ee.Reducer.mean())
          .rename("HI_kriged").clip(region))

    def args(reducer):
        return dict(reducer=reducer, geometry=region, crs=EXPORT_CRS, scale=EXPORT_SCALE, maxPixels=1e13, tileScale=4)
    st = hi.reduceRegion(**args(ee.Reducer.minMax()))
    hmin, hmax = ee.Number(st.get("HI_kriged_min")), ee.Number(st.get("HI_kriged_max"))
    hi_norm = hi.subtract(hmin).divide(hmax.subtract(hmin).max(1e-12)).rename("HI_norm")
    if LABEL_RULE == "percentile":
        alpha = ee.Number(hi_norm.reduceRegion(**args(ee.Reducer.percentile(percentiles=[ALPHA_PERCENTILE],
                                                                            maxRaw=1e6))).values().get(0))
    else:
        alpha = ee.Number(cfg["splitHI"]).subtract(hmin).divide(hmax.subtract(hmin).max(1e-12))
    label = hi_norm.gte(alpha).toByte().rename("class")
    risk = ee.Image(0)
    for i, b in enumerate(RISK_BREAKS, 1):
        risk = risk.where(hi.gte(b), i)
    risk = risk.rename("risk_class").clip(region).toByte()
    return dict(hi=hi, hi_norm=hi_norm, alpha=alpha, label=label, risk=risk, hmin=hmin, hmax=hmax)


# ---------------------------------------------------------------- exports
def basemap_image(cfg):
    """True-colour basemap (RGB, 0-255) and its padded region."""
    x0, y0, x1, y1 = cfg["roi"]
    px, py = (x1 - x0) * BASEMAP_PAD, (y1 - y0) * BASEMAP_PAD
    region = ee.Geometry.Rectangle([x0 - px, y0 - py, x1 + px, y1 + py], "EPSG:4326", False)
    if BASEMAP_EXPORT == "naip":
        col = (ee.ImageCollection("USDA/NAIP/DOQQ").filterBounds(region)
               .filterDate(f"{END_YEAR - 5}-01-01", f"{END_YEAR + 1}-01-01").sort("system:time_start", False))
        img = col.mosaic().select(["R", "G", "B"]).visualize(min=0, max=255)
    else:
        img = (s2_collection(region).filter(ee.Filter.calendarRange(6, 8, "month"))
               .filterDate(f"{END_YEAR - 2}-01-01", f"{END_YEAR + 1}-01-01").median()
               .select(["B4", "B3", "B2"]).visualize(min=0, max=0.25, gamma=1.2))
    return img.toUint8(), region


def export(img, name, region, scale=None):
    scale = scale or EXPORT_SCALE
    if EXPORT_TARGET == "drive":
        task = ee.batch.Export.image.toDrive(image=img, description=name, folder=EXPORT_FOLDER, fileNamePrefix=name,
                                             region=region, crs=EXPORT_CRS, scale=scale, maxPixels=1e13,
                                             fileFormat="GeoTIFF")
        if START_TASKS:
            task.start()
        print("  Drive task:", name, "(started)" if START_TASKS else "(created)")
    elif EXPORT_TARGET == "local":
        import geemap                                       # tiles the request (no 32 MB limit)
        LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        geemap.download_ee_image(img, filename=str(LOCAL_DIR / f"{name}.tif"), region=region, crs=EXPORT_CRS,
                                 scale=scale)
        print("  saved", LOCAL_DIR / f"{name}.tif")


def show_map(name, F, Hz, region):
    """Every band as a layer ('<group> | <band>'), stretched between BAND_STRETCH percentiles (geemap)."""
    import geemap
    feats = F["features"]
    st = feats.reduceRegion(reducer=ee.Reducer.percentile(list(BAND_STRETCH)), geometry=region, scale=20,
                            maxPixels=1e13, bestEffort=True, tileScale=4).getInfo()
    groups = band_groups(feats.bandNames().getInfo())
    m = geemap.Map()
    m.centerObject(region, 14)
    m.add_basemap("HYBRID")
    for grp, bands in groups.items():
        for b in bands:
            lo, hi = st.get(f"{b}_p{BAND_STRETCH[0]}"), st.get(f"{b}_p{BAND_STRETCH[1]}")
            pal, mode = palette_of(b)
            if isinstance(mode, tuple):
                lo, hi = mode
            elif lo is None or hi is None:
                lo, hi = 0, 1
            elif mode == "zero":
                mm = max(abs(lo), abs(hi)) or 1
                lo, hi = -mm, mm
            m.addLayer(feats.select(b), {"min": lo, "max": hi if hi > lo else lo + 1e-6, "palette": pal},
                       f"{grp} | {b}", False)
    m.addLayer(Hz["hi"], {"min": 0, "max": 25, "palette": PAL["hi"]}, "Kriged HI", False)
    m.addLayer(Hz["hi_norm"], {"min": 0, "max": 1, "palette": PAL["hi"]}, "HI normalised", False)
    m.addLayer(Hz["risk"], {"min": 0, "max": 4, "palette": PAL["risk"]}, "Risk class", True)
    m.addLayer(Hz["label"], {"min": 0, "max": 1, "palette": PAL["label"]}, "Reference label", True)
    m.addLayer(wells_of(name, SITES[name]), {"color": "FFD700"}, "Monitoring wells", True)
    return m


def band_groups(names):
    """Feature groups used by the figures and the importance summary (also in hazard_ml.py)."""
    g = {"Vegetation index": [b for b in names if b in VI_NAMES],
         "Sentinel-2": [b for b in names if b in S2_BANDS or b == "QA60"],
         "CSTV": [b for b in names if b.startswith("CSTV_")],
         "Rate of change": [b for b in names if b.endswith("_Rate")],
         "Terrain": [b for b in names if b in ("DEM", "Slope_deg", "Aspect_deg", "Curvature")],
         "Climate": [b for b in names if b in ("Temperature", "Humidity", "Moisture", "PrecipJJA_mm_mean")],
         "Soil": [b for b in names if b in ("pH", "Sand_pct_0_5cm", "Silt_pct_0_5cm", "Clay_pct_0_5cm")],
         "Hydrology": [b for b in names if b in ("UpstreamArea_km2", "HAND_m", "TWI")],
         "Land cover": [b for b in names if b.startswith("DW_")],
         "Seasonal": [b for b in names if b.startswith("NDVI_spring") or b.startswith("NDVI_autumn")
                      or b.startswith("NDVI_harm")]}
    return {k: v for k, v in g.items() if v}


def palette_of(b):
    if b.endswith("_Rate") or "minus_summer" in b:
        return PAL["rate"], "zero"
    if b in VI_NAMES or b in ("NDVI_spring", "NDVI_autumn"):
        return PAL["vegetation"], "stretch"
    if b in S2_BANDS:
        return PAL["reflectance"], "stretch"
    if b == "CSTV_valid":
        return ["ffffff", "000000"], (0, 1)
    if b.startswith("CSTV_"):
        return PAL["cstv"], "stretch"
    fixed = {"DEM": (PAL["dem"], "stretch"), "Slope_deg": (PAL["slope"], "stretch"),
             "Aspect_deg": (PAL["aspect"], (0, 360)), "Curvature": (PAL["diverging"], "zero"),
             "Temperature": (PAL["temperature"], "stretch"), "Humidity": (PAL["humidity"], "stretch"),
             "Moisture": (PAL["moisture"], "stretch"), "pH": (PAL["ph"], "stretch"),
             "Sand_pct_0_5cm": (PAL["sand"], "stretch"), "Silt_pct_0_5cm": (PAL["silt"], "stretch"),
             "Clay_pct_0_5cm": (PAL["clay"], "stretch"), "PrecipJJA_mm_mean": (PAL["precip"], "stretch"),
             "UpstreamArea_km2": (PAL["upstream"], "stretch"), "HAND_m": (PAL["hand"], "stretch"),
             "TWI": (PAL["twi"], "stretch"), "NDVI_harm_phase": (PAL["aspect"], (-math.pi, math.pi)),
             "NDVI_harm_amplitude": (PAL["vegetation"], "stretch")}
    if b in fixed:
        return fixed[b]
    if b.startswith("DW_"):
        return ["ffffff", "397D49"], (0, 1) if b != "DW_LandCover" else (0, 8)
    return PAL["reflectance"], "stretch"


def run_site(name):
    cfg = SITES[name]
    region = roi_of(cfg)
    key = name.replace(" ", "_").lower()
    if BASEMAP_EXPORT:
        img, breg = basemap_image(cfg)
        export(img, f"{key}_basemap", breg, BASEMAP_SCALE)
    if BASEMAP_ONLY:
        return None
    F = make_features(region)
    Hz = make_hazard(name, cfg, region)
    print(f"{name}: {F['features'].bandNames().size().getInfo()} predictors; alpha (HI_norm) "
          f"{Hz['alpha'].getInfo():.3f}; HI {Hz['hmin'].getInfo():.2f}-{Hz['hmax'].getInfo():.2f}")
    if EXPORT_MAIN:
        export(F["features"], f"{key}_Features", region)
        export(Hz["label"], f"{key}_Label", region)
        export(Hz["hi"].toFloat(), f"{key}_HI", region)
        export(Hz["hi_norm"].toFloat(), f"{key}_HI_norm", region)
        export(Hz["risk"], f"{key}_risk_class", region)
    if EXPORT_YEARLY:
        for y, img in zip(YEARS, F["yearly"]):
            export(img, f"{key}_FeatureStack_{y}", region)
            export(Hz["label"], f"{key}_Label_{y}", region)
    return (show_map(name, F, Hz, region) if SHOW_MAP else None)


if __name__ == "__main__":
    init()
    for s in SITES_TO_RUN:
        run_site(s)
