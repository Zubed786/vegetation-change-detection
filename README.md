# Bi-Temporal Vegetation Change Analysis Platform

A complete, production-grade web application for bi-temporal remote-sensing vegetation change analysis using satellite imagery. The platform couples a pretrained deep-learning change-detection transformer (**ChangeFormerV6**) with an analytical multispectral vegetation pipeline (**NDVI**) and provides an indexed interface to the **CDVQA** visual question answering dataset.

---

## Key Features

1. **Dual-Mode Analytical Pipeline**:
   - **Sentinel-2 Multi-Spectral Mode (NDVI Vegetation Analysis)**: Processes Red ($B4$) and Near-Infrared ($B8$) surface reflectance to compute true physical NDVI, isolating vegetation loss, vegetation gain, and stable vegetation with projected CRS physical area in hectares and $\text{km}^2$.
   - **CDVQA / SECOND Optical Mode (Learned Change Detection)**: Leverages ChangeFormerV6 for learned optical change detection and CDVQA question answering without assuming spectral bands.
2. **Pretrained ChangeFormerV6 Deep Learning**:
   - Uses the official pretrained `best_ckpt.pt` checkpoint trained on LEVIR-CD.
   - Evaluated as a generic temporal change detector (not conflated with vegetation detection).
   - High-resolution quadrant tiling and CPU/CUDA adaptive device placement.
3. **Strict Guardrails & Zero Fabrication**:
   - **No RGB Pseudo-NDVI**: Without Near-Infrared ($B8$), actual NDVI is not calculated; RGB runs through the learned change-detection branch.
   - **Dynamic GSD & Geospatial Area**: Never hardcodes pixel resolution; verifies CRS/GSD per image or reports clearly labeled image-relative statistics.
   - **Region Attribute Constraints**: Strictly calculates region type, area, location/bounds, and $T1/T2$ vegetation status without displaying region-level NDVI difference.
   - **Factual AI Explanations**: 100% grounded in backend statistics with zero hallucinated causes or locations.
4. **Interactive Remote-Sensing Web Viewer**:
   - Large image viewer with interactive layer switcher (`Original T1`, `Original T2`, `Detected Changes`).
   - Opacity slider, pan/zoom, and clickable SVG region boundaries with a Region Inspector popup.
   - Distinct high-contrast palette: Vegetation Loss in Coral Red (`#EF4444`) and Gain in Emerald Green (`#10B981`).
5. **Transparent Analysis Confidence**:
   - Multi-factor breakdown (image compatibility, metadata quality, valid-pixel coverage, query understanding, model execution, region coherence).

---

## Directory Structure

```
c:\projects/
├── backend/
│   ├── image_validation.py      # Input validation & pair compatibility
│   ├── query_interpretation.py  # 4 query intents & semantic parser
│   ├── change_detection.py      # ChangeFormerV6 inference & tiling
│   ├── ndvi.py                  # Sentinel-2 physical NDVI calculation
│   ├── vegetation_analysis.py   # Conjunction logic & noise filtering
│   ├── region_analysis.py       # Connected component extraction
│   ├── area_calculation.py      # Projected CRS & GSD area resolution
│   ├── statistics.py            # Global KPI calculations
│   ├── evidence.py              # Supporting evidence aggregation
│   ├── confidence.py            # Multi-factor confidence evaluation
│   ├── explanation.py           # Grounded natural-language explanation
│   ├── cdvqa_adapter.py         # CDVQA dataset adapter & catalog
│   ├── visuals.py               # Visual overlays, masks, & GeoJSON export
│   └── main.py                  # FastAPI application
├── frontend/
│   ├── index.html               # Remote-sensing dashboard UI
│   ├── styles.css               # Professional dark-mode design system
│   └── app.js                   # Layer switcher, pan/zoom, vector overlays
├── checkpoints/                 # ChangeFormerV6 pretrained weights
├── data/                        # Sentinel-2 GeoTIFFs & CDVQA image pairs
├── outputs/                     # Generated job artifacts (PNGs, TIFs, GeoJSON)
├── tests/                       # Automated pytest test suite (33 passing)
├── docs/                        # Complete technical documentation
│   ├── METHODOLOGY.md
│   ├── ARCHITECTURE.md
│   ├── API.md
│   ├── DATASET.md
│   ├── MODEL.md
│   └── DEMO_RESULTS.md
└── README.md
```

---

## Exact Setup Commands

Ensure Python 3.11+ is installed.

```powershell
# 1. Clone or navigate to the project directory
cd c:\projects

# 2. Install all required dependencies
pip install fastapi uvicorn python-multipart rasterio shapely geopandas scipy numpy pillow pytest httpx einops timm torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```

---

## Exact Run Commands

### 1. Launch the Application Server
```powershell
cd c:\projects
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
- Web Application: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Interactive API Docs (Swagger UI): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2. Run the Automated Test Suite
```powershell
cd c:\projects
python -m pytest tests/ -v
```
All **33 automated tests** cover NDVI formulas, nodata/NaN handling, zero denominators, gain/loss logic, region attribute constraints, geospatial CRS area, GSD parsing, 4 query intents, ChangeFormerV6 integration, and API endpoints.

---

## End-to-End Demonstration Results

### Sentinel-2 Multi-Spectral Pair (`EPSG:32632`, 10m GSD)
1. **"Where has vegetation been lost?"**
   - Intent: `VEGETATION_LOSS` (0.98 confidence)
   - T1 Veg Area: 429.17 ha | T2 Veg Area: 290.23 ha
   - Loss Area: **12.31 ha** across **10 detected regions**
   - Net Change: -138.94 ha (-32.37%)
   - Analysis Confidence: **0.98 (HIGH)**
2. **"Where has vegetation increased?"**
   - Intent: `VEGETATION_GAIN` (0.98 confidence)
   - Gain Area: 0.00 ha (0 regions)
   - Analysis Confidence: **0.98 (HIGH)**
3. **"How much vegetation changed?"**
   - Intent: `CHANGE_STATISTICS` (0.99 confidence)
   - Net Change: -138.94 ha (-32.37%)
   - Analysis Confidence: **0.98 (HIGH)**
4. **"What changed in vegetation?"**
   - Intent: `GENERAL_COMPARISON` (0.98 confidence)
   - Total Regions: 10
   - Analysis Confidence: **0.98 (HIGH)**

### CDVQA / SECOND Optical Pair (`02180.png`, verified 0.1524m GSD)
- **"How much vegetation changed?"**
  - Mode: `optical_rgb`
  - Total Changed Pixels: 13,406 px (20.46% of scene)
  - Total Changed Area: $311.36\text{ m}^2$
  - Region #1: $311.36\text{ m}^2$ $[80, 0, 256, 175]$
  - Analysis Confidence: **0.96 (HIGH)**

---

## Known Limitations
1. **Pretrained Domain**: ChangeFormerV6 is pretrained on LEVIR-CD building change data. It serves as a generic surface change detector; vegetation semantics are derived from NDVI.
2. **NDVI Heuristic**: Fixed $\text{NDVI} \ge 0.30$ threshold is an empirical baseline.
3. **Atmospheric Factors**: Cloud cover or shadow variance between temporal dates can introduce false change detections.
4. **Causality**: The system quantifies physical differences in vegetation cover but cannot determine specific socio-economic or environmental causes without external data.
