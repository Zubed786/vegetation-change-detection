# System Architecture & Technical Specifications

## 1. Modular Component Layout

The codebase is strictly modularized into decoupled, single-responsibility components in `c:\projects\backend`:

```
c:\projects/
├── backend/
│   ├── image_validation.py      # Format, dimensions, bands, CRS validation
│   ├── query_interpretation.py  # 4-intent semantic parsing & rejection
│   ├── change_detection.py      # ChangeFormerV6 inference & tiling
│   ├── ndvi.py                  # Physical Sentinel-2 NDVI calculation
│   ├── vegetation_analysis.py   # Conjunction logic & noise filtering
│   ├── region_analysis.py       # Connected component extraction
│   ├── area_calculation.py      # Projected CRS & GSD area resolution
│   ├── statistics.py            # Global KPI calculations
│   ├── evidence.py              # Verifiable evidence aggregation
│   ├── confidence.py            # Transparent multi-factor confidence scoring
│   ├── explanation.py           # Factual explanation generation
│   ├── cdvqa_adapter.py         # CDVQA dataset interface & catalog
│   ├── visuals.py               # Visual overlays, PNGs, and GeoJSON export
│   └── main.py                  # FastAPI application & route endpoints
├── frontend/
│   ├── index.html               # Clean remote-sensing dashboard
│   ├── styles.css               # Professional dark-mode design system
│   └── app.js                   # Interactive layer switcher, pan/zoom, SVG
├── checkpoints/                 # ChangeFormerV6 pretrained weights
├── data/                        # Sentinel-2 & CDVQA bi-temporal pairs
├── outputs/                     # Job artifacts (GeoTIFFs, PNGs, GeoJSON)
├── tests/                       # Pytest automated test suite
└── docs/                        # Complete technical documentation
```

---

## 2. Dual-Mode Architecture

| Feature | Sentinel-2 Multispectral Mode | CDVQA Optical Mode |
| :--- | :--- | :--- |
| **Primary Input** | Multi-band GeoTIFF with Red ($B4$) and NIR ($B8$) | High-resolution 3-channel RGB image pair |
| **Change Detection** | Pretrained ChangeFormerV6 generic CD | Pretrained ChangeFormerV6 generic CD |
| **Vegetation Classification** | True physical NDVI stage ($B8 - B4$) / ($B8 + B4$) | Explicitly disabled (no NIR band present) |
| **Area Reporting** | True physical hectares and $\text{km}^2$ via projected CRS | Verified metadata GSD ($m^2$) or relative stats |
| **Region Classification** | `VEGETATION_LOSS`, `VEGETATION_GAIN`, `STABLE` | `SURFACE_CHANGE` |
| **Visual Overlays** | Coral Red (Loss) & Emerald Green (Gain) | Amber (Surface change) |

---

## 3. Job Lifecycle & Artifact Convention

Every analysis request receives a unique UUID job identifier (`job_<12hex>`) and dedicated output directory under `outputs/`:

```
outputs/job_86b046e40970/
├── t1_original.png           # RGB preview of acquisition T1
├── t2_original.png           # RGB preview of acquisition T2
├── change_highlighted.png    # Composite overlay (Loss in red, Gain in green)
├── change_map.png            # Binary ChangeFormer detection mask
├── change_mask.tif           # GeoTIFF/TIFF change mask preserving CRS
├── vegetation_t1.tif         # T1 vegetation mask (if multispectral)
├── vegetation_t2.tif         # T2 vegetation mask (if multispectral)
├── vegetation_loss.tif       # Vegetation loss mask (if multispectral)
├── vegetation_gain.tif       # Vegetation gain mask (if multispectral)
├── change_regions.geojson    # Vector GeoJSON polygons with attributes
└── analysis.json             # Full structured response dictionary
```

---

## 4. Security & Performance Guardrails

- **File Type & Readability Validation**: Strict MIME/extension checking (`.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`).
- **Upload Size Limits**: Requests buffered safely via FastAPI file streaming.
- **Filename Sanitization**: Uploaded filenames are stripped of path traversal attempts and assigned clean internal stems (`input_t1.ext`, `input_t2.ext`).
- **No Arbitrary Execution**: The query interpreter and explanation generator do not allow external user input to trigger dynamic code execution.
- **CPU & Laptop Efficiency**:
  - CPU PyTorch execution natively configured.
  - Quadrant tiling ensures manageable memory consumption ($< 1.5\text{ GB}$).
  - Vector polygons are simplified for smooth 60fps rendering in browser canvas/SVG.
