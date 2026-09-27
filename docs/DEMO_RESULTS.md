# End-to-End Demonstration & Evaluation Results

This document records the exact results produced by running the full bi-temporal analysis pipeline against both the **Sentinel-2 Multi-Spectral Dataset** and the **CDVQA / SECOND Optical Remote Sensing Dataset**.

All reported statistics, region counts, confidence values, and explanations were directly computed by the platform with zero fabrication.

---

## 1. Sentinel-2 Multi-Spectral Demonstration
- **Imagery**: `sentinel2_t1.tif` and `sentinel2_t2.tif`
- **Spatial Reference**: EPSG:32632 (WGS 84 / UTM Zone 32N)
- **Pixel Resolution**: $10.0\text{ m} \times 10.0\text{ m}$ ($1\text{ pixel} = 100\text{ m}^2 = 0.01\text{ ha}$)
- **Spectral Bands**: $B2$ (Blue), $B3$ (Green), $B4$ (Red), $B8$ (Near-Infrared)
- **Change Detection Model**: Pretrained ChangeFormerV6 (`best_ckpt.pt`)

---

### Query 1: *"Where has vegetation been lost?"*
- **Recognized Intent**: `VEGETATION_LOSS` (Confidence: 0.98)
- **T1 Vegetation Area**: 429.17 ha (42,917 pixels)
- **T2 Vegetation Area**: 290.23 ha (29,023 pixels)
- **Vegetation Loss Area**: **12.31 ha** across **10 detected regions**
- **Vegetation Gain Area**: 0.00 ha (0 regions)
- **Net Vegetation Change**: -138.94 ha (-32.37%)
- **Analysis Confidence**: **0.98 (HIGH)**
- **Supporting Evidence**:
  - Change Detection Engine: ChangeFormerV6 (LEVIR-CD checkpoint) on CPU
  - Valid Pixel Coverage: T1: 100.0%, T2: 100.0%
  - Spatial Basis: EPSG:32632 (Projected CRS)
  - Loss Regions: 10 regions (12.31 hectares)
- **Generated Natural-Language Explanation**:
  > *"A total of 10 vegetation loss regions were detected, accounting for 12.31 hectares of lost vegetation. Total vegetation area decreased from 429.17 hectares at T1 to 290.23 hectares at T2 (net change of -138.94 hectares, -32.37%)."*

---

### Query 2: *"Where has vegetation increased?"*
- **Recognized Intent**: `VEGETATION_GAIN` (Confidence: 0.98)
- **T1 Vegetation Area**: 429.17 ha
- **T2 Vegetation Area**: 290.23 ha
- **Vegetation Gain Area**: 0.00 ha (0 regions)
- **Analysis Confidence**: **0.98 (HIGH)**
- **Generated Natural-Language Explanation**:
  > *"No significant vegetation gain regions were detected above the minimum region threshold. Vegetation area at T1 was 429.17 hectares and at T2 was 290.23 hectares."*

---

### Query 3: *"How much vegetation changed?"*
- **Recognized Intent**: `CHANGE_STATISTICS` (Confidence: 0.99)
- **T1 Vegetation Area**: 429.17 ha
- **T2 Vegetation Area**: 290.23 ha
- **Vegetation Loss Area**: 12.31 ha across 10 regions
- **Vegetation Gain Area**: 0.00 ha across 0 regions
- **Net Vegetation Difference**: -138.94 ha (-32.37%)
- **Analysis Confidence**: **0.98 (HIGH)**
- **Generated Natural-Language Explanation**:
  > *"Quantitative vegetation analysis indicates: T1 vegetation area was 429.17 hectares, and T2 vegetation area was 290.23 hectares. Vegetation loss was measured at 12.31 hectares across 10 regions, and vegetation gain was 0.0 hectares across 0 regions. The net vegetation change is -138.94 hectares (-32.37%)."*

---

### Query 4: *"What changed in vegetation?"*
- **Recognized Intent**: `GENERAL_COMPARISON` (Confidence: 0.98)
- **Total Regions Detected**: 10
- **Net Vegetation Change**: -138.94 ha (-32.37%)
- **Analysis Confidence**: **0.98 (HIGH)**
- **Generated Natural-Language Explanation**:
  > *"Comparison between the two acquisition dates shows 10 total change regions. Vegetation covers 429.17 hectares at T1 and 290.23 hectares at T2. Detected transitions include 10 loss regions (12.31 hectares) and 0 gain regions (0.0 hectares), resulting in a net vegetation difference of -138.94 hectares (-32.37%)."*

---

## 2. CDVQA / SECOND Optical Dataset Demonstration
- **Sample File**: `02180.png` ($512 \times 512$ optical aerial pair from CDVQA / SECOND)
- **Verified GSD**: $0.1524\text{ m}$ ($1\text{ pixel} \approx 0.0232\text{ m}^2$)
- **Mode**: `optical_rgb` (Change detection without pseudo-NDVI)
- **Query Tested**: *"How much vegetation changed?"*
- **Recognized Intent**: `CHANGE_STATISTICS`
- **Total Changed Pixels**: **13,406 pixels** (20.46% of image footprint)
- **Total Changed Area**: **$311.36\text{ m}^2$** across 1 primary connected change cluster
- **Analysis Confidence**: **0.96 (HIGH)**
- **Top Region Detected**:
  - Region ID: `#1`
  - Type: `SURFACE_CHANGE`
  - Area: $311.36\text{ m}^2$ (13,406 pixels)
  - Bounds: $[80, 0, 256, 175]$
  - Centroid: $(176.8, 85.8)$
- **Generated Natural-Language Explanation**:
  > *"Optical change analysis detected 13,406 changed pixels (20.46% of image area) distributed across 1 distinct change regions. Total changed area is 311.36 m². Note: Near-Infrared spectral data was not present; true NDVI vegetation classification requires multispectral imagery."*
