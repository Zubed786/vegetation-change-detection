# Analytical Methodology & Algorithmic Pipeline

## 1. End-to-End Workflow Architecture

```
                    ┌──────────────────────────────────────┐
                    │ T1 Image + T2 Image + Natural Query │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │       Image & Query Validation       │
                    │   (Dimension match, Bands, CRS)      │
                    └──────────────────┬───────────────────┘
                                       │
                    ┌──────────────────┴───────────────────┐
                    ▼                                      ▼
     [ Sentinel-2 Multispectral Mode ]          [ CDVQA / Optical Mode ]
                    │                                      │
                    ├──────────────────┬───────────────────┤
                    │                  │                   │
                    ▼                  ▼                   │
       ┌────────────────────────┐┌───────────────┐         │
       │ Pretrained ChangeFormer││  NDVI Module  │         │
       │  (Generic CD Signal)   ││(B8-B4)/(B8+B4)│         │
       └────────────┬───────────┘└───────┬───────┘         │
                    │                    │                 │
                    ▼                    ▼                 │
       ┌─────────────────────────────────────────┐         │
       │   Conjunction & Vegetation Transition   │         │
       │  (Loss: Veg->NonVeg; Gain: NonVeg->Veg) │         │
       └────────────────────┬────────────────────┘         │
                            │                              │
                            └──────────────┬───────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────┐
                    │      Connected Region Extraction     │
                    │     (Morphology, Noise Filtering)    │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │    Geospatial Area & KPI Engine      │
                    │  (Projected CRS ha / Verified GSD)   │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │ Supporting Evidence & Confidence Eval│
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │  Grounded Natural-Language Answer    │
                    └──────────────────────────────────────┘
```

---

## 2. Step-by-Step Processing Stages

### Stage 1: Input Validation
- **Dimension Matching**: Ensures $T1$ and $T2$ pixel dimensions $(W \times H)$ match identically. Arbitrary resizing, cropping, or warping is rejected.
- **Spectral Band Identification**:
  - In **Sentinel-2 Mode**: Checks for Red ($B4$) and Near-Infrared ($B8$) bands.
  - In **CDVQA Mode**: Checks for 3-band RGB optical channels.
- **Pixel Validity**: Rejects images with $< 40\%$ valid data (preventing corrupt or all-nodata runs).

### Stage 2: Query Interpretation
- Queries are categorized into one of 4 supported intents:
  1. `VEGETATION_LOSS`
  2. `VEGETATION_GAIN`
  3. `CHANGE_STATISTICS`
  4. `GENERAL_COMPARISON`
- Employs weighted regex keyword matching with CDVQA question template support.
- Unsupported or out-of-domain queries are rejected with actionable suggestions.

### Stage 3: Generic Learned Change Detection (ChangeFormerV6)
- Pretrained ChangeFormerV6 (trained on LEVIR-CD) is executed on the normalized optical channels of $T1$ and $T2$.
- Produces a binary surface change mask $M_{\text{change}} \in \{0, 1\}$.
- **Crucial Rule**: The output of ChangeFormer is explicitly labeled as *generic temporal change*, not vegetation change.

### Stage 4: NDVI Analytical Stage (Sentinel-2 Mode Only)
- Computes Normalized Difference Vegetation Index:
  $$\text{NDVI} = \frac{B8 - B4}{B8 + B4}$$
- Dynamic reflectance scaling detects 0–10000 surface reflectance vs 0–255 vs normalized floats.
- Rigorous guardrails protect against zero denominators, NaN, inf, and negative reflectances:
  $$\text{NDVI}(x, y) = 0.0 \quad \text{if } (B8 + B4) \le 10^{-6}$$
- Clamped strictly to theoretical limits $[-1.0, 1.0]$.
- **Rule**: If NIR ($B8$) is missing, no pseudo-NDVI is computed.

### Stage 5: Vegetation Masking & Gain/Loss Conjunction
- Vegetation threshold applied:
  $$V_{T1} = (\text{NDVI}_{T1} \ge 0.30), \quad V_{T2} = (\text{NDVI}_{T2} \ge 0.30)$$
- Conjunction with ChangeFormer change mask $M_{\text{change}}$:
  - **Vegetation Loss**: $V_{T1} \land (\neg V_{T2}) \land M_{\text{change}}$
  - **Vegetation Gain**: $(\neg V_{T1}) \land V_{T2} \land M_{\text{change}}$
  - **Stable Vegetation**: $V_{T1} \land V_{T2} \land (\neg M_{\text{change}})$
  - **Other Change**: $M_{\text{change}} \land \neg(\text{Loss}) \land \neg(\text{Gain})$
- Noise filtering: Connected components smaller than a configurable threshold (default 16 pixels) are eliminated.

### Stage 6: Region-Level Analysis
- Connected components labeled with 8-connectivity.
- For each significant region:
  - `region_id`: Unique integer
  - `region_type`: `VEGETATION_LOSS`, `VEGETATION_GAIN`, or `SURFACE_CHANGE`
  - `area_pixels`: Integer count of member pixels
  - `bounds`: Bounding box $[min_x, min_y, max_x, max_y]$
  - `centroid`: Coordinate $(c_x, c_y)$
  - `t1_vegetation_status`: `"Vegetation"` or `"Non-Vegetation"`
  - `t2_vegetation_status`: `"Vegetation"` or `"Non-Vegetation"`
- **Strict Prohibition**: Does NOT calculate or display $T1$ mean NDVI, $T2$ mean NDVI, or NDVI difference.

### Stage 7: Geospatial & Relative Area Calculation
- **Projected CRS**: When CRS is projected (e.g. UTM meters), computes physical ground area:
  $$\text{Area (ha)} = \frac{\text{Pixel Count} \times (\Delta x \cdot \Delta y)}{10,000}$$
- **Geographic CRS**: If in degrees (EPSG:4326), reprojects centroid to meters before computing physical area.
- **CDVQA Metadata**: When metadata specifies `res_x: '.1524m'`, converts directly to $m^2$.
- **Image-Relative Fallback**: If geospatial metadata is unverified, reports pixel counts and percentages. Does not invent hectare values.

### Stage 8: Supporting Evidence & Transparent Confidence
- Compiles 6 verifiable factors: Image compatibility, metadata quality, valid-pixel coverage, query confidence, model execution, and region coherence.
- Generates a weighted confidence score (0.0 to 1.0) with an explicit verbal rationale.

### Stage 9: Grounded Explanation
- Generates a concise natural-language response strictly from the computed numbers.
- Zero hallucinations: no fabricated numbers, no invented geographic places, no assumed environmental causes.

---

## 3. Documented Limitations

1. **Pretrained Training Domain Discrepancy**: ChangeFormerV6 is pretrained on LEVIR-CD building changes. While its learned attention maps transfer well to surface disturbances, building features may differ from agricultural or natural clearings.
2. **NDVI Thresholding Heuristic**: The fixed threshold ($\text{NDVI} \ge 0.30$) is an empirical heuristic. Densely vegetated canopies vs. sparse scrubland or seasonal phenology may warrant different local thresholds.
3. **Cloud, Shadow & Atmospheric Effects**: Cloud cover, cloud shadows, or haze between acquisition dates can cause false change detections or temporary NDVI drops.
4. **False Positives & Negatives**: Spectral reflectance changes caused by soil moisture or seasonal drying can be flagged as change by optical detectors.
5. **Causality Attribution**: The system detects *bi-temporal differences* in vegetation cover; it cannot determine whether loss was caused by drought, logging, wildfire, urban expansion, or normal agricultural harvesting without external ancillary data.
