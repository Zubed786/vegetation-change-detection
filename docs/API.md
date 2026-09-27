# REST API Specification

The platform exposes a high-performance REST API built on **FastAPI**. Interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`.

---

## Endpoints

### 1. Health Check
```http
GET /health
```
Checks backend readiness, ChangeFormerV6 model status, checkpoint presence, inference device, and CDVQA dataset index.

#### Response (200 OK)
```json
{
  "status": "healthy",
  "timestamp": 1789046179.47,
  "changeformer_model_ready": true,
  "checkpoint_loaded": true,
  "inference_device": "cpu",
  "cdvqa_dataset_indexed": true,
  "cdvqa_images_count": 6400
}
```

---

### 2. Available Demo Samples
```http
GET /api/samples
```
Returns curated demo pairs (Sentinel-2 multispectral and CDVQA optical) with sample questions.

---

### 3. CDVQA Dataset Catalog
```http
GET /api/cdvqa/samples?limit=10
```
Returns image entries, verified resolutions, and diverse questions directly from the indexed CDVQA dataset.

---

### 4. Bi-Temporal Analysis
```http
POST /analyze
```
Processes a bi-temporal image pair and natural-language query.

#### Request (multipart/form-data)
| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `image_t1` | File | Yes | Satellite image at acquisition time T1 |
| `image_t2` | File | Yes | Satellite image at acquisition time T2 |
| `query` | Text | Yes | Natural-language query regarding change |
| `ndvi_threshold`| Float| No (default: 0.30) | Vegetation classification threshold |
| `min_region_size`| Int | No (default: 16) | Minimum connected pixel size to filter noise |

#### Response (200 OK)
```json
{
  "job_id": "job_86b046e40970",
  "mode": "sentinel2_multispectral",
  "query": "Where has vegetation been lost?",
  "intent": {
    "intent": "VEGETATION_LOSS",
    "confidence": 0.98,
    "method": "rule_based"
  },
  "statistics": {
    "mode": "sentinel2_multispectral",
    "total_pixels": 65536,
    "area_unit": "hectares",
    "calculation_method": "projected_crs",
    "t1_vegetation_area": 429.17,
    "t2_vegetation_area": 290.23,
    "vegetation_loss_area": 12.31,
    "vegetation_gain_area": 0.0,
    "net_vegetation_change_area": -138.94,
    "percentage_change": -32.37,
    "loss_region_count": 10,
    "gain_region_count": 0,
    "total_changed_pixels": 1231,
    "total_changed_area": 12.31,
    "total_change_percentage": 1.88,
    "total_regions_count": 10
  },
  "explanation": "A total of 10 vegetation loss regions were detected, accounting for 12.31 hectares of lost vegetation. Total vegetation area decreased from 429.17 hectares at T1 to 290.23 hectares at T2 (net change of -138.94 hectares, -32.37%).",
  "evidence": {
    "total_change_regions": 10,
    "loss_regions_count": 10,
    "gain_regions_count": 0,
    "valid_pixel_coverage_t1": 100.0,
    "valid_pixel_coverage_t2": 100.0,
    "model_execution_info": "ChangeFormerV6 (LEVIR-CD checkpoint) executed on cpu",
    "geospatial_source": "EPSG:32632 (Projected CRS)"
  },
  "confidence": {
    "overall_confidence": 0.98,
    "confidence_tier": "HIGH",
    "explanation": "Analysis confidence is assessed at 98% (HIGH confidence). Strongest supporting factors: Image Compatibility, Metadata Quality, Valid Pixel Coverage, Model Execution."
  },
  "regions": [
    {
      "region_id": 1,
      "region_type": "VEGETATION_LOSS",
      "area_pixels": 250,
      "area_physical": 2.5,
      "area_unit": "ha",
      "bounds": [140, 30, 220, 110],
      "centroid": [180.0, 70.0],
      "t1_vegetation_status": "Vegetation",
      "t2_vegetation_status": "Non-Vegetation"
    }
  ],
  "visual_outputs": {
    "t1_original": "outputs/job_86b046e40970/t1_original.png",
    "t2_original": "outputs/job_86b046e40970/t2_original.png",
    "change_highlighted": "outputs/job_86b046e40970/change_highlighted.png",
    "change_map_png": "outputs/job_86b046e40970/change_map.png",
    "change_mask": "outputs/job_86b046e40970/change_mask.tif",
    "vegetation_loss": "outputs/job_86b046e40970/vegetation_loss.tif",
    "change_regions_geojson": "outputs/job_86b046e40970/change_regions.geojson",
    "analysis_json": "outputs/job_86b046e40970/analysis.json"
  },
  "processing": {
    "total_duration_seconds": 2.14
  }
}
```

---

## Error Handling

### 1. Incompatible Pair (HTTP 400)
```json
{
  "detail": {
    "error": "Image Pair Incompatible",
    "validation_errors": [
      "Dimension mismatch between T1 (256x256) and T2 (300x300). Bi-temporal pairs must represent the exact same pixel dimensions. Arbitrary resizing or cropping is rejected."
    ]
  }
}
```

### 2. Unsupported Query Intent (HTTP 400)
```json
{
  "detail": {
    "error": "Unsupported Query Intent",
    "reason": "Could not match query 'What is the capital of France?' to any supported intent.",
    "supported_intents": [
      "VEGETATION_LOSS",
      "VEGETATION_GAIN",
      "CHANGE_STATISTICS",
      "GENERAL_COMPARISON"
    ]
  }
}
```
