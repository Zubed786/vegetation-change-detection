"""
FastAPI Main Application for Bi-Temporal Remote Sensing & Vegetation Change Analysis.
Integrates:
- Image Validation
- Query Interpretation
- Pretrained ChangeFormerV6 Change Detection
- NDVI Vegetation Analysis (in Sentinel-2 mode)
- Connected Region Extraction & Area Calculation
- Supporting Evidence Compilation
- Transparent Multi-Factor Confidence Scoring
- Factual Natural-Language Explanation Generation
- Visual Artifact & GeoJSON Export
"""

from dataclasses import asdict
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from backend.image_validation import (
    ImageValidationError,
    inspect_single_image,
    validate_bitemporal_pair,
)
from backend.query_interpretation import interpret_query
from backend.change_detection import ChangeDetector
from backend.ndvi import calculate_ndvi, MultispectralBandsRequiredError
from backend.vegetation_analysis import (
    classify_vegetation_changes,
    classify_visible_vegetation_changes,
)
from backend.region_analysis import extract_change_regions
from backend.area_calculation import resolve_area_context
from backend.statistics import calculate_global_statistics
from backend.evidence import compile_evidence
from backend.confidence import evaluate_analysis_confidence
from backend.explanation import generate_factual_explanation
from backend.cdvqa_adapter import CDVQAAdapter
from backend.visuals import array_to_rgb_preview, export_analysis_artifacts


APP_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = APP_ROOT / "outputs"
DATA_DIR = APP_ROOT / "data"
FRONTEND_DIR = APP_ROOT / "frontend"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Bi-Temporal Vegetation & Remote Sensing Change Analysis API",
    version="1.0.0",
    description="Dual-mode bi-temporal analysis platform with ChangeFormerV6, NDVI, and CDVQA dataset support.",
)

# CORS middleware for open development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs, static frontend files, and sample data
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
if DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


@app.get("/")
def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Bi-Temporal Vegetation Analysis API is active. Access /docs for Swagger UI."}


@app.get("/health")
def health_check():
    """System health check endpoint verifying ChangeFormer and environment readiness."""
    try:
        detector = ChangeDetector.get_instance()
        model_ready = True
        checkpoint_loaded = detector.checkpoint_loaded
        device = str(detector.device)
    except Exception as e:
        model_ready = False
        checkpoint_loaded = False
        device = "unknown"

    cdvqa = CDVQAAdapter.get_instance()

    return {
        "status": "healthy",
        "timestamp": time.time(),
        "changeformer_model_ready": model_ready,
        "checkpoint_loaded": checkpoint_loaded,
        "inference_device": device,
        "cdvqa_dataset_indexed": cdvqa.is_loaded,
        "cdvqa_images_count": len(cdvqa.images),
    }


@app.get("/api/samples")
def get_demo_samples():
    """Returns bundled demo image pairs (both Sentinel-2 and CDVQA) with sample questions."""
    samples = [
        {
            "id": "sentinel2_demo",
            "title": "Sentinel-2 Multi-Spectral Bi-Temporal Pair (10m GSD)",
            "mode": "sentinel2_multispectral",
            "description": "Multi-spectral Sentinel-2 imagery with Red (B4) and NIR (B8) bands. UTM Zone 32N projected CRS (EPSG:32632). Real physical hectares.",
            "t1_file": "data/sentinel2_t1.tif",
            "t2_file": "data/sentinel2_t2.tif",
            "sample_queries": [
                "Where has vegetation been lost?",
                "Where has vegetation increased?",
                "How much vegetation changed?",
                "What changed in vegetation?",
                "What percentage of vegetation was lost?",
            ],
        },
        {
            "id": "cdvqa_demo",
            "title": "CDVQA / SECOND High-Resolution Optical Image Pair",
            "mode": "optical_rgb",
            "description": "Bi-temporal aerial/satellite optical RGB pair from CDVQA / SECOND dataset. Learned surface change detection via ChangeFormerV6.",
            "t1_file": "data/cdvqa_pair1_t1.png",
            "t2_file": "data/cdvqa_pair1_t2.png",
            "sample_queries": [
                "How much vegetation changed?",
                "What changed in vegetation?",
                "Did the areas of non-vegetated ground surface change?",
                "Have the areas of trees changed?",
                "What is the percentage of changed areas?",
            ],
        },
    ]
    return {"samples": samples}


@app.get("/api/cdvqa/samples")
def get_cdvqa_dataset_samples(limit: int = 10):
    """Returns samples directly from the cloned CDVQA dataset."""
    adapter = CDVQAAdapter.get_instance()
    catalog = adapter.get_sample_catalogue(limit=limit)
    return {
        "dataset_loaded": adapter.is_loaded,
        "total_images": len(adapter.images),
        "samples": [asdict(s) for s in catalog],
    }


def read_image_data(file_path: str) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[float, ...]], Optional[int]]:
    """
    Reads image raster.
    Returns:
        full_array (C, H, W) or (H, W, C),
        b4_red (H, W) or None,
        b8_nir (H, W) or None,
        transform,
        crs_epsg
    """
    ext = Path(file_path).suffix.lower()
    if HAS_RASTERIO and ext in {".tif", ".tiff"}:
        try:
            with rasterio.open(file_path) as src:
                data = src.read()  # (C, H, W)
                transform = tuple(src.transform) if src.transform else None
                crs_epsg = src.crs.to_epsg() if src.crs else None

                # Find Red and NIR bands if multispectral
                num_bands = src.count
                b4_red = None
                b8_nir = None
                descriptions = [d.upper() if d else "" for d in (src.descriptions or [])]

                for i, desc in enumerate(descriptions):
                    if "B4" in desc or "RED" in desc:
                        b4_red = data[i]
                    elif "B8" in desc or "NIR" in desc:
                        b8_nir = data[i]

                if b4_red is None or b8_nir is None:
                    if num_bands == 4:
                        b4_red = data[2]
                        b8_nir = data[3]
                    elif num_bands in {12, 13}:
                        b4_red = data[3]
                        b8_nir = data[7]
                    elif num_bands == 2:
                        b4_red = data[0]
                        b8_nir = data[1]

                return data, b4_red, b8_nir, transform, crs_epsg
        except Exception:
            pass

    # PIL fallback
    with Image.open(file_path) as img:
        arr = np.array(img)
        return arr, None, None, None, None


@app.post("/analyze")
async def analyze_bitemporal_pair(
    image_t1: UploadFile = File(..., description="Satellite image at time T1"),
    image_t2: UploadFile = File(..., description="Satellite image at time T2"),
    query: str = Form(..., description="Natural-language question about vegetation/change"),
    ndvi_threshold: float = Form(0.30, description="NDVI threshold for vegetation (default 0.30)"),
    min_region_size: int = Form(16, description="Minimum connected region pixel size (default 16)"),
):
    """
    Main bi-temporal analysis endpoint.
    Orchestrates validation, query interpretation, ChangeFormerV6, NDVI analysis,
    region extraction, area calculation, evidence compilation, and factual explanation.
    """
    start_time = time.time()
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    job_dir = OUTPUTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    processing_steps = []

    def record_step(step_name: str, duration_sec: float):
        processing_steps.append({"step": step_name, "duration_seconds": round(duration_sec, 3)})

    # Save uploaded files temporarily in job directory
    step_start = time.time()
    t1_ext = Path(image_t1.filename).suffix.lower() or ".png"
    t2_ext = Path(image_t2.filename).suffix.lower() or ".png"
    t1_temp_path = job_dir / f"input_t1{t1_ext}"
    t2_temp_path = job_dir / f"input_t2{t2_ext}"

    with open(t1_temp_path, "wb") as f:
        shutil.copyfileobj(image_t1.file, f)
    with open(t2_temp_path, "wb") as f:
        shutil.copyfileobj(image_t2.file, f)

    # 1. Image Validation
    val_res = validate_bitemporal_pair(str(t1_temp_path), str(t2_temp_path))
    record_step("Validating images...", time.time() - step_start)

    if not val_res.is_compatible:
        # Clean up files on hard rejection
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Image Pair Incompatible",
                "validation_errors": val_res.errors,
                "warnings": val_res.warnings,
            },
        )

    # 2. Query Interpretation
    step_start = time.time()
    query_interp = interpret_query(query)
    record_step("Interpreting query...", time.time() - step_start)

    if not query_interp.is_supported:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Unsupported Query Intent",
                "reason": query_interp.rejection_reason,
                "supported_intents": [
                    "VEGETATION_LOSS",
                    "VEGETATION_GAIN",
                    "CHANGE_STATISTICS",
                    "GENERAL_COMPARISON",
                ],
            },
        )

    # 3. Read image data & generate visual RGB previews
    step_start = time.time()
    t1_raw, t1_b4, t1_b8, transform, crs_epsg = read_image_data(str(t1_temp_path))
    t2_raw, t2_b4, t2_b8, _, _ = read_image_data(str(t2_temp_path))

    is_multispectral = val_res.mode == "sentinel2_multispectral"
    t1_rgb = array_to_rgb_preview(t1_raw, is_multispectral=is_multispectral)
    t2_rgb = array_to_rgb_preview(t2_raw, is_multispectral=is_multispectral)

    # 4. Change Detection (ChangeFormerV6)
    step_start = time.time()
    detector = ChangeDetector.get_instance()
    # ChangeFormer operates on RGB inputs (or first 3 channels normalized)
    cd_result = detector.detect_changes(
        t1_img=t1_rgb,
        t2_img=t2_rgb,
        threshold=0.50,
        min_region_size=min_region_size,
    )
    record_step("Running ChangeFormerV6...", time.time() - step_start)

    # 5. Vegetation & Change Analysis
    t1_ndvi_res = None
    t2_ndvi_res = None
    veg_masks = None
    loss_mask = None
    gain_mask = None
    t1_veg_mask = None
    t2_veg_mask = None

    if is_multispectral and t1_b4 is not None and t1_b8 is not None and t2_b4 is not None and t2_b8 is not None:
        step_start = time.time()
        t1_ndvi_res = calculate_ndvi(t1_b4, t1_b8)
        t2_ndvi_res = calculate_ndvi(t2_b4, t2_b8)
        record_step("Calculating NDVI...", time.time() - step_start)

        step_start = time.time()
        veg_masks = classify_vegetation_changes(
            t1_ndvi=t1_ndvi_res.ndvi,
            t2_ndvi=t2_ndvi_res.ndvi,
            change_mask=cd_result.change_mask,
            ndvi_threshold=ndvi_threshold,
            min_region_size=min_region_size,
        )
        loss_mask = veg_masks.veg_loss_mask
        gain_mask = veg_masks.veg_gain_mask
        t1_veg_mask = veg_masks.t1_veg_mask
        t2_veg_mask = veg_masks.t2_veg_mask
        record_step("Detecting vegetation gain/loss...", time.time() - step_start)
    else:
        # Optical RGB Mode: Visible Spectrum Vegetation Analysis (VARI, ExG, GLI)
        step_start = time.time()
        veg_masks = classify_visible_vegetation_changes(
            t1_rgb=t1_rgb,
            t2_rgb=t2_rgb,
            change_mask=cd_result.change_mask,
            min_region_size=min_region_size,
        )
        loss_mask = veg_masks.veg_loss_mask
        gain_mask = veg_masks.veg_gain_mask
        t1_veg_mask = veg_masks.t1_veg_mask
        t2_veg_mask = veg_masks.t2_veg_mask
        record_step("Classifying visible vegetation...", time.time() - step_start)

    # 6. Area Context & Resolution
    cdvqa_meta = None
    # Check if this matches a known CDVQA image file
    orig_fn = Path(image_t1.filename).name
    adapter = CDVQAAdapter.get_instance()
    match_img = adapter.find_image_metadata(orig_fn)
    meta_rx = match_img.get("res_x") if match_img else None
    meta_ry = match_img.get("res_y") if match_img else None

    area_ctx = resolve_area_context(
        crs_epsg=crs_epsg,
        crs_wkt=val_res.t1_meta.crs_wkt if val_res.t1_meta else None,
        transform=transform,
        bounds=val_res.t1_meta.bounds if val_res.t1_meta else None,
        metadata_res_x=meta_rx,
        metadata_res_y=meta_ry,
    )
    if not is_multispectral and area_ctx.area_calculation_method == "image_relative":
        area_ctx.area_calculation_method = "Visible-Spectrum Optical Vegetation Index (VARI & ExG)"

    # 7. Region Extraction
    step_start = time.time()
    regions = extract_change_regions(
        loss_mask=loss_mask,
        gain_mask=gain_mask,
        generic_change_mask=cd_result.change_mask,
        pixel_area_ha=area_ctx.pixel_area_ha,
        pixel_area_m2=area_ctx.pixel_area_m2,
        min_region_size=min_region_size,
    )
    record_step("Extracting vegetation regions...", time.time() - step_start)

    loss_region_count = sum(1 for r in regions if r.region_type == "VEGETATION_LOSS")
    gain_region_count = sum(1 for r in regions if r.region_type == "VEGETATION_GAIN")

    # 8. Statistics Calculation
    step_start = time.time()
    stats = calculate_global_statistics(
        mode=val_res.mode,
        total_pixels=cd_result.total_pixels,
        area_ctx=area_ctx,
        t1_veg_pixels=veg_masks.t1_veg_pixels if veg_masks else None,
        t2_veg_pixels=veg_masks.t2_veg_pixels if veg_masks else None,
        loss_pixels=veg_masks.loss_pixels if veg_masks else None,
        gain_pixels=veg_masks.gain_pixels if veg_masks else None,
        loss_region_count=loss_region_count,
        gain_region_count=gain_region_count,
        total_changed_pixels=cd_result.changed_pixels,
        total_regions_count=len(regions),
    )
    record_step("Calculating statistics...", time.time() - step_start)

    # 9. Evidence & Confidence Evaluation
    evidence_res = compile_evidence(
        stats=stats,
        val_res=val_res,
        model_device=cd_result.device_used,
    )
    confidence_res = evaluate_analysis_confidence(
        val_res=val_res,
        query_interp=query_interp,
        stats=stats,
        model_executed_successfully=True,
    )

    # 10. Natural-Language Explanation
    step_start = time.time()
    explanation_text = generate_factual_explanation(
        query_interp=query_interp,
        stats=stats,
    )
    record_step("Generating explanation...", time.time() - step_start)

    # 11. Visual Artifacts Export
    step_start = time.time()
    analysis_dict = {
        "job_id": job_id,
        "mode": val_res.mode,
        "intent": asdict(query_interp),
        "statistics": asdict(stats),
        "evidence": asdict(evidence_res),
        "confidence": asdict(confidence_res),
        "explanation": explanation_text,
    }
    visual_files = export_analysis_artifacts(
        output_dir=job_dir,
        t1_rgb=t1_rgb,
        t2_rgb=t2_rgb,
        change_mask=cd_result.change_mask,
        loss_mask=loss_mask,
        gain_mask=gain_mask,
        t1_veg_mask=t1_veg_mask,
        t2_veg_mask=t2_veg_mask,
        regions=regions,
        analysis_dict=analysis_dict,
        crs_epsg=crs_epsg,
        transform=transform,
    )
    record_step("Preparing visual results...", time.time() - step_start)

    total_processing_duration = round(time.time() - start_time, 2)

    # Prioritize regions matching query intent first, then by area
    target_intent = query_interp.intent if query_interp else None
    if target_intent == "VEGETATION_LOSS":
        sorted_regions = sorted(regions, key=lambda r: (r.region_type != "VEGETATION_LOSS", -r.area_pixels))
    elif target_intent == "VEGETATION_GAIN":
        sorted_regions = sorted(regions, key=lambda r: (r.region_type != "VEGETATION_GAIN", -r.area_pixels))
    else:
        # Prioritize vegetation changes (loss & gain) over generic surface changes
        sorted_regions = sorted(regions, key=lambda r: (r.region_type == "SURFACE_CHANGE", -r.area_pixels))

    # Format region summaries for JSON response (without bulky contours)
    region_summaries = [
        {
            "region_id": r.region_id,
            "region_type": r.region_type,
            "area_pixels": r.area_pixels,
            "area_physical": r.area_physical,
            "area_unit": r.area_unit,
            "bounds": r.bounds,
            "centroid": r.centroid,
            "t1_vegetation_status": r.t1_vegetation_status,
            "t2_vegetation_status": r.t2_vegetation_status,
        }
        for r in sorted_regions[:100]  # Return top 100 in JSON; full vector dataset is in GeoJSON
    ]

    response_payload = {
        "job_id": job_id,
        "mode": val_res.mode,
        "query": query,
        "intent": asdict(query_interp),
        "statistics": asdict(stats),
        "explanation": explanation_text,
        "evidence": asdict(evidence_res),
        "confidence": asdict(confidence_res),
        "regions": region_summaries,
        "total_regions_count": len(regions),
        "visual_outputs": visual_files,
        "processing": {
            "total_duration_seconds": total_processing_duration,
            "steps": processing_steps,
        },
    }

    return JSONResponse(content=response_payload)
