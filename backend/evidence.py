"""
Supporting Evidence Module.
Aggregates verifiable analytical indicators backing the query response:
- Number of detected loss and gain regions
- T1 / T2 vegetation areas
- Vegetation loss / gain areas
- Valid-pixel coverage percentage
- Change-detector execution status and model checkpoint
- Spatial resolution and CRS attribution
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from backend.statistics import QuantitativeStatistics
from backend.image_validation import PairValidationResult


@dataclass
class EvidenceSummary:
    items: List[Dict[str, Any]]
    total_change_regions: int
    loss_regions_count: int
    gain_regions_count: int
    valid_pixel_coverage_t1: float
    valid_pixel_coverage_t2: float
    model_execution_info: str
    geospatial_source: str


def compile_evidence(
    stats: QuantitativeStatistics,
    val_res: PairValidationResult,
    model_device: str,
) -> EvidenceSummary:
    """
    Compiles structured supporting evidence based strictly on backend computations.
    """
    items: List[Dict[str, Any]] = []

    # 1. Model execution evidence
    model_info = f"ChangeFormerV6 (LEVIR-CD checkpoint) executed on {model_device}"
    items.append({
        "label": "Change Detection Engine",
        "value": model_info,
        "category": "model",
    })

    # 2. Pixel validity
    t1_cov = val_res.t1_meta.valid_pixel_percentage if val_res.t1_meta else 100.0
    t2_cov = val_res.t2_meta.valid_pixel_percentage if val_res.t2_meta else 100.0
    items.append({
        "label": "Valid Pixel Coverage",
        "value": f"T1: {t1_cov:.1f}%, T2: {t2_cov:.1f}%",
        "category": "data_quality",
    })

    # 3. Geospatial / Resolution reference
    geo_source = stats.calculation_method
    if val_res.t1_meta and val_res.t1_meta.crs_epsg:
        geo_detail = f"EPSG:{val_res.t1_meta.crs_epsg} (Projected CRS)"
    else:
        geo_detail = f"{stats.calculation_method} ({stats.area_unit})"
    items.append({
        "label": "Spatial Metric Basis",
        "value": geo_detail,
        "category": "geospatial",
    })

    # 4. Mode-specific metrics
    unit = stats.area_unit
    if stats.t1_vegetation_pixels is not None:
        items.append({
            "label": "Detected Loss Regions",
            "value": f"{stats.loss_region_count} regions ({stats.vegetation_loss_area or 0} {unit})",
            "category": "vegetation",
        })
        items.append({
            "label": "Detected Gain Regions",
            "value": f"{stats.gain_region_count} regions ({stats.vegetation_gain_area or 0} {unit})",
            "category": "vegetation",
        })
        items.append({
            "label": "T1 Total Vegetation",
            "value": f"{stats.t1_vegetation_area or 0} {unit} ({stats.t1_vegetation_pixels or 0:,} px)",
            "category": "vegetation",
        })
        items.append({
            "label": "T2 Total Vegetation",
            "value": f"{stats.t2_vegetation_area or 0} {unit} ({stats.t2_vegetation_pixels or 0:,} px)",
            "category": "vegetation",
        })
        items.append({
            "label": "Net Vegetation Shift",
            "value": f"{stats.net_vegetation_change_area or 0} {unit} ({stats.percentage_change or 0:+.2f}%)",
            "category": "vegetation",
        })

    if stats.mode == "optical_rgb":
        items.append({
            "label": "Total Surface Change Regions",
            "value": f"{stats.total_regions_count} regions ({stats.total_change_percentage:.2f}% of image)",
            "category": "optical_change",
        })
        items.append({
            "label": "Total Changed Area",
            "value": f"{stats.total_changed_area or 0} {stats.area_unit} ({stats.total_changed_pixels:,} px)",
            "category": "optical_change",
        })
        if stats.t1_vegetation_pixels is not None:
            items.append({
                "label": "Vegetation Metric Basis",
                "value": "Visible-Spectrum Optical Vegetation Index (VARI & ExG). Near-Infrared (B8) spectral band is required for true Sentinel-2 NDVI.",
                "category": "notice",
            })
        else:
            items.append({
                "label": "Vegetation Semantics Note",
                "value": "Optical RGB mode active. Near-Infrared (B8) spectral band is required for true NDVI vegetation gain/loss classification.",
                "category": "notice",
            })

    return EvidenceSummary(
        items=items,
        total_change_regions=stats.total_regions_count,
        loss_regions_count=stats.loss_region_count,
        gain_regions_count=stats.gain_region_count,
        valid_pixel_coverage_t1=round(t1_cov, 1),
        valid_pixel_coverage_t2=round(t2_cov, 1),
        model_execution_info=model_info,
        geospatial_source=geo_detail,
    )
