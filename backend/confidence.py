"""
Analysis Confidence Indicator Module.
Does not claim generic empirical model accuracy.
Instead, calculates a transparent, factor-based analysis confidence score
measuring the fidelity of the current processing pipeline:
- Image compatibility (dimensions, format, bit depth)
- Metadata quality (CRS, geotransform, GSD verification)
- Valid-pixel coverage (absence of nodata/black voids)
- Query interpretation confidence
- Signal quality / region coherence
- ChangeFormer execution status
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from backend.image_validation import PairValidationResult
from backend.query_interpretation import InterpretationResult
from backend.statistics import QuantitativeStatistics


@dataclass
class FactorScore:
    name: str
    score: float          # In range [0.0, 1.0]
    weight: float
    description: str


@dataclass
class ConfidenceReport:
    overall_confidence: float        # In range [0.0, 1.0]
    confidence_tier: str             # "HIGH", "MEDIUM", "LOW"
    factors: List[Dict[str, Any]]
    explanation: str


def evaluate_analysis_confidence(
    val_res: PairValidationResult,
    query_interp: InterpretationResult,
    stats: QuantitativeStatistics,
    model_executed_successfully: bool = True,
) -> ConfidenceReport:
    """
    Computes transparent multi-factor analysis confidence and verbal explanation.
    """
    factors: List[FactorScore] = []

    # 1. Image Compatibility Factor (Weight: 0.20)
    if val_res.is_compatible:
        if len(val_res.warnings) == 0:
            compat_score = 1.0
            compat_desc = f"Exact dimension match ({val_res.width}x{val_res.height}) with zero compatibility warnings."
        else:
            compat_score = 0.85
            compat_desc = f"Dimensions match ({val_res.width}x{val_res.height}), with minor advisory warnings: {'; '.join(val_res.warnings)}."
    else:
        compat_score = 0.20
        compat_desc = f"Incompatibility errors detected: {'; '.join(val_res.errors)}."
    factors.append(FactorScore("Image Compatibility", compat_score, 0.20, compat_desc))

    # 2. Metadata & Geospatial Quality (Weight: 0.20)
    if val_res.t1_meta and val_res.t1_meta.has_geospatial:
        meta_score = 0.98
        meta_desc = f"Authoritative projected CRS (EPSG:{val_res.t1_meta.crs_epsg or 'WKT'}) and affine transform present."
    elif stats.calculation_method == "verified_metadata_gsd":
        meta_score = 0.88
        meta_desc = "Verified ground sample distance (GSD) parsed directly from dataset metadata."
    else:
        meta_score = 0.70
        meta_desc = "Standard image metadata without projected CRS. Image-relative metrics used transparently."
    factors.append(FactorScore("Metadata Quality", meta_score, 0.20, meta_desc))

    # 3. Valid Pixel Coverage (Weight: 0.15)
    t1_pct = val_res.t1_meta.valid_pixel_percentage if val_res.t1_meta else 100.0
    t2_pct = val_res.t2_meta.valid_pixel_percentage if val_res.t2_meta else 100.0
    min_pct = min(t1_pct, t2_pct)
    if min_pct >= 95.0:
        cov_score = 1.0
        cov_desc = f"Optimal pixel validity (T1: {t1_pct:.1f}%, T2: {t2_pct:.1f}% non-nodata)."
    elif min_pct >= 80.0:
        cov_score = 0.85
        cov_desc = f"Good pixel validity (lowest coverage: {min_pct:.1f}%)."
    else:
        cov_score = max(0.40, min_pct / 100.0)
        cov_desc = f"Reduced pixel coverage (lowest coverage: {min_pct:.1f}% contains nodata/voids)."
    factors.append(FactorScore("Valid Pixel Coverage", cov_score, 0.15, cov_desc))

    # 4. Query Interpretation Confidence (Weight: 0.15)
    q_score = query_interp.confidence if query_interp.is_supported else 0.20
    q_desc = f"Query intent '{query_interp.intent}' recognized with {q_score * 100:.0f}% confidence via {query_interp.method}."
    factors.append(FactorScore("Query Understanding", q_score, 0.15, q_desc))

    # 5. Model Execution Integrity (Weight: 0.15)
    if model_executed_successfully:
        m_score = 1.0
        m_desc = "Pretrained ChangeFormerV6 encoder-decoder executed completely with no runtime errors."
    else:
        m_score = 0.0
        m_desc = "Change detection model execution failed or was interrupted."
    factors.append(FactorScore("Model Execution", m_score, 0.15, m_desc))

    # 6. Signal-to-Noise / Region Coherence (Weight: 0.15)
    if stats.total_regions_count > 0:
        snr_score = 0.90
        snr_desc = f"{stats.total_regions_count} cohesive change regions extracted after morphological noise filtering."
    else:
        snr_score = 0.85
        snr_desc = "Zero significant change regions detected above the minimum region threshold."
    factors.append(FactorScore("Region Coherence", snr_score, 0.15, snr_desc))

    # Compute weighted average
    total_weight = sum(f.weight for f in factors)
    overall_conf = sum(f.score * f.weight for f in factors) / total_weight
    overall_conf = round(min(0.99, max(0.10, overall_conf)), 2)

    if overall_conf >= 0.85:
        tier = "HIGH"
    elif overall_conf >= 0.65:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    # Verbal explanation
    top_strengths = [f.name for f in factors if f.score >= 0.90]
    top_cautions = [f.name for f in factors if f.score < 0.80]

    rationale_parts = [
        f"Analysis confidence is assessed at {overall_conf * 100:.0f}% ({tier} confidence)."
    ]
    if top_strengths:
        rationale_parts.append(f"Strongest supporting factors: {', '.join(top_strengths)}.")
    if top_cautions:
        rationale_parts.append(f"Considerations affecting confidence: {', '.join(top_cautions)}.")

    explanation_str = " ".join(rationale_parts)

    factor_dicts = [
        {
            "factor": f.name,
            "score": round(f.score, 2),
            "weight": f.weight,
            "description": f.description,
        }
        for f in factors
    ]

    return ConfidenceReport(
        overall_confidence=overall_conf,
        confidence_tier=tier,
        factors=factor_dicts,
        explanation=explanation_str,
    )
