"""
Natural-Language Explanation Module.
Generates concise, factual, grounded explanations strictly from structured backend statistics.
Zero hallucinations: does NOT invent numbers, geographical place names, environmental causes,
or unsupported speculations.
Provides a deterministic template-based explanation when LLM is unavailable.
"""

from typing import Any, Dict, Optional
from backend.statistics import QuantitativeStatistics
from backend.query_interpretation import InterpretationResult, QueryIntent


def generate_factual_explanation(
    query_interp: InterpretationResult,
    stats: QuantitativeStatistics,
    llm_callable: Optional[Any] = None,
) -> str:
    """
    Generates a natural-language answer to the user's query strictly using backend numbers.
    """
    # Deterministic factual generator
    intent = query_interp.intent
    unit = stats.area_unit

    if stats.mode == "sentinel2_multispectral":
        t1_area = stats.t1_vegetation_area or 0.0
        t2_area = stats.t2_vegetation_area or 0.0
        loss_area = stats.vegetation_loss_area or 0.0
        gain_area = stats.vegetation_gain_area or 0.0
        net_area = stats.net_vegetation_change_area or 0.0
        pct_change = stats.percentage_change or 0.0
        loss_count = stats.loss_region_count
        gain_count = stats.gain_region_count

        if intent == QueryIntent.VEGETATION_LOSS.value:
            if loss_count > 0:
                text = (
                    f"A total of {loss_count} vegetation loss regions were detected, accounting for "
                    f"{loss_area} {unit} of lost vegetation. Total vegetation area decreased from "
                    f"{t1_area} {unit} at T1 to {t2_area} {unit} at T2 (net change of {net_area} {unit}, "
                    f"{pct_change:+.2f}%)."
                )
            else:
                text = (
                    f"No significant vegetation loss regions were detected above the minimum region threshold. "
                    f"Vegetation remained stable or increased from {t1_area} {unit} to {t2_area} {unit}."
                )

        elif intent == QueryIntent.VEGETATION_GAIN.value:
            if gain_count > 0:
                text = (
                    f"A total of {gain_count} vegetation gain regions were identified, representing "
                    f"{gain_area} {unit} of newly emerged or recovered vegetation. Total vegetation area shifted "
                    f"from {t1_area} {unit} at T1 to {t2_area} {unit} at T2."
                )
            else:
                text = (
                    f"No significant vegetation gain regions were detected above the minimum region threshold. "
                    f"Vegetation area at T1 was {t1_area} {unit} and at T2 was {t2_area} {unit}."
                )

        elif intent == QueryIntent.CHANGE_STATISTICS.value:
            text = (
                f"Quantitative vegetation analysis indicates: T1 vegetation area was {t1_area} {unit}, "
                f"and T2 vegetation area was {t2_area} {unit}. Vegetation loss was measured at {loss_area} {unit} "
                f"across {loss_count} regions, and vegetation gain was {gain_area} {unit} across {gain_count} regions. "
                f"The net vegetation change is {net_area} {unit} ({pct_change:+.2f}%)."
            )

        else:  # GENERAL_COMPARISON or default
            text = (
                f"Comparison between the two acquisition dates shows {stats.total_regions_count} total change regions. "
                f"Vegetation covers {t1_area} {unit} at T1 and {t2_area} {unit} at T2. Detected transitions include "
                f"{loss_count} loss regions ({loss_area} {unit}) and {gain_count} gain regions ({gain_area} {unit}), "
                f"resulting in a net vegetation difference of {net_area} {unit} ({pct_change:+.2f}%)."
            )

    elif stats.t1_vegetation_pixels is not None:
        # Optical RGB Mode with Visible Spectrum Vegetation Analysis
        t1_area = stats.t1_vegetation_area or 0.0
        t2_area = stats.t2_vegetation_area or 0.0
        loss_area = stats.vegetation_loss_area or 0.0
        gain_area = stats.vegetation_gain_area or 0.0
        net_area = stats.net_vegetation_change_area or 0.0
        pct_change = stats.percentage_change or 0.0
        loss_count = stats.loss_region_count
        gain_count = stats.gain_region_count
        chg_area = stats.total_changed_area or 0.0
        chg_count = stats.total_regions_count
        chg_pct = stats.total_change_percentage

        if intent == QueryIntent.VEGETATION_LOSS.value:
            if loss_count > 0:
                text = (
                    f"Visible-spectrum optical vegetation analysis identified {loss_count} vegetation loss regions "
                    f"totaling {loss_area} {unit}, where pre-existing green canopy/vegetation was replaced by "
                    f"built structures or cleared surfaces. Total visible vegetation shifted from {t1_area} {unit} at T1 "
                    f"to {t2_area} {unit} at T2 (net change of {net_area} {unit}, {pct_change:+.2f}%)."
                )
            else:
                text = (
                    f"No significant vegetation loss was detected in the visible spectrum above the minimum region threshold. "
                    f"Visible vegetation area shifted from {t1_area} {unit} at T1 to {t2_area} {unit} at T2."
                )

        elif intent == QueryIntent.VEGETATION_GAIN.value:
            if gain_count > 0:
                text = (
                    f"Visible-spectrum optical vegetation analysis identified {gain_count} vegetation gain regions "
                    f"representing {gain_area} {unit} of newly emerged or recovered vegetation. "
                    f"Total visible vegetation shifted from {t1_area} {unit} at T1 to {t2_area} {unit} at T2 "
                    f"(net change of {net_area} {unit}, {pct_change:+.2f}%)."
                )
            else:
                text = (
                    f"No significant vegetation gain was detected above the minimum region threshold. "
                    f"Visible vegetation area was {t1_area} {unit} at T1 and {t2_area} {unit} at T2."
                )

        elif intent == QueryIntent.CHANGE_STATISTICS.value:
            text = (
                f"Quantitative optical analysis indicates: visible vegetation was {t1_area} {unit} at T1 "
                f"and {t2_area} {unit} at T2. Detected transitions include {loss_count} vegetation loss regions ({loss_area} {unit}) "
                f"and {gain_count} vegetation gain regions ({gain_area} {unit}), within {chg_count} total surface change regions "
                f"covering {chg_area} {unit} ({chg_pct:.2f}% of scene). Net vegetation change is {net_area} {unit} ({pct_change:+.2f}%)."
            )

        else:  # GENERAL_COMPARISON
            text = (
                f"Bi-temporal optical comparison identified {chg_count} total surface change regions ({chg_area} {unit}, {chg_pct:.2f}%). "
                f"Visible vegetation indexing classified {loss_count} vegetation loss regions ({loss_area} {unit}) and "
                f"{gain_count} vegetation gain regions ({gain_area} {unit}), shifting visible vegetation from {t1_area} {unit} at T1 "
                f"to {t2_area} {unit} at T2 (net change: {net_area} {unit}, {pct_change:+.2f}%)."
            )

    else:
        # Optical RGB Mode without vegetation analysis
        chg_area = stats.total_changed_area or 0.0
        chg_pct = stats.total_change_percentage
        chg_count = stats.total_regions_count
        chg_px = stats.total_changed_pixels

        if intent == QueryIntent.CHANGE_STATISTICS.value:
            text = (
                f"Optical change analysis detected {chg_px:,} changed pixels ({chg_pct:.2f}% of image area) "
                f"distributed across {chg_count} distinct change regions. Total changed area is {chg_area} {unit}. "
                "Note: Near-Infrared spectral data was not present; true NDVI vegetation classification requires multispectral imagery."
            )
        else:
            text = (
                f"Bi-temporal optical change detection identified {chg_count} surface change regions, "
                f"representing {chg_pct:.2f}% of the scene ({chg_area} {unit}). "
                "ChangeFormerV6 isolated learned temporal surface shifts between the two image dates."
            )

    return text
