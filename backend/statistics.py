"""
Global Statistics Module.
Calculates:
- T1 vegetation area
- T2 vegetation area
- Vegetation loss area
- Vegetation gain area
- Net vegetation change (T2 - T1)
- Percentage change (((T2 - T1) / T1) * 100)
Handles zero T1 vegetation edge cases and dual-mode (multispectral vs optical) differences.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from backend.area_calculation import AreaContext


@dataclass
class QuantitativeStatistics:
    mode: str                          # "sentinel2_multispectral" or "optical_rgb"
    total_pixels: int
    area_unit: str                     # "ha", "m²", or "pixels"
    calculation_method: str

    # Vegetation statistics (Sentinel-2 multispectral mode)
    t1_vegetation_pixels: Optional[int]
    t2_vegetation_pixels: Optional[int]
    vegetation_loss_pixels: Optional[int]
    vegetation_gain_pixels: Optional[int]
    net_vegetation_change_pixels: Optional[int]

    t1_vegetation_area: Optional[float]
    t2_vegetation_area: Optional[float]
    vegetation_loss_area: Optional[float]
    vegetation_gain_area: Optional[float]
    net_vegetation_change_area: Optional[float]
    percentage_change: Optional[float]

    loss_region_count: int
    gain_region_count: int

    # Generic optical statistics (applicable to both modes)
    total_changed_pixels: int
    total_changed_area: Optional[float]
    total_change_percentage: float
    total_regions_count: int


def calculate_global_statistics(
    mode: str,
    total_pixels: int,
    area_ctx: AreaContext,
    # Multispectral metrics
    t1_veg_pixels: Optional[int] = None,
    t2_veg_pixels: Optional[int] = None,
    loss_pixels: Optional[int] = None,
    gain_pixels: Optional[int] = None,
    loss_region_count: int = 0,
    gain_region_count: int = 0,
    # Generic change metrics
    total_changed_pixels: int = 0,
    total_regions_count: int = 0,
) -> QuantitativeStatistics:
    """
    Computes global statistics with accurate units and safe zero division handling.
    """
    # Helper to convert pixel count to physical area if context allows
    def px_to_area(px: Optional[int]) -> Optional[float]:
        if px is None:
            return None
        if area_ctx.pixel_area_ha is not None and area_ctx.unit_name == "hectares":
            return round(px * area_ctx.pixel_area_ha, 4)
        elif area_ctx.pixel_area_m2 is not None and area_ctx.unit_name == "m²":
            return round(px * area_ctx.pixel_area_m2, 2)
        return float(px)

    total_changed_pct = round((total_changed_pixels / total_pixels * 100.0), 2) if total_pixels > 0 else 0.0
    total_changed_area = px_to_area(total_changed_pixels)

    if t1_veg_pixels is not None and t2_veg_pixels is not None:
        loss_px = loss_pixels or 0
        gain_px = gain_pixels or 0
        net_px = t2_veg_pixels - t1_veg_pixels

        t1_area = px_to_area(t1_veg_pixels)
        t2_area = px_to_area(t2_veg_pixels)
        loss_area = px_to_area(loss_px)
        gain_area = px_to_area(gain_px)

        if t1_area is not None and t2_area is not None:
            net_area = round(t2_area - t1_area, 4 if area_ctx.unit_name == "hectares" else 2)
        else:
            net_area = None

        # Percentage change: ((T2 - T1) / T1) * 100
        if t1_veg_pixels > 0:
            pct_change = round(((t2_veg_pixels - t1_veg_pixels) / t1_veg_pixels) * 100.0, 2)
        else:
            # Handle zero T1 vegetation area
            if t2_veg_pixels > 0:
                pct_change = 100.0  # complete new gain
            else:
                pct_change = 0.0

        return QuantitativeStatistics(
            mode=mode,
            total_pixels=total_pixels,
            area_unit=area_ctx.unit_name,
            calculation_method=area_ctx.area_calculation_method,
            t1_vegetation_pixels=t1_veg_pixels,
            t2_vegetation_pixels=t2_veg_pixels,
            vegetation_loss_pixels=loss_px,
            vegetation_gain_pixels=gain_px,
            net_vegetation_change_pixels=net_px,
            t1_vegetation_area=t1_area,
            t2_vegetation_area=t2_area,
            vegetation_loss_area=loss_area,
            vegetation_gain_area=gain_area,
            net_vegetation_change_area=net_area,
            percentage_change=pct_change,
            loss_region_count=loss_region_count,
            gain_region_count=gain_region_count,
            total_changed_pixels=total_changed_pixels,
            total_changed_area=total_changed_area,
            total_change_percentage=total_changed_pct,
            total_regions_count=total_regions_count,
        )
    else:
        # Optical RGB Mode without vegetation analysis
        return QuantitativeStatistics(
            mode=mode,
            total_pixels=total_pixels,
            area_unit=area_ctx.unit_name,
            calculation_method=area_ctx.area_calculation_method,
            t1_vegetation_pixels=None,
            t2_vegetation_pixels=None,
            vegetation_loss_pixels=None,
            vegetation_gain_pixels=None,
            net_vegetation_change_pixels=None,
            t1_vegetation_area=None,
            t2_vegetation_area=None,
            vegetation_loss_area=None,
            vegetation_gain_area=None,
            net_vegetation_change_area=None,
            percentage_change=None,
            loss_region_count=0,
            gain_region_count=0,
            total_changed_pixels=total_changed_pixels,
            total_changed_area=total_changed_area,
            total_change_percentage=total_changed_pct,
            total_regions_count=total_regions_count,
        )
