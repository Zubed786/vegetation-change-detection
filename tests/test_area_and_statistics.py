"""
Unit tests for area calculation and statistics module.
Tests:
- Projected CRS area calculation (ha and km²)
- Verified metadata GSD parsing (e.g. '.1524m')
- Image-relative statistics fallback (ensuring no invented hectares)
- Global statistics calculation
- Percentage change formula
- Edge case: Zero T1 vegetation area
"""

import pytest
from backend.area_calculation import (
    resolve_area_context,
    parse_cdvqa_resolution,
)
from backend.statistics import calculate_global_statistics


def test_parse_cdvqa_resolution():
    assert parse_cdvqa_resolution(".1524m") == 0.1524
    assert parse_cdvqa_resolution("0.5m") == 0.5
    assert parse_cdvqa_resolution("10m") == 10.0
    assert parse_cdvqa_resolution(None) is None
    assert parse_cdvqa_resolution("invalid") is None


def test_projected_crs_area():
    # 10m x 10m pixel transform in UTM Zone 32N (EPSG:32632)
    # 1 pixel = 100 m² = 0.01 ha
    transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0)
    ctx = resolve_area_context(crs_epsg=32632, transform=transform)

    assert ctx.has_physical_area
    assert ctx.unit_name == "hectares"
    assert ctx.pixel_area_m2 == 100.0
    assert ctx.pixel_area_ha == 0.01
    assert ctx.area_calculation_method == "projected_crs"


def test_verified_cdvqa_metadata_area():
    # CDVQA metadata provides res_x: '.1524m'
    ctx = resolve_area_context(metadata_res_x=".1524m", metadata_res_y=".1524m")
    assert ctx.has_physical_area
    assert ctx.unit_name == "m²"
    assert ctx.area_calculation_method == "verified_metadata_gsd"
    assert round(ctx.pixel_area_m2, 4) == round(0.1524 * 0.1524, 4)


def test_image_relative_fallback_no_invented_hectares():
    # Standard image without CRS and without verified GSD
    ctx = resolve_area_context()
    assert not ctx.has_physical_area
    assert ctx.pixel_area_ha is None
    assert ctx.pixel_area_m2 is None
    assert ctx.unit_name == "pixels (% of image)"
    assert ctx.area_calculation_method == "image_relative"


def test_global_statistics_formulas():
    transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0)
    ctx = resolve_area_context(crs_epsg=32632, transform=transform)

    # 100x100 scene = 10,000 pixels = 100 ha
    # T1 veg = 5,000 px (50 ha)
    # T2 veg = 4,000 px (40 ha)
    # Loss = 1,200 px (12 ha)
    # Gain = 200 px (2 ha)
    stats = calculate_global_statistics(
        mode="sentinel2_multispectral",
        total_pixels=10000,
        area_ctx=ctx,
        t1_veg_pixels=5000,
        t2_veg_pixels=4000,
        loss_pixels=1200,
        gain_pixels=200,
        loss_region_count=4,
        gain_region_count=1,
    )

    assert stats.t1_vegetation_area == 50.0
    assert stats.t2_vegetation_area == 40.0
    assert stats.vegetation_loss_area == 12.0
    assert stats.vegetation_gain_area == 2.0
    assert stats.net_vegetation_change_area == -10.0
    # Percentage change: ((4000 - 5000) / 5000) * 100 = -20.0%
    assert stats.percentage_change == -20.0


def test_zero_t1_vegetation_handling():
    ctx = resolve_area_context()
    # T1 veg is zero, T2 has 500 pixels
    stats = calculate_global_statistics(
        mode="sentinel2_multispectral",
        total_pixels=10000,
        area_ctx=ctx,
        t1_veg_pixels=0,
        t2_veg_pixels=500,
        loss_pixels=0,
        gain_pixels=500,
    )

    # Should safely handle division by zero and report 100% gain
    assert stats.percentage_change == 100.0
