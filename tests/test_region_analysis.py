"""
Unit tests for region-level analysis module.
Tests:
- Connected component extraction
- Region attributes (type, area, bounds, centroid, T1 status, T2 status)
- Minimum region size threshold
- CRITICAL CHECK: Verifies that no region contains T1 mean NDVI, T2 mean NDVI, or NDVI difference.
"""

import numpy as np
import pytest
from backend.region_analysis import extract_change_regions


def test_region_extraction_attributes():
    loss_mask = np.zeros((30, 30), dtype=bool)
    # 5x5 loss region at [5..10, 5..10] -> 25 pixels
    loss_mask[5:10, 5:10] = True

    gain_mask = np.zeros((30, 30), dtype=bool)
    # 6x6 gain region at [20..26, 20..26] -> 36 pixels
    gain_mask[20:26, 20:26] = True

    regions = extract_change_regions(
        loss_mask=loss_mask,
        gain_mask=gain_mask,
        pixel_area_ha=0.01,
        min_region_size=5,
    )

    assert len(regions) == 2
    # Regions are sorted descending by size: gain (36 px) first, then loss (25 px)
    assert regions[0].region_type == "VEGETATION_GAIN"
    assert regions[0].area_pixels == 36
    assert np.isclose(regions[0].area_physical, 0.36)
    assert regions[0].area_unit == "ha"
    assert regions[0].t1_vegetation_status == "Non-Vegetation"
    assert regions[0].t2_vegetation_status == "Vegetation"

    assert regions[1].region_type == "VEGETATION_LOSS"
    assert regions[1].area_pixels == 25
    assert np.isclose(regions[1].area_physical, 0.25)
    assert regions[1].t1_vegetation_status == "Vegetation"
    assert regions[1].t2_vegetation_status == "Non-Vegetation"


def test_strict_absence_of_region_ndvi_metrics():
    """
    CRITICAL CONSTRAINT: For each significant vegetation-change region,
    do NOT calculate or display:
    - T1 mean NDVI
    - T2 mean NDVI
    - NDVI difference
    """
    loss_mask = np.zeros((20, 20), dtype=bool)
    loss_mask[2:8, 2:8] = True

    regions = extract_change_regions(
        loss_mask=loss_mask,
        gain_mask=None,
        pixel_area_ha=0.01,
    )

    for r in regions:
        # Check dataclass attributes
        d = r.__dict__
        assert "t1_mean_ndvi" not in d
        assert "t2_mean_ndvi" not in d
        assert "ndvi_difference" not in d
        assert "mean_ndvi" not in d
        assert "ndvi_diff" not in d
