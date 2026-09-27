"""
Unit tests for vegetation analysis module.
Tests:
- Vegetation thresholding (default >= 0.30)
- Vegetation loss detection (T1 veg & not T2 veg & change)
- Vegetation gain detection (not T1 veg & T2 veg & change)
- Stable vegetation detection (T1 veg & T2 veg & not change)
- Minimum region size noise filtering
"""

import numpy as np
import pytest
from backend.vegetation_analysis import classify_vegetation_changes


def test_vegetation_loss_detection():
    # 20x20 scene
    t1_ndvi = np.full((20, 20), 0.70, dtype=np.float32)  # Vegetation everywhere at T1
    t2_ndvi = np.full((20, 20), 0.70, dtype=np.float32)
    change_mask = np.zeros((20, 20), dtype=np.uint8)

    # 10x10 block in top-left experienced change and lost vegetation at T2
    t2_ndvi[:10, :10] = 0.10  # Below 0.30 threshold
    change_mask[:10, :10] = 1

    masks = classify_vegetation_changes(
        t1_ndvi, t2_ndvi, change_mask, ndvi_threshold=0.30, min_region_size=5
    )

    assert masks.loss_pixels == 100
    assert masks.gain_pixels == 0
    assert masks.stable_pixels == 300
    assert np.all(masks.veg_loss_mask[:10, :10])
    assert not np.any(masks.veg_gain_mask)


def test_vegetation_gain_detection():
    t1_ndvi = np.full((20, 20), 0.15, dtype=np.float32)  # Non-vegetation at T1
    t2_ndvi = np.full((20, 20), 0.15, dtype=np.float32)
    change_mask = np.zeros((20, 20), dtype=np.uint8)

    # 8x8 block gained vegetation at T2
    t2_ndvi[5:13, 5:13] = 0.65  # Above 0.30 threshold
    change_mask[5:13, 5:13] = 1

    masks = classify_vegetation_changes(
        t1_ndvi, t2_ndvi, change_mask, ndvi_threshold=0.30, min_region_size=5
    )

    assert masks.gain_pixels == 64
    assert masks.loss_pixels == 0
    assert np.all(masks.veg_gain_mask[5:13, 5:13])


def test_noise_filtering_min_region_size():
    t1_ndvi = np.full((20, 20), 0.60, dtype=np.float32)
    t2_ndvi = np.full((20, 20), 0.60, dtype=np.float32)
    change_mask = np.zeros((20, 20), dtype=np.uint8)

    # Isolated 2-pixel speckle loss
    t2_ndvi[2, 2] = 0.10
    t2_ndvi[2, 3] = 0.10
    change_mask[2, 2] = 1
    change_mask[2, 3] = 1

    # Large 6x6 real loss block (36 pixels)
    t2_ndvi[10:16, 10:16] = 0.10
    change_mask[10:16, 10:16] = 1

    # Min size 10 should discard the 2-pixel speckle and keep the 36-pixel region
    masks = classify_vegetation_changes(
        t1_ndvi, t2_ndvi, change_mask, ndvi_threshold=0.30, min_region_size=10
    )

    assert not masks.veg_loss_mask[2, 2]
    assert masks.loss_pixels == 36


def test_visible_vegetation_mask_and_classification():
    from backend.vegetation_analysis import (
        compute_visible_vegetation_mask,
        classify_visible_vegetation_changes,
    )

    # Create synthetic RGB scene (20x20)
    # Green vegetation: R=40, G=120, B=50
    t1_rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    t1_rgb[:, :] = [40, 120, 50]

    # Non-vegetated roof/concrete: R=180, G=140, B=110
    t2_rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    t2_rgb[:, :] = [40, 120, 50]
    t2_rgb[:10, :10] = [180, 140, 110]  # Roof built over vegetation in top-left

    vmask_t1 = compute_visible_vegetation_mask(t1_rgb)
    assert np.all(vmask_t1)  # All vegetated

    vmask_t2 = compute_visible_vegetation_mask(t2_rgb)
    assert not np.any(vmask_t2[:10, :10])  # Roof is rejected

    change_mask = np.zeros((20, 20), dtype=np.uint8)
    change_mask[:10, :10] = 1

    masks = classify_visible_vegetation_changes(t1_rgb, t2_rgb, change_mask, min_region_size=5)
    assert masks.loss_pixels == 100
    assert masks.gain_pixels == 0
    assert np.all(masks.veg_loss_mask[:10, :10])

