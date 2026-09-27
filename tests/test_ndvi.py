"""
Unit tests for NDVI calculation module.
Tests:
- Formula correctness: (B8 - B4) / (B8 + B4)
- Nodata & NaN handling
- Zero denominator / black void handling
- Reflectance scaling (0-10000, 0-255, 0-1)
- Exception on missing multispectral NIR band
"""

import numpy as np
import pytest
from backend.ndvi import (
    calculate_ndvi,
    detect_and_normalize_reflectance,
    MultispectralBandsRequiredError,
)


def test_standard_ndvi_formula():
    # Synthetic healthy vegetation: B4=0.05, B8=0.45
    # Expected NDVI = (0.45 - 0.05) / (0.45 + 0.05) = 0.40 / 0.50 = 0.80
    b4 = np.full((10, 10), 0.05, dtype=np.float32)
    b8 = np.full((10, 10), 0.45, dtype=np.float32)

    res = calculate_ndvi(b4, b8)
    assert np.allclose(res.ndvi, 0.80, atol=1e-4)
    assert res.mean_ndvi == 0.80
    assert np.all(res.valid_mask)


def test_zero_denominator_handling():
    # Both bands are zero: denominator = 0
    b4 = np.zeros((10, 10), dtype=np.float32)
    b8 = np.zeros((10, 10), dtype=np.float32)

    res = calculate_ndvi(b4, b8)
    assert np.all(res.ndvi == 0.0)
    assert not np.any(res.valid_mask)


def test_nan_and_inf_handling():
    b4 = np.full((10, 10), 0.10, dtype=np.float32)
    b8 = np.full((10, 10), 0.50, dtype=np.float32)

    # Insert NaNs and Infs
    b4[2, 2] = np.nan
    b8[5, 5] = np.inf

    res = calculate_ndvi(b4, b8)
    assert not res.valid_mask[2, 2]
    assert not res.valid_mask[5, 5]
    assert res.ndvi[2, 2] == 0.0
    assert res.ndvi[5, 5] == 0.0
    # Remaining pixels should be valid and equal to (0.5 - 0.1) / (0.5 + 0.1) = 0.4 / 0.6 = 0.6667
    assert np.isclose(res.ndvi[0, 0], 0.6667, atol=1e-3)


def test_reflectance_scaling_detection():
    # Test Sentinel-2 standard DN scale (10,000 max)
    b_dn = np.array([[2000, 4000], [5000, 8000]], dtype=np.uint16)
    norm, scale = detect_and_normalize_reflectance(b_dn)
    assert scale == "0-10000_surface_reflectance"
    assert np.isclose(norm[0, 0], 0.20, atol=1e-3)

    # Test 8-bit scale (0-255)
    b_8bit = np.array([[50, 100], [150, 200]], dtype=np.uint8)
    norm2, scale2 = detect_and_normalize_reflectance(b_8bit)
    assert scale2 == "0-255_scaled"
    assert np.isclose(norm2[0, 0], 50 / 255.0, atol=1e-3)


def test_multispectral_missing_nir_rejection():
    b4 = np.full((10, 10), 0.10, dtype=np.float32)
    # Passing None for NIR should raise explicit error
    with pytest.raises(MultispectralBandsRequiredError):
        calculate_ndvi(b4, None)
