"""
Integration tests for ChangeFormerV6 change detection module.
Tests:
- Model instantiation
- Pretrained checkpoint loading
- Output shape and probability bounds
"""

import numpy as np
import pytest
from backend.change_detection import ChangeDetector


def test_changeformer_detector_initialization():
    detector = ChangeDetector.get_instance()
    assert detector is not None
    assert detector.checkpoint_loaded, "Pretrained checkpoint best_ckpt.pt should be loaded"


def test_changeformer_inference_shapes_and_bounds():
    detector = ChangeDetector.get_instance()
    # Create two synthetic 256x256 test images
    t1 = np.full((256, 256, 3), 100, dtype=np.uint8)
    t2 = np.full((256, 256, 3), 200, dtype=np.uint8)

    res = detector.detect_changes(t1, t2, threshold=0.50, min_region_size=5)

    assert res.change_mask.shape == (256, 256)
    assert res.change_prob.shape == (256, 256)
    assert res.total_pixels == 256 * 256
    assert np.all(res.change_prob >= 0.0)
    assert np.all(res.change_prob <= 1.0)
    assert set(np.unique(res.change_mask)).issubset({0, 1})
