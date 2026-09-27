"""
Unit tests for image validation module.
Tests:
- Dimension mismatch rejection (prohibits arbitrary resizing, cropping, or warping)
- Format validation (rejects unsupported file formats)
- Valid-pixel percentage checks
- Mode compatibility determination
"""

import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from backend.image_validation import (
    validate_bitemporal_pair,
    inspect_single_image,
    ImageValidationError,
)


@pytest.fixture
def temp_images_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_dimension_mismatch_rejection(temp_images_dir):
    # Create T1 of 256x256 and T2 of 300x300
    p1 = temp_images_dir / "t1.png"
    p2 = temp_images_dir / "t2.png"

    arr1 = np.full((256, 256, 3), 128, dtype=np.uint8)
    arr2 = np.full((300, 300, 3), 128, dtype=np.uint8)

    Image.fromarray(arr1).save(p1)
    Image.fromarray(arr2).save(p2)

    res = validate_bitemporal_pair(str(p1), str(p2))
    assert not res.is_compatible
    assert any("Dimension mismatch" in err for err in res.errors)


def test_valid_pixel_percentage_check(temp_images_dir):
    p1 = temp_images_dir / "t1_mostly_empty.png"
    p2 = temp_images_dir / "t2_normal.png"

    # T1 has only 10% valid pixels (below 40% threshold)
    arr1 = np.zeros((100, 100, 3), dtype=np.uint8)
    arr1[:10, :10, :] = 200
    arr2 = np.full((100, 100, 3), 150, dtype=np.uint8)

    Image.fromarray(arr1).save(p1)
    Image.fromarray(arr2).save(p2)

    res = validate_bitemporal_pair(str(p1), str(p2))
    assert not res.is_compatible
    assert any("too few valid pixels" in err for err in res.errors)


def test_unsupported_file_extension(temp_images_dir):
    p = temp_images_dir / "image.txt"
    p.write_text("not an image")

    with pytest.raises(ImageValidationError):
        inspect_single_image(str(p))


def test_compatible_pair(temp_images_dir):
    p1 = temp_images_dir / "t1.png"
    p2 = temp_images_dir / "t2.png"

    arr = np.full((256, 256, 3), 180, dtype=np.uint8)
    Image.fromarray(arr).save(p1)
    Image.fromarray(arr).save(p2)

    res = validate_bitemporal_pair(str(p1), str(p2))
    assert res.is_compatible
    assert res.width == 256
    assert res.height == 256
    assert res.mode == "optical_rgb"
    assert len(res.errors) == 0
