"""
End-to-End API tests for FastAPI application.
Tests:
- GET /health
- GET /api/samples
- GET /api/cdvqa/samples
- POST /analyze (Sentinel-2 multispectral mode)
- POST /analyze (CDVQA optical mode)
- POST /analyze (Validation rejection on dimension mismatch)
- POST /analyze (Rejection on unsupported query)
"""

import io
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from starlette.testclient import TestClient
from backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["changeformer_model_ready"] is True
    assert data["checkpoint_loaded"] is True


def test_api_samples(client):
    res = client.get("/api/samples")
    assert res.status_code == 200
    samples = res.json()["samples"]
    assert len(samples) >= 2
    assert any(s["mode"] == "sentinel2_multispectral" for s in samples)
    assert any(s["mode"] == "optical_rgb" for s in samples)


def test_api_cdvqa_samples(client):
    res = client.get("/api/cdvqa/samples")
    assert res.status_code == 200
    data = res.json()
    assert data["dataset_loaded"] is True
    assert len(data["samples"]) > 0


    project_root = Path(__file__).resolve().parent.parent
    s1_path = project_root / "data" / "sentinel2_t1.tif"
    s2_path = project_root / "data" / "sentinel2_t2.tif"

    if not s1_path.exists() or not s2_path.exists():
        pytest.skip("Sample Sentinel-2 rasters not found.")

    with open(s1_path, "rb") as f1, open(s2_path, "rb") as f2:
        files = {
            "image_t1": ("sentinel2_t1.tif", f1, "image/tiff"),
            "image_t2": ("sentinel2_t2.tif", f2, "image/tiff"),
        }
        res = client.post(
            "/analyze",
            files=files,
            data={"query": "Where has vegetation been lost?"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["mode"] == "sentinel2_multispectral"
        assert data["intent"]["intent"] == "VEGETATION_LOSS"
        assert "statistics" in data
        assert data["statistics"]["t1_vegetation_area"] is not None
        assert "explanation" in data
        assert "evidence" in data
        assert "confidence" in data
        assert "visual_outputs" in data


def test_api_analyze_rejection_dimension_mismatch(client):
    buf1 = io.BytesIO()
    buf2 = io.BytesIO()

    Image.fromarray(np.full((256, 256, 3), 100, dtype=np.uint8)).save(buf1, format="PNG")
    Image.fromarray(np.full((320, 320, 3), 100, dtype=np.uint8)).save(buf2, format="PNG")

    buf1.seek(0)
    buf2.seek(0)

    files = {
        "image_t1": ("t1.png", buf1, "image/png"),
        "image_t2": ("t2.png", buf2, "image/png"),
    }
    res = client.post("/analyze", files=files, data={"query": "How much vegetation changed?"})
    assert res.status_code == 400
    assert "Dimension mismatch" in str(res.json())


def test_api_analyze_rejection_unsupported_query(client):
    buf1 = io.BytesIO()
    buf2 = io.BytesIO()

    arr = np.full((256, 256, 3), 100, dtype=np.uint8)
    Image.fromarray(arr).save(buf1, format="PNG")
    Image.fromarray(arr).save(buf2, format="PNG")

    buf1.seek(0)
    buf2.seek(0)

    files = {
        "image_t1": ("t1.png", buf1, "image/png"),
        "image_t2": ("t2.png", buf2, "image/png"),
    }
    res = client.post("/analyze", files=files, data={"query": "What is the capital of France?"})
    assert res.status_code == 400
    assert "Unsupported Query Intent" in str(res.json())
