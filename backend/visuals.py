"""
Visual Artifact Generation Module.
Produces displayable browser imagery (PNGs), raster masks (TIFF/GeoTIFF),
GeoJSON vector boundaries, and blended change-highlighted overlays.
Distinguishes vegetation loss (vivid coral/red) and gain (emerald green) clearly.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def array_to_rgb_preview(arr: np.ndarray, is_multispectral: bool = False, red_idx: int = 2) -> np.ndarray:
    """
    Converts raster array to an 8-bit RGB preview (H, W, 3).
    Handles 4-band Sentinel-2 [B2, B3, B4, B8] by selecting B4(Red), B3(Green), B2(Blue) for true color.
    """
    if arr.ndim == 2:
        norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6) * 255.0
        return np.stack([norm.astype(np.uint8)] * 3, axis=-1)

    if arr.shape[0] in [3, 4, 12, 13] and arr.shape[0] < arr.shape[1]:
        # (C, H, W) format
        if is_multispectral and arr.shape[0] >= 4:
            # Sentinel-2 true color: Red=band 3 (idx 2), Green=band 2 (idx 1), Blue=band 1 (idx 0)
            rgb = np.stack([arr[2], arr[1], arr[0]], axis=-1)
        else:
            rgb = np.transpose(arr[:3], (1, 2, 0))
    else:
        # (H, W, C) format
        rgb = arr[:, :, :3]

    # Normalize channels to 0-255 with 2%-98% percentile stretch for optimal visual dynamic range
    out = np.zeros(rgb.shape, dtype=np.uint8)
    for c in range(3):
        ch = rgb[:, :, c].astype(np.float32)
        valid = ch[np.isfinite(ch) & (ch > 0)]
        if valid.size > 0:
            p2 = float(np.percentile(valid, 2))
            p98 = float(np.percentile(valid, 98))
            denom = max(1.0, p98 - p2)
            stretched = np.clip((ch - p2) / denom * 255.0, 0, 255)
            out[:, :, c] = stretched.astype(np.uint8)
        else:
            out[:, :, c] = 0
    return out


def create_change_highlighted_image(
    base_rgb: np.ndarray,
    loss_mask: Optional[np.ndarray],
    gain_mask: Optional[np.ndarray],
    generic_change_mask: Optional[np.ndarray] = None,
    alpha: float = 0.55,
) -> Image.Image:
    """
    Creates a composite image blending the original satellite imagery with:
    - Vegetation Loss in bright Coral Red (#FF3B30)
    - Vegetation Gain in vivid Emerald Green (#34C759)
    - Or Generic Surface Change in Amber (#FF9500)
    """
    h, w = base_rgb.shape[:2]
    overlay = np.array(base_rgb, dtype=np.float32)

    # Color definitions (RGB)
    COLOR_LOSS = np.array([255, 59, 48], dtype=np.float32)     # Vivid Coral Red
    COLOR_GAIN = np.array([52, 199, 89], dtype=np.float32)     # Vibrant Emerald Green
    COLOR_CHANGE = np.array([255, 149, 0], dtype=np.float32)  # Amber

    alpha_veg = min(1.0, alpha + 0.10)
    alpha_surf = max(0.20, alpha - 0.25)

    if generic_change_mask is not None and np.any(generic_change_mask):
        chg_indices = generic_change_mask.astype(bool).copy()
        if loss_mask is not None:
            chg_indices = chg_indices & (~loss_mask.astype(bool))
        if gain_mask is not None:
            chg_indices = chg_indices & (~gain_mask.astype(bool))
        if np.any(chg_indices):
            overlay[chg_indices] = (1 - alpha_surf) * overlay[chg_indices] + alpha_surf * COLOR_CHANGE

    if loss_mask is not None and np.any(loss_mask):
        loss_indices = loss_mask.astype(bool)
        overlay[loss_indices] = (1 - alpha_veg) * overlay[loss_indices] + alpha_veg * COLOR_LOSS

    if gain_mask is not None and np.any(gain_mask):
        gain_indices = gain_mask.astype(bool)
        overlay[gain_indices] = (1 - alpha_veg) * overlay[gain_indices] + alpha_veg * COLOR_GAIN

    img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    return img


def export_analysis_artifacts(
    output_dir: Path,
    t1_rgb: np.ndarray,
    t2_rgb: np.ndarray,
    change_mask: np.ndarray,
    loss_mask: Optional[np.ndarray],
    gain_mask: Optional[np.ndarray],
    t1_veg_mask: Optional[np.ndarray],
    t2_veg_mask: Optional[np.ndarray],
    regions: List[Any],
    analysis_dict: Dict[str, Any],
    crs_epsg: Optional[int] = None,
    transform: Optional[Tuple[float, ...]] = None,
) -> Dict[str, str]:
    """
    Saves visual comparison images, masks, GeoJSON, and analysis JSON into the job output directory.
    Returns relative URLs / file paths for the frontend.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    files: Dict[str, str] = {}

    COLOR_LOSS = np.array([255, 59, 48], dtype=np.float32)
    COLOR_GAIN = np.array([52, 199, 89], dtype=np.float32)
    COLOR_VEG = np.array([34, 197, 94], dtype=np.float32)

    def create_single_overlay(base: np.ndarray, mask: Optional[np.ndarray], color: np.ndarray, alpha: float = 0.55) -> Optional[Image.Image]:
        if mask is None:
            return None
        over = np.array(base, dtype=np.float32)
        idx = mask.astype(bool)
        over[idx] = (1 - alpha) * over[idx] + alpha * color
        return Image.fromarray(np.clip(over, 0, 255).astype(np.uint8))

    # 1. Original T1 Preview
    t1_img = Image.fromarray(t1_rgb)
    t1_path = output_dir / "t1_original.png"
    t1_img.save(t1_path)
    files["t1_original"] = f"outputs/{output_dir.name}/t1_original.png"

    # 2. Original T2 Preview
    t2_img = Image.fromarray(t2_rgb)
    t2_path = output_dir / "t2_original.png"
    t2_img.save(t2_path)
    files["t2_original"] = f"outputs/{output_dir.name}/t2_original.png"

    # 3. Change-Highlighted Image (Overlay on T2)
    highlighted = create_change_highlighted_image(
        base_rgb=t2_rgb,
        loss_mask=loss_mask,
        gain_mask=gain_mask,
        generic_change_mask=change_mask,
    )
    chg_hi_path = output_dir / "change_highlighted.png"
    highlighted.save(chg_hi_path)
    files["change_highlighted"] = f"outputs/{output_dir.name}/change_highlighted.png"

    # 4. Single-layer vegetation and change overlays for direct layer tab inspection
    if t1_veg_mask is not None:
        t1_veg_img = create_single_overlay(t1_rgb, t1_veg_mask, COLOR_VEG)
        if t1_veg_img:
            t1_veg_img.save(output_dir / "t1_vegetation.png")
            files["t1_vegetation"] = f"outputs/{output_dir.name}/t1_vegetation.png"

    if t2_veg_mask is not None:
        t2_veg_img = create_single_overlay(t2_rgb, t2_veg_mask, COLOR_VEG)
        if t2_veg_img:
            t2_veg_img.save(output_dir / "t2_vegetation.png")
            files["t2_vegetation"] = f"outputs/{output_dir.name}/t2_vegetation.png"

    if loss_mask is not None:
        loss_img = create_single_overlay(t2_rgb, loss_mask, COLOR_LOSS)
        if loss_img:
            loss_img.save(output_dir / "vegetation_loss.png")
            files["vegetation_loss_preview"] = f"outputs/{output_dir.name}/vegetation_loss.png"

    if gain_mask is not None:
        gain_img = create_single_overlay(t2_rgb, gain_mask, COLOR_GAIN)
        if gain_img:
            gain_img.save(output_dir / "vegetation_gain.png")
            files["vegetation_gain_preview"] = f"outputs/{output_dir.name}/vegetation_gain.png"

    # 5. Binary Change Map PNG
    chg_map_img = Image.fromarray((change_mask * 255).astype(np.uint8))
    chg_map_path = output_dir / "change_map.png"
    chg_map_img.save(chg_map_path)
    files["change_map_png"] = f"outputs/{output_dir.name}/change_map.png"

    # 5. Helper function to write GeoTIFF if metadata available, else standard TIFF
    def save_tif(name: str, mask_arr: Optional[np.ndarray]):
        if mask_arr is None:
            return
        out_tif = output_dir / f"{name}.tif"
        if HAS_RASTERIO and transform is not None:
            try:
                with rasterio.open(
                    out_tif,
                    "w",
                    driver="GTiff",
                    height=mask_arr.shape[0],
                    width=mask_arr.shape[1],
                    count=1,
                    dtype=np.uint8,
                    crs=f"EPSG:{crs_epsg}" if crs_epsg else None,
                    transform=transform,
                ) as dst:
                    dst.write(mask_arr.astype(np.uint8), 1)
                files[name] = f"outputs/{output_dir.name}/{name}.tif"
                return
            except Exception:
                pass
        # Fallback to PIL TIFF
        im = Image.fromarray((mask_arr * 255).astype(np.uint8))
        im.save(out_tif)
        files[name] = f"outputs/{output_dir.name}/{name}.tif"

    save_tif("change_mask", change_mask)
    if t1_veg_mask is not None:
        save_tif("vegetation_t1", t1_veg_mask)
    if t2_veg_mask is not None:
        save_tif("vegetation_t2", t2_veg_mask)
    if loss_mask is not None:
        save_tif("vegetation_loss", loss_mask)
    if gain_mask is not None:
        save_tif("vegetation_gain", gain_mask)

    # 6. Change Regions GeoJSON
    geojson_features = []
    for r in regions:
        # Convert pixel bounds/contour to GeoJSON format
        coords = [[[p[0], p[1]] for p in r.contour]]
        # Ensure closed polygon ring
        if coords[0] and coords[0][0] != coords[0][-1]:
            coords[0].append(coords[0][0])

        feature = {
            "type": "Feature",
            "properties": {
                "region_id": r.region_id,
                "region_type": r.region_type,
                "area_pixels": r.area_pixels,
                "area_physical": r.area_physical,
                "area_unit": r.area_unit,
                "bounds": list(r.bounds),
                "centroid": list(r.centroid),
                "t1_vegetation_status": r.t1_vegetation_status,
                "t2_vegetation_status": r.t2_vegetation_status,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": coords,
            },
        }
        geojson_features.append(feature)

    geojson_data = {
        "type": "FeatureCollection",
        "features": geojson_features,
    }
    geojson_path = output_dir / "change_regions.geojson"
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2)
    files["change_regions_geojson"] = f"outputs/{output_dir.name}/change_regions.geojson"

    # 7. Analysis JSON
    json_path = output_dir / "analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_dict, f, indent=2)
    files["analysis_json"] = f"outputs/{output_dir.name}/analysis.json"

    return files
