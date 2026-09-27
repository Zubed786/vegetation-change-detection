"""
Region-Level Analysis Module.
Extracts connected change regions and computes region-level attributes:
- region type
- region area
- region location/bounds
- vegetation status at T1
- vegetation status at T2

CRITICAL CONSTRAINT: Does NOT calculate or display:
- T1 mean NDVI
- T2 mean NDVI
- NDVI difference
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import ndimage


@dataclass
class ChangeRegion:
    region_id: int
    region_type: str              # "VEGETATION_LOSS", "VEGETATION_GAIN", or "SURFACE_CHANGE"
    area_pixels: int
    area_physical: Optional[float]  # in hectares if geospatial CRS exists
    area_unit: str                # "ha", "m²", or "pixels (% of image)"
    bounds: Tuple[int, int, int, int]  # (min_x, min_y, max_x, max_y) in pixel coordinates
    centroid: Tuple[float, float]      # (cx, cy)
    t1_vegetation_status: str     # "Vegetation", "Non-Vegetation", or "N/A"
    t2_vegetation_status: str     # "Vegetation", "Non-Vegetation", or "N/A"
    contour: List[List[int]] = field(default_factory=list)  # Simplified boundary polygon for UI


def extract_change_regions(
    loss_mask: Optional[np.ndarray],
    gain_mask: Optional[np.ndarray],
    generic_change_mask: Optional[np.ndarray] = None,
    pixel_area_ha: Optional[float] = None,
    pixel_area_m2: Optional[float] = None,
    min_region_size: int = 16,
    max_regions: int = 200,
) -> List[ChangeRegion]:
    """
    Extracts connected components from loss, gain, or generic change masks.
    Calculates region type, area, location/bounds, and T1/T2 vegetation status.
    Strictly avoids computing or displaying NDVI values per region.
    """
    regions: List[ChangeRegion] = []
    region_id_counter = 1

    tasks = []
    if loss_mask is not None and np.any(loss_mask):
        tasks.append((loss_mask, "VEGETATION_LOSS", "Vegetation", "Non-Vegetation"))
    if gain_mask is not None and np.any(gain_mask):
        tasks.append((gain_mask, "VEGETATION_GAIN", "Non-Vegetation", "Vegetation"))

    if generic_change_mask is not None and np.any(generic_change_mask):
        other_mask = generic_change_mask.astype(bool).copy()
        if loss_mask is not None:
            other_mask = other_mask & (~loss_mask.astype(bool))
        if gain_mask is not None:
            other_mask = other_mask & (~gain_mask.astype(bool))
        if np.any(other_mask):
            t1_s = "Non-Vegetation" if (loss_mask is not None or gain_mask is not None) else "N/A (Optical Mode)"
            t2_s = "Non-Vegetation" if (loss_mask is not None or gain_mask is not None) else "N/A (Optical Mode)"
            tasks.append((other_mask, "SURFACE_CHANGE", t1_s, t2_s))

    for mask, reg_type, t1_status, t2_status in tasks:
        labeled, num_features = ndimage.label(mask.astype(bool))
        if num_features == 0:
            continue

        # Find objects (slices)
        objects = ndimage.find_objects(labeled)

        for i, slc in enumerate(objects):
            if slc is None:
                continue
            lbl = i + 1
            region_pixels = int(np.count_nonzero(labeled[slc] == lbl))
            if region_pixels < min_region_size:
                continue

            y_slice, x_slice = slc
            min_y, max_y = y_slice.start, y_slice.stop
            min_x, max_x = x_slice.start, x_slice.stop

            # Compute centroid
            sub_mask = labeled[slc] == lbl
            cy, cx = ndimage.center_of_mass(sub_mask)
            centroid = (round(min_x + cx, 1), round(min_y + cy, 1))

            # Approximate simplified boundary bounding box or coarse contour for visualization
            contour = [
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
            ]

            # Physical area calculation if GSD or CRS is known
            if pixel_area_ha is not None and pixel_area_ha > 0:
                area_phys = round(region_pixels * pixel_area_ha, 4)
                unit = "ha"
            elif pixel_area_m2 is not None and pixel_area_m2 > 0:
                area_phys = round(region_pixels * pixel_area_m2, 2)
                unit = "m²"
            else:
                area_phys = None
                unit = "pixels"

            regions.append(
                ChangeRegion(
                    region_id=region_id_counter,
                    region_type=reg_type,
                    area_pixels=region_pixels,
                    area_physical=area_phys,
                    area_unit=unit,
                    bounds=(min_x, min_y, max_x, max_y),
                    centroid=centroid,
                    t1_vegetation_status=t1_status,
                    t2_vegetation_status=t2_status,
                    contour=contour,
                )
            )
            region_id_counter += 1

            if len(regions) >= max_regions:
                break
        if len(regions) >= max_regions:
            break

    # Sort regions descending by pixel size
    regions.sort(key=lambda r: r.area_pixels, reverse=True)
    return regions
