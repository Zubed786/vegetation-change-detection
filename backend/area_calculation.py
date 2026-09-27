"""
Geospatial and Image Area Calculation Module.
Handles:
1. True geospatial projection & physical area (hectares, km²) using projected CRS.
2. Verified metadata GSD (e.g. CDVQA res_x / res_y) without hardcoding.
3. Transparent image-relative statistics when geospatial metadata is unavailable.
"""

from dataclasses import dataclass
import math
import re
from typing import Optional, Tuple
from pyproj import CRS, Transformer


@dataclass
class AreaContext:
    has_physical_area: bool
    pixel_area_m2: Optional[float]
    pixel_area_ha: Optional[float]
    unit_name: str                 # "hectares", "m²", or "pixels (% of image)"
    area_calculation_method: str   # "projected_crs", "verified_metadata_gsd", or "image_relative"
    crs_info: Optional[str] = None
    gsd_x: Optional[float] = None
    gsd_y: Optional[float] = None


def parse_cdvqa_resolution(res_str: Optional[str]) -> Optional[float]:
    """
    Parses resolution string from CDVQA metadata such as '.1524m' or '0.5m'.
    Returns float in meters if valid, None otherwise.
    """
    if not res_str or not isinstance(res_str, str):
        return None
    m = re.search(r"([0-9]*\.?[0-9]+)\s*m?", res_str.strip())
    if m:
        try:
            val = float(m.group(1))
            if val > 0:
                return val
        except ValueError:
            pass
    return None


def resolve_area_context(
    crs_epsg: Optional[int] = None,
    crs_wkt: Optional[str] = None,
    transform: Optional[Tuple[float, ...]] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    metadata_res_x: Optional[str] = None,
    metadata_res_y: Optional[str] = None,
) -> AreaContext:
    """
    Determines pixel ground resolution and physical area multipliers.
    Prioritizes projected CRS from raster transform.
    Falls back to verified CDVQA metadata GSD if present.
    Defaults to image-relative statistics if neither is verified.
    """
    # 1. Check Rasterio Transform and CRS
    if transform is not None and len(transform) >= 6:
        dx = abs(transform[0])
        dy = abs(transform[4]) if len(transform) >= 5 else dx

        crs_obj = None
        if crs_epsg:
            try:
                crs_obj = CRS.from_epsg(crs_epsg)
            except Exception:
                pass
        elif crs_wkt:
            try:
                crs_obj = CRS.from_wkt(crs_wkt)
            except Exception:
                pass

        if crs_obj is not None:
            if crs_obj.is_projected:
                # Direct meter dimensions
                pixel_area_m2 = dx * dy
                pixel_area_ha = pixel_area_m2 / 10000.0
                return AreaContext(
                    has_physical_area=True,
                    pixel_area_m2=pixel_area_m2,
                    pixel_area_ha=pixel_area_ha,
                    unit_name="hectares",
                    area_calculation_method="projected_crs",
                    crs_info=crs_obj.name,
                    gsd_x=round(dx, 3),
                    gsd_y=round(dy, 3),
                )
            elif crs_obj.is_geographic and bounds is not None:
                # Coordinates are in degrees (e.g. EPSG:4326)
                # Compute meter length at latitude centroid using spherical approximation
                center_lat = (bounds[1] + bounds[3]) / 2.0
                lat_rad = math.radians(center_lat)
                # 1 deg lat ~ 111,320m; 1 deg lon ~ 111,320m * cos(lat)
                m_per_deg_lat = 111320.0
                m_per_deg_lon = 111320.0 * math.cos(lat_rad)
                pixel_m_x = dx * m_per_deg_lon
                pixel_m_y = dy * m_per_deg_lat
                pixel_area_m2 = pixel_m_x * pixel_m_y
                pixel_area_ha = pixel_area_m2 / 10000.0
                return AreaContext(
                    has_physical_area=True,
                    pixel_area_m2=pixel_area_m2,
                    pixel_area_ha=pixel_area_ha,
                    unit_name="hectares",
                    area_calculation_method="reprojected_geographic_crs",
                    crs_info=f"Geographic {crs_obj.name} (reprojected at {center_lat:.2f}°N)",
                    gsd_x=round(pixel_m_x, 3),
                    gsd_y=round(pixel_m_y, 3),
                )

    # 2. Check verified CDVQA metadata resolution (e.g. res_x: '.1524m')
    res_x = parse_cdvqa_resolution(metadata_res_x)
    res_y = parse_cdvqa_resolution(metadata_res_y) or res_x
    if res_x is not None and res_x > 0:
        pixel_area_m2 = res_x * res_y
        pixel_area_ha = pixel_area_m2 / 10000.0
        return AreaContext(
            has_physical_area=True,
            pixel_area_m2=pixel_area_m2,
            pixel_area_ha=pixel_area_ha,
            unit_name="m²",
            area_calculation_method="verified_metadata_gsd",
            crs_info=None,
            gsd_x=round(res_x, 4),
            gsd_y=round(res_y, 4),
        )

    # 3. Transparent Fallback: Image-relative statistics
    return AreaContext(
        has_physical_area=False,
        pixel_area_m2=None,
        pixel_area_ha=None,
        unit_name="pixels (% of image)",
        area_calculation_method="image_relative",
        crs_info="No geospatial CRS or verified GSD metadata available",
        gsd_x=None,
        gsd_y=None,
    )
