"""
Cadastral boundary loader

Fetches cadastral parcel boundaries from Swiss geo.admin.ch API.
"""

import requests
from shapely.geometry import shape
from typing import Tuple, Optional, Dict


def fetch_boundary_by_egrid(egrid: str) -> Tuple[Optional[object], Optional[Dict]]:
    """
    Fetch the cadastral boundary (Polygon) and metadata for a given EGRID via geo.admin.ch API.
    
    Args:
        egrid: Swiss EGRID identifier
        
    Returns:
        Tuple: (Shapely geometry in EPSG:2056, metadata dict)
    """
    url = "https://api3.geo.admin.ch/rest/services/ech/MapServer/find"
    params = {
        "layer": "ch.kantone.cadastralwebmap-farbe",
        "searchText": egrid,
        "searchField": "egris_egrid",
        "returnGeometry": "true",
        "geometryFormat": "geojson",
        "sr": "2056"
    }
    
    print(f"Fetching boundary for EGRID {egrid}...")
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    
    if not data.get("results"):
        print(f"No results found for EGRID {egrid}")
        return None, None
    
    feature = data["results"][0]
    geometry = shape(feature["geometry"])
    
    # Extract cadastre metadata
    attributes = feature.get("properties", {}) or feature.get("attributes", {})
    
    # Calculate area from geometry (in m² since EPSG:2056 is in meters)
    area_m2 = geometry.area
    
    metadata = {
        "egrid": egrid,
        "canton": attributes.get("ak", ""),
        "parcel_number": attributes.get("number", ""),
        "local_id": attributes.get("identnd", ""),
        "geoportal_url": attributes.get("geoportal_url", ""),
        "realestate_type": attributes.get("realestate_type", ""),
        "area_m2": round(area_m2, 2),
        "perimeter_m": round(geometry.length, 2),
    }
    
    # Print metadata
    if metadata["canton"]:
        print(f"  Canton: {metadata['canton']}")
    if metadata["parcel_number"]:
        print(f"  Parcel Number: {metadata['parcel_number']}")
    print(f"  Area: {metadata['area_m2']:.1f} m² ({metadata['area_m2']/10000:.3f} ha)")
    print(f"  Perimeter: {metadata['perimeter_m']:.1f} m")
    
    return geometry, metadata

