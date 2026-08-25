"""
Spatial station matcher: match bronze stations to GBFS stations by lat/lng proximity.

Bronze CSV stations use short numeric IDs (e.g. "6535.04"),
GBFS stations use UUIDs (e.g. "66dd1f44-0aca-11e7-...") with capacity info.
Since there's no shared ID, we match by geographic proximity.

Uses scipy KDTree for efficient nearest-neighbor lookup.
"""

import os, sys, json
from typing import Dict, Tuple
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Haversine distance between two lat/lng points in kilometers."""
    from math import radians, sin, cos, sqrt, asin
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng/2)**2
    return R * 2 * asin(sqrt(a)) * 1000  # return meters


def build_capacity_lookup(
    gbfs_json_path: str,
    max_distance_m: float = 100.0,
) -> Tuple[Dict[str, int], float]:
    """Match bronze stations to GBFS stations by nearest lat/lng.

    Args:
        gbfs_json_path: Path to station_information.json.
        max_distance_m: Maximum distance (meters) to consider a match.

    Returns:
        Tuple of (bronze_station_id → capacity dict, match_rate).
    """
    from scipy.spatial import cKDTree

    # Load GBFS
    with open(gbfs_json_path) as f:
        data = json.load(f)
    gbfs_stations = data["data"]["stations"]

    gbfs_coords = []
    gbfs_info = []
    for s in gbfs_stations:
        lat = s.get("lat")
        lon = s.get("lon")
        cap = s.get("capacity", 0) or 0
        if lat and lon:
            gbfs_coords.append([lat, lon])
            gbfs_info.append({"name": s.get("name", ""), "capacity": cap})

    gbfs_coords = np.array(gbfs_coords)
    tree = cKDTree(gbfs_coords)

    # This function is called with bronze station coordinates
    # Returns station_id → capacity
    print(f"  GBFS stations with coords: {len(gbfs_coords)}")
    return None  # placeholder — the actual matching needs bronze stations

    # We'll do the matching inside a function that accepts bronze stations DataFrame


def match_stations_from_dataframe(
    bronze_stations_df,
    gbfs_json_path: str,
    max_distance_m: float = 100.0,
) -> Dict[str, int]:
    """Given a pandas DataFrame of bronze stations, return station_id → capacity.

    Args:
        bronze_stations_df: pandas DataFrame with columns [station_id, latitude, longitude].
        gbfs_json_path: Path to station_information.json.
        max_distance_m: Max distance in meters to accept a match.

    Returns:
        Dict mapping bronze station_id → capacity (0 if no match).
    """
    from scipy.spatial import cKDTree

    with open(gbfs_json_path) as f:
        data = json.load(f)
    gbfs_stations = data["data"]["stations"]

    gbfs_coords = []
    gbfs_caps = []
    for s in gbfs_stations:
        lat = s.get("lat")
        lon = s.get("lon")
        if lat and lon:
            gbfs_coords.append([lat, lon])
            gbfs_caps.append(s.get("capacity", 0) or 0)

    tree = cKDTree(gbfs_coords)

    bronze_coords = bronze_stations_df[["latitude", "longitude"]].values
    distances, indices = tree.query(bronze_coords, k=1)

    capacity_map = {}
    matched = 0
    for i, (station_id, dist, idx) in enumerate(
        zip(bronze_stations_df["station_id"], distances, indices)
    ):
        if dist <= max_distance_m / 111000.0:  # rough deg conversion
            # More precise: use haversine check
            lat_b = bronze_stations_df.iloc[i]["latitude"]
            lng_b = bronze_stations_df.iloc[i]["longitude"]
            lat_g, lng_g = gbfs_coords[idx]
            actual_dist = haversine_km(lat_b, lng_b, lat_g, lng_g)
            if actual_dist <= max_distance_m:
                capacity_map[str(station_id)] = gbfs_caps[idx]
                matched += 1
                continue
        capacity_map[str(station_id)] = 0

    match_rate = matched / len(bronze_stations_df) if len(bronze_stations_df) > 0 else 0
    print(f"  Station capacity match: {matched}/{len(bronze_stations_df)} ({match_rate:.1%})")
    print(f"  Non-zero capacities: {sum(1 for v in capacity_map.values() if v > 0)}")
    return capacity_map
