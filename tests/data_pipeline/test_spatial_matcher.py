"""Tests for spatial station matcher."""

import os, sys, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data_pipeline.spatial_matcher import (
    match_stations_from_dataframe, haversine_km,
)


class TestHaversine:
    """Test distance calculation."""

    def test_same_point_zero(self):
        d = haversine_km(40.73, -73.99, 40.73, -73.99)
        assert d == 0.0

    def test_known_distance(self):
        """Times Square to Grand Central ≈ 1.2 km."""
        # Times Square: 40.7580, -73.9855
        # Grand Central: 40.7527, -73.9772
        d = haversine_km(40.7580, -73.9855, 40.7527, -73.9772)
        assert 800 < d < 1500  # roughly 1.1 km


class TestSpatialMatcher:
    """Test station matching logic."""

    def test_match_exact_location(self):
        """Two stations at same lat/lng should match."""
        import pandas as pd

        # Create a temp GBFS JSON
        gbfs = {
            "data": {
                "stations": [
                    {"station_id": "uuid-1", "lat": 40.73, "lon": -73.99, "capacity": 42, "name": "Station A"},
                    {"station_id": "uuid-2", "lat": 40.75, "lon": -73.95, "capacity": 30, "name": "Station B"},
                ]
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(gbfs, f)
            json_path = f.name

        try:
            bronze = pd.DataFrame({
                "station_id": ["s1", "s2"],
                "latitude": [40.73, 40.75],
                "longitude": [-73.99, -73.95],
            })

            cap_map = match_stations_from_dataframe(bronze, json_path, max_distance_m=200.0)

            assert cap_map["s1"] == 42
            assert cap_map["s2"] == 30
        finally:
            os.remove(json_path)

    def test_match_far_away_returns_zero(self):
        """Station too far from any GBFS station should get capacity=0."""
        import pandas as pd

        gbfs = {
            "data": {
                "stations": [
                    {"station_id": "uuid-1", "lat": 40.73, "lon": -73.99, "capacity": 42, "name": "A"},
                ]
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(gbfs, f)
            json_path = f.name

        try:
            bronze = pd.DataFrame({
                "station_id": ["far_station"],
                "latitude": [34.05],  # Los Angeles — 4000 km away
                "longitude": [-118.24],
            })

            cap_map = match_stations_from_dataframe(bronze, json_path, max_distance_m=200.0)

            assert cap_map["far_station"] == 0
        finally:
            os.remove(json_path)
