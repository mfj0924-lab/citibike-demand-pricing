"""Tests for pricing analysis generator."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.pricing.supply_demand_gap import (
    estimate_available_bikes, estimate_available_docks,
    calculate_pricing_multiplier,
)


class TestEstimationFunctions:

    def test_estimate_bikes_half_capacity(self):
        # capacity 50 → ~25 bikes
        assert estimate_available_bikes(50) == 25
        assert estimate_available_bikes(100) == 50

    def test_estimate_bikes_odd_capacity(self):
        # capacity 51 → floor(25) or ceil? should be 25
        assert estimate_available_bikes(51) == 25

    def test_estimate_bikes_zero_capacity(self):
        assert estimate_available_bikes(0) == 15

    def test_estimate_docks_complement(self):
        # total = bikes + docks
        c = 50
        assert estimate_available_bikes(c) + estimate_available_docks(c) == c


class TestPricingMultiplier:

    def test_balanced(self):
        m = calculate_pricing_multiplier(
            bike_gap=5, dock_gap=5,
            available_bikes=20, available_docks=20,
        )
        assert m == 1.0

    def test_severe_shortage(self):
        m = calculate_pricing_multiplier(
            bike_gap=-15, dock_gap=-15,
            available_bikes=5, available_docks=5,
        )
        assert m > 1.5

    def test_surplus_discount(self):
        m = calculate_pricing_multiplier(
            bike_gap=20, dock_gap=20,
            available_bikes=25, available_docks=25,
        )
        assert m < 1.0
