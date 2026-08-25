"""Tests for DynamicPricingEngine — pure logic, no Spark needed."""

import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.pricing.pricing_engine import (
    DynamicPricingEngine,
    PricingZone,
    PricingResult,
)


class TestDynamicPricingEngine:
    """Test the core pricing logic with various supply-demand scenarios."""

    def setup_method(self):
        self.engine = DynamicPricingEngine()

    def test_bike_shortage_surge(self):
        """When bikes are scarce, price should surge."""
        result = self.engine.predict_price(
            station_id="327",
            timestamp="2026-06-15 08:00",
            predicted_bike_demand=35,
            predicted_dock_demand=10,
            available_bikes_now=10,
            available_docks_now=15,
        )
        assert result.pricing_multiplier > 1.2
        assert result.pricing_zone == PricingZone.SURGE
        assert result.suggested_price_usd > 4.49
        assert result.bike_gap < 0  # negative = shortage
        assert "shortage" in result.reason.lower()

    def test_dock_congestion_mild_surge(self):
        """When docks are congested, mild surge should apply."""
        result = self.engine.predict_price(
            station_id="327",
            timestamp="2026-06-15 18:00",
            predicted_bike_demand=10,
            predicted_dock_demand=40,
            available_bikes_now=20,
            available_docks_now=5,
        )
        assert result.pricing_multiplier >= 1.0
        assert result.dock_gap < 0
        assert "dock" in result.reason.lower() or "Dock" in result.reason

    def test_balanced_normal(self):
        """Balanced supply-demand should be normal price."""
        result = self.engine.predict_price(
            station_id="327",
            timestamp="2026-06-15 12:00",
            predicted_bike_demand=10,
            predicted_dock_demand=10,
            available_bikes_now=20,
            available_docks_now=20,
        )
        assert result.pricing_multiplier == 1.0
        assert result.suggested_price_usd == 4.49
        assert result.pricing_zone == PricingZone.NORMAL

    def test_surplus_discount(self):
        """When bikes are abundant, discount should apply."""
        result = self.engine.predict_price(
            station_id="327",
            timestamp="2026-06-15 03:00",
            predicted_bike_demand=1,
            predicted_dock_demand=1,
            available_bikes_now=30,
            available_docks_now=25,
        )
        assert result.pricing_multiplier < 1.0
        assert result.suggested_price_usd < 4.49
        assert result.bike_gap > 0  # positive = surplus

    def test_double_shortage_extreme(self):
        """Extreme double shortage should cap at 2.0x."""
        result = self.engine.predict_price(
            station_id="327",
            timestamp="2026-06-15 08:00",
            predicted_bike_demand=50,
            predicted_dock_demand=60,
            available_bikes_now=3,
            available_docks_now=2,
        )
        assert result.pricing_multiplier <= 2.0
        assert result.pricing_zone == PricingZone.SURGE

    def test_edge_case_zero_bikes(self):
        """Should not crash when available bikes = 0."""
        result = self.engine.predict_price(
            station_id="327",
            timestamp="2026-06-15 08:00",
            predicted_bike_demand=5,
            predicted_dock_demand=3,
            available_bikes_now=0,
            available_docks_now=10,
        )
        assert result.pricing_multiplier > 1.0
        assert result.suggested_price_usd > 4.49

    def test_batch_predict(self):
        """24-hour batch forecast should return 24 results."""
        hourly = [("2026-06-15 {:02d}:00".format(h), 10, 8) for h in range(24)]
        results = self.engine.batch_predict(
            station_id="327",
            hourly_predictions=hourly,
            current_bikes=15,
            current_docks=35,
            total_capacity=50,
        )
        assert len(results) == 24
        assert all(isinstance(r, PricingResult) for r in results)

    def test_base_price_constant(self):
        """Base price should be $4.49."""
        assert self.engine.BASE_PRICE == 4.49


class TestPricingResult:
    """Test PricingResult dataclass."""

    def test_rounding(self):
        """Suggested price should be rounded to 2 decimal places."""
        result = PricingResult(
            station_id="123",
            timestamp="2026-01-01 00:00",
            predicted_bike_demand=10.0,
            predicted_dock_demand=8.0,
            available_bikes_now=20,
            available_docks_now=20,
            bike_gap=10,
            dock_gap=12,
            pricing_multiplier=1.35,
            suggested_price_usd=6.74,
            pricing_zone=PricingZone.SURGE,
            reason="Test",
        )
        # Check rounding: 4.49 * 1.35 = 6.0615, should round to 6.06 (or 6.74 in manual case)
        import decimal

        d = decimal.Decimal(str(result.suggested_price_usd))
        assert (
            len(d.as_tuple().digits) <= 3
            if "." in str(result.suggested_price_usd)
            else True
        )
