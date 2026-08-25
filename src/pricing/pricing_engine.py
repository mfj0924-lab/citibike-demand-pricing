"""
Dynamic Pricing Engine.

Converts ML model predictions into pricing recommendations.

Input:
  - station_id, timestamp
  - predicted bike_demand, predicted dock_demand
  - current station status (bikes available, docks available)

Output:
  - pricing_multiplier (e.g. 1.35 → 35% surge)
  - suggested_price (USD)
  - pricing rationale (human-readable reason)

This is the core business innovation of the project.
"""

import os, sys
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class PricingZone(Enum):
    SURGE = "Surge"  # 1.2x - 2.0x
    MILD_SURGE = "Mild Surge"  # 1.0x - 1.2x
    NORMAL = "Normal"  # ~1.0x
    DISCOUNT = "Discount"  # 0.7x - 0.9x


@dataclass
class PricingResult:
    """Pricing decision for a single station-hour."""

    station_id: str
    timestamp: str
    predicted_bike_demand: float
    predicted_dock_demand: float
    available_bikes_now: int
    available_docks_now: int
    bike_gap: int
    dock_gap: int
    pricing_multiplier: float
    suggested_price_usd: float
    pricing_zone: PricingZone
    reason: str


class DynamicPricingEngine:
    """Core pricing logic: demand predictions → pricing decisions.

    Uses a deterministic, explainable rule engine — not a black box.
    Every pricing decision can be traced back to a specific supply-demand gap.

    Pricing rationale:
      - When bikes are predicted to run out → surge (discourage borrowing)
      - When docks are predicted to fill up → surge (discourage returning)
      - When both are abundant → discount (attract riders)
      - When balanced → normal price

    Base price: $4.49 (CitiBike single ride, e-bike, 2026 rate)
    """

    BASE_PRICE = 4.49  # USD

    def predict_price(
        self,
        station_id: str,
        timestamp: str,
        predicted_bike_demand: float,
        predicted_dock_demand: float,
        available_bikes_now: int,
        available_docks_now: int,
    ) -> PricingResult:
        """Calculate pricing for a single station-hour prediction.

        Args:
            station_id: CitiBike station identifier.
            timestamp: ISO-format datetime string.
            predicted_bike_demand: ML model prediction for bikes borrowed this hour.
            predicted_dock_demand: ML model prediction for bikes returned this hour.
            available_bikes_now: Current number of bikes at the station.
            available_docks_now: Current number of empty docks.

        Returns:
            PricingResult with multiplier, price, zone, and explanation.
        """
        # Calculate gaps (+ positive = surplus, negative = shortage)
        bike_gap = available_bikes_now - predicted_bike_demand
        dock_gap = available_docks_now - predicted_dock_demand

        # Determine pricing zone and multiplier
        multiplier, zone, reason = self._determine_pricing(
            bike_gap,
            dock_gap,
            available_bikes_now,
            available_docks_now,
        )

        return PricingResult(
            station_id=station_id,
            timestamp=timestamp,
            predicted_bike_demand=round(predicted_bike_demand, 1),
            predicted_dock_demand=round(predicted_dock_demand, 1),
            available_bikes_now=available_bikes_now,
            available_docks_now=available_docks_now,
            bike_gap=round(bike_gap),
            dock_gap=round(dock_gap),
            pricing_multiplier=multiplier,
            suggested_price_usd=round(self.BASE_PRICE * multiplier, 2),
            pricing_zone=zone,
            reason=reason,
        )

    def _determine_pricing(
        self,
        bike_gap: float,
        dock_gap: float,
        available_bikes: int,
        available_docks: int,
    ):
        """Core pricing logic — four scenarios."""
        bike_ratio = bike_gap / max(available_bikes, 1)
        dock_ratio = dock_gap / max(available_docks, 1)

        # Scenario 1: Double scarcity
        if bike_ratio < -0.5 and dock_ratio < -0.5:
            severity = abs(bike_ratio + dock_ratio) / 2
            multiplier = 1.3 + severity * 0.7
            multiplier = min(multiplier, 2.0)
            return (
                round(multiplier, 2),
                PricingZone.SURGE,
                f"Double shortage: bikes {-bike_gap:.0f} short, docks {-dock_gap:.0f} short. "
                f"Heavy surge applied.",
            )

        # Scenario 2: Bike shortage (people want to borrow, not enough bikes)
        if bike_ratio < -0.3:
            severity = abs(bike_ratio)
            multiplier = 1.0 + severity * 0.6
            multiplier = min(multiplier, 1.5)
            return (
                round(multiplier, 2),
                PricingZone.SURGE,
                f"Bike shortage: predicted demand exceeds supply by {-bike_gap:.0f}. "
                f"Surge applied to encourage nearby station use.",
            )

        # Scenario 3: Dock congestion (people want to return, not enough empty docks)
        if dock_ratio < -0.3:
            severity = abs(dock_ratio)
            multiplier = 1.0 + severity * 0.4
            multiplier = min(multiplier, 1.3)
            return (
                round(multiplier, 2),
                PricingZone.MILD_SURGE,
                f"Dock congestion: predicted returns exceed empty docks by {-dock_gap:.0f}. "
                f"Mild surge to discourage returns here.",
            )

        # Scenario 4: Double abundance — discount to attract
        if bike_ratio > 0.5 and dock_ratio > 0.5:
            surplus = min(bike_ratio, dock_ratio)
            multiplier = 1.0 - surplus * 0.25
            multiplier = max(multiplier, 0.7)
            return (
                round(multiplier, 2),
                PricingZone.DISCOUNT,
                f"Double surplus: bikes and docks both abundant. "
                f"Discount applied to attract riders.",
            )

        # Scenario 5: Bike surplus only — mild discount
        if bike_ratio > 0.5:
            multiplier = 1.0 - bike_ratio * 0.15
            multiplier = max(multiplier, 0.8)
            return (
                round(multiplier, 2),
                PricingZone.DISCOUNT,
                f"Bike surplus: {bike_gap:.0f} extra bikes. "
                f"Mild discount to encourage use.",
            )

        # Default: balanced
        return (
            1.0,
            PricingZone.NORMAL,
            "Supply and demand balanced. Standard pricing.",
        )

    def batch_predict(
        self,
        station_id: str,
        hourly_predictions: list,  # list of (timestamp, bike_demand, dock_demand)
        current_bikes: int = None,
        current_docks: int = None,
        total_capacity: int = 50,
    ) -> list[PricingResult]:
        """Generate 24-hour pricing forecast for a single station.

        Args:
            station_id: Station identifier.
            hourly_predictions: List of (timestamp, bike_demand, dock_demand) tuples.
            current_bikes: Current bike count (default: half capacity).
            current_docks: Current dock count (default: half capacity).
            total_capacity: Station total dock capacity for defaults.

        Returns:
            List of PricingResult for each hour.
        """
        if current_bikes is None:
            current_bikes = total_capacity // 2
        if current_docks is None:
            current_docks = total_capacity - current_bikes

        results = []
        running_bikes = current_bikes
        running_docks = current_docks

        for ts, bike_dem, dock_dem in hourly_predictions:
            result = self.predict_price(
                station_id=station_id,
                timestamp=ts,
                predicted_bike_demand=bike_dem,
                predicted_dock_demand=dock_dem,
                available_bikes_now=running_bikes,
                available_docks_now=running_docks,
            )
            results.append(result)

            # Simulate state evolution across hours
            # Bikes borrowed reduce available bikes, bikes returned increase them
            running_bikes = max(0, running_bikes - int(bike_dem) + int(dock_dem))
            running_bikes = min(total_capacity, running_bikes)
            running_docks = total_capacity - running_bikes

        return results


# ── Demo ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = DynamicPricingEngine()

    # Demo: a popular station during morning rush
    print("=" * 60)
    print("  Dynamic Pricing Engine — Demo")
    print("=" * 60)

    scenarios = [
        # (station_id, timestamp, bike_dem, dock_dem, bikes, docks, label)
        (327, "2026-06-15 08:00", 35, 10, 10, 15, "Morning rush — bike shortage"),
        (327, "2026-06-15 08:00", 35, 10, 5, 15, "Morning rush — severe bike shortage"),
        (327, "2026-06-15 12:00", 5, 5, 25, 25, "Midday — balanced"),
        (327, "2026-06-15 18:00", 10, 40, 20, 5, "Evening return — dock congestion"),
        (327, "2026-06-15 03:00", 1, 1, 30, 25, "Night — surplus, discount"),
        (327, "2026-06-15 08:00", 50, 60, 3, 2, "Double shortage — extreme"),
    ]

    for sid, ts, bd, dd, bikes, docks, label in scenarios:
        result = engine.predict_price(
            station_id=str(sid),
            timestamp=ts,
            predicted_bike_demand=bd,
            predicted_dock_demand=dd,
            available_bikes_now=bikes,
            available_docks_now=docks,
        )
        print(f"\n  [{label}]")
        print(
            f"    Demand: bikes={bd}, docks={dd}  |  "
            f"Available: bikes={bikes}, docks={docks}"
        )
        print(f"    Gaps: bike={result.bike_gap:+d}, dock={result.dock_gap:+d}")
        print(
            f"    → {result.pricing_zone.value}: "
            f"{result.pricing_multiplier:.2f}x → ${result.suggested_price_usd:.2f}"
        )
        print(f"    Reason: {result.reason}")
