"""SignalAgent — generates trade signals from real market data.

Uses trend direction AND strength (derived from the MA crossover ratio)
to produce calibrated confidence scores instead of hardcoded ones.
"""

from typing import Dict


class SignalAgent:
    """Consumes market data, produces BUY / SELL / HOLD signals."""

    # Minimum confidence required to act (below → HOLD)
    MIN_CONFIDENCE = 0.55

    def generate(self, market_data: Dict) -> Dict:
        """
        Generate a trade signal from market data.

        Args:
            market_data: Dict from MarketAgent.fetch_market_data().
                         Must contain at least 'trend' and 'price'.

        Returns:
            Dict with keys: action, confidence, price, trend, reason
        """
        trend = market_data["trend"]
        price = market_data["price"]

        # Derive confidence from trend strength (MA divergence ratio).
        # A 1% divergence → ~0.12 bonus; 3% divergence → ~0.36 bonus.
        trend_strength = market_data.get("trend_strength", 0.0)

        if trend == "UP":
            confidence = 0.60 + abs(trend_strength) * 12.0
            return {
                "action": "BUY",
                "confidence": round(min(confidence, 0.95), 3),
                "price": price,
                "trend": trend,
                "reason": f"Short MA > Long MA, strength={trend_strength:.4f}",
            }

        if trend == "DOWN":
            confidence = 0.60 + abs(trend_strength) * 12.0
            return {
                "action": "SELL",
                "confidence": round(min(confidence, 0.95), 3),
                "price": price,
                "trend": trend,
                "reason": f"Short MA < Long MA, strength={trend_strength:.4f}",
            }

        return {
            "action": "HOLD",
            "confidence": round(0.50 + abs(trend_strength) * 5.0, 3),
            "price": price,
            "trend": trend,
            "reason": "No clear trend direction",
        }

    def should_act(self, signal: Dict) -> bool:
        """Whether this signal is strong enough to execute."""
        return signal["action"] != "HOLD" and signal["confidence"] >= self.MIN_CONFIDENCE
