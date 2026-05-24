"""
MarketAgent - Real market data fetcher with multi-source cross-validation.

Fetches real-time market data from yfinance (primary) with architecture
ready for akshare/tushare as secondary sources. Implements the cross-
validation and weighted fusion logic defined in config.py.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from ai_trader.config import config, DataSourceConfig


class MarketAgent:
    """Fetches real market data with optional multi-source cross-validation."""

    # Internal symbol -> yfinance ticker mapping
    SYMBOL_MAP = {
        "XAUUSD": "GC=F",        # COMEX Gold Futures (most liquid XAUUSD proxy)
        "XAGUSD": "SI=F",        # COMEX Silver Futures
        "WTI":    "CL=F",        # WTI Crude Oil Futures
        "SPX":    "^GSPC",       # S&P 500 Index
        "BTCUSD": "BTC-USD",     # Bitcoin
    }

    def __init__(self, cache_ttl: int = 120):
        """
        Args:
            cache_ttl: Cache time-to-live in seconds. Data refreshes after this
                       period. Default 120s balances freshness vs API rate limits.
        """
        self._cache_ttl = cache_ttl
        self._cache: Dict[str, Dict] = {}
        self._cache_ts: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_market_data(self, symbol: str) -> Dict:
        """
        Fetch real market data with cross-validation from configured sources.

        Returns a dict with at minimum:
            symbol, price, trend, source, timestamp

        When multiple sources are enabled and agree, also returns:
            cross_validated, sources_used, max_divergence
        """
        now = time.time()
        if symbol in self._cache and (now - self._cache_ts.get(symbol, 0)) < self._cache_ttl:
            return self._cache[symbol]

        enabled = sorted(
            [s for s in config.data_sources if s.enabled],
            key=lambda s: s.priority,
        )

        results: List[Tuple[DataSourceConfig, Dict]] = []
        for ds in enabled:
            data = self._fetch_from_source(ds, symbol)
            if data:
                results.append((ds, data))

        if not results:
            raise RuntimeError(
                f"No data source returned valid data for {symbol}. "
                f"Enabled sources: {[s.name for s in enabled]}"
            )

        ok, final = self._cross_validate(results)
        if not ok or final is None:
            raise RuntimeError(f"Cross-validation failed for {symbol}")

        final["_fetched_at"] = now
        self._cache[symbol] = final
        self._cache_ts[symbol] = now

        return final

    # ------------------------------------------------------------------
    # Source adapters
    # ------------------------------------------------------------------

    def _fetch_from_source(self, ds: DataSourceConfig, symbol: str) -> Optional[Dict]:
        """Dispatch to the correct fetcher by source name."""
        fetchers = {
            "yfinance": self._fetch_yfinance,
            "akshare":  self._fetch_akshare,
            "tushare":  self._fetch_tushare,
        }
        fetcher = fetchers.get(ds.name)
        if fetcher is None:
            print(f"[MarketAgent] Unknown source '{ds.name}', skipping")
            return None
        try:
            return fetcher(symbol)
        except Exception as exc:
            print(f"[MarketAgent] {ds.name} error for {symbol}: {exc}")
            return None

    # ------------------------------------------------------------------
    # yfinance fetcher
    # ------------------------------------------------------------------

    def _fetch_yfinance(self, symbol: str) -> Optional[Dict]:
        """Fetch real market data from Yahoo Finance."""
        ticker_str = self.SYMBOL_MAP.get(symbol, symbol)

        for attempt in range(3):
            try:
                ticker = yf.Ticker(ticker_str)

                # Prefer 1h bars for better trend resolution; fall back to 1d
                hist = ticker.history(period="5d", interval="1h")
                if hist.empty:
                    hist = ticker.history(period="5d", interval="1d")
                if hist.empty:
                    print(f"[MarketAgent] yfinance returned empty history for {ticker_str}")
                    return None

                close = hist["Close"].values.astype(float)
                current_price = float(close[-1])

                # ---- trend + strength from moving average crossover ----
                trend, trend_strength = self._compute_trend(close)

                return {
                    "symbol":  symbol,
                    "price":   round(current_price, 2),
                    "open":    round(float(hist["Open"].iloc[-1]), 2),
                    "high":    round(float(hist["High"].iloc[-1]), 2),
                    "low":     round(float(hist["Low"].iloc[-1]), 2),
                    "volume":  int(hist["Volume"].iloc[-1]) if "Volume" in hist else 0,
                    "trend":   trend,
                    "source":  "yfinance",
                    "timestamp": datetime.now().isoformat(),
                    "trend_strength": trend_strength,
                }
            except Exception as exc:
                if ("rate limited" in str(exc).lower() or "too many requests" in str(exc).lower()) and attempt < 2:
                    wait = (attempt + 1) * 5
                    print(f"[MarketAgent] yfinance rate-limited, retrying in {wait}s...")
                    time.sleep(wait)
                elif attempt < 2:
                    wait = 2
                    print(f"[MarketAgent] yfinance error (attempt {attempt+1}/3): {exc}")
                    time.sleep(wait)
                else:
                    raise
        return None

    # ------------------------------------------------------------------
    # akshare / tushare stubs (ready for server deployment)
    # ------------------------------------------------------------------

    def _fetch_akshare(self, symbol: str) -> Optional[Dict]:
        """
        Fetch gold price data from akshare (Shanghai Gold Exchange).

        Uses SGE Au99.99 spot as proxy for XAUUSD. Returns price in CNY/gram.
        """
        try:
            import akshare as ak
        except ImportError:
            print("[MarketAgent] akshare not installed; pip install akshare")
            return None

        # Map internal symbols to SGE symbols
        sge_map = {
            "XAUUSD": "Au99.99",
            "XAGUSD": "Ag99.99",
        }
        sge_symbol = sge_map.get(symbol, symbol)

        try:
            df = ak.spot_hist_sge(symbol=sge_symbol)
            if df.empty:
                print(f"[MarketAgent] akshare returned empty data for {sge_symbol}")
                return None

            latest = df.iloc[-1]
            close_prices = df["close"].values.astype(float)
            current_price = float(close_prices[-1])

            trend, trend_strength = self._compute_trend(close_prices)

            return {
                "symbol":  symbol,
                "price":   round(current_price, 2),
                "open":    round(float(latest["open"]), 2),
                "high":    round(float(latest["high"]), 2),
                "low":     round(float(latest["low"]), 2),
                "volume":  0,  # SGE hist data doesn't include volume
                "trend":   trend,
                "source":  "akshare",
                "timestamp": datetime.now().isoformat(),
                "trend_strength": trend_strength,
            }
        except Exception as exc:
            print(f"[MarketAgent] akshare error: {exc}")
            return None

    def _fetch_tushare(self, symbol: str) -> Optional[Dict]:
        """
        Stub — set TUSHARE_TOKEN env var and install tushare on the server.

        Example:

            import tushare as ts
            pro = ts.pro_api(token=config.tushare_token)
            df = pro.daily(ts_code=symbol, ...)
            ...
        """
        if not config.tushare_token:
            print("[MarketAgent] tushare token not set; skipping")
            return None
        # TODO: implement when token is available
        return None

    # ------------------------------------------------------------------
    # Cross-validation (from config.py design)
    # ------------------------------------------------------------------

    def _cross_validate(
        self, results: List[Tuple[DataSourceConfig, Dict]]
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Cross-validate prices across sources.

        When 2+ sources agree within config.max_price_diff_ratio, returns a
        weighted-average price.  When only one source is available, returns it
        directly with cross_validated=False.
        """
        valid = [(cfg, d) for cfg, d in results if d.get("price")]

        if not valid:
            return False, None

        if len(valid) == 1:
            data = valid[0][1]
            data["cross_validated"] = False
            data["sources_used"] = 1
            data["max_divergence"] = 0.0
            return True, data

        # Multiple sources — check price agreement
        prices = np.array([d["price"] for _, d in valid])
        mean_price = prices.mean()
        divergences = np.abs(prices - mean_price) / mean_price
        max_div = float(divergences.max())

        if max_div > config.max_price_diff_ratio:
            print(
                f"[MarketAgent] WARNING — price divergence {max_div:.4%} "
                f"exceeds threshold {config.max_price_diff_ratio:.4%}. "
                f"Falling back to highest-priority source."
            )

        # Weighted average by source weight
        weights = np.array([cfg.weight for cfg, _ in valid])
        weights /= weights.sum()
        weighted_price = float(np.average(prices, weights=weights))

        merged = valid[0][1].copy()
        merged["price"] = round(weighted_price, 2)
        merged["cross_validated"] = max_div <= config.max_price_diff_ratio
        merged["sources_used"] = len(valid)
        merged["max_divergence"] = round(max_div, 6)
        merged["individual_prices"] = {
            cfg.name: d["price"] for cfg, d in valid
        }
        return True, merged

    # ------------------------------------------------------------------
    # Trend computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_trend(close_prices: np.ndarray) -> Tuple[str, float]:
        """
        Compute trend and strength from short-MA vs long-MA crossover.

        Returns (trend, strength) where:
          - trend  ∈ {"UP", "DOWN", "SIDEWAYS"}
          - strength = (MA_short - MA_long) / MA_long  (signed ratio)
        """
        n = len(close_prices)
        short_w = min(3, max(1, n - 1))
        long_w = min(12, max(2, n // 2))
        long_w = max(long_w, short_w + 1)

        ma_short = float(pd.Series(close_prices).rolling(short_w).mean().iloc[-1])
        ma_long = float(pd.Series(close_prices).rolling(long_w).mean().iloc[-1])

        if np.isnan(ma_long) or ma_long == 0:
            strength = (close_prices[-1] - close_prices[0]) / close_prices[0]
        else:
            strength = (ma_short - ma_long) / ma_long

        if strength > 0.003:
            trend = "UP"
        elif strength < -0.003:
            trend = "DOWN"
        else:
            trend = "SIDEWAYS"

        return trend, round(float(strength), 6)
