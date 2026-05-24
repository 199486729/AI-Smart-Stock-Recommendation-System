"""AlphaMind A-Share analysis package."""

from .data_fetcher import fetch_all_data
from .market_state import compute_market_state
from .sector_ranker import rank_sectors
from .etf_screener import screen_etfs
from .stock_screener import screen_stocks
from .recommender import generate_recommendations
