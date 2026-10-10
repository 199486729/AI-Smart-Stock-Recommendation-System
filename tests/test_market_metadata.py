from datetime import date, datetime
import unittest

import pandas as pd

from ai_trader.analysis.data_fetcher import get_market_session
from ai_trader.analysis.sector_ranker import rank_sectors


class MarketMetadataTests(unittest.TestCase):
    def test_weekend_uses_latest_trade_date(self):
        session = get_market_session(
            now=datetime(2026, 10, 10, 10, 30),
            latest_market_date="2026-10-09",
            trade_dates={date(2026, 10, 8), date(2026, 10, 9)},
        )

        self.assertEqual(session["status"], "non_trading_day")
        self.assertEqual(session["data_date"], "2026-10-09")
        self.assertFalse(session["is_trading_time"])

    def test_board_main_flow_is_converted_to_hundred_million(self):
        boards = pd.DataFrame(
            {
                "板块名称": ["人工智能"],
                "涨跌幅": [1.61],
                "主力净流入": [152244.7],
                "板块异动总次数": [10],
            }
        )

        result = rank_sectors(boards=boards, top_n=1)

        self.assertEqual(result["top_sectors"][0]["fund_flow"], 15.22)


if __name__ == "__main__":
    unittest.main()
