"""AlphaMind AI Trader V2 — entry point."""

from ai_trader.agent.market_agent import MarketAgent
from ai_trader.agent.signal_agent import SignalAgent
from ai_trader.execution.paper_broker import PaperBroker


def main():
    print("=" * 56)
    print("  AlphaMind AI Trader V2")
    print("=" * 56)

    market = MarketAgent()
    signal = SignalAgent()
    broker = PaperBroker()

    # Fetch real market data
    print("\n[1] Fetching real XAUUSD market data ...")
    data = market.fetch_market_data("XAUUSD")

    print(f"    Symbol:     {data['symbol']}")
    print(f"    Price:      {data['price']}")
    print(f"    Trend:      {data['trend']}")
    print(f"    Source:     {data['source']}")
    print(f"    Timestamp:  {data['timestamp']}")
    if data.get("cross_validated") is not None:
        print(f"    Cross-validated: {data['cross_validated']} "
              f"({data['sources_used']} sources, "
              f"divergence={data['max_divergence']:.6f})")
    if data.get("open"):
        print(f"    OHLCV:      O={data['open']} H={data['high']} "
              f"L={data['low']} C={data['price']} V={data['volume']}")

    # Generate signal
    print("\n[2] Generating trade signal ...")
    trade_signal = signal.generate(data)
    print(f"    Action:     {trade_signal['action']}")
    print(f"    Confidence: {trade_signal['confidence']}")
    print(f"    Reason:     {trade_signal['reason']}")

    # Execute if actionable
    print("\n[3] Executing ...")
    if signal.should_act(trade_signal):
        result = broker.execute(trade_signal)
    else:
        result = {"status": "SKIPPED",
                   "reason": f"Confidence {trade_signal['confidence']} below threshold"}
        print(f"    SKIPPED — {result['reason']}")

    # Summary
    print("\n[4] Account Summary")
    summary = broker.get_summary()
    for k, v in summary.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 56)

    return data, trade_signal, result


if __name__ == "__main__":
    main()
