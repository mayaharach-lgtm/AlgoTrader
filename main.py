import time
import matplotlib.pyplot as plt
from strategy import threshold_strategy, moving_average_strategy, merge_intervals
from load_stock_data import load_data, get_close_prices

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]

def run_with_yahoo(strategy_choice: str, period: str = "6mo"):
    results = {}
    all_prices = {}

    for ticker in TICKERS:
        print(f"\n=== Running strategy for {ticker} ===")
        data = load_data(ticker, period=period)

        if data is None or data.empty:
            print(f"No data for {ticker}, skipping...")
            continue

        # Robust close extraction (Series or MultiIndex → list[float])
        prices = get_close_prices(data, ticker=ticker)

        # Run chosen strategy
        if strategy_choice == "1":
            final_value, final_holdings, actions = threshold_strategy(prices, threshold=0.1)
        else:
            # Make sure your strategy signature matches this call
            final_value, final_holdings, actions = moving_average_strategy(
                prices, short_window=10, long_window=30
            )

        actions = merge_intervals(actions)  # merge consecutive HOLDs
        results[ticker] = final_value
        all_prices[ticker] = (prices, actions)

    print("\n===== Final Results =====")
    for t, v in results.items():
        print(f"{t}: {v}")

    if results:
        best_ticker = max(results, key=results.get)
        print(f"\n🏆 Best performer: {best_ticker} with final value {results[best_ticker]}")

        prices, actions = all_prices[best_ticker]

        # Plot chart for best performer
        plt.figure(figsize=(10, 5))
        plt.plot(prices, label=f"{best_ticker} Price")

        buy_x = [i for action, i, price in actions if action == "BUY"]
        buy_y = [price for action, i, price in actions if action == "BUY"]
        sell_x = [i for action, i, price in actions if action == "SELL"]
        sell_y = [price for action, i, price in actions if action == "SELL"]

        plt.scatter(buy_x, buy_y, marker="^", label="BUY", s=100)
        plt.scatter(sell_x, sell_y, marker="v", label="SELL", s=100)

        plt.title(f"Best Performer: {best_ticker}")
        plt.xlabel("Days")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    print("Choose strategy:")
    print("1. Threshold Strategy")
    print("2. Moving Average Strategy")
    strategy_choice = input("Enter 1 or 2: ").strip()

    period_choice = input("Enter period (e.g., 1mo, 3mo, 6mo, 1y): ").strip() or "6mo"

    # Real-time loop
    interval = int(input("Enter refresh interval in seconds (e.g., 60): ").strip() or "60")

    while True:
        run_with_yahoo(strategy_choice, period=period_choice)
        print(f"\n⏳ Waiting {interval} seconds before next run...\n")
        time.sleep(interval)
