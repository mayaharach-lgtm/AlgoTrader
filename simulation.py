import time
import yfinance as yf
from strategy import threshold_strategy, moving_average_strategy
from portfolio import Portfolio

TICKERS = ["AAPL", "MSFT", "TSLA"]

def get_historical_data(ticker, period="6mo", interval="1d"):
    """Download historical prices from Yahoo Finance"""
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if data.empty or "Close" not in data.columns:
        return []
    return list(data["Close"]) 

def simulate_trading(strategy_choice, refresh_interval=0.5):
    # Initialize portfolio with $1000 cash
    portfolio = Portfolio(cash=1000, tickers=TICKERS)
    latest_prices = {}

    for ticker in TICKERS:
        print(f"\n=== Simulating {ticker} ===")
        prices = get_historical_data(ticker)
        if not prices:
            print(f"No data for {ticker}")
            continue

        for i in range(1, len(prices)):
            current_price = prices[i]
            latest_prices[ticker] = current_price

            # Run chosen strategy on prices so far
            if strategy_choice == "1":
                _, _, actions = threshold_strategy(prices[:i+1], threshold=0.02)
            else:
                _, _, actions = moving_average_strategy(prices[:i+1])

            last_action = actions[-1] if actions else None
            if last_action and last_action[0] != "HOLD":
                action, _, p = last_action
                if action == "BUY":
                    portfolio.buy(ticker, p)
                    print(f"BUY {ticker} at {p}")
                elif action == "SELL":
                    portfolio.sell(ticker, p)
                    print(f"SELL {ticker} at {p}")

            time.sleep(refresh_interval)  # simulate "time passing"

    # At the end of simulation, show summary
    print("\n=== Simulation Finished ===")
    portfolio.summary(latest_prices)

if __name__ == "__main__":
    print("Choose strategy:")
    print("1. Threshold Strategy")
    print("2. Moving Average Strategy")
    choice = input("Enter 1 or 2: ").strip()

    simulate_trading(choice, refresh_interval=0.1)
