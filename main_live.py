import time
import yfinance as yf
from strategy import threshold_strategy, moving_average_strategy
from portfolio import Portfolio
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

def get_latest_price(ticker):
    """Fetch the most recent 1-minute price"""
    data = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
    if data.empty:
        return None
    return float(data["Close"].iloc[-1].item())

if __name__ == "__main__":
    cash = 1000
    portfolio = Portfolio(cash=cash, tickers=[])

    print("Choose tickers option:")
    print("1. Enter manually")
    print("2. Use default (AAPL, MSFT, TSLA)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        tickers_input = input("Enter ticker symbols separated by commas (e.g., AAPL, MSFT, TSLA): ")
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    else:
        tickers = ["AAPL", "MSFT", "TSLA"]

    for t in tickers:
        portfolio.add_ticker(t)

    print(f"\nPortfolio created with {portfolio.tickers} and starting cash = {portfolio.cash}\n")

    print("Choose strategy:")
    print("1. Threshold Strategy")
    print("2. Moving Average Strategy")
    strategy_choice = input("Enter 1 or 2: ").strip()

    interval = int(input("Enter refresh interval in seconds (e.g., 60): "))

    print("\n=== Live Trading Started ===")
    history = {ticker: [] for ticker in portfolio.tickers}

    while True:
        for ticker in portfolio.tickers:
            price = get_latest_price(ticker)
            if not price:
                print(f"No data for {ticker}")
                continue

            history[ticker].append(price)
            print(f"{ticker} latest price: {price:.2f}")

            if strategy_choice == "1":
                final_value, holdings, actions = threshold_strategy(history[ticker][-50:], threshold=0.001)
            else:
                final_value, holdings, actions = moving_average_strategy(history[ticker][-100:])

            last_action = actions[-1] if actions else None
            if last_action and last_action[0] != "HOLD":
                action, i, p = last_action
                if action == "BUY":
                    msg = f"{ticker} signal: {Fore.GREEN}{action}{Style.RESET_ALL} at {p:.2f}"
                elif action == "SELL":
                    msg = f"{ticker} signal: {Fore.RED}{action}{Style.RESET_ALL} at {p:.2f}"
                print(msg)
                with open("signals.log", "a") as f:
                    f.write(msg + "\n")

        print("HOLD (waiting for next candle...)\n")
        time.sleep(interval)
