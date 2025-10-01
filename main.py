import yfinance as yf
from strategy import threshold_strategy, moving_average_strategy
from load_stock_data import load_data, get_close_prices, TOP_10_TICKERS
from portfolio import Portfolio
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

if __name__ == "__main__":
    cash = 1000
    portfolio = Portfolio(cash=cash)

    print("Choose tickers option:")
    print("1. Enter manually")
    print("2. Use TOP 10 tickers")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        tickers_input = input("Enter ticker symbols separated by commas (e.g., AAPL, MSFT, TSLA): ")
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    else:
        tickers = TOP_10_TICKERS

    for t in tickers:
        portfolio.add_ticker(t)

    print(f"\nPortfolio created with {portfolio.tickers} and starting cash = {portfolio.cash}")

    print("\nChoose strategy:")
    print("1. Threshold Strategy")
    print("2. Moving Average Strategy")
    strategy_choice = input("Enter 1 or 2: ").strip()

    for ticker in portfolio.tickers:
        print(f"\n=== Running strategy for {ticker} ===")
        data = load_data(ticker, period="6mo")
        prices = get_close_prices(data, ticker)

        if not prices:
            print(f"No data for {ticker}, skipping...")
            continue

        if strategy_choice == "1":
            final_value, holdings, actions = threshold_strategy(prices, threshold=0.1)
        else:
            final_value, holdings, actions = moving_average_strategy(prices)

        # Print all BUY/SELL actions with colors
        print("Actions:")
        for action, i, price in actions:
            if action == "BUY":
                print(f"{Fore.GREEN}{action}{Style.RESET_ALL} at {price}")
            elif action == "SELL":
                print(f"{Fore.RED}{action}{Style.RESET_ALL} at {price}")

        # Print summary
        print(f"\nFinal Value for {ticker}: {final_value:.2f}")
        print(f"Holdings left: {holdings}")

    print("\n=== Analysis Finished ===")
