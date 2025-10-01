from datetime import datetime
# portfolio.py

class Portfolio:
    def __init__(self, cash, tickers=None):
        """
        Portfolio class to track cash, tickers, and holdings.
        tickers is optional – defaults to an empty list.
        """
        self.cash = cash
        self.tickers = tickers if tickers else []
        self.holdings = {t: 0 for t in self.tickers}

    def add_ticker(self, ticker):
        """Add a new ticker to the portfolio if not already included."""
        if ticker not in self.tickers:
            self.tickers.append(ticker)
            self.holdings[ticker] = 0

    def buy(self, ticker, price, quantity=1):
        """Simulate buying a stock (reduces cash, increases holdings)."""
        cost = price * quantity
        if self.cash >= cost:
            self.cash -= cost
            self.holdings[ticker] += quantity
            print(f"Bought {quantity} of {ticker} at {price}. Cash left: {self.cash}")
        else:
            print("Not enough cash to complete the purchase.")

    def sell(self, ticker, price, quantity=1):
        """Simulate selling a stock (increases cash, decreases holdings)."""
        if self.holdings.get(ticker, 0) >= quantity:
            self.holdings[ticker] -= quantity
            self.cash += price * quantity
            print(f"Sold {quantity} of {ticker} at {price}. Cash now: {self.cash}")
        else:
            print(f"Not enough {ticker} to sell.")

    def portfolio_value(self, current_prices):
        """
        Calculate total portfolio value given a dict of {ticker: price}.
        """
        value = self.cash
        for t, qty in self.holdings.items():
            if t in current_prices:
                value += qty * current_prices[t]
        return value

    def summary(self, current_prices):
        """
        Print a summary of portfolio holdings and total value.
        """
        print("\n=== Portfolio Summary ===")
        print(f"Cash: {self.cash:.2f}")
        for t, qty in self.holdings.items():
            if qty > 0:
                print(f"{t}: {qty} shares (latest price: {current_prices.get(t, 'N/A')})")
        print(f"Total Value: {self.portfolio_value(current_prices):.2f}")
