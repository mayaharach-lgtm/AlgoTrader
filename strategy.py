def threshold_strategy(prices, threshold=0.1):
    cash = 1000
    holdings = 0
    actions = []

    for i in range(1, len(prices)):
        if prices[i] < prices[i - 1] * (1 - threshold):
            holdings += 1
            cash -= prices[i]
            actions.append(("BUY", i, prices[i]))
        elif prices[i] > prices[i - 1] * (1 + threshold) and holdings > 0:
            holdings -= 1
            cash += prices[i]
            actions.append(("SELL", i, prices[i]))
        else:
            actions.append(("HOLD", i, prices[i]))

    total_value = cash + holdings * prices[-1]
    return total_value, holdings, actions


def moving_average_strategy(prices, short_window=10, long_window=30):
    """
    A simple moving average (SMA) strategy.
    Buy when the short moving average crosses above the long moving average.
    Sell when the short moving average crosses below the long moving average.
    """
    cash = 1000
    holdings = 0
    actions = []

    # Compute moving averages
    short_ma = [sum(prices[i-short_window+1:i+1]) / short_window if i >= short_window-1 else None for i in range(len(prices))]
    long_ma = [sum(prices[i-long_window+1:i+1]) / long_window if i >= long_window-1 else None for i in range(len(prices))]

    for i in range(len(prices)):
        if short_ma[i] is None or long_ma[i] is None:
            continue

        # Buy signal
        if short_ma[i] > long_ma[i] and holdings == 0:
            holdings += 1
            cash -= prices[i]
            actions.append(("BUY", i, prices[i]))
            print(f"BUY at {prices[i]} | cash={cash} | holdings={holdings}")

        # Sell signal
        elif short_ma[i] < long_ma[i] and holdings > 0:
            holdings -= 1
            cash += prices[i]
            actions.append(("SELL", i, prices[i]))
            print(f"SELL at {prices[i]} | cash={cash} | holdings={holdings}")

        else:
            actions.append(("HOLD", i, prices[i]))

    total_value = cash + holdings * prices[-1]
    return total_value, holdings, actions



def merge_intervals(actions):
    merged = []
    hold_start = None
    hold_end = None

    for action, day, price in actions:
        if action == "HOLD":
            if hold_start is None:
                hold_start = (day, price)
            hold_end = (day, price)
        else:
            if hold_start is not None:
                merged.append(("HOLD", (hold_start[0], hold_end[0]), (hold_start[1], hold_end[1])))
                hold_start, hold_end = None, None
            merged.append((action, day, price))

    if hold_start is not None:
        merged.append(("HOLD", (hold_start[0], hold_end[0]), (hold_start[1], hold_end[1])))

    return merged
