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


def moving_average_strategy(prices, short_window=5, long_window=20):
    if len(prices) < long_window:
        return 0, 0, []  # not enough data

    cash = 1000
    holdings = 0
    actions = []

    for i in range(long_window, len(prices)):
        short_avg = sum(prices[i-short_window:i]) / short_window
        long_avg = sum(prices[i-long_window:i]) / long_window

        if short_avg > long_avg and cash >= prices[i]:
            holdings += 1
            cash -= prices[i]
            actions.append(("BUY", i, prices[i]))
        elif short_avg < long_avg and holdings > 0:
            holdings -= 1
            cash += prices[i]
            actions.append(("SELL", i, prices[i]))
        else:
            actions.append(("HOLD", i, prices[i]))

    total_value = cash + holdings * prices[-1]
    return total_value, holdings, actions
