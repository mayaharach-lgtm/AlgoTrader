def threshold_strategy(prices, threshold=0.1):
    cash = 1000
    holdings = 0

    for i in range(1, len(prices)):
        if prices[i] < prices[i - 1] * (1 - threshold):
            holdings += 1
            cash -= prices[i]
            print(f"BUY at {prices[i]} | cash={cash} | holdings={holdings}")
        elif prices[i] > prices[i - 1] * (1 + threshold) and holdings > 0:
            holdings -= 1
            cash += prices[i]
            print(f"SELL at {prices[i]} | cash={cash} | holdings={holdings}")
        else:
            print(f"HOLD at {prices[i]} | cash={cash} | holdings={holdings}")

    total_value = cash + holdings * prices[-1]
    return total_value, holdings
