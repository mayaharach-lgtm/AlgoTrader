import csv
from strategy import threshold_strategy

# Load prices from CSV file
prices = []
with open("data/prices.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        prices.append(float(row["price"]))

# Run the threshold strategy
final_value, final_holdings = threshold_strategy(prices, threshold=0.1)

print(f"Final Portfolio Value: {final_value}")
print(f"Final Holdings (stocks owned): {final_holdings}")
