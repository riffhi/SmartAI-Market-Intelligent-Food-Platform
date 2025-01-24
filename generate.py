import pandas as pd
import random
from datetime import datetime, timedelta

# Define parameters
items = ["Pizza", "Burger", "Salad", "Pasta", "Sushi"]
start_date = datetime(2024, 1, 1)
num_entries = 2000

# Generate dataset
data = []
for i in range(num_entries):
    date = start_date + timedelta(days=i)
    item = random.choice(items)
    sales = random.randint(50, 300)  # Random sales between 50 and 300
    day_of_week = date.strftime("%A")
    holiday = random.choice([0, 1]) if day_of_week in ["Saturday", "Sunday"] else 0
    promotion = random.choice([0, 1])
    data.append([date.strftime("%Y-%m-%d"), item, sales, day_of_week, holiday, promotion])

# Create DataFrame
df = pd.DataFrame(data, columns=["date", "item", "sales", "day_of_week", "holiday", "promotion"])

# Save to CSV
df.to_csv("sales_dataset.csv", index=False)

print("Dataset generated and saved as 'sales_dataset.csv'.")
