import pandas as pd
import numpy as np

# Function to generate a synthetic dataset
def generate_food_sales_data(num_items=100):
    # Define some sample food items
    food_items = [
        'Burger', 'Pizza', 'Salad', 'Pasta', 'Sushi', 
        'Tacos', 'Sandwich', 'Wrap', 'Steak', 'Fish & Chips'
    ]
    
    # Generate random data
    data = {
        'item_name': np.random.choice(food_items, num_items),
        'sales_volume': np.random.randint(10, 200, size=num_items),  # Random sales volume between 10 and 200
        'customer_behavior': np.random.rand(num_items),  # Random behavior score between 0 and 1
        'portion_size': np.random.randint(100, 500, size=num_items)  # Random portion sizes between 100g and 500g
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Group by item name to aggregate sales volume and average behaviors
    df = df.groupby('item_name').agg({
        'sales_volume': 'sum',
        'customer_behavior': 'mean',
        'portion_size': 'mean'
    }).reset_index()
    
    return df

# Generate dataset with specified number of items
num_items = 1000  # You can adjust this number based on your requirements
dataset = generate_food_sales_data(num_items)

# Save the dataset to a CSV file
dataset.to_csv('food_sales_data.csv', index=False)

print("Dataset generated and saved to food_sales_data.csv")
