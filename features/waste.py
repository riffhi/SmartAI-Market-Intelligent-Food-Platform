import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import random

# Load data
sales_data = pd.read_csv("food_sales_data.csv")
waste_data = pd.read_csv("food_waste_data.csv")

# Merge datasets for enhanced analysis
data = pd.merge(waste_data, sales_data, left_on='Item', right_on='item_name', how='inner')

# Feature Engineering: Extract key features
features = [
    'DayOfWeek', 'TimeOfDay', 'Weather', 'PastSales', 'CurrentStock', 
    'DiscountOffered', 'CustomerCount', 'customer_behavior', 'portion_size'
]

# Encode categorical variables
data = pd.get_dummies(data, columns=['DayOfWeek', 'TimeOfDay', 'Weather'], drop_first=True)

# Define target variables
waste_target = data['WasteGenerated']
portion_target = data['portion_size']

# Define feature set
X = data.drop(['WasteGenerated', 'item_name', 'Item', 'portion_size'], axis=1)

# Split data into training and testing sets for waste prediction
X_train_waste, X_test_waste, y_train_waste, y_test_waste = train_test_split(
    X, waste_target, test_size=0.2, random_state=42
)

# Train a machine learning model for waste prediction
waste_model = RandomForestRegressor(random_state=42)
waste_model.fit(X_train_waste, y_train_waste)

# Predict waste and evaluate
waste_predictions = waste_model.predict(X_test_waste)
waste_mse = mean_squared_error(y_test_waste, waste_predictions)
print(f"Waste Prediction MSE: {waste_mse}")

# Split data into training and testing sets for portion size prediction
X_train_portion, X_test_portion, y_train_portion, y_test_portion = train_test_split(
    X, portion_target, test_size=0.2, random_state=42
)

# Train a machine learning model for portion size recommendation
portion_model = RandomForestRegressor(random_state=42)
portion_model.fit(X_train_portion, y_train_portion)

# Predict portion sizes and evaluate
portion_predictions = portion_model.predict(X_test_portion)
portion_mse = mean_squared_error(y_test_portion, portion_predictions)
print(f"Portion Size Prediction MSE: {portion_mse}")

# Function to generate recommendations
def generate_recommendations(input_data):
    # Ensure all required categorical columns exist before encoding
    categorical_columns = ['DayOfWeek', 'TimeOfDay', 'Weather']
    for col in categorical_columns:
        if col not in input_data.columns:
            input_data[col] = None  # Add the column with default values

    # Perform one-hot encoding
    input_data = pd.get_dummies(input_data, columns=categorical_columns, drop_first=True)

    # Align columns with the training data
    missing_cols = set(X.columns) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0
    input_data = input_data[X.columns]  # Align columns

    # Make predictions
    waste_pred = waste_model.predict(input_data)
    portion_pred = portion_model.predict(input_data)

    # Create recommendations dataframe
    recommendations = pd.DataFrame({
        'PredictedWaste': waste_pred,
        'RecommendedPortionSize': portion_pred
    })

    return recommendations

# Example usage
example_data = X_test_waste.copy()
recommendations = generate_recommendations(example_data)
print(recommendations.head())


# Function to generate random test data
def generate_random_test_data(num_rows=10):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    times = ['Morning', 'Afternoon', 'Evening', 'Night']
    weathers = ['Sunny', 'Rainy', 'Cloudy', 'Snowy']
    
    random_data = {
        'DayOfWeek': [random.choice(days) for _ in range(num_rows)],
        'TimeOfDay': [random.choice(times) for _ in range(num_rows)],
        'Weather': [random.choice(weathers) for _ in range(num_rows)],
        'PastSales': [random.randint(50, 500) for _ in range(num_rows)],
        'CurrentStock': [random.randint(10, 50) for _ in range(num_rows)],
        'DiscountOffered': [random.randint(0, 50) for _ in range(num_rows)],
        'CustomerCount': [random.randint(20, 100) for _ in range(num_rows)],
        'customer_behavior': [random.uniform(0.4, 0.6) for _ in range(num_rows)],
        'portion_size': [random.uniform(200, 350) for _ in range(num_rows)],
    }
    
    return pd.DataFrame(random_data)

# Generate random test data
random_test_data = generate_random_test_data(num_rows=10)

# Generate recommendations using the random test data
recommendations = generate_recommendations(random_test_data)
print(random_test_data)
print(recommendations)
