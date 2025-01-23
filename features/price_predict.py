import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("small_product_dataset.csv")

# Convert categorical demand to numeric values
df['Demand'] = df['Demand'].map({'Low': 100, 'Medium': 500, 'High': 1000})

# Prepare the data
X = df[['Price']]
y = df['Demand']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data into DMatrix format for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters for XGBoost
params = {
    'objective': 'reg:squarederror',
    'max_depth': 3,
    'eta': 0.1
}

# Train the model
bst = xgb.train(params, dtrain, num_boost_round=100)

# Predict demand based on current prices
df['Predicted_Demand'] = bst.predict(xgb.DMatrix(X))

# Function to adjust prices based on predicted demand
def adjust_prices(row):
    if row['Predicted_Demand'] < 350:
        return row['Price'] * 0.9  # Apply 10% discount if predicted demand is low
    else:
        return row['Price'] * 1.1  # Increase by 10% otherwise

# Apply the function to adjust prices
df['Adjusted_Price'] = df.apply(adjust_prices, axis=1)

# Save updated data to CSV
df.to_csv("adjusted_small_product_prices.csv", index=False)

print("Adjusted prices saved to 'adjusted_small_product_prices.csv'.")