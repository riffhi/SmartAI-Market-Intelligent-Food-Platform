import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error as mae

def analyze_demand(item=None):
    # Load data
    df = pd.read_csv('sales_dataset.csv')

    # Preprocess date
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter data if specific item is provided
    if item:
        df = df[df['item'] == item]
        
        # If no data for the item, return error
        if len(df) == 0:
            return {"error": f"No data found for item: {item}"}

    # Feature engineering
    df['weekday'] = df['date'].dt.dayofweek
    df['weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    # Prepare features
    features = ['weekend', 'month', 'year', 'holiday', 'promotion']
    X = df[features]
    y = df['sales']

    # Ensure enough samples
    if len(X) < 10:
        return {"error": "Insufficient data for analysis"}

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost model (best for prediction)
    model = XGBRegressor(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predict next week's demand
    # Create a sample input for prediction (you might want to customize this)
    next_week_input = pd.DataFrame({
        'weekend': [1],  # Assuming a weekend
        'month': [df['month'].mode().values[0]],  # Most common month
        'year': [df['year'].max()],  # Latest year
        'holiday': [0],  # No holiday
        'promotion': [1]  # With promotion
    })

    # Scale input
    next_week_input_scaled = scaler.transform(next_week_input)

    # Predict demand
    predicted_demand = model.predict(next_week_input_scaled)[0]

    return {
        'Predicted_Demand': int(predicted_demand),
        'Prediction_Details': {
            'Current_Average_Sales': df['sales'].mean(),
            
        }
    }

if __name__ == "__main__":
    print(analyze_demand)