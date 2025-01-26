import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import googlemaps
import gmplot

# Data Preparation Function
def prepare_training_data(raw_data):
    """Prepare the data for training by extracting features and ensuring all required columns exist."""
    processed_data = []
    for _, row in raw_data.iterrows():
        try:
            # Extract relevant features
            processed_data.append({
                'distance': row.get('distance', np.nan),
                'duration': row.get('duration', np.nan),
                'weather_code': row.get('weather_code', np.nan),
                'temperature': row.get('temperature', np.nan),
                'time_of_day': pd.to_datetime(row['delivery_timestamp']).hour,
                'day_of_week': pd.to_datetime(row['delivery_timestamp']).dayofweek,
                'actual_delivery_time': row['actual_delivery_time']
            })
        except Exception as e:
            print(f"Error processing row: {e}")
    
    # Convert to DataFrame
    processed_data = pd.DataFrame(processed_data)
    
    # Drop rows with missing target values
    processed_data.dropna(subset=['actual_delivery_time'], inplace=True)
    return processed_data

# Training Function
def train_with_lightgbm(processed_data):
    features = ['distance', 'duration', 'weather_code', 'temperature', 'time_of_day', 'day_of_week']
    target = 'actual_delivery_time'
    
    X = processed_data[features]
    y = processed_data[target]
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # LightGBM Dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # LightGBM Parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # Train the Model
    print("Training LightGBM model...")
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Predictions and Evaluation
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R^2 Score: {r2}")
    
    # Save the model
    model.save_model('lightgbm_delivery_model.txt')
    print("Model saved to 'lightgbm_delivery_model.txt'.")
    return model

# Visualization Function
def visualize_route(api_key, start_lat, start_lng, end_lat, end_lng):
    """Visualize the route between two points using Google Maps."""
    gmaps = googlemaps.Client(key=api_key)
    directions = gmaps.directions((start_lat, start_lng), (end_lat, end_lng), mode="driving")
    
    # Extract route details
    route = directions[0]['legs'][0]['steps']
    latitudes = [step['end_location']['lat'] for step in route]
    longitudes = [step['end_location']['lng'] for step in route]
    
    # Plot route on Google Maps
    gmap = gmplot.GoogleMapPlotter(start_lat, start_lng, 13)
    gmap.plot(latitudes, longitudes, color='red', edge_width=2.5)
    gmap.draw("delivery_route_map.html")
    print("Route map saved as 'delivery_route_map.html'.")

# Main Execution
if __name__ == "_main_":
    # Load the raw data
    try:
        raw_data = pd.read_csv('synthetic_delivery_data.csv')
    except FileNotFoundError:
        print("File 'synthetic_delivery_data.csv' not found. Please ensure the file is in the working directory.")
        exit()
    
    # Prepare the data
    processed_data = prepare_training_data(raw_data)
    
    if processed_data.empty:
        print("No valid data for training. Please check the data source.")
    else:
        # Train the model
        trained_model = train_with_lightgbm(processed_data)
        
        # Define route details
        API_KEY = 'AIzaSyBKdn9ak2NzS0xEA7nJMpTjmqEPI4JJzZI'  # Replace with your Google Maps API key
        start_lat = 37.7749  # Example: San Francisco latitude
        start_lng = -122.4194  # Example: San Francisco longitude
        end_lat = 34.0522  # Example: Los Angeles latitude
        end_lng = -118.2437  # Example: Los Angeles longitude
        
        # Visualize the route
        visualize_route(API_KEY, start_lat, start_lng, end_lat, end_lng)