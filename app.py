import os
import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from geopy.distance import geodesic

class AIDeliveryOptimizationSystem:
    def __init__(self):
        # Load and prepare datasets
        try:
            self.delivery_data = pd.read_csv("synthetic_delivery_data.csv")
            
            # Calculate additional features
            self.calculate_additional_features()
            
            st.write("Available columns:", list(self.delivery_data.columns))
        except Exception as e:
            st.error(f"Error loading delivery data: {e}")
            self.delivery_data = pd.DataFrame()
        
        # Initialize models
        self.route_optimization_model = None
        self.demand_forecasting_model = None
        
        # Preprocessing attributes
        self.route_scaler = StandardScaler()

    def calculate_additional_features(self):
        """Calculate distance and duration features"""
        # Calculate distance between origin and destination
        self.delivery_data['distance'] = self.delivery_data.apply(
            lambda row: geodesic(
                (row['origin_lat'], row['origin_lon']), 
                (row['dest_lat'], row['dest_lon'])
            ).kilometers, 
            axis=1
        )
        
        # Convert weather conditions to numeric
        weather_mapping = {
            'Sunny': 1, 
            'Cloudy': 2, 
            'Rainy': 3, 
            'Snowy': 4, 
            'Windy': 5
        }
        self.delivery_data['weather_code'] = self.delivery_data['weather_condition'].map(
            weather_mapping
        ).fillna(0)
        
        # Convert timestamp to duration (assuming it represents time taken)
        self.delivery_data['delivery_timestamp'] = pd.to_datetime(
            self.delivery_data['delivery_timestamp']
        )
        
        # Estimated temperature (since not in original dataset)
        temp_mapping = {
            'Sunny': 25, 
            'Cloudy': 20, 
            'Rainy': 15, 
            'Snowy': 0, 
            'Windy': 18
        }
        self.delivery_data['temperature'] = self.delivery_data['weather_condition'].map(
            temp_mapping
        ).fillna(20)

    def preprocess_route_data(self):
        """Prepare data for route optimization"""
        # Select specific features
        route_features = [
            'distance', 'weather_code', 'temperature', 
            'origin_lat', 'origin_lon', 'dest_lat', 'dest_lon'
        ]
        
        # Select and clean route data
        route_df = self.delivery_data[route_features].dropna()
        
        # If not enough data, use all available data
        if len(route_df) < 10:
            st.warning("Insufficient data. Using all available data.")
        
        # Scale features
        X_scaled = self.route_scaler.fit_transform(route_df)
        
        return X_scaled, route_df

    def train_route_optimization_model(self):
        """Train a machine learning model for route optimization"""
        # Preprocess route data
        X_scaled, route_df = self.preprocess_route_data()
        
        # Use actual delivery time as target
        y = self.delivery_data['actual_delivery_time']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest Regressor
        self.route_optimization_model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42
        )
        self.route_optimization_model.fit(X_train, y_train)
        
        return self.route_optimization_model

    def optimize_delivery_route(self, input_route_features):
        """
        Optimize a specific delivery route
        
        Args:
        input_route_features (dict): Dictionary with route features
        
        Returns:
        dict: Optimized route recommendations
        """
        # Ensure model is trained
        if self.route_optimization_model is None:
            self.train_route_optimization_model()
        
        # Prepare input features
        features_df = pd.DataFrame([input_route_features])
        
        # Scale input features
        scaled_features = self.route_scaler.transform(features_df)
        
        # Predict optimal route duration
        predicted_duration = self.route_optimization_model.predict(scaled_features)[0]
        
        # Feature importance 
        feature_importance = pd.DataFrame({
            'feature': [
                'distance', 'weather_code', 'temperature', 
                'origin_lat', 'origin_lon', 'dest_lat', 'dest_lon'
            ],
            'importance': self.route_optimization_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'predicted_duration': predicted_duration,
            'feature_importance': feature_importance
        }

def main():
    st.title("AI-Powered Delivery Optimization System")
    
    # Initialize the optimization system
    system = AIDeliveryOptimizationSystem()
    
    # Sidebar for system controls
    st.sidebar.header("AI Optimization Tools")
    
    # Route Optimization Section
    st.header("🚚 AI Route Optimization")
    
    # Input fields for route features
    col1, col2 = st.columns(2)
    with col1:
        origin_lat = st.number_input("Origin Latitude", min_value=-90.0, max_value=90.0, value=40.7128, step=0.0001)
        origin_lon = st.number_input("Origin Longitude", min_value=-180.0, max_value=180.0, value=-74.0060, step=0.0001)
        weather_code = st.selectbox("Weather Condition", [1, 2, 3, 4, 5], 
                                    format_func=lambda x: {1: 'Sunny', 2: 'Cloudy', 3: 'Rainy', 4: 'Snowy', 5: 'Windy'}[x])
    
    with col2:
        dest_lat = st.number_input("Destination Latitude", min_value=-90.0, max_value=90.0, value=40.7282, step=0.0001)
        dest_lon = st.number_input("Destination Longitude", min_value=-180.0, max_value=180.0, value=-73.7949, step=0.0001)
        temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=20.0)
    
    # Calculate distance
    from geopy.distance import geodesic
    distance = geodesic((origin_lat, origin_lon), (dest_lat, dest_lon)).kilometers
    
    route_features = {
        'distance': distance,
        'weather_code': weather_code,
        'temperature': temperature,
        'origin_lat': origin_lat,
        'origin_lon': origin_lon,
        'dest_lat': dest_lat,
        'dest_lon': dest_lon
    }
    
    if st.button("Optimize Route"):
        route_optimization = system.optimize_delivery_route(route_features)
        
        st.subheader("Route Optimization Results")
        st.write(f"Predicted Delivery Duration: {route_optimization['predicted_duration']:.2f} minutes")
        
        st.subheader("Route Optimization Feature Importance")
        st.dataframe(route_optimization['feature_importance'])

if __name__ == "__main__":
    main()