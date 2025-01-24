import streamlit as st
import pandas as pd
import numpy as np
import random
import xgboost as xgb
import matplotlib.pyplot as  plt
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
from nltk.sentiment import SentimentIntensityAnalyzer

from features.route_optimizer import train_with_lightgbm, prepare_training_data
from features.sentiment import polarity_scores_roberta
from features.vendor import match_vendor
from features.waste import generate_recommendations
from features.demand_forecasting import analyze_demand
from features.price_predict import adjust_prices

def apply_custom_styles():
    """Apply custom CSS styling to improve UI"""
    st.markdown("""
    <style>
    .main {
    background-color: #F0F2F6;
}
.stApp {
    max-width: 100%;
    margin: 0 auto;
}
.stHeader {
    color: #2C3E50;
    font-weight: bold;
}
.stButton>button {
    background-color: #3498DB;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    transition: background-color 0.3s ease;
}
.stButton>button:hover {
    background-color: #2980B9;
}
.stSidebar .stRadio {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
    </style>
    """, unsafe_allow_html=True)

def main():
    apply_custom_styles()
    
    
    st.title("🍽️ SmartAI Market: Intelligent Food Platform")
    st.markdown("""
    ### Your AI-Powered Food Service Optimization Platform
    Leveraging advanced machine learning to enhance food delivery, waste reduction, and pricing strategies.
    """)
    
    # Initialize AI platform
    food_ai = FoodPlatformAI()
    
    # Enhanced sidebar with card-like design
    st.sidebar.title("🤖 AI Features")
    st.sidebar.markdown("Select an intelligent feature to explore:")

     # Feature selection sidebar
    feature = st.sidebar.radio("Select AI Feature", [
        "Route Optimization & Delivery Time Prediction",
        "Vendor Matching", 
        "Sentiment Analysis",
        "Waste Reduction",
        "Demand Forecasting",
        "Price Optimization"
    ])
    
    # Render selected feature's page
    if feature == "Route Optimization & Delivery Time Prediction":
        route_optimization_page(food_ai)
    elif feature == "Vendor Matching":
        vendor_matching_page(food_ai.vendors_df, food_ai.customers_df)
    elif feature == "Sentiment Analysis":
        sentiment_analysis_page(food_ai)
    elif feature == "Waste Reduction":
        waste_reduction_page(food_ai)
    elif feature == "Demand Forecasting":
        demand_forecasting_page(food_ai)
    elif feature == "Price Optimization":
        price_optimization_page(food_ai)

class FoodPlatformAI:
    def __init__(self):
        # Load existing datasets
        try:
            self.vendors_df = pd.read_csv("/mnt/data/large_vendors_dataset.csv")
            self.customers_df = pd.read_csv("/mnt/data/large_customers_dataset.csv")
            self.delivery_data = pd.read_csv("synthetic_delivery_data.csv")
            self.waste_data = pd.read_csv("food_waste_data.csv")
            self.sales_data = pd.read_csv("food_sales_data.csv")
        except FileNotFoundError as e:
            st.error(f"Dataset not found: {e}")
            return None


    def ai_route_optimization(self, start_location, end_location):
        """Enhanced route optimization using LightGBM model"""
        try:
            # Prepare data for route prediction
            processed_data = prepare_training_data(self.delivery_data)
            
            # Train model if not already trained
            route_model = train_with_lightgbm(processed_data)
            
            # Calculate distance and create feature set
            distance = geodesic(start_location, end_location).kilometers
            sample_input = pd.DataFrame({
                'distance': [distance],
                'duration': [distance * 10],  # Approximate duration
                'weather_code': [random.randint(1, 5)],
                'temperature': [random.uniform(20, 35)],
                'time_of_day': [pd.Timestamp.now().hour],
                'day_of_week': [pd.Timestamp.now().dayofweek]
            })
            
            # Predict delivery time
            predicted_time = route_model.predict(sample_input)[0]
            
            return {
                'Predicted Delivery Time (minutes)': f"{predicted_time:.2f} minutes",
                'Distance (km)': f"{distance:.2f} km",
                'Optimized Route': f"Route from {start_location} to {end_location}"
            }
        
        except Exception as e:
            return {"error": str(e)}

    
    def sentiment_analysis_advanced(self, review):
        """Advanced sentiment analysis using VADER and RoBERTa"""
        try:
            # Use NLTK VADER for initial sentiment
            sia = SentimentIntensityAnalyzer()
            vader_scores = sia.polarity_scores(review)
            
            # Use RoBERTa for advanced sentiment
            roberta_scores = polarity_scores_roberta(review)
            
            return {
                'vader_scores': vader_scores,
                'roberta_scores': roberta_scores,
                'combined_sentiment': 'Positive' if vader_scores['compound'] > 0 else 'Negative'
            }
        except Exception as e:
            return {"error": str(e)}

    def waste_reduction_analysis(self, food_items):
        """Use waste reduction recommendations"""
        try:
            # Generate random test data for waste analysis
            random_test_data = pd.DataFrame({
                'DayOfWeek': [random.choice(['Monday', 'Tuesday', 'Wednesday'])],
                'TimeOfDay': [random.choice(['Morning', 'Afternoon', 'Evening'])],
                'Weather': [random.choice(['Sunny', 'Rainy', 'Cloudy'])],
                'PastSales': [random.randint(50, 500)],
                'CurrentStock': [random.randint(10, 50)],
                'DiscountOffered': [random.randint(0, 50)],
                'CustomerCount': [random.randint(20, 100)],
                'customer_behavior': [random.uniform(0.4, 0.6)],
                'portion_size': [random.uniform(200, 350)]
            })
            
            recommendations = generate_recommendations(random_test_data)
            return recommendations.to_dict(orient='records')[0]
        except Exception as e:
            return {"error": str(e)}

    def demand_forecasting(self, item):
        """Use demand forecasting analysis"""
        try:
            # Prepare data for demand analysis
            demand_results = analyze_demand(item)
            return demand_results
        except Exception as e:
            return {"error": str(e)}
        
def price_optimization_page(food_ai):
    st.header("💰 Smart Price Optimization")
    
    # Load the dataset for price optimization
    try:
        df = pd.read_csv("small_product_dataset.csv")
    except FileNotFoundError:
        st.error("Product dataset not found.")
        return

    # Display original prices
    st.subheader("Original Prices")
    st.dataframe(df[['Name', 'Category', 'Price', 'Demand']])

    # Optimize prices
    if st.button("Optimize Prices"):
        # Define more granular demand-based pricing strategy
        def optimize_price(row):
            base_price = row['Price']
            demand = row['Demand']
            
            if demand == 'Low':
                # For low demand, apply a more significant discount
                return base_price * 0.85
            elif demand == 'Medium':
                # For medium demand, apply a slight adjustment
                return base_price * 0.95
            else:  # High demand
                # For high demand, slightly increase price
                return base_price * 1.1

        # Apply price optimization
        df['Adjusted_Price'] = df.apply(optimize_price, axis=1)

        # Add prediction explanation column
        df['Pricing_Rationale'] = df.apply(
            lambda row: f"{'Discounted' if row['Demand'] == 'Low' else 'Slightly Adjusted'} based on {row['Demand']} demand", 
            axis=1
        )

        # Display results
        st.subheader("Price Optimization Results")
        results_df = df[['Name', 'Category', 'Price', 'Demand', 'Adjusted_Price']]
        st.dataframe(results_df)

        # Visualize price changes
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(results_df['Name'], results_df['Price'], label='Original Price', alpha=0.7)
        ax.bar(results_df['Name'], results_df['Adjusted_Price'], label='Adjusted Price', alpha=0.7)
        ax.set_title('Price Optimization Comparison')
        ax.set_xlabel('Products')
        ax.set_ylabel('Price')
        ax.legend()
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)


def route_optimization_page(food_ai):
    st.header("🚚 AI-Powered Route Optimization")
    
    col1, col2 = st.columns(2)
    with col1:
        start_lat = st.number_input("Start Latitude", value=20.0)
        start_lon = st.number_input("Start Longitude", value=73.0)
    with col2:
        end_lat = st.number_input("End Latitude", value=20.1)
        end_lon = st.number_input("End Longitude", value=73.1)
    
    if st.button("Optimize Route"):
        result = food_ai.ai_route_optimization(
            (start_lat, start_lon), 
            (end_lat, end_lon)
        )
        st.json(result)

def vendor_matching_page(vendors_df, customers_df):
    st.header("🤝 Vendor Matching")
    
    st.write("Find the best food vendors")
    customer_preferences = st.selectbox("Your Food Preferences", 
        ['Vegetarian', 'Non-Vegetarian', 'Vegan', 'Gluten-Free'])
    max_delivery_time = st.slider("Max Delivery Time (minutes)", 30, 90, 60)
    
    if st.button("Find Matching Vendors"):
        matched_vendors = match_vendors(
            customer_preferences, 
            max_delivery_time, 
            vendors_df
        )
        st.dataframe(matched_vendors)

def match_vendors(preference, max_time, vendors_df):
    matched = vendors_df[
        (vendors_df['Category'] == preference) & 
        (vendors_df['Delivery_Time'] <= max_time)
    ]
    return matched[['Name', 'Category', 'Delivery_Time']]


def sentiment_analysis_page(food_ai):
    st.header("💬 Advanced Sentiment Analysis")
    
    review = st.text_area("Enter Customer Review")
    
    if st.button("Analyze Sentiment"):
        result = food_ai.sentiment_analysis_advanced(review)
        st.json(result)

def waste_reduction_page(food_ai):
    st.header("♻️ Waste Reduction Analysis")
    
    food_items = ['Pizza', 'Burger', 'Salad', 'Sushi', 'Tacos']
    st.multiselect("Select Food Items", food_items)
    
    if st.button("Analyze Waste Patterns"):
        result = food_ai.waste_reduction_analysis(food_items)
        st.json(result)

def demand_forecasting_page(food_ai):
    st.header("📊 Demand Forecasting")
    
    item = st.selectbox("Select Food Item", 
        ['Pizza', 'Burger', 'Salad', 'Sushi'])
    
    if st.button("Forecast Demand"):
        result = food_ai.demand_forecasting(item)
        st.json(result)

if __name__ == "__main__":
    main()