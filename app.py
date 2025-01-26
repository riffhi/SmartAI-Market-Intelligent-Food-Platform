import streamlit as st
import pandas as pd
import numpy as np
import requests
import random
import xgboost as xgb
import matplotlib.pyplot as  plt
from ast import literal_eval
from geopy.distance import geodesic
import urllib3
from sklearn.model_selection import train_test_split
from nltk.sentiment import SentimentIntensityAnalyzer
import folium
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from streamlit_folium import folium_static
from geopy.distance import geodesic
from features.route_optimizer import train_with_lightgbm, prepare_training_data
from features.sentiment import polarity_scores_roberta
from features.vendor import match_vendor
from features.waste import generate_recommendations
from features.demand_forecasting import analyze_demand
from features.price_predict import adjust_prices
import googlemaps
import polyline

def apply_custom_styles():
    """Apply custom CSS styling to improve UI"""
    st.markdown("""
    <style>
    body {
        background-color: #FDF6E3;
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }
   .stApp {
        max-width: 100%;  # Change from 85% to 100%
        margin: 0;  # Remove margin
        display: flex;  # Use flexbox layout
        background-color: #FFFAF0;
        border-radius: 15px;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
        padding: 20px;
    }
    .stButton>button {
        background-color: #FF6F61;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color:rgb(182, 67, 67);
    }
    .stSidebar {
        background-color: #FFDAC1;
        border-radius: 10px;
        padding:0px;
    }
    .stSidebar .stRadio {
        background-color: #FFF7E6;
       padding:10px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #FF4500;
    }
    
    """, unsafe_allow_html=True)

def main():
    apply_custom_styles()

    st.title("🎉 SmartAI Market: Your Playful Food Platform")
    st.markdown("""
    ### 🌟 Welcome to Your AI-Powered Food Service Optimization Platform
    Let's make food delivery fun, efficient, and sustainable with the power of AI! Explore features below. 🍕🚀
    """)

    # Initialize AI platform
    food_ai = FoodPlatformAI()

    # Sidebar
    st.sidebar.title("🎯 AI Features")
    st.sidebar.markdown("Select a feature to play with:")

     # Feature selection sidebar
    feature = st.sidebar.radio("Select AI Feature", [
        "🌍 Route Optimization & Delivery Time Prediction",
        "🤝 Vendor Matching", 
        "💬 Sentiment Analysis",
        "♻️ Waste Reduction",
        "📈 Demand Forecasting",
        "💰 Price Optimization"
    ])
    
    # Render selected feature's page
    if feature == "🌍 Route Optimization & Delivery Time Prediction":
        route_optimization_page(food_ai)
    elif feature == "🤝 Vendor Matching":
    elif feature == "🤝 Vendor Matching":
        vendor_matching_page(food_ai.vendors_df, food_ai.customers_df)
    elif feature == "💬 Sentiment Analysis":
    elif feature == "💬 Sentiment Analysis":
        sentiment_analysis_page(food_ai)
    elif feature == "♻️ Waste Reduction":
    elif feature == "♻️ Waste Reduction":
        waste_reduction_page(food_ai)
    elif feature == "📈 Demand Forecasting":
    elif feature == "📈 Demand Forecasting":
        demand_forecasting_page(food_ai)
    elif feature == "💰 Price Optimization":
    elif feature == "💰 Price Optimization":
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
        


def add_location_autocomplete(api_key):
    """
    Add Google Places Autocomplete to Streamlit location inputs
    
    Args:
        api_key (str): Google Maps API key
    """
    # Initialize Google Maps Client
    gmaps = googlemaps.Client(key=api_key)

    # Custom JavaScript to handle autocomplete
    autocomplete_script = f"""
    <script>
    const initAutocomplete = () => {{
        const sourceInput = document.querySelector('input[aria-label="Enter Source Address"]');
        const destInput = document.querySelector('input[aria-label="Enter Destination Address"]');
        
        const sourceAutocomplete = new google.maps.places.Autocomplete(sourceInput);
        const destAutocomplete = new google.maps.places.Autocomplete(destInput);
    }};
    </script>
    <script src="https://maps.googleapis.com/maps/api/js?key={api_key}&libraries=places&callback=initAutocomplete" async defer></script>
    """
    
    # Inject the script into Streamlit
    st.markdown(autocomplete_script, unsafe_allow_html=True)


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


def get_location_suggestions(query):
    """Fetch location suggestions using Google Places API with enhanced error handling"""
    API_KEY = 'AIzaSyBKdn9ak2NzS0xEA7nJMpTjmqEPI4JJzZI'
    
    # Disable SSL warnings if needed
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    base_url = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
    params = {
        'input': query,
        'types': 'geocode',
        'key': API_KEY
    }
    
    try:
        # Add timeout and retry mechanism
        response = requests.get(
            base_url, 
            params=params, 
            timeout=10,
            verify=False  # Disable SSL verification if needed
        )
        
        # Check for successful response
        response.raise_for_status()
        
        data = response.json()
        
        if data['status'] == 'OK':
            suggestions = [
                prediction['description'] 
                for prediction in data.get('predictions', [])
            ]
            return suggestions
        
        # Log any API-specific errors
        st.warning(f"Google Places API returned status: {data['status']}")
        return []
    
    except requests.exceptions.RequestException as e:
        st.error(f"Network error fetching suggestions: {e}")
        return []
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return []

def location_autocomplete(label):
    """Create an autocomplete input for locations"""
    query = st.text_input(label)
    
    if query:
        suggestions = get_location_suggestions(query)
        
        if suggestions:
            selected_location = st.selectbox(
                f"Suggested {label}", 
                [""] + suggestions,
                key=label
            )
            
            if selected_location:
                return selected_location
    
    return query

def route_optimization_page(food_ai=None):
    st.header("🚚 Route Optimization & Delivery Time Prediction")
    
    # Fixed API Key
    API_KEY = 'AIzaSyBKdn9ak2NzS0xEA7nJMpTjmqEPI4JJzZI'
    
    try:
        # Initialize Google Maps Client
        gmaps = googlemaps.Client(key=API_KEY)
        
        # Source and Destination Input
        col1, col2 = st.columns(2)
        
        with col1:
            source_address = location_autocomplete("Enter Source Address")
        
        with col2:
            dest_address = location_autocomplete("Enter Destination Address")
        
        # Route Analysis Button
        if st.button("Calculate Route"):
            if not source_address or not dest_address:
                st.error("Please enter both source and destination addresses.")
                return
            
            try:
                # Geocode addresses
                source_geocode = gmaps.geocode(source_address)[0]['geometry']['location']
                dest_geocode = gmaps.geocode(dest_address)[0]['geometry']['location']
                
                # Get directions
                directions = gmaps.directions(
                    source_address, 
                    dest_address, 
                    mode="driving"
                )
                
                # Extract route details
                route = directions[0]['legs'][0]
                
                # Create a map centered on the route
                center_lat = (source_geocode['lat'] + dest_geocode['lat']) / 2
                center_lng = (source_geocode['lng'] + dest_geocode['lng']) / 2
                
                # Create Folium map
                m = folium.Map(location=[center_lat, center_lng], zoom_start=10)
                
                # Add markers for source and destination
                folium.Marker(
                    [source_geocode['lat'], source_geocode['lng']], 
                    popup=source_address,
                    icon=folium.Icon(color='green', icon='play')
                ).add_to(m)
                
                folium.Marker(
                    [dest_geocode['lat'], dest_geocode['lng']], 
                    popup=dest_address,
                    icon=folium.Icon(color='red', icon='stop')
                ).add_to(m)
                
                # Decode route polyline
                route_coords = polyline.decode(directions[0]['overview_polyline']['points'])
                
                # Add route line
                folium.PolyLine(
                    route_coords, 
                    color='blue', 
                    weight=5, 
                    opacity=0.8
                ).add_to(m)
                
                # Display route map
                st.subheader("Route Map")
                folium_static(m)
                
                # Distance conversion and time estimation
                distance_meters = route['distance']['value']
                distance_miles = distance_meters / 1609.34  # Convert meters to miles
                distance_km = distance_meters / 1000  # Convert meters to kilometers

                # Estimated delivery time calculation
                # Assuming average delivery speed of 50 mph (about 80 km/h)
                # Include loading time and potential traffic considerations
                estimated_hours = distance_km / 50
                estimated_minutes = estimated_hours * 60 + 15  # Add 15 minutes loading time

                st.write(f"📏 Distance: {distance_km:.2f} km ({distance_miles:.2f} miles)")
                distance_meters = route['distance']['value']
                distance_miles = distance_meters / 1609.34  # Convert meters to miles
                distance_km = distance_meters / 1000  # Convert meters to kilometers

                estimated_time = route['duration']['text']  # e.g., "1 hour 30 mins"

                st.write(f"⏱️ Estimated Time of Arrival: {estimated_time}")

                
                
                
                # Optional: Display route steps
                if st.checkbox("Show Route Steps"):
                    st.subheader("Route Directions")
                    for step in route['steps']:
                        st.markdown(f"- {step['html_instructions']}")
                
            except Exception as e:
                st.error(f"Error calculating route: {e}")
    
    except Exception as api_error:
        st.error(f"Error initializing Google Maps: {api_error}")



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
    """
    Advanced AI-powered vendor matching using machine learning techniques
    
    Args:
    - preference (str): Customer's food preference
    - max_time (int): Maximum acceptable delivery time
    - vendors_df (pd.DataFrame): DataFrame of vendors
    
    Returns:
    - pd.DataFrame: Matched and ranked vendors
    """
    # Preprocess and feature engineering
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Delivery_Time', 'Inventory']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Category'])
        ])
    
    # Prepare feature matrix
    features = vendors_df.copy()
    features_processed = preprocessor.fit_transform(features)
    
    # Initial filtering based on criteria
    matched_vendors = vendors_df[
        (vendors_df['Category'] == preference) & 
        (vendors_df['Delivery_Time'] <= max_time)
    ]
    
    if matched_vendors.empty:
        return pd.DataFrame(columns=['Name', 'Category', 'Delivery_Time', 'Match_Score'])
    
    # Compute matching scores using cosine similarity
    match_scores = cosine_similarity(
        features_processed[matched_vendors.index],
        features_processed[matched_vendors.index]
    ).mean(axis=1)
    
    # Create result DataFrame with matching scores
    result_df = matched_vendors.copy()
    result_df['Match_Score'] = match_scores
    
    # Enhance with additional ranking factors
    result_df['Ranking_Score'] = (
        0.4 * result_df['Match_Score'] + 
        0.3 * (1 / result_df['Delivery_Time']) + 
        0.3 * (result_df['Inventory'] / result_df['Inventory'].max())
    )
    
    # Sort by ranking score in descending order
    result_df = result_df.sort_values('Ranking_Score', ascending=False)
    
    # Select and return top columns
    return result_df[['Name', 'Category', 'Delivery_Time']]


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
