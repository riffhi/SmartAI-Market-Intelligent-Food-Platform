# import os
# import pandas as pd
# import random
# import numpy as np
# from geopy.distance import geodesic

# output_dir = "/mnt/data"
# os.makedirs(output_dir, exist_ok=True)

# vendors = []
# for i in range(1, 501): 
#     vendor = {
#         "Id": i,
#         "Name": f"Vendor {i}",
#         "Location": (random.uniform(19.0, 21.0), random.uniform(72.0, 75.0)), 
#         "Inventory": random.randint(1, 20),
#         "Delivery_Time": random.randint(30, 90), 
#         "Category": random.choice(['Vegetarian', 'Non-Vegetarian', 'Vegan', 'Gluten-Free']) 
#     }
#     vendors.append(vendor)

# vendors_df = pd.DataFrame(vendors)

# vendors_df.to_csv(os.path.join(output_dir, "large_vendors_dataset.csv"), index=False)

# customers = []
# for i in range(1, 1001):  
#     customer = {
#         "Id": i,
#         "Name": f"Customer {i}",
#         "Location": (random.uniform(19.0, 21.0), random.uniform(72.0, 75.0)),  
#         "Preferences": random.choice(['Vegetarian', 'Non-Vegetarian', 'Vegan', 'Gluten-Free']),  
#         "Max_Delivery_Time": random.randint(30, 60), 
#     }
#     customers.append(customer)

# customers_df = pd.DataFrame(customers)

# customers_df.to_csv(os.path.join(output_dir, "large_customers_dataset.csv"), index=False)

# print("Datasets saved successfully.")

# def calculate_distance(loc1, loc2):
#     return geodesic(loc1, loc2).kilometers

# def match_vendor(customer, vendors_df):
#     matched_vendors = []
#     for _, vendor in vendors_df.iterrows():
#         if customer['Preferences'] == vendor['Category']:
#             distance = calculate_distance(customer['Location'], vendor['Location'])
#             if distance <= 10:  
#                 if vendor['Delivery_Time'] <= customer['Max_Delivery_Time']:  
#                     matched_vendors.append({
#                         'Vendor_Name': vendor['Name'],
#                         'Vendor_Location': vendor['Location'],
#                         'Distance': distance,
#                         'Delivery_Time': vendor['Delivery_Time']
#                     })
#     return matched_vendors

# customer_id = 1  
# customer = customers_df.loc[customers_df['Id'] == customer_id].iloc[0]

# matched_vendors = match_vendor(customer, vendors_df)

# # Display the matched vendors for the customer
# print(f"Matched Vendors for {customer['Name']} (ID: {customer_id}):")
# for vendor in matched_vendors:
#     print(f"Vendor: {vendor['Vendor_Name']}, Distance: {vendor['Distance']:.2f} km, Delivery Time: {vendor['Delivery_Time']} minutes")
# import os
# import pandas as pd
# import random
# import numpy as np
# from geopy.distance import geodesic

# # Step 1: Dataset Generation for Vendors and Customers

# # Create a directory to store datasets if it doesn't exist
# output_dir = "/mnt/data"
# os.makedirs(output_dir, exist_ok=True)

# # Create a list of dummy vendors with attributes: id, name, location (lat, lon), inventory, delivery_time
# vendors = []
# for i in range(1, 501):  # 500 vendors
#     vendor = {
#         "Id": i,
#         "Name": f"Vendor {i}",
#         "Location": (random.uniform(19.0, 21.0), random.uniform(72.0, 75.0)),  # Random lat, lon (in Mumbai)
#         "Inventory": random.randint(1, 20),  # Random inventory count
#         "Delivery_Time": random.randint(30, 90),  # Delivery time in minutes
#         "Category": random.choice(['Vegetarian', 'Non-Vegetarian', 'Vegan', 'Gluten-Free'])  # Food Category
#     }
#     vendors.append(vendor)

# # Create a DataFrame for vendors
# vendors_df = pd.DataFrame(vendors)

# # Save vendors dataset
# vendors_df.to_csv(os.path.join(output_dir, "large_vendors_dataset.csv"), index=False)

# # Create a list of dummy customers with attributes: id, name, location (lat, lon), preferences (category), max_delivery_time
# customers = []
# for i in range(1, 1001):  # 1000 customers
#     customer = {
#         "Id": i,
#         "Name": f"Customer {i}",
#         "Location": (random.uniform(19.0, 21.0), random.uniform(72.0, 75.0)),  # Random lat, lon (in Mumbai)
#         "Preferences": random.choice(['Vegetarian', 'Non-Vegetarian', 'Vegan', 'Gluten-Free']),  # Food Category
#         "Max_Delivery_Time": random.randint(30, 60),  # Max acceptable delivery time in minutes
#     }
#     customers.append(customer)

# # Create a DataFrame for customers
# customers_df = pd.DataFrame(customers)

# # Save customers dataset
# customers_df.to_csv(os.path.join(output_dir, "large_customers_dataset.csv"), index=False)

# print("Datasets saved successfully.")

# # Step 2: Intelligent Vendor Matching

# # Function to calculate the distance between two coordinates (lat, lon)
# def calculate_distance(loc1, loc2):
#     return geodesic(loc1, loc2).kilometers

# # Function to match a customer with vendors based on preferences, distance, and delivery time
# def match_vendor(customer, vendors_df):
#     matched_vendors = []
#     for _, vendor in vendors_df.iterrows():
#         # Check if vendor category matches customer preferences
#         if customer['Preferences'] == vendor['Category']:
#             distance = calculate_distance(customer['Location'], vendor['Location'])
#             if distance <= 10:  # Consider only vendors within 10 km radius
#                 if vendor['Delivery_Time'] <= customer['Max_Delivery_Time']:  # Check delivery time constraint
#                     matched_vendors.append({
#                         'Vendor_Name': vendor['Name'],
#                         'Vendor_Location': vendor['Location'],
#                         'Distance': distance,
#                         'Delivery_Time': vendor['Delivery_Time']
#                     })
#     return matched_vendors

# # Step 3: Test the vendor matching function
# customer_id = 1  # Example: Customer with ID = 1
# customer = customers_df.loc[customers_df['Id'] == customer_id].iloc[0]

# matched_vendors = match_vendor(customer, vendors_df)

# # Display the matched vendors for the customer
# print(f"Matched Vendors for {customer['Name']} (ID: {customer_id}):")
# for vendor in matched_vendors:
#     print(f"Vendor: {vendor['Vendor_Name']}, Distance: {vendor['Distance']:.2f} km, Delivery Time: {vendor['Delivery_Time']} minutes")

import os
import pandas as pd
import numpy as np
import random
from geopy.distance import geodesic
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

def parse_location(loc):
    try:
        if isinstance(loc, str):
            lat, lon = loc.strip('()').split(',')
            return float(lat.strip()), float(lon.strip())
        elif isinstance(loc, tuple):
            return loc
        else:
            raise ValueError(f"Unexpected location format: {loc}")
    except Exception as e:
        print(f"Error parsing location {loc}: {e}")
        return None, None

class AIVendorMatcher:
    def __init__(self, vendors_csv, customers_csv):
        # Load datasets
        self.vendors_df = pd.read_csv(vendors_csv)
        self.customers_df = pd.read_csv(customers_csv)
        
        # Prepare feature engineering and preprocessing
        self.preprocess_data()
    
    def preprocess_data(self):
        # Combine vendor and customer location data for preprocessing
        vendors_features = self.vendors_df.copy()
        vendors_features['data_type'] = 'vendor'
        customers_features = self.customers_df.copy()
        customers_features['data_type'] = 'customer'
        
        # Parse location string to latitude and longitude
        vendors_features['latitude'], vendors_features['longitude'] = zip(*vendors_features['Location'].apply(parse_location))
        customers_features['latitude'], customers_features['longitude'] = zip(*customers_features['Location'].apply(parse_location))
        
        # Combine datasets for joint preprocessing
        combined_data = pd.concat([vendors_features, customers_features])
        
        # Preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['latitude', 'longitude']),
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['Category', 'data_type'])
            ])
    
        # Fit preprocessor
        self.preprocessed_data = self.preprocessor.fit_transform(combined_data)
        
        # Train Nearest Neighbors model for intelligent matching
        self.nn_model = NearestNeighbors(n_neighbors=10, algorithm='ball_tree')
        self.nn_model.fit(self.preprocessed_data)
    
    def calculate_distance(self, loc1, loc2):
        return geodesic(loc1, loc2).kilometers
    
    def match_vendor(self, customer, k_neighbors=5):
        # Convert customer to feature vector
        customer_features = customer.copy()
        customer_features['data_type'] = 'customer'
        
        # Parse location
        customer_features['latitude'], customer_features['longitude'] = parse_location(customer['Location'])
        
        # Prepare customer features for preprocessing
        customer_df = pd.DataFrame([{
            'latitude': customer_features['latitude'],
            'longitude': customer_features['longitude'],
            'Category': customer_features['Preferences'],
            'data_type': 'customer'
        }])
        
        # Preprocess customer data
        preprocessed_customer = self.preprocessor.transform(customer_df)
        
        # Find nearest neighbors
        distances, indices = self.nn_model.kneighbors(preprocessed_customer)
        
        # Filter and rank matched vendors
        matched_vendors = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.vendors_df):
                vendor = self.vendors_df.iloc[idx]
                
                # Additional filtering criteria
                if (vendor['Category'] == customer['Preferences'] and 
                    self.calculate_distance(customer['Location'], vendor['Location']) <= 10 and
                    vendor['Delivery_Time'] <= customer['Max_Delivery_Time']):
                    
                    matched_vendors.append({
                        'Vendor_Name': vendor['Name'],
                        'Distance': self.calculate_distance(customer['Location'], vendor['Location']),
                        'Delivery_Time': vendor['Delivery_Time'],
                        'Matching_Score': 1 / (dist + 1)  # Convert distance to a similarity score
                    })
        
        # Sort vendors by matching score
        matched_vendors.sort(key=lambda x: x['Matching_Score'], reverse=True)
        return matched_vendors[:k_neighbors]

def generate_data(num_vendors=500, num_customers=1000):
    output_dir = "/mnt/data"
    os.makedirs(output_dir, exist_ok=True)

    vendors = []
    for i in range(1, num_vendors + 1):
        vendor = {
            "Id": i,
            "Name": f"Vendor {i}",
            "Location": (random.uniform(19.0, 21.0), random.uniform(72.0, 75.0)),
            "Inventory": random.randint(1, 20),
            "Delivery_Time": random.randint(30, 90),
            "Category": random.choice(['Vegetarian', 'Non-Vegetarian', 'Vegan', 'Gluten-Free'])
        }
        vendors.append(vendor)
    
    customers = []
    for i in range(1, num_customers + 1):
        customer = {
            "Id": i,
            "Name": f"Customer {i}",
            "Location": (random.uniform(19.0, 21.0), random.uniform(72.0, 75.0)),
            "Preferences": random.choice(['Vegetarian', 'Non-Vegetarian', 'Vegan', 'Gluten-Free']),
            "Max_Delivery_Time": random.randint(30, 60)
        }
        customers.append(customer)
    
    vendors_df = pd.DataFrame(vendors)
    customers_df = pd.DataFrame(customers)
    
    vendors_df.to_csv(os.path.join(output_dir, "large_vendors_dataset.csv"), index=False)
    customers_df.to_csv(os.path.join(output_dir, "large_customers_dataset.csv"), index=False)

def match_vendor(customer):
    # Generate data if needed
    generate_data()
    
    # Create AI Matcher
    matcher = AIVendorMatcher(
        vendors_csv=os.path.join("/mnt/data", "large_vendors_dataset.csv"),
        customers_csv=os.path.join("/mnt/data", "large_customers_dataset.csv")
    )

    # Match vendors for a specific customer
    matched_vendors = matcher.match_vendor(customer)
    return matched_vendors

if __name__ == "__main__":
    # Example usage
    customer_id = 1
    matcher = AIVendorMatcher(
        vendors_csv=os.path.join("/mnt/data", "large_vendors_dataset.csv"),
        customers_csv=os.path.join("/mnt/data", "large_customers_dataset.csv")
    )
    customer = matcher.customers_df.loc[matcher.customers_df['Id'] == customer_id].iloc[0]

    matched_vendors = matcher.match_vendor(customer)

    print(f"AI-Matched Vendors for {customer['Name']} (ID: {customer_id}):")
    for vendor in matched_vendors:
        print(f"Vendor: {vendor['Vendor_Name']}, Distance: {vendor['Distance']:.2f} km, "
              f"Delivery Time: {vendor['Delivery_Time']} minutes, "
              f"Matching Score: {vendor['Matching_Score']:.2f}")