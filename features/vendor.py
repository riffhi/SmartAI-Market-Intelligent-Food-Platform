import os
import pandas as pd
import random
import numpy as np
from geopy.distance import geodesic

# Step 1: Dataset Generation for Vendors and Customers

# Create a directory to store datasets if it doesn't exist
output_dir = "/mnt/data"
os.makedirs(output_dir, exist_ok=True)

# Create a list of dummy vendors with attributes: id, name, location (lat, lon), inventory, delivery_time
vendors = []
for i in range(1, 501):  # 500 vendors
    vendor = {
        "Id": i,
        "Name": f"Vendor {i}",
        "Location": (random.uniform(19.0, 21.0), random.uniform(72.0, 75.0)),  # Random lat, lon (in Mumbai)
        "Inventory": random.randint(1, 20),  # Random inventory count
        "Delivery_Time": random.randint(30, 90),  # Delivery time in minutes
        "Category": random.choice(['Vegetarian', 'Non-Vegetarian', 'Vegan', 'Gluten-Free'])  # Food Category
    }
    vendors.append(vendor)

# Create a DataFrame for vendors
vendors_df = pd.DataFrame(vendors)

# Save vendors dataset
vendors_df.to_csv(os.path.join(output_dir, "large_vendors_dataset.csv"), index=False)

# Create a list of dummy customers with attributes: id, name, location (lat, lon), preferences (category), max_delivery_time
customers = []
for i in range(1, 1001):  # 1000 customers
    customer = {
        "Id": i,
        "Name": f"Customer {i}",
        "Location": (random.uniform(19.0, 21.0), random.uniform(72.0, 75.0)),  # Random lat, lon (in Mumbai)
        "Preferences": random.choice(['Vegetarian', 'Non-Vegetarian', 'Vegan', 'Gluten-Free']),  # Food Category
        "Max_Delivery_Time": random.randint(30, 60),  # Max acceptable delivery time in minutes
    }
    customers.append(customer)

# Create a DataFrame for customers
customers_df = pd.DataFrame(customers)

# Save customers dataset
customers_df.to_csv(os.path.join(output_dir, "large_customers_dataset.csv"), index=False)

print("Datasets saved successfully.")

# Step 2: Intelligent Vendor Matching

# Function to calculate the distance between two coordinates (lat, lon)
def calculate_distance(loc1, loc2):
    return geodesic(loc1, loc2).kilometers

# Function to match a customer with vendors based on preferences, distance, and delivery time
def match_vendor(customer, vendors_df):
    matched_vendors = []
    for _, vendor in vendors_df.iterrows():
        # Check if vendor category matches customer preferences
        if customer['Preferences'] == vendor['Category']:
            distance = calculate_distance(customer['Location'], vendor['Location'])
            if distance <= 10:  # Consider only vendors within 10 km radius
                if vendor['Delivery_Time'] <= customer['Max_Delivery_Time']:  # Check delivery time constraint
                    matched_vendors.append({
                        'Vendor_Name': vendor['Name'],
                        'Vendor_Location': vendor['Location'],
                        'Distance': distance,
                        'Delivery_Time': vendor['Delivery_Time']
                    })
    return matched_vendors

# Step 3: Test the vendor matching function
customer_id = 1  # Example: Customer with ID = 1
customer = customers_df.loc[customers_df['Id'] == customer_id].iloc[0]

matched_vendors = match_vendor(customer, vendors_df)

# Display the matched vendors for the customer
print(f"Matched Vendors for {customer['Name']} (ID: {customer_id}):")
for vendor in matched_vendors:
    print(f"Vendor: {vendor['Vendor_Name']}, Distance: {vendor['Distance']:.2f} km, Delivery Time: {vendor['Delivery_Time']} minutes")
import os
import pandas as pd
import random
import numpy as np
from geopy.distance import geodesic

# Step 1: Dataset Generation for Vendors and Customers

# Create a directory to store datasets if it doesn't exist
output_dir = "/mnt/data"
os.makedirs(output_dir, exist_ok=True)

# Create a list of dummy vendors with attributes: id, name, location (lat, lon), inventory, delivery_time
vendors = []
for i in range(1, 501):  # 500 vendors
    vendor = {
        "Id": i,
        "Name": f"Vendor {i}",
        "Location": (random.uniform(19.0, 21.0), random.uniform(72.0, 75.0)),  # Random lat, lon (in Mumbai)
        "Inventory": random.randint(1, 20),  # Random inventory count
        "Delivery_Time": random.randint(30, 90),  # Delivery time in minutes
        "Category": random.choice(['Vegetarian', 'Non-Vegetarian', 'Vegan', 'Gluten-Free'])  # Food Category
    }
    vendors.append(vendor)

# Create a DataFrame for vendors
vendors_df = pd.DataFrame(vendors)

# Save vendors dataset
vendors_df.to_csv(os.path.join(output_dir, "large_vendors_dataset.csv"), index=False)

# Create a list of dummy customers with attributes: id, name, location (lat, lon), preferences (category), max_delivery_time
customers = []
for i in range(1, 1001):  # 1000 customers
    customer = {
        "Id": i,
        "Name": f"Customer {i}",
        "Location": (random.uniform(19.0, 21.0), random.uniform(72.0, 75.0)),  # Random lat, lon (in Mumbai)
        "Preferences": random.choice(['Vegetarian', 'Non-Vegetarian', 'Vegan', 'Gluten-Free']),  # Food Category
        "Max_Delivery_Time": random.randint(30, 60),  # Max acceptable delivery time in minutes
    }
    customers.append(customer)

# Create a DataFrame for customers
customers_df = pd.DataFrame(customers)

# Save customers dataset
customers_df.to_csv(os.path.join(output_dir, "large_customers_dataset.csv"), index=False)

print("Datasets saved successfully.")

# Step 2: Intelligent Vendor Matching

# Function to calculate the distance between two coordinates (lat, lon)
def calculate_distance(loc1, loc2):
    return geodesic(loc1, loc2).kilometers

# Function to match a customer with vendors based on preferences, distance, and delivery time
def match_vendor(customer, vendors_df):
    matched_vendors = []
    for _, vendor in vendors_df.iterrows():
        # Check if vendor category matches customer preferences
        if customer['Preferences'] == vendor['Category']:
            distance = calculate_distance(customer['Location'], vendor['Location'])
            if distance <= 10:  # Consider only vendors within 10 km radius
                if vendor['Delivery_Time'] <= customer['Max_Delivery_Time']:  # Check delivery time constraint
                    matched_vendors.append({
                        'Vendor_Name': vendor['Name'],
                        'Vendor_Location': vendor['Location'],
                        'Distance': distance,
                        'Delivery_Time': vendor['Delivery_Time']
                    })
    return matched_vendors

# Step 3: Test the vendor matching function
customer_id = 1  # Example: Customer with ID = 1
customer = customers_df.loc[customers_df['Id'] == customer_id].iloc[0]

matched_vendors = match_vendor(customer, vendors_df)

# Display the matched vendors for the customer
print(f"Matched Vendors for {customer['Name']} (ID: {customer_id}):")
for vendor in matched_vendors:
    print(f"Vendor: {vendor['Vendor_Name']}, Distance: {vendor['Distance']:.2f} km, Delivery Time: {vendor['Delivery_Time']} minutes")