import os
from dotenv import load_dotenv
import folium
from streamlit_folium import folium_static
import streamlit as st
import requests
import json
import geopy.distance

# Load environment variables from .env file
load_dotenv()

class DeliveryMappingSystem:
    def __init__(self):
        """
        Initialize Delivery Mapping System using .env for API key
        """
        # Retrieve API key from environment variables
        self.api_key = os.getenv('OPENROUTESERVICE_API_KEY')
        
        # Set default configuration
        self.routing_api_url = "https://api.openrouteservice.org/v2/directions/driving-car"
        self.geocoding_api_url = "https://api.openrouteservice.org/geocode/search"
        
        # Validate API key
        if not self.api_key:
            st.warning("OpenRouteService API key not found in .env file. Some functionalities may be limited.")

    def geocode_address(self, address):
        """
        Convert address to latitude and longitude
        
        Args:
            address (str): Address to geocode
        
        Returns:
            tuple: (longitude, latitude) or None if geocoding fails
        """
        if not self.api_key:
            st.error("Geocoding requires an API key")
            return None
        
        try:
            params = {
                'api_key': self.api_key,
                'text': address,
                'size': 1  # Get top result
            }
            response = requests.get(self.geocoding_api_url, params=params)
            
            data = response.json()
            
            if data.get('features'):
                coordinates = data['features'][0]['geometry']['coordinates']
                return coordinates  # [lon, lat]
            else:
                st.error(f"No geocoding results for address: {address}")
        except Exception as e:
            st.error(f"Geocoding error: {e}")
        
        return None

    def calculate_route(self, start_coords, end_coords):
        """
        Calculate route between two coordinates
        
        Args:
            start_coords (tuple): (longitude, latitude) of start point
            end_coords (tuple): (longitude, latitude) of end point
        
        Returns:
            dict: Route information including distance, duration, route geometry
        """
        if not self.api_key:
            st.error("Routing requires an API key")
            return None
        
        try:
            payload = {
                "coordinates": [start_coords, end_coords]
            }
            headers = {
                'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
                'Authorization': self.api_key,
                'Content-Type': 'application/json; charset=utf-8'
            }
            
            # Make the API request
            response = requests.post(self.routing_api_url, json=payload, headers=headers)
            
            # Parse the response
            route_data = response.json()
            
            # Check for errors
            if 'error' in route_data:
                st.error(f"Routing API Error: {route_data['error']['message']}")
                return None
            
            # Extract key route information
            first_feature = route_data['features'][0]
            
            # Safely extract route details
            summary = first_feature['properties']['summary']
            distance = summary['distance'] / 1000  # km
            duration = summary['duration'] / 60  # minutes
            
            route_geometry = first_feature['geometry']['coordinates']
            
            return {
                'distance': distance,
                'duration': duration,
                'route_geometry': route_geometry
            }
        except requests.RequestException as req_err:
            st.error(f"Request error: {req_err}")
        except ValueError as val_err:
            st.error(f"Value error: {val_err}")
        except Exception as e:
            st.error(f"Unexpected routing error: {e}")
        
        return None

    def create_delivery_route_map(self, start_coords, end_coords, route_geometry=None):
        """
        Create interactive map showing delivery route
        
        Args:
            start_coords (tuple): (longitude, latitude) of start point
            end_coords (tuple): (longitude, latitude) of end point
            route_geometry (list, optional): Detailed route coordinates
        
        Returns:
            folium.Map: Interactive map object
        """
        # Determine map center
        center_lat = (start_coords[1] + end_coords[1]) / 2
        center_lon = (start_coords[0] + end_coords[0]) / 2
        
        # Create base map
        delivery_map = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=10
        )
        
        # Add start marker
        folium.Marker(
            [start_coords[1], start_coords[0]], 
            popup='Start Location',
            icon=folium.Icon(color='green', icon='play')
        ).add_to(delivery_map)
        
        # Add end marker
        folium.Marker(
            [end_coords[1], end_coords[0]], 
            popup='Destination',
            icon=folium.Icon(color='red', icon='flag')
        ).add_to(delivery_map)
        
        # Draw route if geometry provided
        if route_geometry:
            # Convert coordinates for folium (lat, lon)
            route_points = [(point[1], point[0]) for point in route_geometry]
            
            # Add route polyline
            folium.PolyLine(
                route_points, 
                color='blue', 
                weight=5, 
                opacity=0.8
            ).add_to(delivery_map)
        
        return delivery_map

def main():
    st.title("🗺️ Delivery Route Visualization")
    
    # Initialize mapping system
    mapping_system = DeliveryMappingSystem()
    
    # Input sections
    col1, col2 = st.columns(2)
    
    with col1:
        start_address = st.text_input("Enter Start Address", placeholder="123 Main St, New York, NY 10001, USA")
    
    with col2:
        end_address = st.text_input("Enter Destination Address", placeholder="456 Broadway, New York, NY 10013, USA")
    
    if st.button("Generate Route"):
        # Geocode addresses
        start_coords = mapping_system.geocode_address(start_address)
        end_coords = mapping_system.geocode_address(end_address)
        
        if start_coords and end_coords:
            # Calculate route
            route_info = mapping_system.calculate_route(start_coords, end_coords)
            
            if route_info:
                # Display route details
                st.subheader("Route Details")
                st.write(f"Distance: {route_info['distance']:.2f} km")
                st.write(f"Estimated Duration: {route_info['duration']:.2f} minutes")
                
                # Create and display map
                delivery_map = mapping_system.create_delivery_route_map(
                    start_coords, 
                    end_coords, 
                    route_info['route_geometry']
                )
                
                folium_static(delivery_map)
            else:
                st.error("Failed to calculate route. Please check your addresses and API configuration.")
        else:
            st.error("Unable to geocode one or both addresses. Please check the addresses.")

if __name__ == "__main__":
    main()