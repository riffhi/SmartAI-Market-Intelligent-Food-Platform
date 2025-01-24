# 🍽️ SmartAI Market: Intelligent Food Service Optimization Platform

## 🌟 Project Overview
SmartAI Market is a cutting-edge AI-powered platform designed to revolutionize the food service industry through intelligent data-driven solutions. By leveraging advanced machine learning techniques, the platform addresses critical challenges in food delivery, vendor management, and operational efficiency.

## 🚀 Key Features

### 1. Intelligent Route Optimization
- Predict accurate delivery times
- Minimize transportation costs
- Optimize courier routes using machine learning
- Real-time route adjustments based on dynamic factors

### 2. Smart Vendor Matching
- Advanced algorithm to match customers with ideal food vendors
- Considers dietary preferences, location, and delivery constraints
- Personalized vendor recommendations

### 3. Advanced Sentiment Analysis
- Multi-model sentiment detection using:
  - VADER (Rule-based sentiment analysis)
  - RoBERTa (Deep learning sentiment model)
- Comprehensive review insights
- Actionable customer feedback interpretation

### 4. Waste Reduction Intelligence
- Predictive analytics for food waste
- Recommend optimal portion sizes
- Dynamic inventory management
- Machine learning-driven recommendations

### 5. Demand Forecasting
- Predict future food item sales
- Account for variables like:
  - Day of week
  - Holidays
  - Promotions
  - Historical sales data

### 6. Dynamic Price Optimization
- AI-powered pricing strategies
- Real-time price adjustments
- Maximize revenue while maintaining customer satisfaction

## 🛠️ Technical Architecture

### Backend Technologies
- **Programming Language**: Python
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning Libraries**:
  - Scikit-learn
  - XGBoost
  - LightGBM
  - CatBoost

### AI & NLP Technologies
- NLTK
- Transformers (Hugging Face)
- RoBERTa Sentiment Model

### Geospatial & Distance Calculations
- GeoPy for location-based matching
- Geodesic distance calculations

## 📦 Project Structure
```
riffhi-rubix-25_36_kaizen/
│
├── app.py               # Main Streamlit application
├── main.py              # FastAPI backend
├── requirements.txt     # Project dependencies
│
├── features/
│   ├── delivery_time.py        # Delivery time prediction
│   ├── demand_forecasting.py   # Sales demand prediction
│   ├── price_predict.py        # Dynamic pricing strategies
│   ├── route_optimizer.py      # Route optimization module
│   ├── sentiment.py            # Review sentiment analysis
│   ├── vendor.py               # Vendor matching algorithms
│   └── waste.py                # Waste reduction recommendations
│
├── static/              # Static web assets
│   └── css/
│       └── style.css
│
├── templates/           # HTML templates
│   └── home.html
│
└── datasets/            # Generated synthetic datasets
    ├── adjusted_small_product_prices.csv
    ├── food_sales_data.csv
    ├── food_waste_data.csv
    ├── large_customers_dataset.csv
    ├── large_vendors_dataset.csv
    ├── sales_dataset.csv
    └── small_product_dataset.csv
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation Steps
1. Clone the repository
   ```bash
   git clone https://github.com/riffhi/rubix-25_36_kaizen.git
   cd rubix-25_36_kaizen
   ```

2. Create virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit application
   ```bash
   streamlit run app.py
   ```

## 🤝 Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.

## 🏆 Future Roadmap
- Enhanced multi-model sentiment analysis
- Real-time machine learning model updates
- Expanded geospatial vendor matching
- Advanced anomaly detection in food service operations

## 📞 Contact
Riddhi Bhanushali - youremail@example.com

Project Link: [https://github.com/riffhi/rubix-25_36_kaizen](https://github.com/riffhi/rubix-25_36_kaizen)
