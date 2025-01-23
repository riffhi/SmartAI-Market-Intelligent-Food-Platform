from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import pandas as pd
import numpy as np
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb

app = FastAPI()

# Allow CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (e.g., HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure templates
templates = Jinja2Templates(directory="templates")

# Define models for data input
class DeliveryData(BaseModel):
    data: list[dict]

class PriceData(BaseModel):
    data: list[dict]

class RouteData(BaseModel):
    data: list[dict]

# Placeholder for pre-trained models
regression_models = {
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor(),
    "CatBoost": CatBoostRegressor(verbose=False)
}

# Utility function for preprocessing delivery data
def preprocess_delivery_data(df):
    # Similar preprocessing logic as in the delivery_time.py
    pass

# Utility function for training LightGBM
def train_with_lightgbm(processed_data):
    # Similar logic as in the route_optimizer.py
    pass

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/predict_delivery_time")
def predict_delivery_time(delivery_data: DeliveryData):
    try:
        df = pd.DataFrame(delivery_data.data)
        if df.empty:
            raise HTTPException(status_code=400, detail="Input data is empty.")

        # Preprocess data
        X_transformed, preprocessor = preprocess_delivery_data(df)

        # Select a model for demonstration (e.g., Random Forest)
        model = regression_models["Random Forest"]
        predictions = model.predict(X_transformed)

        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_and_adjust_prices")
def predict_and_adjust_prices(price_data: PriceData):
    try:
        df = pd.DataFrame(price_data.data)
        if df.empty:
            raise HTTPException(status_code=400, detail="Input data is empty.")

        # Convert categorical demand to numeric
        df['Demand'] = df['Demand'].map({'Low': 100, 'Medium': 500, 'High': 1000})
        
        # Prepare data for XGBoost
        X = df[['Price']]
        dmatrix = xgb.DMatrix(X)

        # Load the XGBoost model
        bst = xgb.Booster()
        bst.load_model("xgboost_model.json")

        # Predict demand
        df['Predicted_Demand'] = bst.predict(dmatrix)

        # Adjust prices
        def adjust_prices(row):
            if row['Predicted_Demand'] < 350:
                return row['Price'] * 0.9
            else:
                return row['Price'] * 1.1

        df['Adjusted_Price'] = df.apply(adjust_prices, axis=1)

        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize_routes")
def optimize_routes(route_data: RouteData):
    try:
        df = pd.DataFrame(route_data.data)
        if df.empty:
            raise HTTPException(status_code=400, detail="Input data is empty.")

        # Prepare training data
        processed_data = train_with_lightgbm(df)
        
        if processed_data.empty:
            raise HTTPException(status_code=400, detail="Processed data is empty.")

        return {"message": "Route optimization completed.", "data": processed_data.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthcheck")
def healthcheck():
    return {"status": "OK"}