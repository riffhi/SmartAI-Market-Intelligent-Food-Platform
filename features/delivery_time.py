import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load the dataset
df = pd.read_csv("synthetic_delivery_data.csv")

# Preprocessing function
def preprocess_delivery_data(df):
    cat_features = [feature for feature in df.columns if df[feature].dtype == 'O']
    num_features = [feature for feature in df.columns if df[feature].dtype != 'O']

    or_columns = ['weather_condition', 'traffic_condition', 'actual_delivery_time']
    oh_columns = ['delivery_id', 'delivery_timestamp']
    transform_columns = ['origin_lat', 'origin_lon', 'dest_lat', 'dest_lon']

    # Remove non-existing columns
    or_columns = [col for col in or_columns if col in df.columns]
    oh_columns = [col for col in oh_columns if col in df.columns]
    transform_columns = [col for col in transform_columns if col in df.columns]

    # Define transformers
    numeric_transformer = StandardScaler()
    oh_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    transform_pipe = Pipeline(steps=[
        ('transformer', PowerTransformer(method='yeo-johnson'))
    ])

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("OneHotEncoder", oh_transformer, oh_columns),
            ("Ordinal_Encoder", ordinal_encoder, or_columns),
            ("Transformer", transform_pipe, transform_columns),
            ("StandardScaler", numeric_transformer, [col for col in num_features
                                                     if col not in transform_columns + oh_columns + or_columns])
        ],
        remainder='passthrough'
    )

    X_transformed = preprocessor.fit_transform(df)

    if X_transformed.ndim == 1:
        X_transformed = X_transformed.reshape(-1, 1)

    return X_transformed, preprocessor

# Regression evaluation function
def evaluate_regression_models(X, y, models):
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models_list = []
    r2_list = []
    rmse_list = []
    mae_list = []

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)

        models_list.append(model_name)
        r2_list.append(test_r2)
        rmse_list.append(test_rmse)
        mae_list.append(test_mae)

        print(f"Model performance for {model_name}")
        print(f"Training R² Score: {train_r2:.4f}")
        print(f"Test R² Score: {test_r2:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print("="*50)

    report = pd.DataFrame({
        'Model Name': models_list,
        'R² Score': r2_list,
        'RMSE': rmse_list,
        'MAE': mae_list
    }).sort_values('R² Score', ascending=False)

    return report

# Prepare data
y = df['actual_delivery_time']
X = df.drop('actual_delivery_time', axis=1)

X_transformed, preprocessor = preprocess_delivery_data(X)

# Define regression models
regression_models = {
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor(),
    "CatBoost": CatBoostRegressor(verbose=False)
}

# Evaluate models
base_model_report = evaluate_regression_models(X_transformed, y, regression_models)

# Print the report
print("\nModel Evaluation Report:")
print(base_model_report)
