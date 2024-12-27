import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Load the dataset
df = pd.read_csv("Bhatsa_dam.csv")

# Print the initial column names for reference
print("Initial Column Names:", df.columns)

# Rename columns (update this to match your dataset)
df.columns = [
    "release_bmc_tmc", "release_escape_gate", "%release_escape_gate", 
    "release_irrigation_canal", "%release_irrigation", 
    "release_spillway_river", "%release_spillway", 
    "leakage_gallery", "%leakage_gallery", "total_release", 
    "reservoir_evaporation_losses", "other_measured_leakages", 
    "reservoir_water_level_end_month", "gross_storage_end_month", 
    "%gross_storage_end_month", "calculated_inflow_month"
    # Add or adjust column names if your dataset has more columns
]

# Target variables for prediction
target_irrigation = "%release_irrigation"
target_hydropower = "%release_spillway"

# Features
features = [
    "gross_storage_end_month", "calculated_inflow_month",
    "reservoir_evaporation_losses", "total_release"
]

# Ensure the dataset has no missing values
df = df.dropna()

# Split data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Function to fit SARIMAX model and predict
def fit_sarimax(train_data, test_data, feature, target):
    # SARIMAX requires both exogenous features and endog target series
    sarimax_model = SARIMAX(
        train_data[target],
        exog=train_data[feature],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12)
    )
    model_fit = sarimax_model.fit(disp=False)

    # Forecast on the test data
    forecast = model_fit.forecast(steps=len(test_data), exog=test_data[feature])
    
    return forecast, model_fit

# Predict %release_irrigation
forecast_irrigation, model_irrigation = fit_sarimax(
    train, test, features, target_irrigation
)

# Predict %release_hydropower
forecast_hydropower, model_hydropower = fit_sarimax(
    train, test, features, target_hydropower
)

# Evaluate results
def evaluate_model(test_data, forecast, target):
    mse = mean_squared_error(test_data[target], forecast)
    print(f"Mean Squared Error for {target}: {mse}")
    return mse

# Evaluate irrigation prediction
evaluate_model(test, forecast_irrigation, target_irrigation)

# Evaluate hydropower prediction
evaluate_model(test, forecast_hydropower, target_hydropower)

# Plot results
def plot_results(test_data, forecast, target, title):
    plt.figure(figsize=(12, 6))
    plt.plot(test_data[target].values, label="Actual", color="blue")
    plt.plot(forecast, label="Forecast", color="orange")
    plt.title(title)
    plt.legend()
    plt.show()

# Plot for irrigation
plot_results(test, forecast_irrigation, target_irrigation, "Irrigation Prediction")

# Plot for hydropower
plot_results(test, forecast_hydropower, target_hydropower, "Hydropower Prediction")

# Optimality Check: Ensure water availability in case of reduced inflow
def check_optimality(forecast, available_storage, critical_storage=20):
    """
    Ensure water release is optimal for the next year.
    Args:
    - forecast: Predicted release percentages.
    - available_storage: Current storage level.
    - critical_storage: Minimum storage to ensure availability (default = 20%).
    """
    future_usage = sum(forecast)
    if available_storage - future_usage < critical_storage:
        print("Warning: Predicted usage exceeds safe storage levels!")
    else:
        print("Predicted usage is within optimal levels.")

# Example check
check_optimality(forecast_irrigation, df["gross_storage_end_month"].iloc[-1])
check_optimality(forecast_hydropower, df["gross_storage_end_month"].iloc[-1])
