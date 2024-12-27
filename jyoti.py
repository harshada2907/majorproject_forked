import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the dataset
data = pd.read_csv('Bhatsa_Dam.csv')

# Create the date column based on Year and Month
if 'Year' in data.columns and 'Month' in data.columns:
    data['date'] = data['Year'].astype(str) + '-' + data['Month']
    data['date'] = pd.to_datetime(data['date'], format='%Y-%b-%y', errors='coerce')
    print("Successfully created the 'date' column!")
else:
    print("Error: 'Year' or 'Month' column is missing.")
    exit()

# Handle missing dates
if data['date'].isna().sum() > 0:
    print("Warning: Some dates could not be parsed. They will be dropped.")
    data.dropna(subset=['date'], inplace=True)

# Create a new feature to capture the monsoon months (June, July, August, September)
data['is_monsoon'] = data['Month'].isin([6, 7, 8, 9]).astype(int)

# Define the exogenous features (input parameters)
exogenous_features = [
    'Reservoir water level on start of month (m)',
    'Gross storage on end of month (Mcum)',  # Corrected name
    'Calculated inflow of month (Mcum)',
    'Reservoir evaporation on losses (Mcum)',
    'Other measured Leakages (Mcum)',
    'Release through spillway river (Mcum)',
    'Release through escape gate',
    'is_monsoon'  # Adding the new 'is_monsoon' feature to capture seasonality
]

# Define target variables
target_irr = '%release for irrigation'
target_hydro = 'Release for BMC/TMC other through power house (Mcum)'  # Correct column name for hydropower

# Check if target variables exist
if target_irr not in data.columns or target_hydro not in data.columns:
    print(f"Error: '{target_irr}' or '{target_hydro}' not found in the dataset.")
    exit()

# Split data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Separate features and targets for irrigation and hydropower
X_train_irr, X_test_irr = train[exogenous_features], test[exogenous_features]
y_train_irr, y_test_irr = train[target_irr], test[target_irr]

X_train_hydro, X_test_hydro = train[exogenous_features], test[exogenous_features]
y_train_hydro, y_test_hydro = train[target_hydro], test[target_hydro]

# Train Random Forest Regressor models for irrigation and hydropower
model_irr = RandomForestRegressor()
model_irr.fit(X_train_irr, y_train_irr)

model_hydro = RandomForestRegressor()
model_hydro.fit(X_train_hydro, y_train_hydro)

# Make predictions using Random Forest
y_pred_irr = model_irr.predict(X_test_irr)
y_pred_hydro = model_hydro.predict(X_test_hydro)

# Accuracy of the Random Forest models (for both irrigation and hydropower)
mae_irr = mean_absolute_error(y_test_irr, y_pred_irr)
mse_irr = mean_squared_error(y_test_irr, y_pred_irr)
r2_irr = r2_score(y_test_irr, y_pred_irr)

mae_hydro = mean_absolute_error(y_test_hydro, y_pred_hydro)
mse_hydro = mean_squared_error(y_test_hydro, y_pred_hydro)
r2_hydro = r2_score(y_test_hydro, y_pred_hydro)

# Plotting the actual vs predicted values for irrigation and hydropower
plt.figure(figsize=(12, 6))

# Plot for irrigation
plt.subplot(1, 2, 1)
plt.plot(test['date'], y_test_irr, label='Actual % Release for Irrigation', color='blue')
plt.plot(test['date'], y_pred_irr, label='Predicted % Release for Irrigation', color='red')
plt.title('Irrigation Release: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('% Release')
plt.legend()

# Plot for hydropower
plt.subplot(1, 2, 2)
plt.plot(test['date'], y_test_hydro, label='Actual Hydropower Release', color='blue')
plt.plot(test['date'], y_pred_hydro, label='Predicted Hydropower Release', color='red')
plt.title('Hydropower Release: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Release (Mcum)')
plt.legend()

plt.tight_layout()
plt.show()

# Print the accuracy of the Random Forest model
print("Random Forest Model Accuracy for Irrigation:")
print(f"MAE: {mae_irr:.2f}, MSE: {mse_irr:.2f}, R2: {r2_irr:.2f}")

print("\nRandom Forest Model Accuracy for Hydropower:")
print(f"MAE: {mae_hydro:.2f}, MSE: {mse_hydro:.2f}, R2: {r2_hydro:.2f}")

# SARIMAX Model for Time Series Forecasting
# Predicting the next month's irrigation and hydropower releases using SARIMAX

# Create SARIMAX model for irrigation
sarimax_irr = SARIMAX(train[target_irr], exog=train[exogenous_features], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarimax_result_irr = sarimax_irr.fit(disp=False)

# Create SARIMAX model for hydropower
sarimax_hydro = SARIMAX(train[target_hydro], exog=train[exogenous_features], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarimax_result_hydro = sarimax_hydro.fit(disp=False)

# Predicting with SARIMAX
sarimax_pred_irr = sarimax_result_irr.predict(start=test.index[0], end=test.index[-1], exog=test[exogenous_features])
sarimax_pred_hydro = sarimax_result_hydro.predict(start=test.index[0], end=test.index[-1], exog=test[exogenous_features])

# Plot SARIMAX predictions
plt.figure(figsize=(12, 6))

# Plot for irrigation (SARIMAX)
plt.subplot(1, 2, 1)
plt.plot(test['date'], y_test_irr, label='Actual % Release for Irrigation', color='blue')
plt.plot(test['date'], sarimax_pred_irr, label='Predicted % Release for Irrigation (SARIMAX)', color='green')
plt.title('Irrigation Release: SARIMAX Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('% Release')
plt.legend()

# Plot for hydropower (SARIMAX)
plt.subplot(1, 2, 2)
plt.plot(test['date'], y_test_hydro, label='Actual Hydropower Release', color='blue')
plt.plot(test['date'], sarimax_pred_hydro, label='Predicted Hydropower Release (SARIMAX)', color='green')
plt.title('Hydropower Release: SARIMAX Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Release (Mcum)')
plt.legend()

plt.tight_layout()
plt.show()

# Print SARIMAX results
print("\nSARIMAX Model Results for Irrigation:")
print(f"MAE: {mean_absolute_error(y_test_irr, sarimax_pred_irr):.2f}, MSE: {mean_squared_error(y_test_irr, sarimax_pred_irr):.2f}, R2: {r2_score(y_test_irr, sarimax_pred_irr):.2f}")

print("\nSARIMAX Model Results for Hydropower:")
print(f"MAE: {mean_absolute_error(y_test_hydro, sarimax_pred_hydro):.2f}, MSE: {mean_squared_error(y_test_hydro, sarimax_pred_hydro):.2f}, R2: {r2_score(y_test_hydro, sarimax_pred_hydro):.2f}")

# User input for prediction
print("\nEnter the parameters for the prediction:")

# Input features for prediction (do not ask for target variables)
reservoir_water_level = float(input("Reservoir water level on start of month (m): "))
gross_storage = float(input("Gross storage on end of month (Mcum): "))
inflow = float(input("Calculated inflow of month (Mcum): "))
evaporation_losses = float(input("Reservoir evaporation on losses (Mcum): "))
leakages = float(input("Other measured Leakages (Mcum): "))
spillway_release = float(input("Release through spillway river (Mcum): "))
escape_gate_release = float(input("Release through escape gate: "))

# Create a DataFrame with the input values to predict the output
input_data = pd.DataFrame({
    'Reservoir water level on start of month (m)': [reservoir_water_level],
    'Gross storage on end of month (Mcum)': [gross_storage],
    'Calculated inflow of month (Mcum)': [inflow],
    'Reservoir evaporation on losses (Mcum)': [evaporation_losses],
    'Other measured Leakages (Mcum)': [leakages],
    'Release through spillway river (Mcum)': [spillway_release],
    'Release through escape gate': [escape_gate_release],
    'is_monsoon': [int(input("Is it a monsoon month? (1 for Yes, 0 for No): "))]  # Take user input for monsoon season
})

# Predict using both models
pred_irr_input = model_irr.predict(input_data)
pred_hydro_input = model_hydro.predict(input_data)

# Output the predictions
print("\nPredicted values for the input data:")
print(f"Predicted % Release for Irrigation: {pred_irr_input[0]:.2f}")
print(f"Predicted Hydropower Release (Mcum): {pred_hydro_input[0]:.2f}")
