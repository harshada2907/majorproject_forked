import pandas as pd
import numpy as np

# Seed for reproducibility
np.random.seed(42)

# Define parameters
num_samples = 2000

# Generate synthetic data
data = {
    'Temperature (°C)': np.random.uniform(15, 40, num_samples),  # Random temperature between 15 and 40
    'Rainfall (mm)': np.random.uniform(0, 300, num_samples),  # Random rainfall between 0 and 300 mm
    'Humidity (%)': np.random.uniform(20, 100, num_samples),  # Random humidity between 20% and 100%
    'Soil pH': np.random.uniform(4.0, 8.5, num_samples),  # Random soil pH between 4.0 and 8.5
    'Soil Type': np.random.choice(['Clay', 'Sandy', 'Loamy', 'Silty'], num_samples),  # Random soil type
    'Irrigation Type': np.random.choice(['Drip', 'Flood', 'Sprinkler'], num_samples),  # Random irrigation type
}

# Create DataFrame
crop_data = pd.DataFrame(data)

# Define target variable (Crop Type) based on more detailed conditions
conditions = [
    (crop_data['Temperature (°C)'] < 20) & (crop_data['Rainfall (mm)'] > 150) & (crop_data['Soil pH'] < 6.0),  # Rice
    (crop_data['Temperature (°C)'] >= 20) & (crop_data['Temperature (°C)'] < 30) & (crop_data['Rainfall (mm)'] > 100) & (crop_data['Soil Type'] == 'Loamy'),  # Wheat
    (crop_data['Temperature (°C)'] >= 25) & (crop_data['Rainfall (mm)'] <= 80) & (crop_data['Humidity (%)'] > 60),  # Maize
    (crop_data['Temperature (°C)'] >= 30) & (crop_data['Rainfall (mm)'] < 50) & (crop_data['Soil Type'] == 'Sandy'),  # Cotton
    (crop_data['Temperature (°C)'] < 25) & (crop_data['Humidity (%)'] < 50) & (crop_data['Soil pH'] >= 6.0),  # Sugarcane
    (crop_data['Temperature (°C)'] < 30) & (crop_data['Rainfall (mm)'] > 200),  # Pulses
    (crop_data['Soil Type'] == 'Clay') & (crop_data['Humidity (%)'] > 70),  # Potatoes
    (crop_data['Temperature (°C)'] >= 15) & (crop_data['Rainfall (mm)'] >= 100) & (crop_data['Soil pH'] <= 6.5),  # Tomatoes
    (crop_data['Humidity (%)'] > 50) & (crop_data['Soil pH'] >= 5.5) & (crop_data['Rainfall (mm)'] >= 100),  # Spinach
    (crop_data['Temperature (°C)'] > 25) & (crop_data['Rainfall (mm)'] >= 50) & (crop_data['Soil Type'] == 'Silty'),  # Grapes
    (crop_data['Temperature (°C)'] < 30) & (crop_data['Humidity (%)'] > 80) & (crop_data['Soil Type'] == 'Loamy'),  # Peas
    (crop_data['Temperature (°C)'] > 20) & (crop_data['Rainfall (mm)'] < 60) & (crop_data['Soil pH'] >= 6.0),  # Lettuce
    (crop_data['Temperature (°C)'] < 25) & (crop_data['Rainfall (mm)'] > 200) & (crop_data['Soil Type'] == 'Clay'),  # Carrots
]

# Updated crop types corresponding to the conditions, including fruits and more pulses
choices = [
    'Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane', 
    'Pulses', 'Potatoes', 'Tomatoes', 'Spinach', 
    'Grapes', 'Peas', 'Lettuce', 'Carrots'
]

# Assign crop type based on conditions, default to 'Others'
crop_data['Crop Type'] = np.select(conditions, choices, default='Others')

# Save to CSV
crop_data.to_csv('expanded_crop_prediction_dataset.csv', index=False)

print("Dataset generated and saved as 'expanded_crop_prediction_dataset.csv'.")
