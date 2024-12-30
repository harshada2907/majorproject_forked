# import the libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# load the data
data = pd.read_csv("Bhatsa_Dam.csv")
print(data)

# check for null data
print(data.isnull().sum())

#features
features = data[["Reservoir water level on start of month (m)", 
	"Effective gross storage on start on month (Mcum)", 
	"Release through escape gate", 
	"Release through spillway river (Mcum)", 
	"Leakage through gallery (Mcum)", 
	"Reservoir evaporation on losses (Mcum)", 
	"Other measured Leakages (Mcum)", 
	"Reservoir water level on end of month (m)", 
	"Gross storage on end of month (Mcum)", 
	"Calculated inflow of month (Mcum)"]]

# print the features
print(features)

# target
target = data[["Release for irrigation through canal (Mcum)", 
	"Release for BMC/TMC other through power house (Mcum)"]]

# print the target
print(target)

# Principal Component Analysis for dimensionality reduction
pca = PCA(n_components = 10)
pfeatures = pca.fit_transform(features)
print(pfeatures)
print(pfeatures.shape)

# train test split
x_train, x_test, y_train, y_test = train_test_split(pfeatures, target, random_state = 42)


# model building
model = RandomForestRegressor(n_estimators = 150, random_state = 42)
model.fit(x_train, y_train)


# traning score of model
s1 = model.score(x_train, y_train)
print("Training Score is: ",  round(s1, 2)*100, "%")

# testing score of model
s2 = model.score(x_test, y_test)
print("Testing Score is: ", round(s2, 2)*100, "%")

print("Please enter the following features: ")

re_wa_start = float(input("Enter the reservoir water level at the start of the month: "))
eff_sto_start = float(input("Enter the effective gross storage on the start of the month: "))
esc_gate = float(input("Enter the release of water through escape gate: "))
sp_river = float(input("Enter the release of water through spillway river: "))
lea_gal = float(input("Enter the leakage through gallery: "))
re_eva = float(input("Enter the reservoir evaporation losses: "))
ot_lea = float(input("Enter the other measured leakages: "))
re_wa_end = float(input("Enter the reservoir water level at the end of the month: "))
gross_sto_end = float(input("Enter the gross storage at the end of the month: "))
inflow = float(input("Enter the calculated inflow of month: "))

df = [[re_wa_start, eff_sto_start, esc_gate, sp_river, lea_gal, re_eva, ot_lea, re_wa_end, gross_sto_end, inflow
]]

	



