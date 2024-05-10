import pandas as pd

# Load in data
df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv")

# DATA PREPARATION
# Final column, logS, is the output value or y value
# y = f(x) where the remaining columns are x values

# Data separation as x and y
# y = final column of csv file, titled logS
# with this data, y encapsulates the solubility values that correspond to various chemical descriptors in the dataset
y = df['logS']
# x = remaining columns, axis=1 creates column values, axis=1 tells pandas that I want to drop a column, not a row (axis=0)
x = df.drop('logS', axis=1)

#  Data splitting
from sklearn.model_selection import train_test_split

# Conventional code for data splitting to prepare for linear regression. 
# This creates a training set and a test set with 80% of the data in the training set and 20% in the test set (test_size=0.2)
# random_state=100 ensures that the data is split in the same way each time the code is run
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# MODEL BUILDING
# Linear Regression
from sklearn.linear_model import LinearRegression

# Training the model
# lr becomes our instance of LinearRegression()
lr = LinearRegression()
# pass in training data to the model. The model learns the coefficients (parameters) for each feature in X_train that best fit the data based on minimizing the sum of the squares of the residuals (the differences between observed values in the dataset, and the values predicted by the linear approximation)
lr.fit(X_train, y_train)

# Applying the model to make a prediction
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

# print(y_lr_train_pred)
# print(y_lr_test_pred)

# Evaluating the model
from sklearn.metrics import mean_squared_error, r2_score

# Mean Squared Error:
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
# Squared correlation coefficient:
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

# print("LR Train MSE: ", lr_train_mse)
# print("LR Train R2: ", lr_train_r2)
# print("LR Test MSE: ", lr_test_mse)
# print("LR Test R2: ", lr_test_r2)

lr_results = pd.DataFrame(["Linear Regression", lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ["Method", "Training MSE", "Training R2", "Test MSE", "Test R2"]

# print(lr_results)

# Random Forest
from sklearn.ensemble import RandomForestRegressor

# Training the model
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(X_train, y_train)

RandomForestRegressor(max_depth=2, random_state=100)

# Applying the model to make a prediction
y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)

# Evaluating the model
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

rf_results = pd.DataFrame(["Random Forest", rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ["Method", "Training MSE", "Training R2", "Test MSE", "Test R2"]

# Model Comparison
model_comparison = pd.concat([lr_results, rf_results], axis=0).reset_index(drop=True)

# print(model_comparison)

# Data Visualization of prediction results
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(5,5))
plt.scatter(x=y_train, y=y_lr_train_pred, alpha=0.5, color="blue")

z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)

plt.plot(y_train, p(y_train), color="green")
plt.ylabel("Predicted LogS")
plt.xlabel("Experimental LogS")

# plt.show()