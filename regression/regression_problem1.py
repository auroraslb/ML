#Import
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Extract data
X = np.load('Xtrain_Regression_Part1.npy')
y = np.load('Ytrain_Regression_Part1.npy')

#Create training and validation sets
X_train, X_validate, y_train, y_validate = train_test_split(X, y)

#Create model
linear_regression_model = LinearRegression()

#Train model
linear_regression_model.fit(X_train, y_train)

#Predict for validation set
linear_regression_predictions = linear_regression_model.predict(X_validate)

#Evaluate predictions
linear_regression_mse = mean_squared_error(linear_regression_predictions, y_validate)
