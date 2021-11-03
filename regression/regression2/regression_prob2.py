#Import
import numpy as np
from sklearn.linear_model import HuberRegressor

#Extract data
X = np.load('Xtrain_Regression_Part2.npy')
y = np.load('Ytrain_Regression_Part2.npy')
X_test = np.load('Xtest_Regression_Part2.npy')

## Huber Regression
huber_mod = HuberRegressor(epsilon=1.15)

# Training
huber_mod.fit(X, y.reshape(len(y),))

# Predict for test
huber_reg_pred = huber_mod.predict(X_test)

# Save predictions to file
np.save('predictions.npy', huber_reg_pred)