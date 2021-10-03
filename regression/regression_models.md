Models mentioned in lecture notes:
----------------------------------
- Nearest Neighbor (Value is set to nearest)
- K Nearest Neighbor (Value set to average of the nearest)
- Linear Regression (Very sensitive to outliers)
- Polynomial Regression (Becomes numerically unstable when order of polynomial is increased. Also, higher order may result in overfitting)
- Ridge Regression (Assumes training data has zero mean! Penalizes use of large coefficients)
- The Lasso (Penalizes large error less)

We need to:
-----------
1. Implement the different models
2. Implement one or multiple ways of solving the optimization problem
3. Split training set into training and validation set
4. Train the different methods, test on the validation set, and measure mean squared error
5. Choose the one with the best results, and train it using the whole training set
6. Use the trained model on the given test set, and hand in the results
7. Write regression part of the report while implementing the models
