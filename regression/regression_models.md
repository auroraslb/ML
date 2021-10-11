Models mentioned in lecture notes:
----------------------------------
- Nearest Neighbor (Value is set to nearest)
- K Nearest Neighbor (Value set to average of the nearest)
- Linear Regression (Very sensitive to outliers)
- Polynomial Regression (Becomes numerically unstable when order of polynomial is increased. Also, higher order may result in overfitting)
- Ridge Regression (Assumes training data has zero mean! Penalizes use of large coefficients)
- The Lasso (Penalizes large error less)

Other models:
-------------
- Decision Tree Regression (Breaks data into smaller sets, but changes in data change the entire structure and may cause it to become unstable)
- Random Forest (Uses multiple decision trees)
- Support Vector Machines (Can solve both linear and non-linear problems. SVMs are not at all suitable for predicting values for large training sets. SVM fails when data has more noise)
- Bayesian Linear Regression (Like both Linear Regression and Ridge Regression but more stable than the simple Linear Regression)
- Stepwise Regression (Used when we deal with multiple independent variables.)
- Neural Network Regression (You all must be aware of the power of neural networks in making predictions/assumptions)
- Elastic Net Regression (ElasticNet is hybrid of Lasso and Ridge Regression techniques. It is trained with L1 and L2 prior as regularizer. Elastic-net is useful when there are multiple features which are correlated)

Models we MAY check out:
------------------------
- JackKnife Regression
- Ecological Regression

We need to:
-----------
1. Implement the different models
2. Implement one or multiple ways of solving the optimization problem
3. Split training set into training and validation set
4. Train the different methods, test on the validation set, and measure mean squared error
5. Choose the one with the best results, and train it using the whole training set
6. Use the trained model on the given test set, and hand in the results
7. Write regression part of the report while implementing the models
