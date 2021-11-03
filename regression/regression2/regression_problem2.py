#Import
import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split

#Extract data
X = np.load('Xtrain_Regression_Part1.npy')
y = np.load('Ytrain_Regression_Part1.npy')
X_test = np.load('Xtest_Regression_Part1.npy')

# Defining necessary functions

def outliers(df, feature):

    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.7 * IQR
    upper = Q3 + 1.7 * IQR

    outlier_list = df.index[ (df[feature] < lower) | (df[feature] > upper) ]

    return outlier_list

def remove_outlier(df, index_list):
    ls = sorted(set(index_list))
    df = df.drop(ls)
    return df


# Turning the numpy array of X into pandas dataframe
list_of_features = ['Column_1','Column_2','Column_3','Column_4','Column_5','Column_6','Column_7','Column_8','Column_9','Column_10','Column_11','Column_12','Column_13','Column_14','Column_15','Column_16','Column_17','Column_18','Column_19','Column_20']
df_raw = pd.DataFrame(X, columns = list_of_features)
y_df_raw = pd.DataFrame(y, columns = ['y'])

index_list = []

for feature in list_of_features:
    index_list.extend(outliers(df_raw, feature))

# Remove outliers
X_cleaned = remove_outlier(df_raw, index_list)
y_cleaned = remove_outlier(y_df_raw, index_list)

# Turning back to numpy array
X_np_cleaned = X_cleaned.to_numpy()
y_np_cleaned = y_cleaned.to_numpy()

#Create training and validation sets
#X_train, X_validate, y_train, y_validate = train_test_split(X_np_cleaned, y_np_cleaned.ravel())

# RANSAC
ransac_regmod = RANSACRegressor()

# Training
ransac_regmod.fit(X_cleaned, y_cleaned)

# Predict for validation set
ransac_reg_pred = ransac_regmod.predict(X_test)

# Save predictions to file
np.save('predictions.npy', ransac_reg_pred)