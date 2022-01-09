import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import metrics


# LOADING DATA AND LABELS

data = pd.read_csv("data_new2.csv", names=["a", "b","c", "d", "e", "D", "F"])

features = data.copy()
labels = features.pop("F")


features = np.array(features)
labels = np.array(labels)


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 400, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=None, bootstrap=False)

# Train the model on training data
rf.fit(X_train, y_train)



# Use the forest's predict method on the test data
predictions = rf.predict(X_test)
print(predictions)
print(y_test)
# Calculate the absolute errors
errors = abs(predictions - y_test)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'Newtons.')


# Calculate mean absolute percentage error (MAPE)
# mape = 100 * (errors / y_test)
# # Calculate and display accuracy
# accuracy = 100 - np.mean(mape)
# print('Accuracy:', round(accuracy, 2), '%.')



# {'n_estimators': 400, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False}



