#This document contains the final model, which reached 86% accuracy
import pandas as pd
import sklearn as skl
from sklearn.ensemble import RandomForestClassifier

# Reading in the data
data = pd.read_csv('/users/kristiandampedersen/documents/mobile_ml_proj/data/train.csv')
test = pd.read_csv('/users/kristiandampedersen/documents/mobile_ml_proj/data/test.csv')

# Defining targets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X = data.drop('price_range', axis=1)
y = data['price_range']
X_train, x_val, y_train, y_val = train_test_split(X, y, random_state=1)

#Final model
best_model = RandomForestClassifier(random_state=1)
best_model.fit(X_train,y_train)

# Evaluating our model
y_pred = best_model.predict(x_val)
a_score = accuracy_score(y_pred, y_val)
print(a_score)