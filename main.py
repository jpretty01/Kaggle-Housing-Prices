# Jeremy Pretty
# Kaggle Competition Housing Prices
# 11 Feb 24
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import numpy as np

# Load the datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Preprocessing - Imputing missing values
imputer = SimpleImputer(strategy='median')
train_features = train_data.select_dtypes(include=[np.number]).drop(['SalePrice'], axis=1)
train_target = train_data['SalePrice']
X_train = imputer.fit_transform(train_features)
y_train = train_target

# Model Training - RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Preprocessing - Test data
test_features = test_data.select_dtypes(include=[np.number])
X_test = imputer.transform(test_features)

# Making predictions on the test data
test_predictions = model.predict(X_test)

# Creating the submission file
submission = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': test_predictions})
submission.to_csv('house_price_predictions.csv', index=False)
