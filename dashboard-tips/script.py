# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# modeling imports
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV # Linear Regression Model
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, SplineTransformer, OneHotEncoder #Z-score variables, Polynomial
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error #model evaluation
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
# pipeline imports
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error #model evaluation
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut

# %%
# models
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor # Decision Tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # random forest
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor # gradient boosting
import xgboost as xg
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# %%
df = pd.read_csv('data\lotwize_case.csv')
valid_df = pd.read_csv('data\lotwize_case_validation-1.csv')

# %%
schools = pd.read_csv('data\school_data.csv')

# %%
schools.columns

# %%
predictors = [#'schools/1/rating',
#'schools/0/distance',
#'schools/0/name',
'mortgageRates/thirtyYearFixedRate',
#'nearbyHomes/1/price',
'bathrooms',
'homeType',
'adTargets/sqft',
'bedrooms',
'zipcode',
'yearBuilt',
'photoCount',
'longitude',
'latitude',
'city',
'price'
]

# %%
new_df = df[predictors]
new_df.shape

# %%
zipcodes = df[['zipcode', 'price']]

# %%
unique_schools_by_zipcode = schools.groupby('Zip')['School Name'].nunique().reset_index()
unique_schools_by_zipcode.columns = ['Zip', 'unique_school_count']
school_count = unique_schools_by_zipcode.rename(columns = {'Zip': 'zipcode', 'unique_school_count': 'school_count'})

# %%
new_df = pd.merge(new_df, school_count, on = 'zipcode')

# %%
new_df.shape

# %%
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on the Earth.
    
    Parameters:
    lat1, lon1: Latitude and Longitude of the first point.
    lat2, lon2: Latitude and Longitude of the second point.
    
    Returns:
    Distance between the two points in kilometers.
    """
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of Earth in kilometers (mean radius)
    r = 6371.0
    
    # Calculate the distance
    distance = c * r
    
    return distance

# %%
def school_finder(row):
    zipcode = row['zipcode']
    home_lat = row['latitude']
    home_long = row['longitude']
    schools_in_area = schools[schools['Zip'] == zipcode][['School Name', 'Longitude', 'Latitude']]
    min_distance = 80000000000
    for school in schools_in_area['School Name']:
        school_lat = schools_in_area[schools_in_area['School Name'] == school]['Latitude']
        school_long = schools_in_area[schools_in_area['School Name'] == school]['Longitude']
        distance = haversine_distance(home_lat, home_long, school_lat, school_long)
        if distance < min_distance:
            min_distance = distance
    return min_distance

# %%
new_df.isnull().sum()

# %%
new_df = new_df.dropna().reset_index()
new_df.shape

# %%
import warnings
warnings.filterwarnings('ignore')
new_df['closest_school_distance'] = new_df.apply(school_finder, axis = 1)

# %%
new_df.shape

# %%
new_df = new_df.drop(columns = ['index', 'longitude', 'latitude'])


# %%
crime = pd.read_csv("data\crime_data\ca_offenses_by_city.csv")

# %%
crime.head()

# %%
crime = crime[['Population', 'Violent crime', 'Property crime', 'City']]

# %%
#dividing the number of crimes reported to law enforcement by the total population and then multiplying by 100,000

# %%
def removeCommas(string):
    new_string = string.replace(",","")
    return new_string

# %%
crime['Population'] = crime['Population'].apply(removeCommas)
crime['Violent crime'] = crime['Violent crime'].apply(removeCommas)
crime['Property crime'] = crime['Property crime'].apply(removeCommas)

# %%
crime['Population'] = crime['Population'].astype(int)
crime['Violent crime'] = crime['Violent crime'].astype(int)
crime['Property crime'] = crime['Property crime'].astype(int)
crime.dtypes

# %%
crime['crime rate'] = (((crime['Violent crime'] + crime['Property crime']) / crime['Population']) * 1000)

# %%
crime = crime.rename(columns = {'City': 'city'})
crime = crime[['city','crime rate']]

# %%
new_df = pd.merge(new_df, crime, on = 'city')

# %%
income = pd.read_csv('data\income\kaggle_income.csv',engine='python',encoding='latin1')

# %%
income = income[income['State_ab'] == 'CA']

# %%
income

# %%
income = income.rename(columns = {'Zip_Code': 'zipcode', 'Median': 'Median Household Income'})
income = income[['zipcode', 'Median Household Income']]

# %%
new_df = pd.merge(new_df, income, on = 'zipcode')

# %%
new_df = new_df.drop(columns = ['city'])

# %%
unique_df = new_df.drop_duplicates()
unique_df.shape

# %%
predictors = new_df.columns.tolist()
contin = new_df.columns.tolist()
len(predictors)

# %%
predictors.remove('price')
contin.remove('homeType')
#contin.remove('schools/0/name')
contin.remove('zipcode')
contin.remove('price')
cat = ['homeType', 'zipcode']

# %%
new_df['zipcode'] = new_df["zipcode"].astype("category")
new_df['homeType'] = new_df["homeType"].astype("category")
#new_df['schools/0/name'] = new_df["schools/0/name"].astype("category")

# %%
len(predictors)

# %%
X = new_df[predictors]
y = new_df["price"]

# %%
contin = predictors
contin.remove('homeType')
contin.remove('zipcode')
contin

# %%
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 101)
xgb = xg.XGBRegressor(n_estimators = 1000, max_depth = 3, random_state = 101, booster = 'gbtree', learning_rate = 0.01, enable_categorical=True, 
                      eval_metric = "rmse", early_stopping_rounds=50, reg_lambda = 1, reg_alpha = 0.5)
xgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=True
)

# Make predictions and evaluate
y_pred = xgb.predict(X_test)
val_error = mean_squared_error(y_test, y_pred)
print(f"Validation error: {val_error}")

# %%
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 101)

# %%
xgb = xg.XGBRegressor(n_estimators = 428, max_depth = 3, random_state = 101, booster = 'gbtree', 
                      learning_rate = 0.01, enable_categorical=True)

# model validation
kf = KFold(n_splits = len(predictors))
results = cross_val_score(xgb, X, y, cv=kf)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

mse = {"train": [], "test": []}
mae = {"train": [], "test": []}
mape = {"train": [], "test": []}
r2 = {"train": [], "test": []}

for train, test in kf.split(X):
    X_train = X.iloc[train]
    X_test  = X.iloc[test]
    y_train = y[train]
    y_test  = y[test]

    # fit
    xgb.fit(X_train,y_train)

    # predict
    y_pred_train = xgb.predict(X_train)
    y_pred_test = xgb.predict(X_test)

    # assess
    mse["train"].append(mean_squared_error(y_train,y_pred_train))
    mse["test"].append(mean_squared_error(y_test,y_pred_test))
    
    mae["train"].append(mean_absolute_error(y_train,y_pred_train))
    mae["test"].append(mean_absolute_error(y_test,y_pred_test))

    mape["train"].append(mean_absolute_percentage_error(y_train,y_pred_train))
    mape["test"].append(mean_absolute_percentage_error(y_test,y_pred_test))

    r2["train"].append(r2_score(y_train,y_pred_train))
    r2["test"].append(r2_score(y_test,y_pred_test))

print("Train MSEs:", mse["train"])
print("Test MSEs :", mse["test"])
print("Train MSE :", np.mean(mse["train"]))
print("Test MSE  :", np.mean(mse["test"]))

print("Train MAEs:", mae["train"])
print("Test MAEs :", mae["test"])
print("Train MAE :", np.mean(mae["train"]))
print("Test MAE  :", np.mean(mae["test"]))

print("Train MAPEs:", mape["train"])
print("Test MAPEs :", mape["test"])
print("Train MAPE :", np.mean(mape["train"]))
print("Test MAPE  :", np.mean(mape["test"]))

print("Train R2s:", r2["train"])
print("Test R2s :", r2["test"])
print("Train R2 :", np.mean(r2["train"]))
print("Test R2  :", np.mean(r2["test"]))

# %%
new_df = new_df.drop(columns = ['zipcode'])

# %%
X = new_df[predictors]
y = new_df["price"]

# %%
contin = predictors
predictors

# %%
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
categorical_features = [#'zipcode'
    'homeType']  # Replace with your categorical feature names
continuous_features = [contin]  # Replace with your continuous feature names
# Preprocessing for categorical data
continuous_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('cont', continuous_transformer, continuous_features)
    ])

"""
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 101)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)
"""
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 101)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xg.XGBRegressor(n_estimators = 428, max_depth = 3, random_state = 101, booster = 'gbtree', 
                    learning_rate = 0.01, enable_categorical=True, 
                    eval_metric = "rmse", reg_lambda = 1, reg_alpha = 0.5,))
])
"""
pipeline.named_steps['model'].fit(
    pipeline.named_steps['preprocessor'].fit_transform(X_train), y_train,
    eval_set=[(pipeline.named_steps['preprocessor'].transform(X_val), y_val)],
    early_stopping_rounds=10,
    verbose=True
)
"""

"""
xgb = xg.XGBRegressor(n_estimators = 428, max_depth = 3, random_state = 101, booster = 'gbtree', 
                    learning_rate = 0.01, enable_categorical=True, 
                    eval_metric = "rmse", early_stopping_rounds=10, reg_lambda = 1, reg_alpha = 0.5,)

xgb.fit(X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=True
            )
"""
xgb.fit(X_train, y_train)
# model validation
kf = KFold(n_splits = len(predictors))
results = cross_val_score(xgb, X, y, cv=kf)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

mse = {"train": [], "test": []}
mae = {"train": [], "test": []}
mape = {"train": [], "test": []}
r2 = {"train": [], "test": []}

for train, test in kf.split(X):
    X_train = X.iloc[train]
    X_test  = X.iloc[test]
    y_train = y[train]
    y_test  = y[test]

    # fit
    xgb.fit(X_train,y_train)

    # predict
    y_pred_train = xgb.predict(X_train)
    y_pred_test = xgb.predict(X_test)

    # assess
    mse["train"].append(mean_squared_error(y_train,y_pred_train))
    mse["test"].append(mean_squared_error(y_test,y_pred_test))
    
    mae["train"].append(mean_absolute_error(y_train,y_pred_train))
    mae["test"].append(mean_absolute_error(y_test,y_pred_test))

    mape["train"].append(mean_absolute_percentage_error(y_train,y_pred_train))
    mape["test"].append(mean_absolute_percentage_error(y_test,y_pred_test))

    r2["train"].append(r2_score(y_train,y_pred_train))
    r2["test"].append(r2_score(y_test,y_pred_test))

print("Train MSEs:", mse["train"])
print("Test MSEs :", mse["test"])
print("Train MSE :", np.mean(mse["train"]))
print("Test MSE  :", np.mean(mse["test"]))

print("Train MAEs:", mae["train"])
print("Test MAEs :", mae["test"])
print("Train MAE :", np.mean(mae["train"]))
print("Test MAE  :", np.mean(mae["test"]))

print("Train MAPEs:", mape["train"])
print("Test MAPEs :", mape["test"])
print("Train MAPE :", np.mean(mape["train"]))
print("Test MAPE  :", np.mean(mape["test"]))

print("Train R2s:", r2["train"])
print("Test R2s :", r2["test"])
print("Train R2 :", np.mean(r2["train"]))
print("Test R2  :", np.mean(r2["test"]))

# %%
new_df = pd.merge(new_df, zipcodes, on = 'price')


# %%
#zipcodes.dtypes


# %%
#new_df['zipcode'] = new_df['zipcode'].astype(str)
#new_df['zipcode'] = new_df['zipcode'].apply(lambda x: "zip: " + x )

# %%
#new_df  = new_df.drop(columns=['zipcode'])
#new_df.head()

# %%



