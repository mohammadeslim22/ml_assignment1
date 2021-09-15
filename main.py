import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# to show all columns and rows when print ,head(10)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

train_data = pd.read_csv('./flight_delay.csv')

# print(x_train.head(10))
# print(y_train.head(10))

types = train_data.dtypes
print("Number categorical featues:", sum(types == 'object'))
print(types)

"""
convert string objects of 'depature time' & 'arrival time' to datetime data type
"""
# x_train["Scheduled depature time"] = pd.to_datetime(x_train["Scheduled depature time"],
#                                                     format='%Y-%m-%dT%H:%M:%SZ',
#                                                     errors='coerce')
# x_train["Scheduled arrival time"] = pd.to_datetime(x_train["Scheduled arrival time"],
#                                                    format='%Y-%m-%dT%H:%M:%SZ',
#                                                    errors='coerce')
train_data['Scheduled depature time'] = pd.to_datetime(train_data['Scheduled depature time'])
train_data['Scheduled arrival time'] = pd.to_datetime(train_data['Scheduled arrival time'])

months = train_data['Scheduled depature time'].dt.month
day_of_months = train_data['Scheduled depature time'].dt.day
hours = train_data['Scheduled depature time'].dt.hour
to_one_hot = train_data['Scheduled depature time'].dt.day_name()
days = pd.get_dummies(to_one_hot)


#splitting the data
x_train, x_test, y_train, y_test = train_test_split(train_data.drop(['Delay'], axis = 1), train_data['Delay'],test_size=0.2)
# x_train = train_data.drop('Delay', axis=1)
# y_train = train_data['Delay']
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

print(y_test.head(100))

print(y_test.head(100))
# types = x_train.dtypes
# print("Number categorical featues:", sum(types == 'object'))
# print(types)





# One-hot-encoding of categorical feature
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
cat_feats = ['Depature Airport','Destination Airport']
encoder.fit(x_train[cat_feats])

def ohe_new_features(df, features_name, encoder):
    new_feats = encoder.transform(df[features_name])
    new_cols = pd.DataFrame(new_feats, dtype=int)
    new_df = pd.concat([df, new_cols], axis=1)
    new_df.drop(features_name, axis=1, inplace=True)
    return new_df

x_train = ohe_new_features(x_train,cat_feats,encoder)

types = x_train.dtypes
print("Number categorical featues:", sum(types == 'object'))
print(types)

# print(x_train.head(100))