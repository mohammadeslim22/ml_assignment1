import pandas as pd
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# to show all columns and rows when print ,head(10)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

train_data = pd.read_csv('./flight_delay.csv')

types = train_data.dtypes
print("Number categorical featues:", sum(types == 'object'))
print(types)

"""
convert string objects of 'depature time' & 'arrival time' to datetime data type
"""

train_data['Scheduled depature time'] = pd.to_datetime(train_data['Scheduled depature time'])
train_data['Scheduled arrival time'] = pd.to_datetime(train_data['Scheduled arrival time'])
print(train_data.head(3))
# convert time for departure
depature_years = pd.DataFrame(train_data['Scheduled depature time'].dt.year).rename(
    columns={'Scheduled depature time': 'scheduled_depature_year'})
depature_months = pd.DataFrame(train_data['Scheduled depature time'].dt.month).rename(
    columns={'Scheduled depature time': 'scheduled_depature_month'})
depature_day_of_months = pd.DataFrame(train_data['Scheduled depature time'].dt.day).rename(
    columns={'Scheduled depature time': 'depature_day_month'})
depature_hours = pd.DataFrame(train_data['Scheduled depature time'].dt.hour).rename(
    columns={'Scheduled depature time': 'depature_hour'})
to_one_hot = train_data['Scheduled depature time'].dt.day_name()
depature_days = pd.get_dummies(to_one_hot).rename(
    columns={'Friday': 'depature_Friday', 'Monday': 'depature_Monday', 'Saturday': 'depature_Saturday',
             'Sunday': 'depature_Sunday', 'Thursday': 'depature_Thursday', 'Tuesday': 'depature_Tuesday',
             'Wednesday': 'depature_Wednesday'})
print("_______________________depature years___________________________________")
print(depature_years.head(3))
print("_______________________depature months___________________________________")
print(depature_months.head(3))
print("_______________________depature day_of_months___________________________________")
print(depature_day_of_months.head(3))
print("_______________________depature hours___________________________________")
print(depature_hours.head(3))
print("_______________________depature days___________________________________")
print(depature_days.head(3))
# convert time for arrival
arrival_years = pd.DataFrame(train_data['Scheduled arrival time'].dt.year).rename(
    columns={'Scheduled arrival time': 'scheduled_arrival_year'})
arrival_months = pd.DataFrame(train_data['Scheduled arrival time'].dt.month).rename(
    columns={'Scheduled arrival time': 'scheduled_arrival_month'})
arrival_day_of_months = pd.DataFrame(train_data['Scheduled arrival time'].dt.day).rename(
    columns={'Scheduled arrival time': 'arrival_day_month'})
arrival_hours = pd.DataFrame(train_data['Scheduled arrival time'].dt.hour).rename(
    columns={'Scheduled arrival time': 'arrival_hour'})
to_one_hot = train_data['Scheduled arrival time'].dt.day_name()
arrival_days = pd.get_dummies(to_one_hot).rename(
    columns={'Friday': 'arrival_Friday', 'Monday': 'arrival_Monday', 'Saturday': 'arrival_Saturday',
             'Sunday': 'arrival_Sunday', 'Thursday': 'arrival_Thursday', 'Tuesday': 'arrival_Tuesday',
             'Wednesday': 'arrival_Wednesday'})
print("_______________________depature years___________________________________")
print(arrival_years.head(3))
print("_______________________arrival months___________________________________")
print(arrival_months.head(3))
print("_______________________arrival day_of_months___________________________________")
print(arrival_day_of_months.head(3))
print("_______________________arrival hours___________________________________")
print(arrival_hours.head(3))
print("_______________________arrival days___________________________________")
print(arrival_days.head(3))




def daypart(hour):
    if hour in [2, 3, 4, 5]:
        return "dawn"
    elif hour in [6, 7, 8, 9]:
        return "morning"
    elif hour in [10, 11, 12, 13]:
        return "noon"
    elif hour in [14, 15, 16, 17]:
        return "afternoon"
    elif hour in [18, 19, 20, 21]:
        return "evening"
    else:
        return "midnight"


arrival_raw_dayparts = train_data['Scheduled arrival time'].dt.hour.apply(daypart)
depature_raw_dayparts = train_data['Scheduled depature time'].dt.hour.apply(daypart)
#
depature_dayparts  = pd.get_dummies(depature_raw_dayparts).rename(
    columns={'afternoon': 'depature_afternoon', 'dawn': 'depature_dawn', 'evening': 'arrival_evening',
             'midnight': 'depature_midnight', 'morning': 'depature_morning', 'noon': 'depature_noon'})
arrival_dayparts = pd.get_dummies(arrival_raw_dayparts).rename(
    columns={'afternoon': 'arrival_afternoon', 'dawn': 'arrival_dawn', 'evening': 'arrival_evening',
             'midnight': 'arrival_midnight', 'morning': 'arrival_morning', 'noon': 'arrival_noon'})
print("_______________________arrival dayparts___________________________________")
print(arrival_dayparts.head(3))
print("_______________________depature dayparts___________________________________")
print(depature_dayparts.head(3))


train_data = train_data.drop(['Scheduled arrival time', 'Scheduled depature time'], axis=1)
print(train_data.head(3))


frames = [train_data,depature_years, depature_months, depature_day_of_months, depature_hours, depature_days,depature_dayparts,arrival_years, arrival_months,
          arrival_day_of_months, arrival_hours, arrival_days,arrival_dayparts]

print("_______________________ final data frame  ___________________________________")
train_data = pd.concat(frames, axis=1)

print(train_data.head(3))
# splitting the data
x_train, x_test, y_train, y_test = train_test_split(train_data.drop(['Delay'], axis=1), train_data['Delay'],
                                                    test_size=0.2)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# print(y_test.head(100))
imputer = SimpleImputer(strategy='most_frequent')
imputer.fit(x_train)
x_train = pd.DataFrame(imputer.transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(imputer.transform(x_test), columns=x_test.columns)

# One-hot-encoding of categorical feature
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
cat_feats = ['Depature Airport', 'Destination Airport']
encoder.fit(x_train[cat_feats])


def ohe_new_features(df, features_name, encoder):
    new_feats = encoder.transform(df[features_name])
    new_cols = pd.DataFrame(new_feats, dtype=int)
    new_df = pd.concat([df, new_cols], axis=1)
    new_df.drop(features_name, axis=1, inplace=True)
    return new_df


x_train = ohe_new_features(x_train, cat_feats, encoder)
x_test = ohe_new_features(x_test, cat_feats, encoder)

x_train_types = x_train.dtypes
print("Number categorical featues:", sum(types == 'object'))
# print(x_train_types)

x_test_types = x_test.dtypes
print("Number categorical featues:", sum(types == 'object'))
# print(x_test_types)

# print(x_train.head(100))
