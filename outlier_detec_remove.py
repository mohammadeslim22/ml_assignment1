import numpy as np
import pandas as pd


train_data = pd.read_csv('./flight_delay.csv')


def removeOutliers(dataframe):
    print("---------------------- before editing------------------- ")
    print(dataframe['Delay'].std())
    print(dataframe['Delay'].mean())
    print("Highest allowed", dataframe['Delay'].mean() + 3 * dataframe['Delay'].std())
    print("Lowest allowed", dataframe['Delay'].mean() - 3 * dataframe['Delay'].std())
    print(dataframe['Delay'].describe())
    upper_limit = dataframe['Delay'].mean() + 3 * dataframe['Delay'].std()
    lower_limit = dataframe['Delay'].mean() - 3 * dataframe['Delay'].std()
    print(train_data[(train_data['Delay'] > upper_limit) | (train_data['Delay'] < lower_limit)])
    print(train_data.shape)
    print("---------------------- after editing------------------- ")

    new_train_data = dataframe[(dataframe['Delay'] < upper_limit) & (dataframe['Delay'] > lower_limit)]
    print(dataframe.shape)
    print(dataframe['Delay'].describe())
    print(new_train_data['Delay'].describe())

    # Other way to crop the data frame

    dataframe['Delay'] = np.where(
        dataframe['Delay'] > upper_limit,
        upper_limit,
        np.where(
            dataframe['Delay'] < lower_limit,
            lower_limit,
            dataframe['Delay']
        )
    )
    print(dataframe['Delay'].describe())
    return new_train_data
