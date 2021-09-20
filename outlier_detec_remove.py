import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train_data = pd.read_csv('./flight_delay.csv')

def removeOutliers(datafframe):
    print("---------------------- before editing------------------- ")
    print(datafframe['Delay'].std())
    print(datafframe['Delay'].mean())
    print("Highest allowed",datafframe['Delay'].mean() + 3*datafframe['Delay'].std())
    print("Lowest allowed",datafframe['Delay'].mean() - 3*datafframe['Delay'].std())
    print(datafframe['Delay'].describe())
    print(train_data[(train_data['Delay'] > 144.6) | (train_data['Delay'] < -124.77)])
    print(train_data.shape)
    print("---------------------- after editing------------------- ")
    upper_limit = datafframe['Delay'].mean() + 3 * datafframe['Delay'].std()
    lower_limit = datafframe['Delay'].mean() - 3 * datafframe['Delay'].std()
    new_train_data = datafframe[(datafframe['Delay'] < upper_limit) & (datafframe['Delay'] > lower_limit)]
    print(datafframe.shape)
    print(datafframe['Delay'].describe())
    print(new_train_data['Delay'].describe())

    upper_limit = datafframe['Delay'].mean() + 3*datafframe['Delay'].std()
    lower_limit = datafframe['Delay'].mean() - 3*datafframe['Delay'].std()

    datafframe['Delay'] = np.where(
        datafframe['Delay']>upper_limit,
        upper_limit,
        np.where(
            datafframe['Delay']<lower_limit,
            lower_limit,
            datafframe['Delay']
        )
    )
    print("------4")
    print(datafframe['Delay'].describe())
    return new_train_data