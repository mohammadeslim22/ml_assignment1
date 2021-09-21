import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
# import mplcyberpunk
# style.use('cyberpunk')
from ml_assignment1 import Feature_Engineering

train_data = pd.read_csv('./flight_delay.csv')
print(train_data.describe())

# train_data.plot(x='Scheduled arrival time', y='Delay', style='o')
# plt.title('Arrival Airport  vs Delay')
# plt.xlabel('Destination Airport')
# plt.ylabel('Delay')
# plt.show()


# fig, ax = plt.subplots()
# # create bar chart
# ax.bar(train_data['Destination Airport'], train_data['Delay'])
# # set title and labels
# ax.set_title('Destination Airport  vs Delay')
# ax.set_xlabel('Destination Airport')
# ax.set_ylabel('Delay')
# plt.show()

# style.use('ggplot')
# train_data.plot(kind = 'bar', legend = False)
# plt.title('Destination Airport  vs Delay',color = 'black')
# plt.xticks(color = 'black')
# plt.yticks(color = 'black')
# plt.xlabel('Destination Airport',color = 'black')
# plt.ylabel('Delay',color = 'black')
# plt.savefig('bar_vertical.png')
# plt.show()

# plt.stem(train_data['Destination Airport'], train_data['Delay'],use_line_collection="true")
# plt.title('Destination Airport vs Delay', color = 'black')
# plt.xlabel('Destination Airport', color = 'black')
# plt.ylabel('Delay', color = 'black')
# plt.xticks(color = 'black')
# plt.yticks(color = 'black')
# plt.savefig('Destination Airport Steam.png')
# plt.show()

# sns.boxplot(train_data['Destination Airport'],train_data['Delay'])

# plt.figure(figsize=(16,5))
# plt.subplot(1,2,1)
# sns.distplot(train_data['Delay'])
# plt.subplot(1,2,2)
# sns.distplot(Feature_Engineering.x_train["scheduled_arrival_year"])
# plt.show()

# create a figure and axis
# fig, ax = plt.subplots()
# # plot each data-point
# for i in range(len(train_data['Delay'])):
#     ax.scatter(train_data['Delay'][i], train_data['Depature Airport'][i])
# # set a title and labels
# ax.set_title('Iris Dataset')
# ax.set_xlabel('sepal_length')
# ax.set_ylabel('sepal_width')
# plt.show()

# fig, ax = plt.subplots()
# # plot histogram
# ax.hist(train_data['Delay'])
# # set title and labels
# ax.set_title('Delay amount Frequency')
# ax.set_xlabel('Delay')
# ax.set_ylabel('Frequency')
# plt.show()

# fig, ax = plt.subplots()
# # count the occurrence of each class
# data = train_data['Destination Airport'].value_counts()
# # get x and y data
# points = data.index
# frequency = data.values
# # create bar chart
# ax.bar(points, frequency)
# # set title and labels
# ax.set_title('Wine Review Scores')
# ax.set_xlabel('Points')
# ax.set_ylabel('Frequency')
# plt.show()

plt.scatter(Feature_Engineering.train_data['flight duration'],Feature_Engineering.train_data['Delay']/60)
plt.title('flight duration to Delay')
plt.xlabel('Flight duration')
plt.ylabel('Delay')
plt.show()