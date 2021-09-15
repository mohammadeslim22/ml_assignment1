import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
# import mplcyberpunk
# style.use('cyberpunk')

train_data = pd.read_csv('./flight_delay.csv')
print(train_data.describe())

# train_data.plot(x='Depature Airport', y='Delay', style='o')
# plt.title('Destination Airport  vs Delay')
# plt.xlabel('Depature Airport')
# plt.ylabel('Delay')
# plt.show()


# fig, ax = plt.subplots()
# # create bar chart
# ax.bar(train_data['Destination Airport'], train_data['Delay'])
# # set title and labels
# ax.set_title('Destination Airport  vs Delay')
# ax.set_xlabel('Destination Airport')
# ax.set_ylabel('Delay')

# style.use('ggplot')
# train_data.loc().plot(kind = 'bar', legend = False)
# plt.title('Destination Airport  vs Delay',color = 'black')
# plt.xticks(color = 'black')
# plt.yticks(color = 'black')
# plt.xlabel('Destination Airport',color = 'black')
# plt.ylabel('Delay',color = 'black')
# plt.savefig('bar_vertical.png')
#
# plt.show()

plt.stem(train_data['Destination Airport'], train_data['Delay'],use_line_collection="true")
plt.title('Destination Airport vs Delay', color = 'black')
plt.xlabel('Destination Airport', color = 'black')
plt.ylabel('Delay', color = 'black')
plt.xticks(color = 'black')
plt.yticks(color = 'black')
plt.savefig('lollipop.png')
plt.show()