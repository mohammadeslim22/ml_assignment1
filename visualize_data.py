import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import style
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
# import mplcyberpunk
from sklearn.preprocessing import StandardScaler
import Feature_Engineering

train_data = pd.read_csv('./flight_delay.csv')
print(train_data.describe())

train_data.plot(x='Scheduled arrival time', y='Delay', style='o')
plt.title('Arrival Airport  vs Delay')
plt.xlabel('Destination Airport')
plt.ylabel('Delay')
plt.show()


fig, ax = plt.subplots()
# create bar chart
ax.bar(train_data['Destination Airport'], train_data['Delay'])
# set title and labels
ax.set_title('Destination Airport  vs Delay')
ax.set_xlabel('Destination Airport')
ax.set_ylabel('Delay')
plt.show()

style.use('ggplot')
train_data.plot(kind = 'bar', legend = False)
plt.title('Destination Airport  vs Delay',color = 'black')
plt.xticks(color = 'black')
plt.yticks(color = 'black')
plt.xlabel('Destination Airport',color = 'black')
plt.ylabel('Delay',color = 'black')
plt.savefig('bar_vertical.png')
plt.show()

plt.stem(train_data['Destination Airport'], train_data['Delay'],use_line_collection="true")
plt.title('Destination Airport vs Delay', color = 'black')
plt.xlabel('Destination Airport', color = 'black')
plt.ylabel('Delay', color = 'black')
plt.xticks(color = 'black')
plt.yticks(color = 'black')
plt.savefig('Destination Airport Steam.png')
plt.show()

sns.boxplot(train_data['Destination Airport'],train_data['Delay'])

plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(train_data['Delay'])
plt.subplot(1,2,2)
sns.distplot(Feature_Engineering.x_train["scheduled_arrival_year"])
plt.show()

# create a figure and axis
fig, ax = plt.subplots()
# plot each data-point
for i in range(len(train_data['Delay'])):
    ax.scatter(train_data['Delay'][i], train_data['Depature Airport'][i])
# set a title and labels
ax.set_title('Iris Dataset')
ax.set_xlabel('sepal_length')
ax.set_ylabel('sepal_width')
plt.show()

fig, ax = plt.subplots()
# plot histogram
ax.hist(train_data['Delay'])
# set title and labels
ax.set_title('Delay amount Frequency')
ax.set_xlabel('Delay')
ax.set_ylabel('Frequency')
plt.show()

fig, ax = plt.subplots()
# count the occurrence of each class
data = train_data['Destination Airport'].value_counts()
# get x and y data
points = data.index
frequency = data.values
# create bar chart
ax.bar(points, frequency)
# set title and labels
ax.set_title('Wine Review Scores')
ax.set_xlabel('Points')
ax.set_ylabel('Frequency')
plt.show()

plt.scatter(Feature_Engineering.train_data['flight duration'],Feature_Engineering.train_data['Delay']/60)
plt.title('flight duration to Delay')
plt.xlabel('Flight duration')
plt.ylabel('Delay')
plt.show()

dim_reducer = PCA(n_components=2)
df_reduced = dim_reducer.fit_transform(Feature_Engineering.x_train)
plt.scatter(df_reduced[:,0],df_reduced[:,1],s=0.1)
plt.title("PCA 2 dimentions")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()


X = Feature_Engineering.train_data.drop('Delay',axis=1)
y = Feature_Engineering.train_data['Delay']
sc = StandardScaler()

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

pca = PCA(n_components=3)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

ex_variance=np.var(X_pca,axis=0)
ex_variance_ratio = ex_variance/np.sum(ex_variance)
ex_variance_ratio


Xax = X_pca[:,0]
Yax = X_pca[:,1]
Zax = X_pca[:,2]

cdict = {0:'red',1:'green'}
labl = {0:'Malignant',1:'Benign'}
marker = {0:'*',1:'o'}
alpha = {0:.3, 1:.5}

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')

fig.patch.set_facecolor('white')
for l in np.unique(y):
    ix=np.where(y==l)
    ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], s=1,label=labl[l], marker=marker[l], alpha=alpha[l])
# for loop ends
ax.set_xlabel("First Principal Component", fontsize=14)
ax.set_ylabel("Second Principal Component", fontsize=14)
ax.set_zlabel("Third Principal Component", fontsize=14)

ax.legend()
plt.show()