from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures #to convert the original features into their higher order terms
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import sklearn
import Feature_Engineering

x = Feature_Engineering.x_train
y = Feature_Engineering.y_train
print("x colomx ",x.columns)
print("y colomx ",y.shape)
poly = PolynomialFeatures(degree=1)
x_poly = poly.fit_transform(x)
poly.fit(x_poly,y)
print("x_poly colomx ",len(x_poly))
linear = LinearRegression()
linear.fit(x_poly,y)
print("x_test columns",Feature_Engineering.x_test.shape)
y_pred=linear.predict(Feature_Engineering.x_test)

plt.scatter(x[0],y)
plt.plot(x[0],y_pred,color='red')


print('Mean Absolute Error:', metrics.mean_absolute_error(Feature_Engineering.y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Feature_Engineering.y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Feature_Engineering.y_test, y_pred)))
sklearn.metrics.r2_score(Feature_Engineering.y_test, y_pred, sample_weight=None, multioutput='uniform_average')