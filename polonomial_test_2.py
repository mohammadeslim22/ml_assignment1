from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures  # to convert the original features into their higher order terms
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import sklearn
import Feature_Engineering

from sklearn.preprocessing import PolynomialFeatures

x_train = np.asarray(Feature_Engineering.x_train).astype(np.float32)
y_train = np.asarray(Feature_Engineering.y_train).astype(np.float32)
y_test = np.asanyarray(Feature_Engineering.y_test)
x_test = np.asarray(Feature_Engineering.x_test).astype(np.float32)
# y_train=np.asanyarray(Feature_Engineering.y_train)
poly = PolynomialFeatures(degree=1)
train_x_poly = poly.fit_transform(Feature_Engineering.x_train)
print(train_x_poly)

clf = LinearRegression()
y_train_ = clf.fit(train_x_poly, y_train)
# The coefficients
print('Coefficients: ', clf.coef_)
print('Intercept: ', clf.intercept_)

test_x_poly = poly.fit_transform(Feature_Engineering.x_test)
test_y_ = clf.predict(test_x_poly)

# print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - y_test)))
# print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - y_test) ** 2))
# print("R2-score: %.2f" % sklearn.metrics.r2_score(y_test, test_y_))
print('Mean Absolute Error:', metrics.mean_absolute_error(Feature_Engineering.y_test, test_y_))
print('Mean Squared Error:', metrics.mean_squared_error(Feature_Engineering.y_test, test_y_))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Feature_Engineering.y_test, test_y_)))
sklearn.metrics.r2_score(Feature_Engineering.y_test, test_y_, sample_weight=None, multioutput='uniform_average')
# x = Feature_Engineering.x_train
# y = Feature_Engineering.y_train
# print("x colomx ",x.columns)
# print("y colomx ",y.shape)
# poly = PolynomialFeatures(degree=1)
# x_poly = poly.fit_transform(x)
# poly.fit(x_poly,y)
# print("x_poly colomx ",len(x_poly))
# linear = LinearRegression()
# linear.fit(x_poly,y)
# print("x_test columns",Feature_Engineering.x_test.shape)
# y_pred=linear.predict(Feature_Engineering.x_test)
#
# plt.scatter(x[0],y)
# plt.plot(x[0],y_pred,color='red')


# print('Mean Absolute Error:', metrics.mean_absolute_error(Feature_Engineering.y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(Feature_Engineering.y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Feature_Engineering.y_test, y_pred)))
# sklearn.metrics.r2_score(Feature_Engineering.y_test, y_pred, sample_weight=None, multioutput='uniform_average')
