from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import Feature_Engineering
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.preprocessing import PolynomialFeatures


def ActivatePolynomialRegression():

    y_test = Feature_Engineering.test['Delay']
    x_test = Feature_Engineering.test[['flight duration']]
    X_test = Feature_Engineering.test.drop('Delay', axis=1)
    y_train = Feature_Engineering.train['Delay']
    x_train = Feature_Engineering.train[['flight duration']]
    X_train = Feature_Engineering.train.drop('Delay', axis=1)

    scaler = RobustScaler()
    scaler.fit_transform(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print("x_train.shape", x_train.shape)
    print("x_test.shape", x_test.shape)
    print("y_train.shape", y_train.shape)
    print("y_test.shape", y_test.shape)

    x_train = x_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)
    poly = PolynomialFeatures(degree=2)
    train_x_poly = poly.fit_transform(x_train)
    clf = LinearRegression()
    train_y_ = clf.fit(train_x_poly, y_train)
    # The coefficients
    print('Coefficients: ', clf.coef_)
    print('Intercept: ', clf.intercept_)
    test_x_poly = poly.fit_transform(x_test)
    test_y_ = clf.predict(test_x_poly)

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, test_y_))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, test_y_))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, test_y_)))
    print('R2-score:', sklearn.metrics.r2_score(y_test, test_y_))

    plt.scatter(x_train['flight duration'], x_train['Delay'], color='blue', s=1)
    XX = np.arange(0.0, 25.0, 0.1)
    yy = clf.intercept_ + clf.coef_[1] * XX + clf.coef_[2] * np.power(XX, 2)
    plt.plot(XX, yy, '-r')
    plt.xlabel("Flight Duration")
    plt.ylabel("Delay")
ActivatePolynomialRegression()