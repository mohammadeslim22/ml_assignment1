from sklearn.linear_model import LinearRegression
import Feature_Engineering
import pandas as pd
import sklearn
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


def AcitvateLinearRegression():
    regressor = LinearRegression()
    regressor.fit(Feature_Engineering.x_train, Feature_Engineering.y_train)
    print(f"Model intercept : {regressor.intercept_}")
    print(f"Model coefficients : {regressor.coef_}")

    y_pred = regressor.predict(Feature_Engineering.x_test)
    eval_df = pd.DataFrame({'Actual': Feature_Engineering.y_test, 'Predicted': y_pred})
    print(eval_df)

    print(type(Feature_Engineering.x_train['depature_midnight']))
    print(type(Feature_Engineering.y_train))
    print(Feature_Engineering.x_train.head(3))
    print("_______________________________")
    print(Feature_Engineering.y_train.head(3))
    plt.scatter(Feature_Engineering.x_train[0], Feature_Engineering.y_train)
    # m, b = np.polyfit(Feature_Engineering.x_test[0], y_pred, 1)
    # plt.plot(Feature_Engineering.x_test[0], m * Feature_Engineering.x_test[0] + b)
    plt.plot(Feature_Engineering.x_test[0], y_pred, color='red')
    plt.show()

    print('Mean Absolute Error:', metrics.mean_absolute_error(Feature_Engineering.y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(Feature_Engineering.y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Feature_Engineering.y_test, y_pred)))
    sklearn.metrics.r2_score(Feature_Engineering.y_test, y_pred, sample_weight=None, multioutput='uniform_average')

AcitvateLinearRegression()