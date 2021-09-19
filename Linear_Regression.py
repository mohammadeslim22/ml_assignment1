from sklearn.linear_model import LinearRegression
import Feature_Engineering
import pandas as pd
import sklearn
from sklearn import metrics
import numpy as np
def AcitvateLinearRegression():
    regressor = LinearRegression()
    regressor.fit(Feature_Engineering.x_train, Feature_Engineering.y_train)
    print(f"Model intercept : {regressor.intercept_}")
    print(f"Model coefficients : {regressor.coef_}")

    y_pred = regressor.predict(Feature_Engineering.x_test)
    eval_df = pd.DataFrame({'Actual': Feature_Engineering.y_test, 'Predicted': y_pred})
    print(eval_df)



    print('Mean Absolute Error:', metrics.mean_absolute_error(Feature_Engineering.y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(Feature_Engineering.y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Feature_Engineering.y_test, y_pred)))
    sklearn.metrics.r2_score(Feature_Engineering.y_test, y_pred, sample_weight=None, multioutput='uniform_average')