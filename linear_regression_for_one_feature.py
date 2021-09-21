from sklearn.linear_model import LinearRegression
import pandas as pd
import Feature_Engineering
from sklearn import metrics
import numpy as np
import sklearn


y_test=Feature_Engineering.test['Delay']
x_test=Feature_Engineering.test[['flight duration']]
y_train=Feature_Engineering.train['Delay']
x_train=Feature_Engineering.train[['flight duration']]

x_train=np.asarray(x_train).astype(np.float32)
y_train=np.asarray(y_train).astype(np.float32)
x_test=np.asarray(x_test).astype(np.float32)
y_test=np.asarray(y_test).astype(np.float32)

x_train=x_train.reshape(-1, 1)
x_test=x_test.reshape(-1,1)
y_train=y_train.reshape(-1, 1)
y_test=y_test.reshape(-1,1)

regressor = LinearRegression()
regressor.fit(x_train, y_train)
print(f"Model intercept : {regressor.intercept_}")
print(f"Model coefficients : {regressor.coef_}")

y_pred = regressor.predict(x_test)


print('Mean Absolute Error:', metrics.mean_absolute_error(Feature_Engineering.y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Feature_Engineering.y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Feature_Engineering.y_test, y_pred)))
sklearn.metrics.r2_score(Feature_Engineering.y_test, y_pred, sample_weight=None, multioutput='uniform_average')