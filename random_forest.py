from sklearn.ensemble import RandomForestRegressor
import Feature_Engineering
import pandas as pd
import sklearn
from sklearn import metrics
import numpy as np

regressor = RandomForestRegressor(n_estimators=100, random_state=0)

# fit the regressor with x and y data
regressor.fit(Feature_Engineering.x_train, Feature_Engineering.y_train)
y_pred = regressor.predict(Feature_Engineering.x_test)  # test the output by changing values
eval_df = pd.DataFrame({'Actual': Feature_Engineering.y_test, 'Predicted': y_pred})
print(eval_df)

print('Mean Absolute Error:', metrics.mean_absolute_error(Feature_Engineering.y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Feature_Engineering.y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Feature_Engineering.y_test, y_pred)))
sklearn.metrics.r2_score(Feature_Engineering.y_test, y_pred, sample_weight=None, multioutput='uniform_average')
