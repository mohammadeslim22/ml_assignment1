from sklearn.ensemble import RandomForestRegressor
import Feature_Engineering
import pandas as pd
import sklearn
from sklearn import metrics
import numpy as np
from sklearn import datasets, ensemble
from sklearn.metrics import r2_score


params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}

gb_reg = ensemble.GradientBoostingRegressor(**params)
gb_reg.fit(Feature_Engineering.x_train, Feature_Engineering.y_train)

# fit the regressor with x and y data
y_pred = gb_reg.predict(Feature_Engineering.x_test)
eval_df = pd.DataFrame({'Actual': Feature_Engineering.y_test, 'Predicted': y_pred})
print(eval_df)


# define MAPE function
def mape(true, predicted):
    inside_sum = np.abs(predicted - true) / true
    return round(100 * np.sum(inside_sum) / inside_sum.size, 2)


#print(f"GB model MSE is {round(mean_squared_error(y_true, y_pred),2)}")
print(f"GB model MAPE is {mape(Feature_Engineering.y_test, y_pred)} %")
print(f"GB model R2 is {round(r2_score(Feature_Engineering.y_test, y_pred)* 100 , 2)} %")


print('Mean Absolute Error:', metrics.mean_absolute_error(Feature_Engineering.y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Feature_Engineering.y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Feature_Engineering.y_test, y_pred)))
sklearn.metrics.r2_score(Feature_Engineering.y_test, y_pred, sample_weight=None, multioutput='uniform_average')

