import matplotlib.pylab as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import Feature_Engineering


from sklearn.linear_model import Lasso, Ridge

lasso = Lasso()
ridge = Ridge()
lasso.fit(Feature_Engineering.x_train, Feature_Engineering.y_train)
ridge.fit(Feature_Engineering.x_train, Feature_Engineering.y_train)

print("Lasso Coefficient", lasso.coef_)
print("Ridge Coefficient", ridge.coef_)


alphas = [-2.2, -2, -1.5, 1.3, 1.2, 1.1, 1, 0.3, 0.1]
losses = []
for alpha in alphas:
    # Write (5 lines): create a Lasso regressor with the alpha value.
    # Fit it to the training set, then get the prediction of the validation set (x_val).
    # calculate the mean sqaured error loss, then append it to the losses array
    lasso = Lasso(alpha=alpha)
    lasso.fit(Feature_Engineering.x_train, Feature_Engineering.y_train)
    y_pred = lasso.predict(Feature_Engineering.x_test)
    mse = mean_squared_error(Feature_Engineering.y_test, y_pred)
    losses.append(mse)
plt.plot(alphas, losses)
plt.title("Lasso alpha value selection")
plt.xlabel("alpha")
plt.ylabel("Mean squared error")
plt.show()

best_alpha = alphas[np.argmin(losses)]
print("Best value of alpha:", best_alpha)


lasso = Lasso(best_alpha)
lasso.fit(Feature_Engineering.x_train, Feature_Engineering.y_train)
y_pred = lasso.predict(Feature_Engineering.x_test)
print("MSE on testset:", mean_squared_error(Feature_Engineering.y_test, y_pred))