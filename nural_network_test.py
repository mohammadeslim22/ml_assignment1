from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import Feature_Engineering
import numpy as np
import sklearn
from sklearn import metrics

print(Feature_Engineering.x_train.shape)
def ActivateNeuralNetwork():
    model2 = Sequential()
    model2.add(Dense(20, input_dim=37, activation='relu'))
    model2.add(Dense(10, activation='relu'))
    # model2.add(Dense(128, input_dim=37, activation='relu'))
    # model2.add(Dense(64, activation='relu'))
    # model2.add(Dense(32, activation='relu'))
    model2.add(Dense(1, activation='relu'))
    model2.compile(loss='mean_absolute_error',
                   optimizer='adam',
                   metrics=['accuracy'])
    model2.fit(Feature_Engineering.x_train, Feature_Engineering.y_train, epochs=8, batch_size=32)
    predicted = model2.predict(Feature_Engineering.x_test, batch_size=128)

    print('Mean Absolute Error:', metrics.mean_absolute_error(Feature_Engineering.y_test, predicted))
    print('Mean Squared Error:', metrics.mean_squared_error(Feature_Engineering.y_test, predicted))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Feature_Engineering.y_test, predicted)))
    sklearn.metrics.r2_score(Feature_Engineering.y_test, predicted, sample_weight=None, multioutput='uniform_average')

ActivateNeuralNetwork()