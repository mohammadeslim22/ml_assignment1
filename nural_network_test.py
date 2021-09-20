from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import Feature_Engineering
import numpy as np
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Feature_Engineering.prepareData()

print(Feature_Engineering.x_train.shape)
model2 = Sequential()
# model2.add(Dense(2, input_dim=36, activation='relu'))
# model2.add(Dense(1, input_dim=36, activation='relu'))
model2.add(Dense(128, input_dim=36, activation='relu'))
model2.add(Dense(64, activation='relu'))
model2.add(Dense(32, activation='relu'))
model2.add(Dense(1, activation='relu'))
model2.compile(loss='mean_absolute_error',
               optimizer='adam',
               metrics=['accuracy'])
model2.fit(Feature_Engineering.x_train, Feature_Engineering.y_train, epochs=5, batch_size=32)
predicted = model2.predict(Feature_Engineering.x_test, batch_size=128)

print('Mean Absolute Error:', metrics.mean_absolute_error(Feature_Engineering.y_test, predicted))
print('Mean Squared Error:', metrics.mean_squared_error(Feature_Engineering.y_test, predicted))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Feature_Engineering.y_test, predicted)))
sklearn.metrics.r2_score(Feature_Engineering.y_test, predicted, sample_weight=None, multioutput='uniform_average')
# X = np.asarray(Feature_Engineering.x_train).astype('float32').reshape(208057850,387)
# y = np.asarray(Feature_Engineering.y_train).astype('float32').reshape(1,387)
#
# # X_train, X_test, y_train, y_test = train_test_split(
# #   X,
# #   y,
# #   test_size=0.20
# # )
# def create_model():
#     model = Sequential([
#         Dense(64, activation='relu', input_shape=(4,)),
#         Dense(128, activation='relu'),
#         Dense(128, activation='relu'),
#         Dense(128, activation='relu'),
#         Dense(64, activation='relu'),
#         Dense(64, activation='relu'),
#         Dense(64, activation='relu'),
#         Dense(3, activation='softmax')
#     ])
#     return model
#
# model = create_model()
# model.summary()
#
# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )
#
# history = model.fit(
#     X,
#     y,
#     epochs=200,
#     validation_split=0.25,
#     batch_size=40,
#     verbose=2
# )
