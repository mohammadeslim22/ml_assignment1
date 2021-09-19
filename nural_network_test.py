from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import Feature_Engineering
import numpy as np
from sklearn.model_selection import train_test_split


# Feature_Engineering.prepareData()

X = np.asarray(Feature_Engineering.x_train).astype('float32').reshape(208057850,387)
y = np.asarray(Feature_Engineering.y_train).astype('float32').reshape(1,387)

# X_train, X_test, y_train, y_test = train_test_split(
#   X,
#   y,
#   test_size=0.20
# )
def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(4,)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    return model

model = create_model()
model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X,
    y,
    epochs=200,
    validation_split=0.25,
    batch_size=40,
    verbose=2
)