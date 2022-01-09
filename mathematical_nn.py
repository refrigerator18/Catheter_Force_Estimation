import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
import datetime


epochs = 300

batch_size = 12
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0010)


# LOADING DATA AND LABELS
# data = pd.read_csv("test.csv", names=["a", "b", "D", "F"])
data = pd.read_csv("data_new2.csv", names=["a", "b", "c", "d", "e", "D", "F"])

features = data.copy()
labels = features.pop("F")

features = np.array(features)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=2)

normalizer = preprocessing.Normalization(axis=-1)
normalizer.adapt(np.array(X_train))

model = Sequential([
    normalizer,
    layers.Dense(6, activation='relu', input_shape=(6,)),

    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),

    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(1),
])

# MODEL
# model = Sequential([
# 	normalizer,
# 	layers.Dense(6, activation='relu', input_shape=(6,)),
#     layers.Dense(32, activation='relu'),
#     layers.Dropout(0.2),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(1),
# ])


# CALL BACKS
# save_callback = callbacks.ModelCheckpoint(
#     'checkpoints', save_weights_only=True, monitor='val_r_square', save_best_only=True
# )

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# COMPILING AND TRAINING
model.compile(optimizer=optimizer, loss=tf.losses.MeanSquaredError(), metrics=[tf.keras.metrics.RootMeanSquaredError(), tfa.metrics.RSquare(dtype=tf.float32, y_shape=(1,)), tf.keras.metrics.MeanAbsoluteError()])
model.summary()


history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), callbacks=tensorboard_callback)


# METRICS
# rmse_avg = 0
# r_avg = 0
# mae_avg = 0

# for _ in range(3):
#     history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
#     rmse_avg += min(history.history['val_root_mean_squared_error'])
#     r_avg += max(history.history['val_r_square'])
#     mae_avg += min(history.history['val_mean_absolute_error'])


# rmse_avg = rmse_avg / 3
# r_avg = r_avg / 3
# mae_avg = mae_avg / 3

# print("")
# print("")
# print("RMSE AVG: " + str(rmse_avg))
# print("R SQUARE AVG: " + str(r_avg))
# print("MAE AVG: " + str(mae_avg))

print("rsquare " + str(max(history.history['val_r_square'])))
print("MAE " + str(min(history.history['val_mean_absolute_error'])))


# GRAPHING
acc = history.history['mean_absolute_error']
val_acc = history.history['val_mean_absolute_error']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training MAE')
plt.plot(epochs_range, val_acc, label='Validation MAE')
plt.legend(loc='upper right')
plt.title('Training and Validation Accuracy')

plt.xlabel("Epochs")
plt.ylabel("Mean Average Error")

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss MSE')
# plt.plot(epochs_range, val_loss, label='Validation Loss MSE')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
plt.show()


# # GRAPHING R SQUARE METRIC
# acc = history.history['r_square']
# val_acc = history.history['val_r_square']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training RSquare')
# plt.plot(epochs_range, val_acc, label='Validation RSquare')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy (RSquare)')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss MSE')
# plt.plot(epochs_range, val_loss, label='Validation Loss MSE')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss (MSE)')
# plt.show()
