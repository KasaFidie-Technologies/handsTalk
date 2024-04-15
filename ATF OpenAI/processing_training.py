# PROCESS THE DATA AND TRAINING

# importing some dependencies

import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from keras.applications import MobileNetV2
from keras.optimizers import Adam
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import mediapipe as mp


# Fetch for the data in PATH

DATA_PATH = os.path.join('MP_Data')

# Actions
actions = np.array(['is', 'What'])


# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

label_map = {label:num for num, label in enumerate(actions)}
# print(label_map)

sequences, labels = [], []
for action in actions:
    for sequence in range(sequence_length):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))


X = np.array(sequences)

# converting labels to one hot endcoded representation
y = to_categorical(labels).astype(int)

# Training and test splite
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

res = (.7, 0.2, 0.1)

actions[np.argmax(res)]
actions[np.argmax(y_test[0])]


# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])
# # print(model.summary())

# Make prediction
res = model.predict(X_test)

actions[np.argmax(res[0])]
print(actions[np.argmax(res[0])])

actions[np.argmax(res[0])]
print(actions[np.argmax(res[0])])

# # # # Save model
# model.save('my_model.keras')


# # Evaluate using confusion matrix and accuracy

# yhat = model.predict(X_test)

# ytrue = np.argmax(y_test, axis=1).tolist()
# yhat = np.argmax(yhat, axis=1).tolist()

# # print(ytrue)
# # print(yhat)

# print(multilabel_confusion_matrix(ytrue, yhat))

# print(accuracy_score(ytrue, yhat))

# # print(np.expand_dims(X_test[0], axis=0).shape)
