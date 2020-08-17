#!/bin/env python

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
import matplotlib.pyplot as plt

import make_dataset


input_size = 32 * 32
hidden1 = 32
hidden2 = 10
epochs = 10
classes = 4

training_generator = make_dataset.DataGenerator(validation=False)
validation_generator = make_dataset.DataGenerator(validation=True)


model = Sequential()
model.add(Dense(hidden1, input_dim=input_size, activation='relu'))
model.add(Dense(hidden2, activation='relu'))
model.add(Dense(classes, activation='softmax'))

# Compilation
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer='sgd')
model.summary()

history = model.fit_generator(generator=training_generator,
                              validation_data=validation_generator,
                              use_multiprocessing=True,
                              workers=1, epochs=epochs)

epochs_list = list(range(epochs))

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(epochs_list, history.history['val_loss'], 'bo-')
ax1.set_title('val_loss')

ax2.plot(epochs_list, history.history['val_accuracy'], 'ro-')
ax2.set_title('val_accuracy')

plt.tight_layout()

plt.show()