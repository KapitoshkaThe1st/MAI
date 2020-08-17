#!/bin/env python

from keras.models import Sequential
from keras import layers
from keras.models import Model
from keras.utils import np_utils
import matplotlib.pyplot as plt

import make_dataset


input_size = (32, 32, 1)
epochs = 10
classes = 4

training_generator = make_dataset.DataGenerator(
    validation=False, dim=input_size)
validation_generator = make_dataset.DataGenerator(
    validation=True, dim=input_size)


model = Sequential()

model.add(layers.Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=input_size))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(classes, activation='softmax'))


# Compilation
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer='adam')
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