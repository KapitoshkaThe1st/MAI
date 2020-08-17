import numpy as np
import cv2 as cv
import keras
import os

path = '../img'


class DataGenerator(keras.utils.Sequence):
    'Keras data generator'

    def __init__(self, validation=False, batch_size=4, dim=(32 * 32,),
                 n_classes=4, shuffle=False):
        'Initialization'
        self.dim = dim
        self.cnt = 0
        self.batch_size = batch_size

        self.labels = {}
        self.list_IDs = []

        self.n_classes = n_classes
        self.shuffle = shuffle

        for directory in [x for x in os.listdir(path) if os.path.isdir('/'.join([path, x]))]:
            cropdir_name = '/'.join([path, directory, 'crop'])
            for subdir in os.listdir(cropdir_name):
                if (validation == True) and (subdir != '1'):
                    continue
                if (validation == False) and (subdir == '1'):
                    continue
                subdir_name = '/'.join([cropdir_name, subdir])
                for image_file in os.listdir(subdir_name):
                    self.list_IDs.append('/'.join([subdir_name, image_file]))
                    self.labels[self.list_IDs[-1]] = int(directory) - 1

        self.list_IDs = np.array(self.list_IDs)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = cv.imread(ID, 0)

            X[i, ] = img.reshape(*self.dim) / 255
            # Store class
            y[i] = self.labels[ID]
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
