import keras

import numpy as np


class DataGenerator(keras.utils.Sequence):
    def __init__(self, text, labels, text_mapper, batch_size=16, shuffle=True):
        self.batch_size = batch_size
        self.text = text
        self.labels = labels
        self.shuffle = shuffle
        self.text_mapper = text_mapper

        self.indexes = np.arange(len(self.labels))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data

        :return X: dictionary with keys: values
            words_input: np.array with shape (batch_size, max_sent_len)
            chars_input: np.array with shape (batch_size, max_sent_len, max_word_len)
                y: np.array with shape (batch_size, )
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        # todo: if loss plateaus, increase batch size
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """ Generates data containing batch_size samples """
        # text samples
        text_samples = [self.text[i] for i in indexes]
        labels = [self.labels[i] for i in indexes]

        X = self.text_mapper.texts_to_x(text_samples)
        y = np.array(labels)

        return X, y
