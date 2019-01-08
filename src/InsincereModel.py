import keras.backend as K
import logging
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping

from src.Embedding import Embedding
from src.LRFinder import LRFinder
from src.OneCycleLR import OneCycleLR


class InsincereModel:
    def __init__(self, data, name=None, loss='binary_crossentropy'):
        self.data = data
        self.name = name
        self.embedding = None
        self.model = None
        self.history = None
        self.loss = loss
        self.lr_finder = None

    def load_embedding(self, embedding_file='../input/embeddings/glove.840B.300d/glove.840B.300d.txt'):
        self.embedding = Embedding(self.data)
        self.embedding.load(embedding_file)

    def set_embedding(self, embedding):
        if type(embedding) is str:
            self.load_embedding(embedding)
        else:
            self.embedding = embedding

    def blend_embeddings(self, embeddings, cleanup=False):
        """Average embedding matrix given list of embedding files."""
        if self.embedding is None:
            self.set_embedding(embeddings[0])
        embedding_matrices = list()
        for emb in embeddings:
            embedding_matrices.append(emb.embedding_matrix)
        blend = np.mean(embedding_matrices, axis=0)
        self.embedding.embedding_matrix = blend
        if cleanup:
            for e in embeddings:
                e.cleanup()
        return blend

    def concat_embeddings(self, embeddings, cleanup=False):
        self.embedding.embedding_matrix = np.concatenate(tuple([e.embedding_matrix for e in embeddings]), axis=1)
        if cleanup:
            for e in embeddings:
                e.cleanup()
        return self.embedding.embedding_matrix

    @staticmethod
    def f1_score(y_true, y_pred):
        def recall(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def define_model(self):
        raise NotImplementedError

    def print(self):
        print(self.model.summary())

    def _get_callbacks(self, epochs, batch_size, minimum_lr=1e-8, maximum_lr=1.0e-1):
        num_samples = self.data.train_X.shape[0]
        self.lr_finder = LRFinder(num_samples, batch_size,
                               minimum_lr, maximum_lr,
                               # validation_data=(X_val, Y_val),
                               lr_scale='exp', save_dir='.')
        lr_manager = OneCycleLR(num_samples, epochs, batch_size, maximum_lr,
                                end_percentage = 0.1,
                                maximum_momentum = 0.95, minimum_momentum = 0.85)
        check_point = ModelCheckpoint('model.hdf5', monitor="val_f1_score", mode="max",
                                      verbose=True, save_best_only=True)
        early_stop = EarlyStopping(monitor="val_f1_score", mode="max", patience=3, verbose=True)
        return [self.lr_finder, lr_manager, check_point, early_stop]

    def fit(self,
            train_indices=None,
            val_indices=None,
            pseudo_labels=False,
            batch_size=1024,
            epochs=10,
            save_curve=True,
            curve_file_suffix=None):
        logging.info("Fitting model...")
        if pseudo_labels:
            train_x, train_y = self.data.full_X, self.data.full_y
            val_x, val_y = self.data.val_X, self.data.val_y
            if self.data.custom_features:
                train_features, val_features = self.data.train_features, self.data.test_features
        else:
            if train_indices is not None:
                train_x = self.data.train_X[train_indices]
                train_y = self.data.train_y[train_indices]
                if self.data.custom_features:
                    train_features = self.data.train_features[train_indices]
            else:
                train_x = self.data.train_X
                train_y = self.data.train_y
                if self.data.custom_features:
                    train_features = self.data.train_features
            if val_indices is not None:
                val_x = self.data.train_X[val_indices]
                val_y = self.data.train_y[val_indices]
                if self.data.custom_features:
                    val_features = self.data.train_features[val_indices]
            else:
                val_x = self.data.val_X
                val_y = self.data.val_y
                if self.data.custom_features:
                    val_features = self.data.val_features
        callbacks = self._get_callbacks(epochs, batch_size)
        if self.data.custom_features:
            train_x = [train_x, train_features]
            val_x = [val_x, val_features]
        self.history = self.model.fit(train_x,
                                      train_y,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      validation_data=(val_x, val_y),
                                      callbacks=callbacks)
        if save_curve:
            self.lr_finder.plot_schedule(filename="lr_schedule_" + str(self.name) + ".png")
            filename = 'training_curve'
            if self.name:
                filename += '_' + self.name
            if curve_file_suffix:
                filename += '_' + curve_file_suffix
            filename += '.png'
            self.print_curve(filename)

    def print_curve(self, filename='training_curve.png'):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='best')
        plt.savefig(filename)
        plt.close()

    def predict(self, x, batch_size=1024):
        logging.info("Predicting ...")
        prediction = self.model.predict(x, batch_size=batch_size, verbose=1)
        return prediction

    def cleanup(self):
        self.embedding.cleanup()
