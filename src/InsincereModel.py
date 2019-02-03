import keras.backend as K
import logging
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint

from src.Embedding import Embedding
from src.LRFinder import LRFinder
from src.OneCycleLR import OneCycleLR
from src.config import config_insincere_model
from src.data_generator import DataGenerator


class InsincereModel:
    def __init__(self, data, corpus_info, text_mapper, batch_size=16, name=None, loss='binary_crossentropy'):
        self.data = data
        self.corpus_info = corpus_info
        self.text_mapper = text_mapper
        self.name = name
        self.embedding = None
        self.model = None
        self.history = None
        self.loss = loss
        self.lr_finder = None
        self.config = config_insincere_model
        self.batch_size = batch_size

    def load_embedding(self, embedding_file='../input/embeddings/glove.840B.300d/glove.840B.300d.txt'):
        self.embedding = Embedding(self.corpus_info.word_counts)
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
        if self.embedding is None:
            self.set_embedding(embeddings[0])
        self.embedding.embedding_matrix = np.concatenate(tuple([e.embedding_matrix for e in embeddings]), axis=1)
        self.embedding.embed_size = self.embedding.embedding_matrix.shape[1]
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

    def _get_callbacks(self, epochs, batch_size, checkpoint=False, one_cycle=False):
        config = self.config.get('callbacks')
        early_stop = EarlyStopping(monitor=config.get('early_stopping').get('monitor'),
                                   mode=config.get('early_stopping').get('mode'),
                                   patience=1,
                                   verbose=config.get('early_stopping').get('verbose'),
                                   restore_best_weights=True)
        cbs = [early_stop]
        if one_cycle:
            num_samples = len(self.data.train_qs)
            self.lr_finder = LRFinder(num_samples, batch_size)
            lr_manager = OneCycleLR(num_samples, epochs, batch_size)
            cbs += [self.lr_finder, lr_manager]
        if checkpoint:
            check_point = ModelCheckpoint('model.hdf5',
                                          monitor=config.get('checkpoint').get('monitor'),
                                          mode=config.get('checkpoint').get('mode'),
                                          verbose=config.get('checkpoint').get('verbose'),
                                          save_best_only=config.get('checkpoint').get('save_best_only'))
            cbs += [check_point]

        return cbs

    def fit(self, curve_file_suffix=None):
        logging.info("Fitting model...")
        self.model.summary()
        config = self.config.get('fit')

        train_generator = DataGenerator(text=self.data.train_qs, labels=self.data.train_labels,
                                        text_mapper=self.text_mapper, batch_size=self.batch_size)
        val_generator = DataGenerator(text=self.data.val_qs, labels=self.data.val_labels,
                                      text_mapper=self.text_mapper, batch_size=self.batch_size)

        callbacks = self._get_callbacks(config.get('epochs'), config.get('batch_size'))

        self.model.fit_generator(generator=train_generator, epochs=10, verbose=1, callbacks=callbacks,
                                 validation_data=val_generator, max_queue_size=10,  # why not make this >>>
                                 workers=1,
                                 use_multiprocessing=False,
                                 shuffle=True)


        # self.history = self.model.fit(x=train_x,
        #                               y=train_y,
        #                               epochs=2,
        #                               batch_size=32,
        #                               validation_data=(val_x, val_y),
        #                               callbacks=callbacks)

        if config.get('save_curve'):
            if self.lr_finder:
                self.lr_finder.plot_schedule(filename="lr_schedule_" + str(self.name) + ".png")
            filename = 'training_curve'
            if self.name:
                filename += '_' + self.name
            if curve_file_suffix:
                filename += '_' + curve_file_suffix
            filename += '.png'
            # self.print_curve(filename)

    def predict_subset(self, subset='train'):
        if subset == 'train':
            questions = self.data.train_qs
        elif subset == 'val':
            questions = self.data.val_qs
        elif subset == 'test':
            questions = self.data.get_questions(subset)

        # input_x = self.prepare_model_inputs(questions)
        # preds = self.predict(input_x)
        data_gen = DataGenerator(text=questions, text_mapper=self.text_mapper, shuffle=False)
        preds = self.model.predict_generator(data_gen, workers=2, use_multiprocessing=True, max_queue_size=100)
        return preds

    def print_curve(self, filename='training_curve.png'):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='best')
        plt.savefig(filename)
        plt.close()

    def predict(self, x):
        logging.info("Predicting ...")
        batch_size = self.config.get('predict').get('batch_size')
        verbose = self.config.get('predict').get('verbose')
        prediction = self.model.predict(x, batch_size=batch_size, verbose=verbose)
        return prediction

    def cleanup(self):
        self.embedding.cleanup()

    def prepare_model_inputs(self, questions):
        model_input = self.text_mapper.texts_to_x(questions)
        words_input = model_input['words_input']
        chars_input = model_input['chars_input']
        char_feats_input = model_input['chars_feats_input']
        return {'words_input': words_input, 'chars_input': chars_input, 'chars_feats_input': char_feats_input}
