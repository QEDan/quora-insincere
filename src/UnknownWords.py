import logging
import numpy as np
from keras import Input, Model
from keras.layers import Embedding as EmbeddingLayer, Bidirectional, CuDNNLSTM, Dense, LSTM
from sklearn.model_selection import train_test_split


class UnknownWords:
    def __init__(self, char_mapper, embedding, word_vocab, max_word_len, text_mapper,
                 loss='mean_squared_error'):
        self.char_mapper = char_mapper
        self.max_word_len = max_word_len
        self.embedding = embedding
        self.word_vocab = word_vocab
        self.loss = loss
        self.text_mapper = text_mapper
        self.model = None

    def word_to_x(self, word):
        """ Handles mapping one word into model inputs """
        chars_x = np.zeros(self.max_word_len)

        for char_ind, char in enumerate(word[:self.max_word_len]):
            chars_x[char_ind] = self.char_mapper.get_symbol_index(char)

        return chars_x

    def define_model(self):
        max_word_len = self.max_word_len
        char_vocab_size = self.char_mapper.get_vocab_len()
        chars_input = Input(shape=(max_word_len,), name='chars_input', dtype='int32')
        m = EmbeddingLayer(input_dim=char_vocab_size,
                           output_dim=32,
                           input_length=max_word_len,
                           mask_zero=True)(chars_input)
        m = Bidirectional(LSTM(64))(m)
        m = Dense(self.embedding.embedding_matrix.shape[1])(m)
        self.model = Model(inputs=chars_input, outputs=m)
        self.model.compile(loss=self.loss, optimizer='adam', metrics=['accuracy'])
        return self.model

    def training_data(self):
        known_words = self.embedding.known_words
        known_words_x = np.array([self.word_to_x(word) for word in known_words])
        known_words_y = np.array([self.embedding.embeddings_index.get(word) for word in known_words])
        train_X, val_X, train_y, val_y = train_test_split(known_words_x, known_words_y, test_size=0.02)
        return train_X, val_X, train_y, val_y

    def fit(self):
        logging.info('Fitting UnknownWords model...')
        train_X, val_X, train_y, val_y = self.training_data()
        self.model.fit(x=train_X, y=train_y, epochs=4, batch_size=16, verbose=2, validation_data=(val_X, val_y))

    def predict(self, words):
        input_x = np.array([self.word_to_x(word) for word in words])
        predictions = self.model.predict(input_x)
        return predictions

    def improve_embedding(self):
        logging.info('Improving unknown embeddings with predicted values...')
        unknown_words = self.embedding.unknown_words
        for word in unknown_words:
            matrix_ind = self.text_mapper.get_word_ind(word)
            pred_embedding = self.predict(np.array([word]))
            self.embedding.embedding_matrix[matrix_ind] = pred_embedding[0]
