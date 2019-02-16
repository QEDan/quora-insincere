import logging
import numpy as np
from keras import Input, Model
from keras.layers import Embedding as EmbeddingLayer, Bidirectional, CuDNNLSTM, Dense
from sklearn.model_selection import train_test_split


class UnknownWords:
    def __init__(self, text_mapper, embedding, max_words=5000,
                 loss='mean_squared_error'):
        self.text_mapper = text_mapper
        self.embedding = embedding
        self.max_words = max_words
        self.loss = loss
        self.model = None

    def texts_to_x(self, word):
        """ Handles mapping one text doc into model inputs """
        chars_x = np.zeros(self.text_mapper.max_word_len)

        for char_ind, char in enumerate(word[:self.text_mapper.max_word_len]):
            chars_x[char_ind] = self.text_mapper.char_mapper.get_symbol_index(char)

        return chars_x

    def define_model(self):
        max_word_len = self.text_mapper.max_word_len
        char_vocab_size = self.text_mapper.char_mapper.get_vocab_len()
        chars_input = Input(shape=(max_word_len,), name='chars_input', dtype='int32')
        m = EmbeddingLayer(input_dim=char_vocab_size,
                           output_dim=64,
                           input_length=max_word_len)(chars_input)
        m = Bidirectional(CuDNNLSTM(64))(m)
        m = Dense(self.embedding.embedding_matrix.shape[1])(m)
        self.model = Model(inputs=chars_input, outputs=m)
        self.model.compile(loss=self.loss, optimizer='adam', metrics=['accuracy'])
        return self.model

    def sample_training_data(self, sentences):
        n_tokens = 0
        words = set()
        while n_tokens < self.max_words:
            sample_sentence = np.random.choice(sentences)
            words = words.union(sample_sentence.split())
            n_tokens = len(words)
        list_words = list(words)
        list_tokens = [self.texts_to_x(word) for word in list_words]
        list_embeddings = [self.embedding.embeddings_index[t]
                           if t in self.embedding.embeddings_index.keys()
                           else np.zeros(self.embedding.embed_size)
                           for t in list_words
                           ]
        array_tokens = np.array(list_tokens)
        array_embeddings = np.array(list_embeddings)
        train_X, val_X, train_y, val_y = train_test_split(array_tokens, array_embeddings, test_size=0.2)
        return train_X, val_X, train_y, val_y

    def fit(self, sentences):
        logging.info('Fitting UnknownWords model...')
        train_X, val_X, train_y, val_y = self.sample_training_data(sentences)
        self.model.fit(x=train_X, y=train_y, epochs=10, batch_size=256)

    def predict(self, words):
        input_x = np.array([self.texts_to_x(word) for word in words])
        predictions = self.model.predict(input_x)
        return predictions

    def improve_embedding(self):
        logging.info('Improving unknown embeddings with predicted values...')
        for i, word in enumerate(self.embedding.word_vocab):
            if word in self.embedding.unknown_words:
                pred_embedding = self.predict(np.array([word]))
                self.embedding.embedding_matrix[i] = pred_embedding[0]

