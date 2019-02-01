from keras.layers import Bidirectional, CuDNNLSTM, Reshape, Conv2D, MaxPool2D, \
    Concatenate, Flatten, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate,Conv1D, MaxPooling1D
from keras.layers import Dense, Input, Embedding as EmbeddingLayer, Dropout, Conv2D
from keras.models import Model

from Attention import Attention
from InsincereModel import InsincereModel, InsincereModelV2

import logging
import matplotlib
from pprint import pprint

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
import spacy

from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold

from Data import Data, DataV2, CorpusInfo
from data_mappers import TextMapper
from Embedding import Embedding
from Ensemble import Ensemble
from Models import *  # Make all models available for easy script generation.
from config import random_state as SEED, config_main as config

import tensorflow as tf
import keras.backend as K

class LSTMModel(InsincereModel):
    def define_model(self, model_config=None):
        if model_config is None:
            model_config = self.default_config()
        inp = Input(shape=(self.data.maxlen,))
        x = EmbeddingLayer(self.embedding.nb_words,
                           self.embedding.embed_size,
                           weights=[self.embedding.embedding_matrix],
                           trainable=False)(inp)
        x = Bidirectional(CuDNNLSTM(model_config['lstm_size'], return_sequences=True))(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        concat_layers = [avg_pool, max_pool]
        inputs = [inp]
        if self.data.custom_features:
            inp_features = Input(shape=(len(self.data.custom_features),))
            concat_layers += [inp_features]
            inputs += [inp_features]
        x = concatenate([avg_pool, max_pool, inp_features])
        x = Dense(model_config['dense_size'], activation="relu")(x)
        x = Dropout(model_config['dropout_rate'])(x)
        x = Dense(1, activation="sigmoid")(x)
        self.model = Model(inputs=inputs, outputs=x)
        self.model.compile(loss=self.loss, optimizer='sgd', metrics=['accuracy', self.f1_score])
        return self.model

    def default_config(self):
        config = {'lstm_size': 64,
                  'dense_size': 64,
                  'dropout_rate': 0.1,
                  }
        return config


class LSTMModelAttention(InsincereModel):
    def define_model(self, model_config=None):
        if model_config is None:
            model_config = self.default_config()
        inp = Input(shape=(self.data.maxlen,))
        x = EmbeddingLayer(self.embedding.nb_words,
                           self.embedding.embed_size,
                           weights=[self.embedding.embedding_matrix],
                           trainable=False)(inp)
        x = Bidirectional(CuDNNLSTM(model_config['lstm_size'], return_sequences=True))(x)
        x = Attention(self.data.maxlen)(x)
        inputs = [inp]
        if self.data.custom_features:
            inp_features = Input(shape=(len(self.data.custom_features),))
            x = concatenate([x, inp_features])
            x = Dense(model_config['dense_size_1'], activation="relu")(x)
            inputs += [inp_features]
        x = Dense(model_config['dense_size_2'], activation="relu")(x)
        x = Dropout(model_config['dropout_rate'])(x)
        x = Dense(1, activation="sigmoid")(x)
        self.model = Model(inputs=inputs, outputs=x)
        self.model.compile(loss=self.loss, optimizer='sgd', metrics=['accuracy', self.f1_score])
        return self.model

    def default_config(self):
        config = {'lstm_size': 64,
                  'dense_size_1': 32,
                  'dense_size_2': 16,
                  'dropout_rate': 0.1,
                  }
        return config


class CNNModel(InsincereModel):
    def define_model(self, model_config=None):
        if model_config is None:
            model_config = self.default_config()
        filter_sizes = model_config['filter_sizes']
        num_filters = model_config['num_filters']
        inp = Input(shape=(self.data.maxlen,))
        x = EmbeddingLayer(self.embedding.nb_words, self.embedding.embed_size,
                           weights=[self.embedding.embedding_matrix])(inp)
        x = Reshape((self.data.maxlen, self.embedding.embed_size, 1))(x)
        maxpool_pool = []
        inputs = [inp]
        for i in range(len(filter_sizes)):
            conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], self.embedding.embed_size),
                          kernel_initializer='he_normal', activation='elu')(x)
            maxpool_pool.append(MaxPool2D(pool_size=(self.data.maxlen - filter_sizes[i] + 1, 1))(conv))
        z = Concatenate(axis=1)(maxpool_pool)
        z = Flatten()(z)
        z = Dropout(model_config['dropout_rate'])(z)
        if self.data.custom_features:
            inp_features = Input(shape=(len(self.data.custom_features),))
            z = concatenate([z, inp_features])
            z = Dense(model_config['dense_size'], activation='relu')(z)
            inputs += [inp_features]
        outp = Dense(1, activation="sigmoid")(z)
        self.model = Model(inputs=inputs, outputs=outp)
        self.model.compile(loss=self.loss, optimizer='sgd', metrics=['accuracy', self.f1_score])

        return self.model

    def default_config(self):
        config = {'filter_sizes': [1, 2, 3, 5],
                  'num_filters': 36,
                  'dropout_rate': 0.1,
                  'dense_size': 32}
        return config


from keras.layers import TimeDistributed
from keras.layers import Dropout, GlobalMaxPooling1D, Concatenate

class BiLSTMCharCNNModel(InsincereModelV2):

    def __init__(self, data, corpus_info, text_mapper):
        super().__init__(data, corpus_info, text_mapper)
        self.model = self.define_model()

    def define_model(self, model_config=None):
        # if model_config is None:
        #     model_config = self.default_config()


        max_sent_len = self.text_mapper.max_sent_len
        max_word_len = self.text_mapper.max_word_len
        word_vocab_size = self.text_mapper.word_mapper.get_vocab_len()
        char_vocab_size = self.text_mapper.char_mapper.get_vocab_len()

        chars_input = Input(shape=(max_sent_len, max_word_len), name='chars_input', dtype='int64')

        char_features = char_level_feature_model(chars_input, max_word_len, char_vocab_size)

        words_input = Input(shape=(max_sent_len, ), name='words_input', dtype='int64')
        words_embedding = EmbeddingLayer(input_dim=word_vocab_size, output_dim=300, input_length=max_sent_len)(words_input)

        word_rep = Concatenate()([char_features, words_embedding])

        x = Conv1D(filters=64, kernel_size=3, activation='relu')(word_rep)
        x = GlobalMaxPooling1D()(x)
        x = Dense(16)(x)
        # x = Flatten()(char_sum)
        preds = Dense(1, activation='sigmoid')(x)

        inputs = [chars_input, words_input]

        self.model = Model(inputs=inputs, outputs=preds)
        self.model.compile(loss=self.loss, optimizer='sgd', metrics=['accuracy', self.f1_score])

        return self.model

    # def cnn_char_arch(self, word_char_input):
    #     char_vocab_size = self.text_mapper.char_mapper.get_vocab_len()
    #
    #     char_embedding_input = Input(shape=char_vocab_size)
    #     char_embedding = EmbeddingLayer(inpu_dim)

from keras.layers import GlobalMaxPooling2D

class CharCNNWordModel(InsincereModelV2):
    """ this is an experiment to check that character convolutions are outputting as expected """
    def __init__(self, data, corpus_info, text_mapper):
        super().__init__(data, corpus_info, text_mapper)
        self.model = self.define_model()

    def define_model(self, model_config=None):
        # if model_config is None:
        #     model_config = self.default_config()

        max_sent_len = self.text_mapper.max_sent_len
        max_word_len = self.text_mapper.max_word_len
        char_vocab_size = self.text_mapper.char_mapper.get_vocab_len()

        # load in character input
        chars_input = Input(shape=(max_sent_len, max_word_len), name='chars_input', dtype='int64')

        # time distributed applies the same layer to each time step (for each word)
        chars_words_embedding = TimeDistributed(EmbeddingLayer(char_vocab_size, output_dim=16, input_length=max_word_len))(chars_input)

        # todo: add another input here with additional character information (caps, number, punc, etc)

        # do one dimensional convolutions over each word. filter size will determine size of vector for each word (if globalpool)
        char_conv = TimeDistributed(Conv1D(filters=500, kernel_size=3))(chars_words_embedding)

        # represent each filter with it's max value -  each filter looks for one feature
        x = TimeDistributed(GlobalMaxPooling1D())(char_conv)
        preds = Dense(1, activation='sigmoid')(x)

        inputs = [chars_input]

        self.model = Model(inputs=inputs, outputs=preds)
        self.model.compile(loss=self.loss, optimizer='adam', metrics=['accuracy', self.f1_score])

        return self.model

    # def cnn_char_arch(self, word_char_input):
    #     char_vocab_size = self.text_mapper.char_mapper.get_vocab_len()
    #
    #     char_embedding_input = Input(shape=char_vocab_size)
    #     char_embedding = EmbeddingLayer(inpu_dim)


def char_level_feature_model(input_layer, max_word_len, char_vocab_size):
    chars_words_embedding = TimeDistributed(EmbeddingLayer(char_vocab_size, output_dim=16, input_length=max_word_len))(input_layer)
    char_conv = TimeDistributed(Conv1D(filters=500, kernel_size=3))(chars_words_embedding)
    x = TimeDistributed(GlobalMaxPooling1D())(char_conv)
    return x

# dev_size = config.get('dev_size')
# data = DataV2()
#
# nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])
# corpus_info = CorpusInfo(data.get_questions(subset='train'), nlp)
# word_counts = corpus_info.word_counts
# char_counts = corpus_info.char_counts
#
# text_mapper = TextMapper(word_counts=word_counts, char_counts=char_counts, word_threshold=10, max_word_len=20,
#                          char_threshold=350, max_sent_len=100, nlp=nlp, word_lowercase=True, char_lowercase=True)
#
# # embeddings = load_embeddings(word_counts, embedding_files)
# # save_unknown_words(data, embeddings, max_words=200)
# # models_all = list()
# # for model in config.get('models'):
# #     model_class = globals()[model.get('class')]
# #     models_all.extend(cross_validate(model_class,
# #                                      data,
# #                                      embeddings,
# #                                      model_config=model.get('args')))
#
# # model = CharCNNWordModel(data, corpus_info, text_mapper)
# model = BiLSTMCharCNNModel(data, corpus_info, text_mapper)
#
# model.model.summary()
#



































