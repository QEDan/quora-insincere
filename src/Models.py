import matplotlib
from keras import Input, Model

from src.InsincereModel import InsincereModel

matplotlib.use('Agg')

from src.Models import *  # Make all models available for easy script generation.

from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import Layer
from keras.layers import TimeDistributed, Embedding as EmbeddingLayer, Bidirectional, CuDNNLSTM, Dense, Conv1D
from keras.layers import GlobalMaxPooling1D, Concatenate, BatchNormalization, Dropout, SpatialDropout1D, CuDNNGRU
from keras.layers import GlobalAveragePooling1D, Add
import keras.backend as K
import numpy as np

# https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-652-lb

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


class BiLSTMCharCNNModel(InsincereModel):
    def define_model(self, model_config=None):
        # if model_config is None:
        #     model_config = self.default_config()

        max_sent_len = self.text_mapper.max_sent_len
        max_word_len = self.text_mapper.max_word_len
        word_vocab_size = self.text_mapper.word_mapper.get_vocab_len()
        char_vocab_size = self.text_mapper.char_mapper.get_vocab_len()

        chars_input = Input(shape=(max_sent_len, max_word_len), name='chars_input', dtype='int64')
        char_feats_input = Input(shape=(max_sent_len, max_word_len, self.text_mapper.char_mapper.num_add_feats),
                                 name='chars_feats_input', dtype='float32')
        char_features = char_level_feature_model(chars_input, char_feats_input, max_word_len, char_vocab_size)

        words_input = Input(shape=(max_sent_len,), name='words_input', dtype='int64')
        matrix_shape = self.embedding.embedding_matrix.shape
        if self.embedding is not None:
            un_trainable_words_embedding = EmbeddingLayer(input_dim=word_vocab_size,
                                                          output_dim=matrix_shape[1],
                                                          input_length=max_sent_len,
                                                          weights=[self.embedding.embedding_matrix],
                                                          trainable=False)(words_input)
            if True:
                regularized_word_embeddings = EmbeddingLayer(input_dim=word_vocab_size,
                                                             output_dim=matrix_shape[1],
                                                             input_length=max_sent_len,
                                                             weights=np.zeros(matrix_shape),
                                                             trainable=False)(words_input)
                words_embedding = Add()([un_trainable_words_embedding, regularized_word_embeddings])
            else:
                words_embedding = un_trainable_words_embedding
            word_rep = Concatenate()([char_features, words_embedding, trainable_words_embedding])
        else:
            word_rep = Concatenate()([char_features, trainable_words_embedding])

        x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(word_rep)
        y = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)

        atten_1 = Attention(max_sent_len)(x)  # skip connect
        atten_2 = Attention(max_sent_len)(y)
        avg_pool = GlobalAveragePooling1D()(y)
        max_pool = GlobalMaxPooling1D()(y)

        conc = Concatenate()([atten_1, atten_2, avg_pool, max_pool])
        conc = Dense(16, activation="relu")(conc)
        conc = Dropout(0.1)(conc)

        # x = Conv1D(filters=100, kernel_size=2)(x)
        # max_x = GlobalMaxPooling1D()(x)
        # avg_x = GlobalAveragePooling1D()(x)
        # x = Concatenate()([max_x, avg_x])
        # x = Dense(16)(x)
        # x = Flatten()(char_sum)
        preds = Dense(1, activation='sigmoid')(conc)

        inputs = [chars_input, words_input, char_feats_input]

        self.model = Model(inputs=inputs, outputs=preds)
        self.model.compile(loss=self.loss, optimizer='adam', metrics=['accuracy', self.f1_score])

        return self.model


def char_level_feature_model(char_input, char_feat_input, max_word_len, char_vocab_size):
    chars_words_embedding = TimeDistributed(EmbeddingLayer(char_vocab_size,
                                                           output_dim=16,
                                                           input_length=max_word_len))(char_input)
    x = TimeDistributed(SpatialDropout1D(0.1))(chars_words_embedding)
    char_rep = Concatenate()([x, char_feat_input])
    conv_outputs = []
    # todo: tune these conv kernels
    conv_kernels = [[32, 2], [32, 3], [32, 4], [32, 5]]
    for num_filter, kernel_size in conv_kernels:
        char_conv = TimeDistributed(Conv1D(filters=num_filter, kernel_size=kernel_size, activation='relu'))(char_rep)
        batch_norm = TimeDistributed(BatchNormalization())(char_conv)
        # todo: put batch norm after max pooling?
        x = TimeDistributed(GlobalMaxPooling1D())(batch_norm)
        conv_outputs.append(x)
    x = Concatenate()(conv_outputs)
    # x = Dense(100, activation='relu')(x)
    # x = Dropout(0.3)(x)
    # x = Dense(50, activation='relu')(x)
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
# model.define_model()
# model.model.summary()
# # # # #
