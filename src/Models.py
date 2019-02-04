import matplotlib
from keras import Input, Model

from src.InsincereModel import InsincereModel

matplotlib.use('Agg')

from src.Models import *  # Make all models available for easy script generation.

from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import Layer
from keras.layers import TimeDistributed, Embedding as EmbeddingLayer, Bidirectional, CuDNNLSTM, Dense, Conv1D
from keras.layers import GlobalMaxPooling1D, Concatenate, BatchNormalization, Dropout, SpatialDropout1D, CuDNNGRU
from keras.layers import GlobalAveragePooling1D, Add, Activation, Average, Maximum, Multiply, Dot
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
    """
    notes:

    dropout should never be used before any batchnorm, only after
    batchnorm momentum should be decreased when using larger batch sizes

    """


    def define_model(self, model_config=None):
        # todo: look up adjusting batch norm momentum (talked about in forums)

        # corpus params
        cps = {
            'max_sent_len': self.text_mapper.max_sent_len,
            'max_word_len': self.text_mapper.max_word_len,
            'word_vocab_size': self.text_mapper.word_mapper.get_vocab_len(),
            'char_vocab_size':self.text_mapper.char_mapper.get_vocab_len()
        }

        # model inputs
        chars_input = Input(shape=(cps['max_sent_len'], cps['max_word_len']), name='chars_input', dtype='int64')
        char_feats_input = Input(shape=(cps['max_sent_len'], cps['max_word_len'],
                                        self.text_mapper.char_mapper.num_add_feats),
                                 name='chars_feats_input', dtype='float32')
        sent_feats_input = Input(shape=(self.text_mapper.num_sent_feats,), name="sent_feats_input", dtype='float32')
        words_input = Input(shape=(cps['max_sent_len'],), name='words_input', dtype='int64')
        words_feats_input = Input(shape=(cps['max_sent_len'], self.text_mapper.word_mapper.num_add_feats),
                                  name='words_feats_input', dtype='float32')

        model_inputs = {
            "chars_input": chars_input,
            "char_feats_input": char_feats_input,
            "sent_feats_input": sent_feats_input,
            "words_input": words_input,
            "words_feats_input": words_feats_input,
        }

        # load this large matrix only once, and then reuse if when needed
        word_emb = self.get_word_embedding_layer(model_inputs, cps)

        lstm_pred, lstm_logits = lstm_model(model_inputs, cps, word_emb)
        conv_pred, conv_logits = conv_model(model_inputs, cps, word_emb)

        ensemble_weights = ensemble_weights_model(model_inputs, cps, word_emb, lstm_logits, conv_logits)

        ensemble_preds = Concatenate()([lstm_pred, conv_pred])
        final_pred = Dot(axes=-1)([ensemble_weights, ensemble_preds])

        inputs = list(model_inputs.values())
        preds = [lstm_pred, conv_pred, final_pred]
        self.model = Model(inputs=inputs, outputs=preds)
        return self.model

    def get_word_embedding_layer(self, inputs, cps):
        # are we using pretrained weights?
        if self.embedding is not None:
            # todo: make this trainable at the end
            matrix_shape = self.embedding.embedding_matrix.shape
            untrainable_word_embedding = EmbeddingLayer(input_dim=cps['word_vocab_size'],
                                                        output_dim=matrix_shape[1],
                                                        input_length=cps['max_sent_len'],
                                                        weights=[self.embedding.embedding_matrix],
                                                        trainable=False,
                                                        name='static_word_emb')(inputs['words_input'])
            # You can make this next embedding trainable to apply a loss to changing these word representations
            # consider making this trainable at the very end.

            # regularized_word_embedding = EmbeddingLayer(input_dim=word_vocab_size,
            #                                             output_dim=matrix_shape[1],
            #                                             input_length=max_sent_len,
            #                                             weights=[np.zeros(matrix_shape)],
            #                                             todo: tune this regularization
            #                                             todo: add function to turn training on/off (access by name)
            #                                             embeddings_regularizer=regularizers.l2(0.1),
            #                                             trainable=False,
            #                                             name='reg_word_emb')(inputs['words_input'])

            # uncomment out as appropriate
            # word_emb = Add()([untrainable_word_embedding, regularized_word_embedding])
            word_emb = untrainable_word_embedding

        # do we want to train from scratch? (most likely, no), but for fast pipeline checks - don't need emb loading
        else:
            trainable_lstm_embedding = EmbeddingLayer(input_dim=cps['word_vocab_size'],
                                                      output_dim=50,
                                                      input_length=cps['max_sent_len'],
                                                      trainable=True)(inputs['words_input'])
            word_emb = trainable_lstm_embedding
        return word_emb


def lstm_model(inputs, cps, word_emb):
    word_rep = word_rep_with_char_info(inputs, cps, word_emb)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(word_rep)
    lstm_out = Bidirectional(CuDNNGRU(64))(x)
    lstm_logits = Dense(1, activation='linear')(lstm_out)
    lstm_pred = Activation('sigmoid', name='lstm_pred')(lstm_logits)
    return lstm_pred, lstm_logits


def conv_model(inputs, cps, word_emb):
    word_rep = word_rep_with_char_info(inputs, cps, word_emb)
    conv_cell_out = conv_cell(word_rep)

    # add additional sentence features? can comment in/out
    conv_cell_out = Concatenate()([conv_cell_out, inputs['sent_feats_input']])

    # conv_out = Concatenate()([conv_out, sent_feats_input])
    conv_dense = Dense(64, activation='relu')(conv_cell_out)
    # maybe add another dense layer?
    conv_dense = Dropout(0.2)(conv_dense)
    conv_logits = Dense(1, activation='linear')(conv_dense)
    conv_pred = Activation('sigmoid', name='conv_pred')(conv_logits)
    return conv_pred, conv_logits


def ensemble_weights_model(inputs, cps, word_emb, lstm_logits, conv_logits):
    # todo: make this simpler

    word_rep = word_rep_with_char_info(inputs, cps, word_emb)
    x = Bidirectional(CuDNNLSTM(16, return_sequences=True))(word_rep)
    conv_cell_out = conv_cell([x, word_rep])

    # add additional sentence features? can comment in/out
    conv_cell_out = Concatenate()([conv_cell_out, inputs['sent_feats_input'],
                                   lstm_logits, conv_logits])

    conv_dense = Dense(32, activation='relu')(conv_cell_out)
    ensemble_weights = Dense(2, activation='softmax')(conv_dense)
    return ensemble_weights


def word_rep_with_char_info(inputs, cps, word_emb):

    char_features = char_level_feature_model(inputs, cps)

    # Can optionally allow each word_rep to have some of it's word embedding trainable by including this:
    # and then concatenating at the end

    # trainable_char_embedding = EmbeddingLayer(input_dim=word_vocab_size, output_dim=10,
    #                                           input_length=max_sent_len)(words_input)
    # trainable_char_embedding = SpatialDropout1D(0.1)(trainable_char_embedding)

    word_emb = SpatialDropout1D(0.1)(word_emb)
    word_rep = Concatenate()([char_features, word_emb, inputs["words_feats_input"]])
    # word_rep = Concatenate()([char_features, word_emb, inputs["words_feats_input"], trainable_conv_embedding])

    return word_rep


def char_level_feature_model(inputs, cps, outdim=128):

    chars_words_embedding = TimeDistributed(EmbeddingLayer(cps['char_vocab_size'],
                                                           output_dim=16,
                                                           # embeddings_regularizer=regularizers.l1(),
                                                           input_length=cps['max_word_len']))(inputs['chars_input'])
    x = TimeDistributed(SpatialDropout1D(0.1))(chars_words_embedding)
    char_rep = Concatenate()([x, inputs['char_feats_input']])
    conv_outputs = []
    # todo: tune these conv kernels
    conv_kernels = [[32, 1], [32, 2], [32, 3], [32, 4]]
    for num_filter, kernel_size in conv_kernels:
        char_conv = TimeDistributed(Conv1D(filters=num_filter, kernel_size=kernel_size))(char_rep)
        x = TimeDistributed(BatchNormalization())(char_conv)
        x = TimeDistributed(Activation('relu'))(x)
        # x = TimeDistributed(Dropout(0.1))(x)
        m = TimeDistributed(GlobalMaxPooling1D())(x)
        conv_outputs.append(m)
    conv_outs = Concatenate()(conv_outputs)
    feats_1 = Dense(outdim)(conv_outs)
    feats_1 = BatchNormalization()(feats_1)
    feats_1 = Activation('relu')(feats_1)
    # feats_2 = Dense(outdim)(feats_1)
    # feats_2 = BatchNormalization()(feats_2)
    # feats_2 = Activation('relu')(feats_2)
    # x = Concatenate()([feats_1, feats_2])
    return feats_1


def conv_cell(input_layers, conv_kernels=[[32, 1], [32, 2], [32, 3], [32, 4]]):
    conv_outputs = []

    if not isinstance(input_layers, list):
        input_layers = [input_layers]
    for num_filter, kernel_size in conv_kernels:
        # todo: consider adding output of bilstm back here
        # why is conv model weak? but we don't want to affect learning (turn off backprop?)
        for i in input_layers:
            char_conv = Conv1D(filters=num_filter, kernel_size=kernel_size)(i)
            batch_norm = BatchNormalization()(char_conv)
            activation = Activation('relu')(batch_norm)
            conv_features = []
            conv_features.append(GlobalMaxPooling1D()(activation))
            # conv_features.append(GlobalAveragePooling1D()(activation))
            if len(conv_features) > 1:
                conv_feats = Concatenate()(conv_features)
            else:
                conv_feats = conv_features[0]
            conv_outputs.append(conv_feats)
    conv_cell_out = Concatenate()(conv_outputs)
    return conv_cell_out
