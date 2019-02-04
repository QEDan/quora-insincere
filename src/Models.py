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
        # if model_config is None:
        #     model_config = self.default_config()

        max_sent_len = self.text_mapper.max_sent_len
        max_word_len = self.text_mapper.max_word_len
        word_vocab_size = self.text_mapper.word_mapper.get_vocab_len()
        char_vocab_size = self.text_mapper.char_mapper.get_vocab_len()

        chars_input = Input(shape=(max_sent_len, max_word_len), name='chars_input', dtype='int64')
        char_feats_input = Input(shape=(max_sent_len, max_word_len, self.text_mapper.char_mapper.num_add_feats),
                                 name='chars_feats_input', dtype='float32')
        lstm_char_features = char_level_feature_model(chars_input, char_feats_input, max_word_len, char_vocab_size)
        conv_char_features = char_level_feature_model(chars_input, char_feats_input, max_word_len, char_vocab_size)
        sent_char_features = char_level_feature_model(chars_input, char_feats_input, max_word_len,
                                                      char_vocab_size, outdim=32)
        sent_feats_input = Input(shape=(self.text_mapper.num_sent_feats,), name="sent_feats_input", dtype='float32')
        words_input = Input(shape=(max_sent_len,), name='words_input', dtype='int64')
        # trainable_lstm_embedding = EmbeddingLayer(input_dim=word_vocab_size, output_dim=10,
        #                                           input_length=max_sent_len)(words_input)

        # todo: is this doing something useful? maybe each model should get their own?
        trainable_char_embedding = EmbeddingLayer(input_dim=word_vocab_size, output_dim=10,
                                                  input_length=max_sent_len)(words_input)
        trainable_char_embedding = SpatialDropout1D(0.1)(trainable_char_embedding)

        if self.embedding is not None:
            matrix_shape = self.embedding.embedding_matrix.shape
            # todo: make this trainable at the end
            untrainable_word_embedding = EmbeddingLayer(input_dim=word_vocab_size,
                                                        output_dim=matrix_shape[1],
                                                        input_length=max_sent_len,
                                                        weights=[self.embedding.embedding_matrix],
                                                        trainable=False,
                                                        name='static_word_emb')(words_input)
            # regularized_word_embedding = EmbeddingLayer(input_dim=word_vocab_size,
            #                                             output_dim=matrix_shape[1],
            #                                             input_length=max_sent_len,
            #                                             weights=[np.zeros(matrix_shape)],
            #                                             # embeddings_regularizer=regularizers.l2(0.1),  # todo: tune this regularization
            #                                             trainable=False,
            #                                             name='reg_word_emb')(words_input)
            # word_emb = Add()([untrainable_word_embedding, regularized_word_embedding])
            word_emb = untrainable_word_embedding
            lstm_embedding = SpatialDropout1D(0.1)(word_emb)
            conv_embedding = SpatialDropout1D(0.1)(word_emb)
            lstm_rep = Concatenate()([lstm_char_features, lstm_embedding])
            conv_rep = Concatenate()([conv_char_features, conv_embedding])
            # lstm_rep = Concatenate()([lstm_char_features, lstm_embedding, trainable_lstm_embedding])
            # conv_rep = Concatenate()([conv_char_features, conv_embedding, trainable_char_embedding])
        else:
            trainable_lstm_embedding = EmbeddingLayer(input_dim=word_vocab_size,
                                                      output_dim=50,
                                                      input_length=max_sent_len,
                                                      trainable=False)(words_input)
            trainable_conv_embedding = EmbeddingLayer(input_dim=word_vocab_size,
                                                      output_dim=50,
                                                      input_length=max_sent_len,
                                                      trainable=False)(words_input)
            lstm_embedding = SpatialDropout1D(0.1)(trainable_lstm_embedding)
            conv_embedding = SpatialDropout1D(0.1)(trainable_conv_embedding)
            lstm_rep = Concatenate()([lstm_char_features, lstm_embedding])
            conv_rep = Concatenate()([conv_char_features, conv_embedding])

        words_feats_input = Input(shape=(max_sent_len, self.text_mapper.word_mapper.num_add_feats),
                                  name='words_feats_input', dtype='float32')
        lstm_rep = Concatenate()([lstm_rep, words_feats_input])
        conv_rep = Concatenate()([conv_rep, words_feats_input])

        x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(lstm_rep)
        lstm_out = Bidirectional(CuDNNGRU(64))(x)
        lstm_logits = Dense(1, activation='linear')(lstm_out)
        lstm_pred = Activation('sigmoid', name='lstm_pred')(lstm_logits)

        conv_outputs = []
        conv_kernels = [[16, 1], [16, 2], [16, 3], [16, 4]]
        for num_filter, kernel_size in conv_kernels:
            # todo: consider adding output of bilstm back here - conv model is too weak, but we don't want to affect learning (turn off backprop?)
            for i in [conv_rep]:
                char_conv = Conv1D(filters=num_filter, kernel_size=kernel_size)(i)
                batch_norm = BatchNormalization()(char_conv)
                activation = Activation('relu')(batch_norm)
                global_max = GlobalMaxPooling1D()(activation)
                # global_avg = GlobalAveragePooling1D()(activation)
                # path_out = Concatenate()([global_max, global_avg])
                conv_outputs.append(global_max)

        conv_out = Concatenate()(conv_outputs)
        # conv_out = Concatenate()([conv_out, sent_feats_input])
        conv_dense = Dense(128, activation='relu')(conv_out)
        # todo: look up adjusting batch norm momentum (talked about in forums)
        # todo: maybe add another dense layer?
        conv_dense = Dropout(0.2)(conv_dense)
        conv_logits = Dense(1, activation='linear')(conv_dense)
        conv_pred = Activation('sigmoid', name='conv_pred')(conv_logits)

        error_feats = Concatenate()([sent_char_features, words_feats_input, trainable_char_embedding])
        conv_outputs = []

        conv_kernels = [[16, 1], [16, 2], [16, 3], [16, 4]]
        for num_filter, kernel_size in conv_kernels:
            # todo: consider adding output of bilstm back here - conv model is too weak,
            #  but don't want to affect lstm backprop (turn this off? how?)
            for i in [error_feats]:
                char_conv = Conv1D(filters=num_filter, kernel_size=kernel_size)(i)
                batch_norm = BatchNormalization()(char_conv)
                activation = Activation('relu')(batch_norm)
                # todo: try maxpool and more conv layers instead
                global_max = GlobalMaxPooling1D()(activation)
                global_avg = GlobalAveragePooling1D()(activation)
                path_out = Concatenate()([global_max, global_avg])
                conv_outputs.append(path_out)

        conv_out = Concatenate()(conv_outputs)

        # todo: add sentence level faetures here (ie how many unknown words, how many caps, how many words,
        #  average_length word etc - we need the final layer to learn when/how the other models are correct/incorrect)
        # todo: remove conv_dense and lstm_out from here I think

        sent_rep = Concatenate()([lstm_logits, conv_logits, sent_feats_input, conv_out])  # what are some more features to add?
        sent_rep = Dense(64, activation='relu')(sent_rep)
        sent_rep = Dropout(0.3)(sent_rep)
        sent_rep = Dense(32, activation='relu')(sent_rep)
        sent_rep = Dropout(0.2)(sent_rep)
        model_weight = Dense(2, activation='softmax', name='model_weight')(sent_rep)

        ensemble_preds = Concatenate()([lstm_pred, conv_pred])
        final_pred = Dot(axes=-1)([model_weight, ensemble_preds])

        inputs = [chars_input, words_input, char_feats_input, words_feats_input, sent_feats_input]
        preds = [lstm_pred, conv_pred, final_pred]
        self.model = Model(inputs=inputs, outputs=preds)
        return self.model


def char_level_feature_model(char_input, char_feat_input, max_word_len, char_vocab_size, outdim=100):
    chars_words_embedding = TimeDistributed(EmbeddingLayer(char_vocab_size,
                                                           output_dim=16,
                                                           # embeddings_regularizer=regularizers.l1(),
                                                           input_length=max_word_len))(char_input)
    x = TimeDistributed(SpatialDropout1D(0.1))(chars_words_embedding)
    char_rep = Concatenate()([x, char_feat_input])
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
    feats_1 = Dense(128)(conv_outs)
    feats_1 = BatchNormalization()(feats_1)
    feats_1 = Activation('relu')(feats_1)
    # feats_2 = Dense(outdim)(feats_1)
    # feats_2 = BatchNormalization()(feats_2)
    # feats_2 = Activation('relu')(feats_2)
    # x = Concatenate()([feats_1, feats_2])
    return feats_2

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
