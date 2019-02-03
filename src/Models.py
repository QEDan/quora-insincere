import matplotlib
from keras import Input, Model

from src.InsincereModel import InsincereModel

matplotlib.use('Agg')

from src.Models import *  # Make all models available for easy script generation.

from keras.layers import TimeDistributed, Embedding as EmbeddingLayer, Bidirectional, CuDNNLSTM, Dense, Conv1D
from keras.layers import GlobalMaxPooling1D, Concatenate, BatchNormalization, Dropout


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
        if self.embedding is not None:
            words_embedding = EmbeddingLayer(input_dim=word_vocab_size, output_dim=self.embedding.embedding_matrix.shape[1],
                                             input_length=max_sent_len,
                                             weights=[self.embedding.embedding_matrix] if self.embedding else None,
                                             trainable=False)(words_input)
        else:
            words_embedding = EmbeddingLayer(input_dim=word_vocab_size, output_dim=10,
                                             input_length=max_sent_len)(words_input)

        word_rep = Concatenate()([char_features, words_embedding])

        # todo: maybe bidirectional lstm is just too slow, can try a deeper convolutional network
        x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(word_rep)
        x = Bidirectional(CuDNNLSTM(64))(x)
        # x = Conv1D(filters=100, kernel_size=2)(x)
        # max_x = GlobalMaxPooling1D()(x)
        # avg_x = GlobalAveragePooling1D()(x)
        # x = Concatenate()([max_x, avg_x])
        # x = Dense(16)(x)
        # x = Flatten()(char_sum)
        preds = Dense(1, activation='sigmoid')(x)

        inputs = [chars_input, words_input, char_feats_input]

        self.model = Model(inputs=inputs, outputs=preds)
        self.model.compile(loss=self.loss, optimizer='adam', metrics=['accuracy', self.f1_score])

        return self.model


class CharCNNWordModel(InsincereModel):
    """ this is an experiment to check that character convolutions are outputting as expected """
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
        char_conv = TimeDistributed(Conv1D(filters=300, kernel_size=3))(chars_words_embedding)

        # represent each filter with it's max value -  each filter looks for one feature
        x = TimeDistributed(GlobalMaxPooling1D())(char_conv)
        preds = Dense(1, activation='sigmoid')(x)

        inputs = [chars_input]

        self.model = Model(inputs=inputs, outputs=preds)
        self.model.compile(loss=self.loss, optimizer='adam', metrics=['accuracy', self.f1_score])

        return self.model


def char_level_feature_model(char_input, char_feat_input, max_word_len, char_vocab_size):
    chars_words_embedding = TimeDistributed(EmbeddingLayer(char_vocab_size,
                                                           output_dim=16,
                                                           input_length=max_word_len))(char_input)
    # todo: add additional char features here
    char_rep = Concatenate()([chars_words_embedding, char_feat_input])
    conv_outputs = []
    # todo: tune these conv kernels
    conv_kernels = [[32, 2], [32, 3], [32, 4]]
    for num_filter, kernel_size in conv_kernels:
        char_conv = TimeDistributed(Conv1D(filters=num_filter, kernel_size=kernel_size))(char_rep)
        batch_norm = TimeDistributed(BatchNormalization())(char_conv)
        # todo: add dropout or batchnorm here? global average pooling?
        x = TimeDistributed(GlobalMaxPooling1D())(batch_norm)
        conv_outputs.append(x)
    char_convs_out = Concatenate()(conv_outputs)
    x = Dense(100)(char_convs_out)
    x = Dropout(0.3)(x)
    x = Dense(50)(x)
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
