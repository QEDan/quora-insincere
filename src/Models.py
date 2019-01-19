from keras.layers import Bidirectional, CuDNNLSTM, Reshape, Conv2D, MaxPool2D, \
    Concatenate, Flatten, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate
from keras.layers import Dense, Input, Embedding as EmbeddingLayer, Dropout
from keras.models import Model

from src.Attention import Attention
from src.InsincereModel import InsincereModel


class LSTMModel(InsincereModel):
    def define_model(self):
        inp = Input(shape=(self.data.maxlen,))
        x = EmbeddingLayer(self.embedding.nb_words,
                           self.embedding.embed_size,
                           weights=[self.embedding.embedding_matrix],
                           trainable=False)(inp)
        x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        concat_layers = [avg_pool, max_pool]
        inputs = [inp]
        if self.data.custom_features:
            inp_features = Input(shape=(len(self.data.custom_features),))
            concat_layers += [inp_features]
            inputs += [inp_features]
        x = concatenate([avg_pool, max_pool, inp_features])
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(1, activation="sigmoid")(x)
        self.model = Model(inputs=inputs, outputs=x)
        self.model.compile(loss=self.loss, optimizer='sgd', metrics=['accuracy', self.f1_score])
        return self.model


class LSTMModelAttention(InsincereModel):
    def define_model(self):
        inp = Input(shape=(self.data.maxlen,))
        x = EmbeddingLayer(self.embedding.nb_words,
                           self.embedding.embed_size,
                           weights=[self.embedding.embedding_matrix],
                           trainable=False)(inp)
        x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
        x = Attention(self.data.maxlen)(x)
        inputs = [inp]
        if self.data.custom_features:
            inp_features = Input(shape=(len(self.data.custom_features),))
            x = concatenate([x, inp_features])
            x = Dense(32, activation="relu")(x)
            inputs += [inp_features]
        x = Dense(16, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(1, activation="sigmoid")(x)
        self.model = Model(inputs=inputs, outputs=x)
        self.model.compile(loss=self.loss, optimizer='sgd', metrics=['accuracy', self.f1_score])
        return self.model


class CNNModel(InsincereModel):
    def define_model(self):
        filter_sizes = [1, 2, 3, 5]
        num_filters = 36
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
        z = Dropout(0.1)(z)
        if self.data.custom_features:
            inp_features = Input(shape=(len(self.data.custom_features),))
            z = concatenate([z, inp_features])
            z = Dense(32, activation='relu')(z)
            inputs += [inp_features]
        outp = Dense(1, activation="sigmoid")(z)
        self.model = Model(inputs=inputs, outputs=outp)
        self.model.compile(loss=self.loss, optimizer='sgd', metrics=['accuracy', self.f1_score])

        return self.model