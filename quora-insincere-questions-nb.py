import gc
import time

import keras.backend as K
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import traceback
from gensim.models import KeyedVectors
from keras.engine import Layer
from keras.layers import Bidirectional, CuDNNLSTM, initializers, regularizers, constraints
from keras.layers import Dense, Input, Embedding, Dropout
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self, train_path="../input/train.csv", test_path="../input/test.csv"):
        self.train_path = train_path
        self.test_path = test_path
        self.train_df = None
        self.test_df = None
        self.val_df = None
        self.train_X = None
        self.val_X = None
        self.test_X = None
        self.full_X = None
        self.full_y = None
        self.train_y = None
        self.val_y = None
        self.maxlen = None
        self.tokenizer = None
        self.max_feature = None

    def load(self, dev_size=None):
        logging.info("Loading data...")
        if dev_size is not None:
            logging.warning("Using dev set of size=" + str(dev_size))
        self.train_df = pd.read_csv(self.train_path, nrows=dev_size)
        self.test_df = pd.read_csv(self.test_path, nrows=dev_size)
        logging.info("Train shape : {}".format(self.train_df.shape))
        logging.info("Test shape : {}".format(self.test_df.shape))

    @staticmethod
    def _remove_stops(sentence):
        stop = set(stopwords.words('english'))
        filtered = list()
        for w in sentence.split(" "):
            if w not in stop:
                filtered.append(w)
        return " ".join(filtered)

    def preprocess_questions(self, questions, remove_stop_words=False):
        questions = questions.str.lower()
        questions = questions.fillna("_na_")
        if remove_stop_words:
            questions = questions.apply(self._remove_stops)
        return questions

    def preprocessing(self):
        logging.info("Preprocessing data...")
        for df in [self.train_df, self.test_df]:
            df['question_text'] = self.preprocess_questions(df['question_text'])
        self.split()
        self.get_xs_ys()
        self.tokenize()
        self.pad_sequences()

    def split(self, test_size=0.1, random_state=2018):
        logging.info("Train/Eval split...")
        self.train_df, self.val_df = train_test_split(self.train_df, test_size=test_size, random_state=random_state)

    def get_xs_ys(self):
        self.train_X = self.train_df["question_text"].values
        self.val_X = self.val_df["question_text"].values
        self.test_X = self.test_df["question_text"].values
        self.train_y = self.train_df['target'].values
        self.val_y = self.val_df['target'].values

    def tokenize(self, max_feature=50000):
        logging.info("Tokenizing...")
        tokenizer = Tokenizer(num_words=max_feature)
        tokenizer.fit_on_texts(list(self.train_X))
        self.train_X = tokenizer.texts_to_sequences(self.train_X)
        self.val_X = tokenizer.texts_to_sequences(self.val_X)
        self.test_X = tokenizer.texts_to_sequences(self.test_X)
        self.tokenizer = tokenizer
        self.max_feature = max_feature

    def pad_sequences(self, maxlen=100):
        logging.info("Padding Sequences...")
        self.train_X = pad_sequences(self.train_X, maxlen=maxlen)
        self.val_X = pad_sequences(self.val_X, maxlen=maxlen)
        self.test_X = pad_sequences(self.test_X, maxlen=maxlen)
        self.maxlen = maxlen

    def add_pseudo_data(self, pred_test_y):
        logging.warning("Using pseudo data...")
        self.full_X = np.vstack([self.train_X, self.val_X, self.test_X])
        self.full_y = np.vstack([self.train_y.reshape((len(self.train_y), 1)),
                                 self.val_y.reshape((len(self.val_y), 1)), pred_test_y])


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
        self.built = False
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
        return input_shape[0], self.features_dim


class RNNModel:
    def __init__(self, data, name=None):
        self.data = data
        self.name = name
        self.embeddings_index = None
        self.embed_size = None
        self.embedding_matrix = None
        self.embedding_vector = None
        self.nb_words = None
        self.model = None
        self.history = None

    def load_embedding(self, embedding_file='../input/embeddings/glove.840B.300d/glove.840B.300d.txt'):
        logging.info("loading embedding : " + embedding_file)

        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')
        if "wiki-news" in embedding_file:
            self.embeddings_index = dict(get_coefs(*o.split(" "))
                                         for i, o in enumerate(open(embedding_file)) if len(o) > 100)
        elif "glove" in embedding_file:
            self.embeddings_index = dict(get_coefs(*o.split(" ")) for i, o in enumerate(open(embedding_file)))
        elif "paragram" in embedding_file:
            self.embeddings_index = dict(get_coefs(*o.split(" ")) for i, o in
                                         enumerate(open(embedding_file, encoding="utf8", errors='ignore'))
                                         if len(o) > 100)
        elif "GoogleNews" in embedding_file:
            self.embeddings_index = {}
            wv_from_bin = KeyedVectors.load_word2vec_format(embedding_file, binary=True)
            for i, (word, vector) in enumerate(zip(wv_from_bin.vocab, wv_from_bin.vectors)):
                coefs = np.asarray(vector, dtype='float32')
                self.embeddings_index[word] = coefs
        else:
            raise ValueError("Unsupported embedding file: " + embedding_file)

        try:
            all_embs = np.stack(self.embeddings_index.values())
        except ValueError as e:
            logging.error(e)
            tb = traceback.format_exc()
            logging.error(tb)
            logging.info("len(self.embeddings_index.values()): " + str(len(self.embeddings_index.values())))
            logging.info("type(self.embeddings_index.values()[0]): " + str(type(list(self.embeddings_index.values())[0])))
            logging.info("first few self.embeddings_index.values(): " + str(list(self.embeddings_index.values())[:5]))
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        self.embed_size = all_embs.shape[1]

        word_index = self.data.tokenizer.word_index
        self.nb_words = min(self.data.max_feature, len(word_index))
        self.embedding_matrix = np.random.normal(emb_mean, emb_std, (self.nb_words, self.embed_size))
        for word, i in word_index.items():
            if i >= self.data.max_feature:
                continue
            embedding_vector = self.embeddings_index.get(word)
            if self.embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector
        return self.embedding_matrix

    def blend_embeddings(self, embedding_files):
        """Average embedding matrix given list of embedding files."""
        embedding_matrices = list()
        for emb in embedding_files:
            embedding_matrices.append(self.load_embedding(emb))
        blend = np.mean(embedding_matrices, axis=0)
        self.embedding_matrix = blend
        return blend

    @staticmethod
    def f1_score(y_true, y_pred):
        c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
        c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

        if c3 == 0:
            return 0

        precision = c1 / c2
        recall = c1 / c3

        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

    def define_model(self):
        inp = Input(shape=(self.data.maxlen,))
        x = Embedding(self.nb_words, self.embed_size, weights=[self.embedding_matrix], trainable=False)(inp)
        x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
        x = Attention(self.data.maxlen)(x)
        x = Dense(16, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(1, activation="sigmoid")(x)
        self.model = Model(inputs=inp, outputs=x)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', self.f1_score])
        return self.model

    def print(self):
        print(self.model.summary())

    def fit(self, pseudo_labels=False, batch_size=512, epochs=4, save_curve=True):
        logging.info("Fitting model...")
        if pseudo_labels:
            train_x, train_y = self.data.full_X, self.data.full_y
        else:
            train_x, train_y = self.data.train_X, self.data.train_y
        self.history = self.model.fit(train_x, train_y,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      validation_data=(self.data.val_X, self.data.val_y))
        if save_curve:
            filename = 'training_curve'
            if self.name: filename += '_' + self.name
            filename += '.png'
            self.print_curve(filename)

    def print_curve(self, filename='training_curve.png'):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(filename)

    def predict(self, x, batch_size=1024):
        logging.info("Predicting ...")
        prediction = self.model.predict([x], batch_size=batch_size, verbose=1)
        return prediction

    def cleanup(self):
        logging.info("Releasing memory...")
        del self.embeddings_index, self.embedding_matrix
        gc.collect()
        time.sleep(10)


class Ensemble:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        logging.info("Predicting with ensemble, size=" + str(len(self.models)))
        predictions = list()
        for m in self.models:
            predictions.append(m.predict(X))
            logging.debug(type(predictions[-1]))
        logging.debug(type(predictions), type(predictions[0]))
        logging.debug(len(predictions), predictions[0].shape)
        logging.debug([p.shape for p in predictions if p is not None])
        logging.debug(["None: " + str(i) for i, p in enumerate(predictions) if p is None])
        avg_pred = np.mean(predictions, axis=0)
        return avg_pred


def find_best_threshold(preds, y):
    logging.info("Finding the best threshold...")
    best_thresh = -100
    best_score = -100
    for thresh in np.arange(0.1, 0.501, 0.01):
        thresh = np.round(thresh, 2)
        score = metrics.f1_score(y, (preds > thresh).astype(int))
        if score > best_score:
            best_thresh = thresh
        logging.info("F1 score at threshold {0} is {1}".format(thresh, score))
    return best_thresh


def write_predictions(data, preds, thresh=0.5):
    logging.info("Writing predictions ...")
    preds = (preds > thresh).astype(int)
    out_df = pd.DataFrame({"qid": data.test_df["qid"].values})
    out_df['prediction'] = preds
    out_df.to_csv("submission.csv", index=False)


def print_diagnostics(y_true, y_pred, persist=True):
    try:
        cfn_matrix = metrics.confusion_matrix(y_true, y_pred)
    except ValueError:
        logging.warning("Warning: mix of binary and continuous targets used. Searching for best threshold.")
        thresh = find_best_threshold(y_pred, y_true)
        logging.warning("Applying threshold {} to predictions.".format(thresh))
        y_pred = (y_pred > thresh).astype(int)
        cfn_matrix = metrics.confusion_matrix(y_true, y_pred)
    with open('diagnostics.txt', 'w') if persist else None as f:
        print("Confusion Matrix", file=f)
        print(cfn_matrix, file=f)
        print("-"*40, file=f)
        print("F1 score: " + str(metrics.f1_score(y_true, y_pred)), file=f)
        print("MCC score: " + str(metrics.matthews_corrcoef(y_true, y_pred)), file=f)
        print("precision: " + str(metrics.precision_score(y_true, y_pred)), file=f)
        print("Recall: " + str(metrics.recall_score(y_true, y_pred)), file=f)


def get_wrongest(X, y_true, y_pred, num_wrongest=5):
    logging.info("Finding the worst predictions...")
    df = pd.DataFrame({'qid': X['qid'],
                       'question_text': X['question_text'],
                       'y_true': y_true,
                       'y_pred': y_pred.reshape(len(y_pred))})
    df['prediction_error'] = df['y_true'] - df['y_pred']
    df = df.sort_values('prediction_error')
    return df[df['y_true'] == 0].head(num_wrongest), df[df['y_true'] == 1].tail(num_wrongest)


def print_wrongest(X, y_true, y_pred, num_wrongest=100, print=False, persist=True):
    def print_row(row):
        print("Q:" + row['question_text'])
        print("qid: " + row['qid'])
        print("Target: " + str(row['y_true']))
        print("Prediction: " + str(row['y_pred']))
        print("-"*40)

    wrongest_fps, wrongest_fns = get_wrongest(X, y_true, y_pred, num_wrongest=num_wrongest)
    if print:
        print("Wrongest {} false positives:".format(num_wrongest))
        print("-" * 40)
        for i, row in wrongest_fps.iterrows():
            print_row(row)
        print()
        print("Wrongest {} false negatives:".format(num_wrongest))
        print("-" * 40)
        for i, row in wrongest_fns.iterrows():
            print_row(row)
    if persist:
        wrongest_fps.to_csv("wrongest_fps.csv", index=False)
        wrongest_fns.to_csv("wrongest_fns.csv", index=False)
    return wrongest_fps, wrongest_fns


def main():
    embeddings = ['../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin',
                  '../input/embeddings/glove.840B.300d/glove.840B.300d.txt',
                  '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec',
                  '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
                  ]
    dev_size = None  # set dev_size=None for full-scale runs
    data = Data()
    data.load(dev_size=dev_size)
    data.preprocessing()
    model = RNNModel(data)
    model.blend_embeddings(embeddings)
    model.define_model()
    model.print()
    model.fit()
    pred_val_y = model.predict(data.val_X)
    model.cleanup()
    print_wrongest(data.val_df, data.val_y, pred_val_y)
    thresh = find_best_threshold(pred_val_y, data.val_y)
    print_diagnostics(data.val_y, (pred_val_y > thresh).astype(int))
    pred_y_test = model.predict(data.test_X)
    data.add_pseudo_data(pred_y_test)
    model.fit(pseudo_labels=True)
    pred_y_test = model.predict(data.test_X)
    write_predictions(data, pred_y_test, thresh)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S')
    main()
    logging.info("Done!")
