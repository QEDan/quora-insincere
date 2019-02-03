import logging
import matplotlib
from pprint import pprint

from src.UnknownWords import UnknownWords

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
import spacy
import pickle
import os

from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold

from src.Data import Data, CorpusInfo
from src.data_mappers import TextMapper
from src.Embedding import Embedding
from src.Models import *  # Make all models available for easy script generation.
from src.config import random_state as SEED, config_main as config

np.random.seed(SEED)
tf.set_random_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)


def find_best_threshold(y_proba, y_true, plot=False):
    logging.info("Finding best threshold...")
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.001)
    F = 2.0 / (1.0/precision + 1.0/recall)
    best_score = np.max(F)
    best_th = thresholds[np.argmax(F)]
    logging.info("Best score = {}. Best threshold = {}".format(best_score, best_th))
    if plot:
        plt.plot(thresholds, F, '-b')
        plt.plot([best_th], [best_score], '*r')
        plt.savefig('threshold.png')
        plt.close()
    return best_th


def write_predictions(data, preds, thresh=0.5):
    logging.info("Writing predictions ...")
    preds = (preds > thresh).astype(int)
    out_df = pd.DataFrame({"qid": data.test_df["qid"].values})
    out_df['prediction'] = preds
    out_df.to_csv("submission.csv", index=False)


def print_diagnostics(y_true, y_pred, file_suffix='', persist=True):
    try:
        cfn_matrix = metrics.confusion_matrix(y_true, y_pred)
    except ValueError:
        logging.warning("Warning: mix of binary and continuous targets used. Searching for best threshold.")
        thresh = find_best_threshold(y_pred, y_true)
        logging.warning("Applying threshold {} to predictions.".format(thresh))
        y_pred = (y_pred > thresh).astype(int)
        cfn_matrix = metrics.confusion_matrix(y_true, y_pred)
    with open('diagnostics' + file_suffix + '.txt', 'w') if persist else None as f:
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


def print_wrongest(X, y_true, y_pred, num_wrongest=100, print_them=False, persist=True, file_suffix=None):
    def print_row(row):
        print("Q:" + row['question_text'])
        print("qid: " + row['qid'])
        print("Target: " + str(row['y_true']))
        print("Prediction: " + str(row['y_pred']))
        print("-"*40)

    wrongest_fps, wrongest_fns = get_wrongest(X, y_true, y_pred, num_wrongest=num_wrongest)
    if print_them:
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
        filename = 'wrongest'
        if file_suffix:
            filename += '_' + file_suffix
        wrongest_fps.to_csv(filename + '_fps.csv', index=False)
        wrongest_fns.to_csv(filename + '_fns.csv', index=False)
    return wrongest_fps, wrongest_fns


def cross_validate(model_class, data, embeddings, n_splits=4, show_wrongest=True, model_config=None,):
    logging.info("Cross validating model {} using {} folds...".format(model_class.__name__, str(n_splits)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    models = list()
    for i, (train, test) in enumerate(skf.split(data.train_X, data.train_y)):
        logging.info("Running Fold {} of {}".format(i + 1, n_splits))
        models.append(None)
        cv_name = model_class.__name__ + '_cv_' + str(i)
        models[-1] = model_class(data=data, name=cv_name)
        models[-1].blend_embeddings(embeddings)
        models[-1].define_model(model_config)
        models[-1].fit(train_indices=train, val_indices=test, curve_file_suffix=str(i))
        if data.custom_features:
            predict_X = [data.train_X[test], data.train_features[test]]
        else:
            predict_X = [data.train_X[test]]
        pred_y_val = models[-1].predict(predict_X)
        print_diagnostics(data.train_y[test], pred_y_val, file_suffix='_' + cv_name)
        if show_wrongest:
            print_wrongest(data.train_df.iloc[test],
                           data.train_y[test],
                           pred_y_val,
                           num_wrongest=20,
                           persist=True,
                           file_suffix=models[-1].name)
    return models


def load_embeddings(word_vocab, embedding_files, keep_index=True):
    embeddings = list()
    for f in embedding_files:
        embeddings.append(Embedding(word_vocab=word_vocab))
        embeddings[-1].load(f)
        if not keep_index:
            embeddings[-1].cleanup_index()
    return embeddings


def save_unknown_words(data, embeddings, max_words=None):
    vocab = data.get_train_vocab()
    nb_words = 0
    for v in vocab.items():
        nb_words += v[1]
    for emb in embeddings:
        unknown_words = emb.check_coverage(vocab)
        df_unknown_words = pd.DataFrame(unknown_words, columns=['word', 'count'])\
            .sort_values('count', ascending=False)
        df_unknown_words['frequency'] = df_unknown_words['count'] / nb_words
        df_unknown_words = df_unknown_words.head(max_words)
        df_unknown_words.to_csv('unknown_words_' + emb.name + '.csv', index=False)


def cleanup_models(models):
    for m in models:
        m.cleanup()

def save_configs():
    from src.config import random_state, \
        config_data, \
        config_insincere_model, \
        config_lrfinder, \
        config_one_cycle, \
        config_main
    config_dict = locals().copy()
    with open('configs.txt', 'wt') as f:
        pprint(config_dict, stream=f)
    logging.info('Configurations: ')
    logging.info(pprint(config_dict))

import sys
from pympler import asizeof
def main():

    embedding_files = config.get('embedding_files')
    dev_size = config.get('dev_size')
    # dev_size = 50000
    data = Data()
    data.load(dev_size)
    data.split()

    nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])

    corpus_info = CorpusInfo(data.get_questions(subset='train'), nlp)

    word_counts = corpus_info.word_counts
    char_counts = corpus_info.char_counts

    text_mapper = TextMapper(word_counts=word_counts, char_counts=char_counts, word_threshold=10, max_word_len=12,
                             char_threshold=350, max_sent_len=70, nlp=nlp, word_lowercase=True, char_lowercase=True)

    word_vocab = text_mapper.get_words_vocab()

    embeddings = load_embeddings(word_vocab, embedding_files)

    unknown_word_models = [UnknownWords(text_mapper, embedding) for embedding in embeddings]
    for model in unknown_word_models:
        model.define_model()
        model.fit(data.train_qs)
        model.improve_embedding()
    # save_unknown_words(data, embeddings, max_words=200)
    # models_all = list()
    # for model in config.get('models'):
    #     model_class = globals()[model.get('class')]
    #     models_all.extend(cross_validate(model_class,
    #                                      data,
    #                                      embeddings,
    #                                      model_config=model.get('args')))

    model = BiLSTMCharCNNModel(data=data, corpus_info=corpus_info, text_mapper=text_mapper, batch_size=128)
    model.blend_embeddings(embeddings)

    model.define_model()
    model.fit()

    # cleanup_models([model])  # embedding/memory cleanup

    val_preds = model.predict_subset(subset='val')

    # ensemble_cv = Ensemble(m odels_all)
    # train_X = [data.train_X]
    # val_X = [data.val_X]
    # test_X = [data.test_X]
    # if data.custom_features:
    #     train_X += [data.train_features]
    #     val_X += [data.val_features]
    #     test_X += [data.test_features]

    # find the best threshold

    # pred_train_y = ensemble_cv.predict_linear_regression(train_X, data.train_y, train_X)
    val_y = np.array(data.val_labels)
    thresh = find_best_threshold(val_preds, val_y)

    # pred_val_y = ensemble_cv.predict_linear_regression(train_X, data.train_y, val_X)
    print_diagnostics(val_y, (val_preds > thresh).astype(int))
    # pred_y_test = ensemble_cv.predict_linear_regression(train_X, data.train_y, test_X)
    pred_y_test = model.predict_subset('test')
    write_predictions(data, pred_y_test, thresh)


if __name__ == "__main__":
    logging.getLogger()
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S')
    save_configs()
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config_tf.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config_tf)
    main()
    sess.close()
    logging.info("Done!")
