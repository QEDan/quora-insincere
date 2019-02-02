from collections import defaultdict

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import config_data
from src.text_cleaning import clean_contractions, clean_specials, clean_spelling, \
    clean_acronyms, clean_non_dictionary, clean_numbers


class Data:
    """ Loads and preprocesses data """
    def __init__(self, train_path="../input/train.csv", test_path="../input/test.csv",
                 text_col='question_text', id_col='qid', label_col='target'):

        self.text_col = text_col
        self.id_col = id_col
        self.label_col = label_col
        self.train_path = train_path
        self.test_path = test_path

        self.train_df = None
        self.test_df = None

    def load(self, dev_size=None):
        logging.info("Loading data...")
        if dev_size is not None:
            logging.warning("Using dev set of size=" + str(dev_size))
        self.train_df = pd.read_csv(self.train_path, nrows=dev_size)
        self.test_df = pd.read_csv(self.test_path, nrows=dev_size)
        logging.info("Train shape : {}".format(self.train_df.shape))
        logging.info("Test shape : {}".format(self.test_df.shape))

    def split(self):
        self.train_qs, self.val_qs, self.train_labels, self.val_labels = self.get_training_split()
        return self.train_qs, self.val_qs, self.train_labels, self.val_labels

    @staticmethod
    def preprocessing(questions):

        questions = questions.fillna("_na_")
        preprocess_config = config_data.get('preprocess')
        case_sensitive = not preprocess_config.get('lower_case')
        if preprocess_config.get('lower_case'):
            questions = questions.str.lower()
        # trouble removing stop words before we have tokenized the text, this has to happen later
        # if preprocess_config.get('remove_stop_words'):
            # questions = questions.apply(remove_stops)
        if preprocess_config.get('remove_contractions'):
            questions = questions.apply(lambda x: clean_contractions(x))
        if preprocess_config.get('remove_specials'):
            questions = questions.apply(lambda x: clean_specials(x))
        if preprocess_config.get('correct_spelling'):
            questions = questions.apply(lambda x: clean_spelling(x, case_sensitive=case_sensitive))
        if preprocess_config.get('replace_acronyms'):
            questions = questions.apply(lambda x: clean_acronyms(x, case_sensitive=case_sensitive))
        if preprocess_config.get('replace_non_words'):
            questions = questions.apply(lambda x: clean_non_dictionary(x, case_sensitive=case_sensitive))
        if preprocess_config.get('replace_numbers'):
            questions = questions.apply(lambda x: clean_numbers(x))
        return questions

    def get_questions(self, subset='train'):
        # todo: add functionality to only get data with a certain label (if we want to fine tune word embeddings...)
        if subset == 'train':
            data = list(self.train_df[self.text_col])
        if subset == 'test':
            data = list(self.test_df[self.text_col])
        if subset == 'all':
            data = list(self.train_df[self.text_col]) + list(self.test_df[self.text_col])
        return data

    def get_training_labels(self):
        labels = self.train_df.loc[:, self.label_col].values
        return labels

    def get_training_split(self, test_size=0.1, seed=0):
        train_qs, val_qs, train_labels, val_labels = train_test_split(self.train_df[self.text_col].tolist(),
                                                                      self.train_df[self.label_col].tolist(),
                                                                      stratify=self.train_df[self.label_col].tolist(),
                                                                      test_size=test_size,
                                                                      random_state=seed)
        return train_qs, val_qs, train_labels, val_labels


class CorpusInfo:
    """ Calculates corpus information to be referenced during feature engineering later """
    # todo: pass in a general tokenizer (so that it matches the tokenizer in the rest of the pipeline)
    # todo: how can we make this run faster? can be parallelized...?
    def __init__(self, questions, nlp, word_lowercase=True, char_lowercase=True):
        self.nlp = nlp
        self.word_lowercase = word_lowercase
        self.char_lowercase = char_lowercase

        self.word_counts = []
        self.char_counts = []
        self.sent_lengths = []
        self.word_lengths = []

        self.calc_corpus_info(questions)

    def calc_corpus_info(self, questions):
        word_counters = defaultdict(int)
        char_counters = defaultdict(int)

        for question in questions:
            tokenized_question = self.nlp(question)
            self.sent_lengths.append(len(tokenized_question))
            for token in tokenized_question:
                text = token.text
                self.word_lengths.append(len(text))
                word_to_count = text.lower() if self.word_lowercase else text
                word_counters[word_to_count] += 1
                for char in text:
                    char_to_count = char.lower() if self.char_lowercase else char
                    char_counters[char_to_count] += 1

        self.word_counts = sorted(word_counters.items(), key=lambda x: x[1], reverse=True)
        self.char_counts = sorted(char_counters.items(), key=lambda x: x[1], reverse=True)

    def plot_word_lengths(self, max_len):
        plt.hist(self.word_lengths, bins=np.arange(0, max_len, 2), cumulative=True, normed=1)
        plt.hlines(0.975, 0, 30, colors='red')

    def plot_sent_lengths(self, max_len):
        plt.hist(self.sent_lengths, bins=np.arange(0, max_len, 2), cumulative=True, normed=1)
        plt.hlines(0.975, 0, 30, colors='red')

