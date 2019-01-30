from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from text_cleaning import clean_contractions, clean_specials, clean_spelling, clean_acronyms, clean_non_dictionary,\
    clean_numbers
from collections import defaultdict
from config import config_data, random_state
from data_mappers import TextMapper
from data_generator import DataGenerator

import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd
import re
import spacy

pd.options.display.max_colwidth = -1
pd.options.display.max_columns = 10


class DataV2:
    """ Loads and preprocesses data """

    def __init__(self, train_path="../input/train.csv", test_path="../input/test.csv",
                 text_col='question_text', id_col='qid', label_col='target'):

        self.text_col = text_col
        self.id_col = id_col
        self.label_col = label_col

        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)

        # self.train_df['preprocessed_text'] = self.preprocessing(self.train_df['question_text'])
        # self.test_df['preprocessed_text'] = self.preprocessing(self.test_df['question_text'])

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
        # todo: add functionality to only get data with a certain label
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

    def get_training_split(self, seed=0):
        train_qs, val_qs, train_labels, val_labels = train_test_split(self.train_df[self.text_col].tolist(),
                                                                      self.train_df[self.label_col].tolist(),
                                                                      stratify=self.train_df[self.label_col].tolist(),
                                                                      random_state=seed)
        return train_qs, val_qs, train_labels, val_labels


class CorpusInfo:
    """ Calculates corpus information to be referenced during feature engineering later """
    # todo: pass in a general tokenizer (so that it matches the tokenizer in the rest of the pipeline)
    # todo: how can we make this run faster? can be parallelized...?
    def __init__(self, questions, nlp, word_lowercase=True, char_lowercase=False):
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


# nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])
# data = DataV2()
#
# train_qs, val_qs, train_labels, val_labels = data.get_training_split()
#
# corpus_info = CorpusInfo(questions=data.get_questions()[:10000], nlp=nlp)
#
# word_counts = corpus_info.word_counts
# char_counts = corpus_info.char_counts
#
# text_mapper = TextMapper(word_counts=word_counts, char_counts=char_counts, word_threshold=10, max_word_len=20,
#                          char_threshold=350, max_sent_len=100, nlp=nlp, word_lowercase=True, char_lowercase=True)
#
# sample_question = data.train_df.sample()['question_text'].values[0]
#
# b = DataGenerator(train_qs, train_labels, text_mapper)
#
# out = b[0]
#
# def test_data_mapping(text_mapper, data):
#
#     sample_question = data.train_df.sample()['question_text'].values[0]
#     print("Sample Question:\n{}".format(sample_question))
#     words_input, chars_input = text_mapper.text_to_x(sample_question)
#
#     print("word_x rep:\n{}".format(words_input))
#     print("char_x rep:\n{}".format(chars_input))
#
#     words_x_to_text = text_mapper.x_to_words(words_input)
#     chars_x_to_text = text_mapper.x_to_chars(chars_input)
#
#     print("word_x_to_text:\n{}".format(words_x_to_text))
#     print("char_x_to_text:\n{}".format(chars_x_to_text))
#
#
# test_data_mapping(text_mapper, data)


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
        self.custom_features = None
        self.train_features = None
        self.val_features = None
        self.test_features = None
        self.feature_scaler = None
        self.config = config_data

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

    def preprocess_questions(self, questions):
        questions = questions.fillna("_na_")
        preprocess_config = self.config.get('preprocess')
        case_sensitive = not preprocess_config.get('lower_case')
        if preprocess_config.get('lower_case'):
            questions = questions.str.lower()
        if preprocess_config.get('remove_stop_words'):
            questions = questions.apply(self._remove_stops)
        if preprocess_config.get('remove_contractions'):
            questions = questions.apply(lambda x: self.clean_contractions(x))
        if preprocess_config.get('remove_specials'):
            questions = questions.apply(lambda x: self.clean_specials(x))
        if preprocess_config.get('correct_spelling'):
            questions = questions.apply(lambda x: self.clean_spelling(x, case_sensitive=case_sensitive))
        if preprocess_config.get('replace_acronyms'):
            questions = questions.apply(lambda x: self.clean_acronyms(x, case_sensitive=case_sensitive))
        if preprocess_config.get('replace_non_words'):
            questions = questions.apply(lambda x: self.clean_non_dictionary(x, case_sensitive=case_sensitive))
        if preprocess_config.get('replace_numbers'):
            questions = questions.apply(lambda x: self.clean_numbers(x))
        return questions

    def preprocessing(self):
        logging.info("Preprocessing data...")
        for df in [self.train_df, self.test_df]:
            if self.config.get('preprocess').get('use_custom_features'):
                df = self.add_features(df)
            df['question_text'] = self.preprocess_questions(df['question_text'])
        if self.config.get('preprocess').get('use_custom_features'):
            self.scale_features()
        self.split()
        self.get_xs_ys()
        self.tokenize()
        self.pad_sequences()

    def split(self):
        logging.info("Train/Eval split...")
        self.train_df, self.val_df = train_test_split(self.train_df,
                                                      test_size=self.config.get('test_size'),
                                                      random_state=random_state)
        if self.custom_features:
            self.train_features = self.train_df[self.custom_features].values
            self.val_features = self.val_df[self.custom_features].values
            self.test_features = self.test_df[self.custom_features].values

    def get_xs_ys(self):
        self.train_X = self.train_df["question_text"].values
        self.val_X = self.val_df["question_text"].values
        self.test_X = self.test_df["question_text"].values
        self.train_y = self.train_df['target'].values
        self.val_y = self.val_df['target'].values

    def tokenize(self):
        logging.info("Tokenizing...")
        max_feature = self.config.get('max_feature')
        tokenizer = Tokenizer(num_words=max_feature)
        tokenizer.fit_on_texts(list(self.train_X))
        self.train_X = tokenizer.texts_to_sequences(self.train_X)
        self.val_X = tokenizer.texts_to_sequences(self.val_X)
        self.test_X = tokenizer.texts_to_sequences(self.test_X)
        self.tokenizer = tokenizer
        self.max_feature = max_feature

    def append_features(self):
        self.train_X = [self.train_X, np.array(self.train_df[self.custom_features])]
        self.val_X = [self.val_X, np.array(self.val_df[self.custom_features])]
        self.test_X = [self.test_X, np.array(self.test_df[self.custom_features])]

    def pad_sequences(self):
        logging.info("Padding Sequences...")
        maxlen = self.config.get('max_seq_len')
        self.train_X = pad_sequences(self.train_X, maxlen=maxlen)
        self.val_X = pad_sequences(self.val_X, maxlen=maxlen)
        self.test_X = pad_sequences(self.test_X, maxlen=maxlen)
        self.maxlen = maxlen

    def add_pseudo_data(self, pred_test_y):
        logging.warning("Using pseudo data...")
        self.full_X = np.vstack([self.train_X, self.val_X, self.test_X])
        self.full_y = np.vstack([self.train_y.reshape((len(self.train_y), 1)),
                                 self.val_y.reshape((len(self.val_y), 1)), pred_test_y])

    @staticmethod
    def clean_contractions(text):
        contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                               "could've": "could have", "couldn't": "could not", "didn't": "did not",
                               "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
                               "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
                               "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                               "I'd": "i would", "I'd've": "I would have", "I'll": "i will", "I'll've": "i will have",
                               "I'm": "i am", "I've": "i have", "i'd": "i would", "i'd've": "i would have",
                               "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have",
                               "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
                               "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
                               "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                               "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                               "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                               "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                               "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                               "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                               "she'll've": "she will have", "she's": "she is", "should've": "should have",
                               "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                               "so's": "so as", "this's": "this is", "that'd": "that would",
                               "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                               "there'd've": "there would have", "there's": "there is", "here's": "here is",
                               "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                               "they'll've": "they will have", "they're": "they are", "they've": "they have",
                               "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
                               "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",
                               "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                               "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is",
                               "when've": "when have", "where'd": "where did", "where's": "where is",
                               "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                               "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
                               "will've": "will have", "won't": "will not", "won't've": "will not have",
                               "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                               "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
                               "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
                               "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                               "you're": "you are", "you've": "you have"}
        specials = ["’", "‘", "´", "`"]
        for s in specials:
            text = text.replace(s, "'")
        text = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in text.split(" ")])
        return text

    @staticmethod
    def clean_specials(text):
        punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
        puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=',
                  '#', '*', '+', '\\', '•', '~', '@', '£',
                  '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′',
                  'Â', '█', '½', 'à', '…',
                  '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║',
                  '―', '¥', '▓', '—', '‹', '─',
                  '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è',
                  '¸', '¾', 'Ã', '⋅', '‘', '∞',
                  '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï',
                  'Ø', '¹', '≤', '‡', '√', ]
        punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                         "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                         '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-',
                         'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }
        for p in punct_mapping:
            text = text.replace(p, punct_mapping[p])
        for p in set(list(punct) + puncts) - set(punct_mapping.keys()):
            text = text.replace(p, f' {p} ')

        specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '',
                    'है': ''}
        for s in specials:
            text = text.replace(s, specials[s])
        return text

    @staticmethod
    def clean_spelling(text, case_sensitive=False):
        misspell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling',
                         'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',
                         'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize',
                         'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What',
                         'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can',
                         'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I',
                         'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation',
                         'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis',
                         'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017',
                         '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess',
                         "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization',
                         'demonitization': 'demonetization', 'demonetisation': 'demonetization'}
        for word in misspell_dict.keys():
            if case_sensitive:
                text = text.replace(word, misspell_dict[word])
            else:
                re_insensitive = re.compile(re.escape(word), re.IGNORECASE)
                text = re_insensitive.sub(misspell_dict[word], text)
        return text

    @staticmethod
    def clean_acronyms(text, case_sensitive=False):
        acronym_dict = {'upsc': 'union public service commission',
                        'aiims': 'all india institute of medical sciences',
                        'cgl': 'graduate level examination',
                        'icse': 'indian school certificate exam',
                        'iiit': 'indian institute of information technology',
                        'cgpa': 'cumulative grade point average',
                        'ielts': 'international english language training system',
                        'ncert': 'national council of education research training',
                        'isro': 'indian space research organization',
                        'clat': 'common law admission test',
                        'ibps': 'institute of banking personnel selection',
                        'iiser': 'indian institute of science education and research',
                        'iisc': 'indian institute of science',
                        'iims': 'indian institutes of management',
                        'cpec': 'china pakistan economic corridor'

                        }
        for word in acronym_dict.keys():
            if case_sensitive:
                text = text.replace(word, acronym_dict[word])
            else:
                re_insensitive = re.compile(re.escape(word), re.IGNORECASE)
                text = re_insensitive.sub(acronym_dict[word], text)
        return text

    @staticmethod
    def clean_non_dictionary(text, case_sensitive=False):
        replace_dict = {'quorans': 'users',
                        'quoran': 'user',
                        'jio': 'phone manufacturer',
                        'manipal': 'city',
                        'bitsat': 'exam',
                        'mtech': 'technical university',
                        'pilani': 'town',
                        'bhu': 'university',
                        'h1b': 'visa',
                        'redmi': 'phone manufacturer',
                        'nift': 'university',
                        'kvpy': 'exam',
                        'thanos': 'comic villain',
                        'paytm': 'payment system',
                        'comedk': 'medical consortium',
                        'accenture': 'management consulting company',
                        'llb': 'bachelor of laws',
                        'ignou': 'university',
                        'dtu': 'university',
                        'aadhar': 'social number',
                        'lenovo': 'computer manufacturer',
                        'gmat': 'exam',
                        'kiit': 'institute of technology',
                        'shopify': 'music streaming',
                        'fitjee': 'exam',
                        'kejriwal': 'politician',
                        'wbjee': 'exam',
                        'pgdm': 'master of business administration',
                        'trudeau': 'politician',
                        'nri': 'research institute',
                        'deloitte': 'accounting company',
                        'jinping': 'politician',
                        'bcom': 'bachelor of commerce',
                        'mcom': 'masters of commerce',
                        'virat': 'athlete',
                        'kcet': 'television network',
                        'wipro': 'information technology company',
                        'articleship': 'internship',
                        'comey': 'law enforcement director',
                        'jnu': 'university',
                        'acca': 'chartered accountants',
                        'aakash': 'phone manufacturer',
                        'brexit': 'british succession',
                        'crypto': 'digital currency',
                        'cryptocurrency': 'digital currency',
                        'cryptocurrencies': 'digital currencies',
                        'etherium': 'digital currency',
                        'bitcoin': 'digital currency',
                        'viteee': 'exam',
                        'iocl': 'indian oil company',
                        'nmims': 'management school',
                        'rohingya': 'myanmar people',
                        'fortnite': 'videogame',
                        'upes': 'university',
                        'nsit': 'university',
                        'coinbase': 'digital currency exchange'
                        }
        for word in replace_dict.keys():
            if case_sensitive:
                text = text.replace(word, replace_dict[word])
            else:
                re_insensitive = re.compile(re.escape(word), re.IGNORECASE)
                text = re_insensitive.sub(replace_dict[word], text)
        return text

    @staticmethod
    def clean_numbers(text, min_magnitude=2, max_magnitude=10):
        for n in range(min_magnitude, max_magnitude):
            text = re.sub('[0-9]{' + str(n) + '}', '#'*n, text)
        return text

    def add_features(self, df):
        df['question_text'] = df['question_text'].apply(lambda x: str(x))
        df['total_length'] = df['question_text'].apply(len)
        df['capitals'] = df['question_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
        df['caps_vs_length'] = df.apply(lambda row: float(row['capitals']) / float(row['total_length']),
                                        axis=1)
        df['num_words'] = df.question_text.str.count('\S+')
        df['num_unique_words'] = df['question_text'].apply(
            lambda comment: len(set(w for w in comment.split())))
        df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
        df['caps_vs_length'] = df['caps_vs_length'].fillna(0)
        df['words_vs_unique'] = df['words_vs_unique'].fillna(0)
        df['num_exclamation_marks'] = df['question_text'].apply(lambda comment: comment.count('!'))
        df['num_question_marks'] = df['question_text'].apply(lambda comment: comment.count('?'))
        df['num_smilies'] = df['question_text'].apply(
            lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))
        self.custom_features = ['total_length', 'capitals', 'caps_vs_length', 'num_words',
                                'num_unique_words', 'words_vs_unique', 'caps_vs_length',
                                'words_vs_unique', 'num_exclamation_marks', 'num_question_marks', 'num_smilies']
        return df

    def scale_features(self):
        self.feature_scaler = StandardScaler()
        features = self.train_df[self.custom_features]
        test_features = self.test_df[self.custom_features]
        self.feature_scaler.fit(features)
        self.train_df[self.custom_features] = self.feature_scaler.transform(features)
        self.test_df[self.custom_features] = self.feature_scaler.transform(test_features)

    def get_train_vocab(self):
        sentences = self.train_df['question_text'].apply(lambda x: x.split()).values
        vocab = {}
        for sentence in sentences:
            for word in sentence:
                try:
                    vocab[word] += 1
                except KeyError:
                    vocab[word] = 1
        return vocab
