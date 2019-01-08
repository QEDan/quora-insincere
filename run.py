#!/usr/bin/env python


import contextlib as __stickytape_contextlib

@__stickytape_contextlib.contextmanager
def __stickytape_temporary_dir():
    import tempfile
    import shutil
    dir_path = tempfile.mkdtemp()
    try:
        yield dir_path
    finally:
        shutil.rmtree(dir_path)

with __stickytape_temporary_dir() as __stickytape_working_dir:
    def __stickytape_write_module(path, contents):
        import os, os.path, errno

        def make_package(path):
            parts = path.split("/")
            partial_path = __stickytape_working_dir
            for part in parts:
                partial_path = os.path.join(partial_path, part)
                if not os.path.exists(partial_path):
                    os.mkdir(partial_path)
                    open(os.path.join(partial_path, "__init__.py"), "w").write("\n")
                    
        make_package(os.path.dirname(path))
        
        full_path = os.path.join(__stickytape_working_dir, path)
        with open(full_path, "w") as module_file:
            module_file.write(contents)

    import sys as __stickytape_sys
    __stickytape_sys.path.insert(0, __stickytape_working_dir)

    __stickytape_write_module('''src/data.py''', '''import logging\nimport numpy as np\nimport pandas as pd\nimport re\nfrom keras.preprocessing.sequence import pad_sequences\nfrom keras.preprocessing.text import Tokenizer\nfrom nltk.corpus import stopwords\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\n\n\nclass Data:\n    def __init__(self, train_path="../input/train.csv", test_path="../input/test.csv"):\n        self.train_path = train_path\n        self.test_path = test_path\n        self.train_df = None\n        self.test_df = None\n        self.val_df = None\n        self.train_X = None\n        self.val_X = None\n        self.test_X = None\n        self.full_X = None\n        self.full_y = None\n        self.train_y = None\n        self.val_y = None\n        self.maxlen = None\n        self.tokenizer = None\n        self.max_feature = None\n        self.custom_features = None\n        self.train_features = None\n        self.val_features = None\n        self.test_features = None\n        self.feature_scaler = None\n\n    def load(self, dev_size=None):\n        logging.info("Loading data...")\n        if dev_size is not None:\n            logging.warning("Using dev set of size=" + str(dev_size))\n        self.train_df = pd.read_csv(self.train_path, nrows=dev_size)\n        self.test_df = pd.read_csv(self.test_path, nrows=dev_size)\n        logging.info("Train shape : {}".format(self.train_df.shape))\n        logging.info("Test shape : {}".format(self.test_df.shape))\n\n    @staticmethod\n    def _remove_stops(sentence):\n        stop = set(stopwords.words('english'))\n        filtered = list()\n        for w in sentence.split(" "):\n            if w not in stop:\n                filtered.append(w)\n        return " ".join(filtered)\n\n    def preprocess_questions(self, questions,\n                             lower_case=False,\n                             remove_stop_words=False,\n                             remove_contractions=True,\n                             remove_specials=True,\n                             correct_spelling=True,\n                             replace_acronyms=True,\n                             replace_non_words=True,\n                             replace_numbers=False):\n        questions = questions.fillna("_na_")\n        case_sensitive = not lower_case\n        if lower_case:\n            questions = questions.str.lower()\n        if remove_stop_words:\n            questions = questions.apply(self._remove_stops)\n        if remove_contractions:\n            questions = questions.apply(lambda x: self.clean_contractions(x))\n        if remove_specials:\n            questions = questions.apply(lambda x: self.clean_specials(x))\n        if correct_spelling:\n            questions = questions.apply(lambda x: self.clean_spelling(x, case_sensitive=case_sensitive))\n        if replace_acronyms:\n            questions = questions.apply(lambda x: self.clean_acronyms(x, case_sensitive=case_sensitive))\n        if replace_non_words:\n            questions = questions.apply(lambda x: self.clean_non_dictionary(x, case_sensitive=case_sensitive))\n        if replace_numbers:\n            questions = questions.apply(lambda x: self.clean_numbers(x))\n        return questions\n\n    def preprocessing(self, lower_case=False, use_custom_features=True):\n        logging.info("Preprocessing data...")\n        for df in [self.train_df, self.test_df]:\n            if use_custom_features:\n                df = self.add_features(df)\n            df['question_text'] = self.preprocess_questions(df['question_text'], lower_case=lower_case)\n        if use_custom_features:\n            self.scale_features()\n        self.split()\n        self.get_xs_ys()\n        self.tokenize()\n        self.pad_sequences()\n\n    def split(self, test_size=0.1, random_state=2018):\n        logging.info("Train/Eval split...")\n        self.train_df, self.val_df = train_test_split(self.train_df, test_size=test_size, random_state=random_state)\n        if self.custom_features:\n            self.train_features = self.train_df[self.custom_features].values\n            self.val_features = self.val_df[self.custom_features].values\n            self.test_features = self.test_df[self.custom_features].values\n\n    def get_xs_ys(self):\n        self.train_X = self.train_df["question_text"].values\n        self.val_X = self.val_df["question_text"].values\n        self.test_X = self.test_df["question_text"].values\n        self.train_y = self.train_df['target'].values\n        self.val_y = self.val_df['target'].values\n\n    def tokenize(self, max_feature=50000):\n        logging.info("Tokenizing...")\n        tokenizer = Tokenizer(num_words=max_feature)\n        tokenizer.fit_on_texts(list(self.train_X))\n        self.train_X = tokenizer.texts_to_sequences(self.train_X)\n        self.val_X = tokenizer.texts_to_sequences(self.val_X)\n        self.test_X = tokenizer.texts_to_sequences(self.test_X)\n        self.tokenizer = tokenizer\n        self.max_feature = max_feature\n\n    def append_features(self):\n        self.train_X = [self.train_X, np.array(self.train_df[self.custom_features])]\n        self.val_X = [self.val_X, np.array(self.val_df[self.custom_features])]\n        self.test_X = [self.test_X, np.array(self.test_df[self.custom_features])]\n\n    def pad_sequences(self, maxlen=100):\n        logging.info("Padding Sequences...")\n        self.train_X = pad_sequences(self.train_X, maxlen=maxlen)\n        self.val_X = pad_sequences(self.val_X, maxlen=maxlen)\n        self.test_X = pad_sequences(self.test_X, maxlen=maxlen)\n        self.maxlen = maxlen\n\n    def add_pseudo_data(self, pred_test_y):\n        logging.warning("Using pseudo data...")\n        self.full_X = np.vstack([self.train_X, self.val_X, self.test_X])\n        self.full_y = np.vstack([self.train_y.reshape((len(self.train_y), 1)),\n                                 self.val_y.reshape((len(self.val_y), 1)), pred_test_y])\n\n    @staticmethod\n    def clean_contractions(text):\n        contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",\n                               "could've": "could have", "couldn't": "could not", "didn't": "did not",\n                               "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",\n                               "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",\n                               "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",\n                               "I'd": "i would", "I'd've": "I would have", "I'll": "i will", "I'll've": "i will have",\n                               "I'm": "i am", "I've": "i have", "i'd": "i would", "i'd've": "i would have",\n                               "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have",\n                               "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will",\n                               "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",\n                               "mayn't": "may not", "might've": "might have", "mightn't": "might not",\n                               "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",\n                               "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",\n                               "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",\n                               "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",\n                               "she'd": "she would", "she'd've": "she would have", "she'll": "she will",\n                               "she'll've": "she will have", "she's": "she is", "should've": "should have",\n                               "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",\n                               "so's": "so as", "this's": "this is", "that'd": "that would",\n                               "that'd've": "that would have", "that's": "that is", "there'd": "there would",\n                               "there'd've": "there would have", "there's": "there is", "here's": "here is",\n                               "they'd": "they would", "they'd've": "they would have", "they'll": "they will",\n                               "they'll've": "they will have", "they're": "they are", "they've": "they have",\n                               "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",\n                               "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",\n                               "weren't": "were not", "what'll": "what will", "what'll've": "what will have",\n                               "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is",\n                               "when've": "when have", "where'd": "where did", "where's": "where is",\n                               "where've": "where have", "who'll": "who will", "who'll've": "who will have",\n                               "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",\n                               "will've": "will have", "won't": "will not", "won't've": "will not have",\n                               "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",\n                               "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",\n                               "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",\n                               "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",\n                               "you're": "you are", "you've": "you have"}\n        specials = ["\u2019", "\u2018", "\xb4", "`"]\n        for s in specials:\n            text = text.replace(s, "'")\n        text = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in text.split(" ")])\n        return text\n\n    @staticmethod\n    def clean_specials(text):\n        punct = "/-'?!.,#$%\\'()*+-/:;<=>@[\\\\]^_`{|}~" + '""\u201c\u201d\u2019' + '\u221e\u03b8\xf7\u03b1\u2022\xe0\u2212\u03b2\u2205\xb3\u03c0\u2018\u20b9\xb4\xb0\xa3\u20ac\\\xd7\u2122\u221a\xb2\u2014\u2013&'\n        puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=',\n                  '#', '*', '+', '\\\\', '\u2022', '~', '@', '\xa3',\n                  '\xb7', '_', '{', '}', '\xa9', '^', '\xae', '`', '<', '\u2192', '\xb0', '\u20ac', '\u2122', '\u203a', '\u2665', '\u2190', '\xd7', '\xa7', '\u2033', '\u2032',\n                  '\xc2', '\u2588', '\xbd', '\xe0', '\u2026',\n                  '\u201c', '\u2605', '\u201d', '\u2013', '\u25cf', '\xe2', '\u25ba', '\u2212', '\xa2', '\xb2', '\xac', '\u2591', '\xb6', '\u2191', '\xb1', '\xbf', '\u25be', '\u2550', '\xa6', '\u2551',\n                  '\u2015', '\xa5', '\u2593', '\u2014', '\u2039', '\u2500',\n                  '\u2592', '\uff1a', '\xbc', '\u2295', '\u25bc', '\u25aa', '\u2020', '\u25a0', '\u2019', '\u2580', '\xa8', '\u2584', '\u266b', '\u2606', '\xe9', '\xaf', '\u2666', '\xa4', '\u25b2', '\xe8',\n                  '\xb8', '\xbe', '\xc3', '\u22c5', '\u2018', '\u221e',\n                  '\u2219', '\uff09', '\u2193', '\u3001', '\u2502', '\uff08', '\xbb', '\uff0c', '\u266a', '\u2569', '\u255a', '\xb3', '\u30fb', '\u2566', '\u2563', '\u2554', '\u2557', '\u25ac', '\u2764', '\xef',\n                  '\xd8', '\xb9', '\u2264', '\u2021', '\u221a', ]\n        punct_mapping = {"\u2018": "'", "\u20b9": "e", "\xb4": "'", "\xb0": "", "\u20ac": "e", "\u2122": "tm", "\u221a": " sqrt ", "\xd7": "x", "\xb2": "2",\n                         "\u2014": "-", "\u2013": "-", "\u2019": "'", "_": "-", "`": "'", '\u201c': '"', '\u201d': '"', '\u201c': '"', "\xa3": "e",\n                         '\u221e': 'infinity', '\u03b8': 'theta', '\xf7': '/', '\u03b1': 'alpha', '\u2022': '.', '\xe0': 'a', '\u2212': '-',\n                         '\u03b2': 'beta', '\u2205': '', '\xb3': '3', '\u03c0': 'pi', }\n        for p in punct_mapping:\n            text = text.replace(p, punct_mapping[p])\n        for p in set(list(punct) + puncts) - set(punct_mapping.keys()):\n            text = text.replace(p, f' {p} ')\n\n        specials = {'\\u200b': ' ', '\u2026': ' ... ', '\\ufeff': '', '\u0915\u0930\u0928\u093e': '',\n                    '\u0939\u0948': ''}\n        for s in specials:\n            text = text.replace(s, specials[s])\n        return text\n\n    @staticmethod\n    def clean_spelling(text, case_sensitive=False):\n        misspell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling',\n                         'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',\n                         'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize',\n                         'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What',\n                         'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can',\n                         'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I',\n                         'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation',\n                         'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis',\n                         'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017',\n                         '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess',\n                         "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization',\n                         'demonitization': 'demonetization', 'demonetisation': 'demonetization'}\n        for word in misspell_dict.keys():\n            if case_sensitive:\n                text = text.replace(word, misspell_dict[word])\n            else:\n                re_insensitive = re.compile(re.escape(word), re.IGNORECASE)\n                text = re_insensitive.sub(misspell_dict[word], text)\n        return text\n\n    @staticmethod\n    def clean_acronyms(text, case_sensitive=False):\n        acronym_dict = {'upsc': 'union public service commission',\n                        'aiims': 'all india institute of medical sciences',\n                        'cgl': 'graduate level examination',\n                        'icse': 'indian school certificate exam',\n                        'iiit': 'indian institute of information technology',\n                        'cgpa': 'cumulative grade point average',\n                        'ielts': 'international english language training system',\n                        'ncert': 'national council of education research training',\n                        'isro': 'indian space research organization',\n                        'clat': 'common law admission test',\n                        'ibps': 'institute of banking personnel selection',\n                        'iiser': 'indian institute of science education and research',\n                        'iisc': 'indian institute of science',\n                        'iims': 'indian institutes of management',\n                        'cpec': 'china pakistan economic corridor'\n\n                        }\n        for word in acronym_dict.keys():\n            if case_sensitive:\n                text = text.replace(word, acronym_dict[word])\n            else:\n                re_insensitive = re.compile(re.escape(word), re.IGNORECASE)\n                text = re_insensitive.sub(acronym_dict[word], text)\n        return text\n\n    @staticmethod\n    def clean_non_dictionary(text, case_sensitive=False):\n        replace_dict = {'quorans': 'users',\n                        'quoran': 'user',\n                        'jio': 'phone manufacturer',\n                        'manipal': 'city',\n                        'bitsat': 'exam',\n                        'mtech': 'technical university',\n                        'pilani': 'town',\n                        'bhu': 'university',\n                        'h1b': 'visa',\n                        'redmi': 'phone manufacturer',\n                        'nift': 'university',\n                        'kvpy': 'exam',\n                        'thanos': 'comic villain',\n                        'paytm': 'payment system',\n                        'comedk': 'medical consortium',\n                        'accenture': 'management consulting company',\n                        'llb': 'bachelor of laws',\n                        'ignou': 'university',\n                        'dtu': 'university',\n                        'aadhar': 'social number',\n                        'lenovo': 'computer manufacturer',\n                        'gmat': 'exam',\n                        'kiit': 'institute of technology',\n                        'shopify': 'music streaming',\n                        'fitjee': 'exam',\n                        'kejriwal': 'politician',\n                        'wbjee': 'exam',\n                        'pgdm': 'master of business administration',\n                        'trudeau': 'politician',\n                        'nri': 'research institute',\n                        'deloitte': 'accounting company',\n                        'jinping': 'politician',\n                        'bcom': 'bachelor of commerce',\n                        'mcom': 'masters of commerce',\n                        'virat': 'athlete',\n                        'kcet': 'television network',\n                        'wipro': 'information technology company',\n                        'articleship': 'internship',\n                        'comey': 'law enforcement director',\n                        'jnu': 'university',\n                        'acca': 'chartered accountants',\n                        'aakash': 'phone manufacturer',\n                        'brexit': 'british succession',\n                        'crypto': 'digital currency',\n                        'cryptocurrency': 'digital currency',\n                        'cryptocurrencies': 'digital currencies',\n                        'etherium': 'digital currency',\n                        'bitcoin': 'digital currency',\n                        'viteee': 'exam',\n                        'iocl': 'indian oil company',\n                        'nmims': 'management school',\n                        'rohingya': 'myanmar people',\n                        'fortnite': 'videogame',\n                        'upes': 'university',\n                        'nsit': 'university',\n                        'coinbase': 'digital currency exchange'\n                        }\n        for word in replace_dict.keys():\n            if case_sensitive:\n                text = text.replace(word, replace_dict[word])\n            else:\n                re_insensitive = re.compile(re.escape(word), re.IGNORECASE)\n                text = re_insensitive.sub(replace_dict[word], text)\n        return text\n\n    @staticmethod\n    def clean_numbers(text, min_magnitude=2, max_magnitude=10):\n        for n in range(min_magnitude, max_magnitude):\n            text = re.sub('[0-9]{' + str(n) + '}', '#'*n, text)\n        return text\n\n    def add_features(self, df):\n        df['question_text'] = df['question_text'].apply(lambda x: str(x))\n        df['total_length'] = df['question_text'].apply(len)\n        df['capitals'] = df['question_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))\n        df['caps_vs_length'] = df.apply(lambda row: float(row['capitals']) / float(row['total_length']),\n                                        axis=1)\n        df['num_words'] = df.question_text.str.count('\\S+')\n        df['num_unique_words'] = df['question_text'].apply(\n            lambda comment: len(set(w for w in comment.split())))\n        df['words_vs_unique'] = df['num_unique_words'] / df['num_words']\n        df['caps_vs_length'] = df['caps_vs_length'].fillna(0)\n        df['words_vs_unique'] = df['words_vs_unique'].fillna(0)\n        df['num_exclamation_marks'] = df['question_text'].apply(lambda comment: comment.count('!'))\n        df['num_question_marks'] = df['question_text'].apply(lambda comment: comment.count('?'))\n        df['num_smilies'] = df['question_text'].apply(\n            lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))\n        self.custom_features = ['total_length', 'capitals', 'caps_vs_length', 'num_words',\n                                'num_unique_words', 'words_vs_unique', 'caps_vs_length',\n                                'words_vs_unique', 'num_exclamation_marks', 'num_question_marks', 'num_smilies']\n        return df\n\n    def scale_features(self):\n        self.feature_scaler = StandardScaler()\n        features = self.train_df[self.custom_features]\n        test_features = self.test_df[self.custom_features]\n        self.feature_scaler.fit(features)\n        self.train_df[self.custom_features] = self.feature_scaler.transform(features)\n        self.test_df[self.custom_features] = self.feature_scaler.transform(test_features)\n\n    def get_train_vocab(self):\n        sentences = self.train_df['question_text'].apply(lambda x: x.split()).values\n        vocab = {}\n        for sentence in sentences:\n            for word in sentence:\n                try:\n                    vocab[word] += 1\n                except KeyError:\n                    vocab[word] = 1\n        return vocab\n''')
    __stickytape_write_module('''src/__init__.py''', '''''')
    import gc
    import time
    
    import keras.backend as K
    import logging
    import matplotlib.pyplot as plt
    import numpy as np
    import operator
    import os
    import pandas as pd
    import random
    import tensorflow as tf
    import traceback
    import warnings
    from gensim.models import KeyedVectors
    from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
    from keras.engine import Layer
    from keras.layers import Bidirectional, CuDNNLSTM, initializers, regularizers, constraints, Reshape, Conv2D, MaxPool2D, \
        Concatenate, Flatten, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate
    from keras.layers import Dense, Input, Embedding as EmbeddingLayer, Dropout
    from keras.models import Model
    from sklearn import metrics
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import precision_recall_curve
    from sklearn.model_selection import StratifiedKFold
    
    from src.data import Data
    
    SEED = 42
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    
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
    
    
    class OneCycleLR(Callback):
        def __init__(self, num_samples, num_epochs, batch_size, max_lr,
                     end_percentage=0.1, scale_percentage=None,
                     maximum_momentum=0.95, minimum_momentum=0.85,
                     verbose=True):
            """ This callback implements a cyclical learning rate policy (CLR).
            This is a special case of Cyclic Learning Rates, where we have only 1 cycle.
            After the completion of 1 cycle, the learning rate will decrease rapidly to
            100th its initial lowest value.
            # Arguments:
                num_samples: Integer. Number of sample points in the dataset
                num_epochs: Integer. Number of training epochs
                batch_size: Integer. Batch size per training epoch
                max_lr: Float. Initial learning rate. This also sets the
                    starting learning rate (which will be 10x smaller than
                    this), and will increase to this value during the first cycle.
                end_percentage: Float. The percentage of all the epochs of training
                    that will be dedicated to sharply decreasing the learning
                    rate after the completion of 1 cycle. Must be between 0 and 1.
                scale_percentage: Float or None. If float, must be between 0 and 1.
                    If None, it will compute the scale_percentage automatically
                    based on the `end_percentage`.
                maximum_momentum: Optional. Sets the maximum momentum (initial)
                    value, which gradually drops to its lowest value in half-cycle,
                    then gradually increases again to stay constant at this max value.
                    Can only be used with SGD Optimizer.
                minimum_momentum: Optional. Sets the minimum momentum at the end of
                    the half-cycle. Can only be used with SGD Optimizer.
                verbose: Bool. Whether to print the current learning rate after every
                    epoch.
            # Reference
                - [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, weight_decay, and weight decay](https://arxiv.org/abs/1803.09820)
                - [Super-Convergence: Very Fast Training of Residual Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120)
            """
            super(OneCycleLR, self).__init__()
    
            if end_percentage < 0. or end_percentage > 1.:
                raise ValueError("`end_percentage` must be between 0 and 1")
    
            if scale_percentage is not None and (scale_percentage < 0. or scale_percentage > 1.):
                raise ValueError("`scale_percentage` must be between 0 and 1")
    
            self.num_samples = num_samples
            self.num_epochs = num_epochs
            self.batch_size = batch_size
            self.num_samples_per_batch = max(num_samples // batch_size, 3)
            self.initial_lr = max_lr
            self.end_percentage = end_percentage
            self.scale = float(scale_percentage) if scale_percentage is not None else float(end_percentage)
            self.max_momentum = maximum_momentum
            self.min_momentum = minimum_momentum
            self.verbose = verbose
    
            self.num_iterations = self.num_epochs * self.num_samples_per_batch
            self.mid_cycle_id = int(self.num_iterations * ((1. - end_percentage)) / float(2))
    
            if self.max_momentum is not None and self.min_momentum is not None:
                self._update_momentum = True
            else:
                self._update_momentum = False
    
            self.clr_iterations = 0.
            self.history = {}
    
        def _reset(self):
            """
            Reset the callback.
            """
            self.clr_iterations = 0.
            self.history = {}
    
        def compute_lr(self):
            """
            Compute the learning rate based on which phase of the cycle it is in.
            - If in the first half of training, the learning rate gradually increases.
            - If in the second half of training, the learning rate gradually decreases.
            - If in the final `end_percentage` portion of training, the learning rate
                is quickly reduced to near 100th of the original min learning rate.
            # Returns:
                the new learning rate
            """
            if self.clr_iterations > 2 * self.mid_cycle_id:
                current_percentage = (self.clr_iterations - 2 * self.mid_cycle_id)
                current_percentage /= float((self.num_iterations - 2 * self.mid_cycle_id))
                new_lr = self.initial_lr * (1. + (current_percentage * (1. - 100.) / 100.)) * self.scale
    
            elif self.clr_iterations > self.mid_cycle_id:
                current_percentage = 1. - (self.clr_iterations - self.mid_cycle_id) / self.mid_cycle_id
                new_lr = self.initial_lr * (1. + current_percentage * (self.scale * 100 - 1.)) * self.scale
    
            else:
                current_percentage = self.clr_iterations / self.mid_cycle_id
                new_lr = self.initial_lr * (1. + current_percentage * (self.scale * 100 - 1.)) * self.scale
    
            if self.clr_iterations == self.num_iterations:
                self.clr_iterations = 0
    
            return new_lr
    
        def compute_momentum(self):
            """
             Compute the momentum based on which phase of the cycle it is in.
            - If in the first half of training, the momentum gradually decreases.
            - If in the second half of training, the momentum gradually increases.
            - If in the final `end_percentage` portion of training, the momentum value
                is kept constant at the maximum initial value.
            # Returns:
                the new momentum value
            """
            if self.clr_iterations > 2 * self.mid_cycle_id:
                new_momentum = self.max_momentum
    
            elif self.clr_iterations > self.mid_cycle_id:
                current_percentage = 1. - ((self.clr_iterations - self.mid_cycle_id) / float(self.mid_cycle_id))
                new_momentum = self.max_momentum - current_percentage * (self.max_momentum - self.min_momentum)
    
            else:
                current_percentage = self.clr_iterations / float(self.mid_cycle_id)
                new_momentum = self.max_momentum - current_percentage * (self.max_momentum - self.min_momentum)
    
            return new_momentum
    
        def on_train_begin(self, logs={}):
            logs = logs or {}
    
            self._reset()
            K.set_value(self.model.optimizer.lr, self.compute_lr())
    
            if self._update_momentum:
                if not hasattr(self.model.optimizer, 'momentum'):
                    raise ValueError("Momentum can be updated only on SGD optimizer !")
    
                new_momentum = self.compute_momentum()
                K.set_value(self.model.optimizer.momentum, new_momentum)
    
        def on_batch_end(self, epoch, logs=None):
            logs = logs or {}
    
            self.clr_iterations += 1
            new_lr = self.compute_lr()
    
            self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
            K.set_value(self.model.optimizer.lr, new_lr)
    
            if self._update_momentum:
                if not hasattr(self.model.optimizer, 'momentum'):
                    raise ValueError("Momentum can be updated only on SGD optimizer !")
    
                new_momentum = self.compute_momentum()
    
                self.history.setdefault('momentum', []).append(K.get_value(self.model.optimizer.momentum))
                K.set_value(self.model.optimizer.momentum, new_momentum)
    
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)
    
        def on_epoch_end(self, epoch, logs=None):
            if self.verbose:
                if self._update_momentum:
                    print(" - lr: %0.5f - momentum: %0.2f " % (self.history['lr'][-1],
                                                               self.history['momentum'][-1]))
    
                else:
                    print(" - lr: %0.5f " % (self.history['lr'][-1]))
    
    
    class LRFinder(Callback):
    
        def __init__(self, num_samples, batch_size,
                     minimum_lr=1e-5, maximum_lr=10.,
                     lr_scale='exp', validation_data=None,
                     validation_sample_rate=5,
                     stopping_criterion_factor=4.,
                     loss_smoothing_beta=0.98,
                     save_dir=None, verbose=True):
            """
            This class uses the Cyclic Learning Rate history to find a
            set of learning rates that can be good initializations for the
            One-Cycle training proposed by Leslie Smith in the paper referenced
            below.
            A port of the Fast.ai implementation for Keras.
            # Note
            This requires that the model be trained for exactly 1 epoch. If the model
            is trained for more epochs, then the metric calculations are only done for
            the first epoch.
            # Interpretation
            Upon visualizing the loss plot, check where the loss starts to increase
            rapidly. Choose a learning rate at somewhat prior to the corresponding
            position in the plot for faster convergence. This will be the maximum_lr lr.
            Choose the max value as this value when passing the `max_val` argument
            to OneCycleLR callback.
            Since the plot is in log-scale, you need to compute 10 ^ (-k) of the x-axis
            # Arguments:
                num_samples: Integer. Number of samples in the dataset.
                batch_size: Integer. Batch size during training.
                minimum_lr: Float. Initial learning rate (and the minimum).
                maximum_lr: Float. Final learning rate (and the maximum).
                lr_scale: Can be one of ['exp', 'linear']. Chooses the type of
                    scaling for each update to the learning rate during subsequent
                    batches. Choose 'exp' for large range and 'linear' for small range.
                validation_data: Requires the validation dataset as a tuple of
                    (X, y) belonging to the validation set. If provided, will use the
                    validation set to compute the loss metrics. Else uses the training
                    batch loss. Will warn if not provided to alert the user.
                validation_sample_rate: Positive or Negative Integer. Number of batches to sample from the
                    validation set per iteration of the LRFinder. Larger number of
                    samples will reduce the variance but will take longer time to execute
                    per batch.
                    If Positive > 0, will sample from the validation dataset
                    If Megative, will use the entire dataset
                stopping_criterion_factor: Integer or None. A factor which is used
                    to measure large increase in the loss value during training.
                    Since callbacks cannot stop training of a model, it will simply
                    stop logging the additional values from the epochs after this
                    stopping criterion has been met.
                    If None, this check will not be performed.
                loss_smoothing_beta: Float. The smoothing factor for the moving
                    average of the loss function.
                save_dir: Optional, String. If passed a directory path, the callback
                    will save the running loss and learning rates to two separate numpy
                    arrays inside this directory. If the directory in this path does not
                    exist, they will be created.
                verbose: Whether to print the learning rate after every batch of training.
            # References:
                - [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, weight_decay, and weight decay](https://arxiv.org/abs/1803.09820)
            """
            super(LRFinder, self).__init__()
    
            if lr_scale not in ['exp', 'linear']:
                raise ValueError("`lr_scale` must be one of ['exp', 'linear']")
    
            if validation_data is not None:
                self.validation_data = validation_data
                self.use_validation_set = True
    
                if validation_sample_rate > 0 or validation_sample_rate < 0:
                    self.validation_sample_rate = validation_sample_rate
                else:
                    raise ValueError("`validation_sample_rate` must be a positive or negative integer other than o")
            else:
                self.use_validation_set = False
                self.validation_sample_rate = 0
    
            self.num_samples = num_samples
            self.batch_size = batch_size
            self.initial_lr = minimum_lr
            self.final_lr = maximum_lr
            self.lr_scale = lr_scale
            self.stopping_criterion_factor = stopping_criterion_factor
            self.loss_smoothing_beta = loss_smoothing_beta
            self.save_dir = save_dir
            self.verbose = verbose
    
            self.num_batches_ = num_samples // batch_size - 1
            self.current_lr_ = minimum_lr
    
            if lr_scale == 'exp':
                self.lr_multiplier_ = (maximum_lr / float(minimum_lr)) ** (1. / float(self.num_batches_))
            else:
                extra_batch = int((num_samples % batch_size) != 0)
                self.lr_multiplier_ = np.linspace(minimum_lr, maximum_lr, num=self.num_batches_ + extra_batch)
    
            # If negative, use entire validation set
            if self.validation_sample_rate < 0:
                self.validation_sample_rate = self.validation_data[0].shape[0] // batch_size
    
            self.current_batch_ = 0
            self.current_epoch_ = 0
            self.best_loss_ = 1e6
            self.running_loss_ = 0.
    
            self.history = {}
    
        def on_train_begin(self, logs=None):
            self.current_epoch_ = 1
            K.set_value(self.model.optimizer.lr, self.initial_lr)
    
            warnings.simplefilter("ignore")
    
        def on_epoch_begin(self, epoch, logs=None):
            self.current_batch_ = 0
    
            if self.current_epoch_ > 1:
                warnings.warn("\n\nLearning rate finder should be used only with a single epoch. "
                              "Hereafter, the callback will not measure the losses.\n\n")
    
        def on_batch_begin(self, batch, logs=None):
            self.current_batch_ += 1
    
        def on_batch_end(self, batch, logs=None):
            if self.current_epoch_ > 1:
                return
    
            if self.use_validation_set:
                X, Y = self.validation_data[0], self.validation_data[1]
    
                # use 5 random batches from test set for fast approximate of loss
                num_samples = self.batch_size * self.validation_sample_rate
    
                if num_samples > X.shape[0]:
                    num_samples = X.shape[0]
    
                idx = np.random.choice(X.shape[0], num_samples, replace=False)
                x = X[idx]
                y = Y[idx]
    
                values = self.model.evaluate(x, y, batch_size=self.batch_size, verbose=False)
                loss = values[0]
            else:
                loss = logs['loss']
    
            # smooth the loss value and bias correct
            running_loss = self.loss_smoothing_beta * loss + (1. - self.loss_smoothing_beta) * loss
            running_loss = running_loss / (1. - self.loss_smoothing_beta ** self.current_batch_)
    
            # stop logging if loss is too large
            if self.current_batch_ > 1 and self.stopping_criterion_factor is not None and (
                    running_loss > self.stopping_criterion_factor * self.best_loss_):
    
                if self.verbose:
                    print(" - LRFinder: Skipping iteration since loss is %d times as large as best loss (%0.4f)" % (
                        self.stopping_criterion_factor, self.best_loss_
                    ))
                return
    
            if running_loss < self.best_loss_ or self.current_batch_ == 1:
                self.best_loss_ = running_loss
    
            current_lr = K.get_value(self.model.optimizer.lr)
    
            self.history.setdefault('running_loss_', []).append(running_loss)
            if self.lr_scale == 'exp':
                self.history.setdefault('log_lrs', []).append(np.log10(current_lr))
            else:
                self.history.setdefault('log_lrs', []).append(current_lr)
    
            # compute the lr for the next batch and update the optimizer lr
            if self.lr_scale == 'exp':
                current_lr *= self.lr_multiplier_
            else:
                current_lr = self.lr_multiplier_[self.current_batch_ - 1]
    
            K.set_value(self.model.optimizer.lr, current_lr)
    
            # save the other metrics as well
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)
    
            if self.verbose:
                if self.use_validation_set:
                    print(" - LRFinder: val_loss: %1.4f - lr = %1.8f " % (values[0], current_lr))
                else:
                    print(" - LRFinder: lr = %1.8f " % current_lr)
    
        def on_epoch_end(self, epoch, logs=None):
            if self.save_dir is not None and self.current_epoch_ <= 1:
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
    
                losses_path = os.path.join(self.save_dir, 'losses.npy')
                lrs_path = os.path.join(self.save_dir, 'lrs.npy')
    
                np.save(losses_path, self.losses)
                np.save(lrs_path, self.lrs)
    
                if self.verbose:
                    print("\tLR Finder : Saved the losses and learning rate values in path : {%s}" % (self.save_dir))
    
            self.current_epoch_ += 1
    
            warnings.simplefilter("default")
    
        def plot_schedule(self, filename="lr_schedule.png", clip_beginning=None, clip_endding=None):
            """
            Plots the schedule from the callback itself.
            # Arguments:
                clip_beginning: Integer or None. If positive integer, it will
                    remove the specified portion of the loss graph to remove the large
                    loss values in the beginning of the graph.
                clip_endding: Integer or None. If negative integer, it will
                    remove the specified portion of the ending of the loss graph to
                    remove the sharp increase in the loss values at high learning rates.
            """
            try:
                import matplotlib.pyplot as plt
                plt.style.use('seaborn-white')
            except ImportError:
                print("Matplotlib not found. Please use `pip install matplotlib` first.")
                return
    
            if clip_beginning is not None and clip_beginning < 0:
                clip_beginning = -clip_beginning
    
            if clip_endding is not None and clip_endding > 0:
                clip_endding = -clip_endding
    
            losses = self.losses
            lrs = self.lrs
    
            if clip_beginning:
                losses = losses[clip_beginning:]
                lrs = lrs[clip_beginning:]
    
            if clip_endding:
                losses = losses[:clip_endding]
                lrs = lrs[:clip_endding]
    
            plt.plot(lrs, losses)
            plt.gca().set_yscale('log')
            plt.title('Learning rate vs Loss')
            plt.xlabel('log(learning rate)')
            plt.ylabel('log(loss)')
            plt.savefig(filename)
    
        @classmethod
        def restore_schedule_from_dir(cls, directory, clip_beginning=None, clip_endding=None):
            """
            Loads the training history from the saved numpy files in the given directory.
            # Arguments:
                directory: String. Path to the directory where the serialized numpy
                    arrays of the loss and learning rates are saved.
                clip_beginning: Integer or None. If positive integer, it will
                    remove the specified portion of the loss graph to remove the large
                    loss values in the beginning of the graph.
                clip_endding: Integer or None. If negative integer, it will
                    remove the specified portion of the ending of the loss graph to
                    remove the sharp increase in the loss values at high learning rates.
            Returns:
                tuple of (losses, learning rates)
            """
            if clip_beginning is not None and clip_beginning < 0:
                clip_beginning = -clip_beginning
    
            if clip_endding is not None and clip_endding > 0:
                clip_endding = -clip_endding
    
            losses_path = os.path.join(directory, 'losses.npy')
            lrs_path = os.path.join(directory, 'lrs.npy')
    
            if not os.path.exists(losses_path) or not os.path.exists(lrs_path):
                print("%s and %s could not be found at directory : {%s}" % (
                    losses_path, lrs_path, directory
                ))
    
                losses = None
                lrs = None
    
            else:
                losses = np.load(losses_path)
                lrs = np.load(lrs_path)
    
                if clip_beginning:
                    losses = losses[clip_beginning:]
                    lrs = lrs[clip_beginning:]
    
                if clip_endding:
                    losses = losses[:clip_endding]
                    lrs = lrs[:clip_endding]
    
            return losses, lrs
    
        @classmethod
        def plot_schedule_from_file(cls, directory, clip_beginning=None, clip_endding=None):
            """
            Plots the schedule from the saved numpy arrays of the loss and learning
            rate values in the specified directory.
            # Arguments:
                directory: String. Path to the directory where the serialized numpy
                    arrays of the loss and learning rates are saved.
                clip_beginning: Integer or None. If positive integer, it will
                    remove the specified portion of the loss graph to remove the large
                    loss values in the beginning of the graph.
                clip_endding: Integer or None. If negative integer, it will
                    remove the specified portion of the ending of the loss graph to
                    remove the sharp increase in the loss values at high learning rates.
            """
            try:
                import matplotlib.pyplot as plt
                plt.style.use('seaborn-white')
            except ImportError:
                print("Matplotlib not found. Please use `pip install matplotlib` first.")
                return
    
            losses, lrs = cls.restore_schedule_from_dir(directory,
                                                        clip_beginning=clip_beginning,
                                                        clip_endding=clip_endding)
    
            if losses is None or lrs is None:
                return
            else:
                plt.plot(lrs, losses)
                plt.title('Learning rate vs Loss')
                plt.xlabel('learning rate')
                plt.ylabel('loss')
                plt.show()
    
        @property
        def lrs(self):
            return np.array(self.history['log_lrs'])
    
        @property
        def losses(self):
            return np.array(self.history['running_loss_'])
    
    
    class Embedding:
        def __init__(self, data):
            self.embeddings_index = None
            self.nb_words = None
            self.embeddings_index = None
            self.embed_size = None
            self.embedding_matrix = None
            self.data = data
            self.name = None
    
        def load(self, embedding_file='../input/embeddings/glove.840B.300d/glove.840B.300d.txt'):
            logging.info("loading embedding : " + embedding_file)
            self.name = embedding_file.split('/')[3]
    
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
                logging.debug("len(self.embeddings_index.values()): "
                              + str(len(self.embeddings_index.values())))
                logging.debug("type(self.embeddings_index.values()[0]): "
                              + str(type(list(self.embeddings_index.values())[0])))
                logging.debug("first few self.embeddings_index.values(): "
                              + str(list(self.embeddings_index.values())[:5]))
                raise
            emb_mean, emb_std = all_embs.mean(), all_embs.std()
            self.embed_size = all_embs.shape[1]
    
            word_index = self.data.tokenizer.word_index
            self.nb_words = min(self.data.max_feature, len(word_index))
            self.embedding_matrix = np.random.normal(emb_mean, emb_std, (self.nb_words, self.embed_size))
            for word, i in word_index.items():
                if i >= self.nb_words:
                    continue
                embedding_vector = self.embeddings_index.get(word)
                if embedding_vector is not None:
                    self.embedding_matrix[i] = embedding_vector
            return self.embedding_matrix
    
        def check_coverage(self, vocab):
            known_words = {}
            unknown_words = {}
            nb_known_words = 0
            nb_unknown_words = 0
            for word in vocab.keys():
                try:
                    known_words[word] = self.embeddings_index[word]
                    nb_known_words += vocab[word]
                except:
                    unknown_words[word] = vocab[word]
                    nb_unknown_words += vocab[word]
                    pass
    
            logging.info('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))
            logging.info('Found embeddings for  {:.2%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))
            unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]
            return unknown_words
    
        def cleanup(self):
            logging.info("Releasing memory...")
            try:
                del self.embeddings_index, self.embedding_matrix
                gc.collect()
                time.sleep(10)
            except AttributeError:
                logging.warning('embeddings index not found. They were probably already cleaned up.')
    
        def cleanup_index(self):
            logging.info("Releasing memory...")
            try:
                del self.embeddings_index
                gc.collect()
                time.sleep(10)
            except AttributeError:
                logging.warning('embeddings index not found. They were probably already cleaned up.')
    
    
    class InsincereModel:
        def __init__(self, data, name=None, loss='binary_crossentropy'):
            self.data = data
            self.name = name
            self.embedding = None
            self.model = None
            self.history = None
            self.loss = loss
            self.lr_finder = None
    
        def load_embedding(self, embedding_file='../input/embeddings/glove.840B.300d/glove.840B.300d.txt'):
            self.embedding = Embedding(self.data)
            self.embedding.load(embedding_file)
    
        def set_embedding(self, embedding):
            if type(embedding) is str:
                self.load_embedding(embedding)
            else:
                self.embedding = embedding
    
        def blend_embeddings(self, embeddings, cleanup=False):
            """Average embedding matrix given list of embedding files."""
            if self.embedding is None:
                self.set_embedding(embeddings[0])
            embedding_matrices = list()
            for emb in embeddings:
                embedding_matrices.append(emb.embedding_matrix)
            blend = np.mean(embedding_matrices, axis=0)
            self.embedding.embedding_matrix = blend
            if cleanup:
                for e in embeddings:
                    e.cleanup()
            return blend
    
        def concat_embeddings(self, embeddings, cleanup=False):
            self.embedding.embedding_matrix = np.concatenate(tuple([e.embedding_matrix for e in embeddings]), axis=1)
            if cleanup:
                for e in embeddings:
                    e.cleanup()
            return self.embedding.embedding_matrix
    
        @staticmethod
        def f1_score(y_true, y_pred):
            def recall(y_true, y_pred):
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
                recall = true_positives / (possible_positives + K.epsilon())
                return recall
    
            def precision(y_true, y_pred):
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
                precision = true_positives / (predicted_positives + K.epsilon())
                return precision
    
            precision = precision(y_true, y_pred)
            recall = recall(y_true, y_pred)
            return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    
        def define_model(self):
            raise NotImplementedError
    
        def print(self):
            print(self.model.summary())
    
        def _get_callbacks(self, epochs, batch_size, minimum_lr=1e-8, maximum_lr=1.0e-1):
            num_samples = self.data.train_X.shape[0]
            self.lr_finder = LRFinder(num_samples, batch_size,
                                   minimum_lr, maximum_lr,
                                   # validation_data=(X_val, Y_val),
                                   lr_scale='exp', save_dir='.')
            lr_manager = OneCycleLR(num_samples, epochs, batch_size, maximum_lr,
                                    end_percentage = 0.1,
                                    maximum_momentum = 0.95, minimum_momentum = 0.85)
            check_point = ModelCheckpoint('model.hdf5', monitor="val_f1_score", mode="max",
                                          verbose=True, save_best_only=True)
            early_stop = EarlyStopping(monitor="val_f1_score", mode="max", patience=3, verbose=True)
            return [self.lr_finder, lr_manager, check_point, early_stop]
    
        def fit(self,
                train_indices=None,
                val_indices=None,
                pseudo_labels=False,
                batch_size=1024,
                epochs=10,
                save_curve=True,
                curve_file_suffix=None):
            logging.info("Fitting model...")
            if pseudo_labels:
                train_x, train_y = self.data.full_X, self.data.full_y
                val_x, val_y = self.data.val_X, self.data.val_y
                if self.data.custom_features:
                    train_features, val_features = self.data.train_features, self.data.test_features
            else:
                if train_indices is not None:
                    train_x = self.data.train_X[train_indices]
                    train_y = self.data.train_y[train_indices]
                    if self.data.custom_features:
                        train_features = self.data.train_features[train_indices]
                else:
                    train_x = self.data.train_X
                    train_y = self.data.train_y
                    if self.data.custom_features:
                        train_features = self.data.train_features
                if val_indices is not None:
                    val_x = self.data.train_X[val_indices]
                    val_y = self.data.train_y[val_indices]
                    if self.data.custom_features:
                        val_features = self.data.train_features[val_indices]
                else:
                    val_x = self.data.val_X
                    val_y = self.data.val_y
                    if self.data.custom_features:
                        val_features = self.data.val_features
            callbacks = self._get_callbacks(epochs, batch_size)
            if self.data.custom_features:
                train_x = [train_x, train_features]
                val_x = [val_x, val_features]
            self.history = self.model.fit(train_x,
                                          train_y,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          validation_data=(val_x, val_y),
                                          callbacks=callbacks)
            if save_curve:
                self.lr_finder.plot_schedule(filename="lr_schedule_" + str(self.name) + ".png")
                filename = 'training_curve'
                if self.name:
                    filename += '_' + self.name
                if curve_file_suffix:
                    filename += '_' + curve_file_suffix
                filename += '.png'
                self.print_curve(filename)
    
        def print_curve(self, filename='training_curve.png'):
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='best')
            plt.savefig(filename)
            plt.close()
    
        def predict(self, x, batch_size=1024):
            logging.info("Predicting ...")
            prediction = self.model.predict(x, batch_size=batch_size, verbose=1)
            return prediction
    
        def cleanup(self):
            self.embedding.cleanup()
    
    
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
    
    
    class Ensemble:
        def __init__(self, models):
            self.models = models
    
        def predict_average(self, X):
            logging.info("Predicting with ensemble average, size=" + str(len(self.models)))
            predictions = list()
            for m in self.models:
                predictions.append(m.predict(X))
                logging.debug(type(predictions[-1]))
            avg_pred = np.mean(predictions, axis=0)
            return avg_pred
    
        def predict_linear_regression(self, X_train, y_train, X_predict):
            predictions_train = [model.predict(X_train) for model in self.models]
            X = np.asarray(predictions_train)
            X = X[..., 0]
            reg = LinearRegression().fit(X.T, y_train)
            predictions_predict = [model.predict(X_predict) for model in self.models]
            prediction_lin_reg = np.sum([predictions_predict[i] * reg.coef_[i]
                                         for i in range(len(predictions_predict))], axis=0)
            return prediction_lin_reg
    
    
    def find_best_threshold(y_proba, y_true, plot=False):
        logging.info("Finding best threshold...")
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        thresholds = np.append(thresholds, 1.001)
        F = 2 / (1/precision + 1/recall)
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
    
    
    def cross_validate(model_class, data, embeddings, n_splits=3, show_wrongest=True):
        logging.info("Cross validating model {} using {} folds...".format(model_class.__name__, str(n_splits)))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        models = list()
        for i, (train, test) in enumerate(skf.split(data.train_X, data.train_y)):
            logging.info("Running Fold {} of {}".format(i + 1, n_splits))
            models.append(None)
            cv_name = model_class.__name__ + '_cv_' + str(i)
            models[-1] = model_class(data, name=cv_name)
            models[-1].blend_embeddings(embeddings)
            models[-1].define_model()
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
    
    
    def load_embeddings(data, embedding_files, keep_index=True):
        embeddings = list()
        for f in embedding_files:
            embeddings.append(Embedding(data))
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
    
    
    def main():
        embedding_files = [
                           # '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin',
                           '../input/embeddings/glove.840B.300d/glove.840B.300d.txt',
                           '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec',
                           '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
                          ]
        dev_size = 500  # set dev_size=None for full-scale runs
        data = Data()
        data.load(dev_size=dev_size)
        data.preprocessing(lower_case=True)
        embeddings = load_embeddings(data, embedding_files)
        save_unknown_words(data, embeddings, max_words=200)
        # models_all = LSTMModel(data=data)
        # models_all.blend_embeddings(embeddings)
        # models_all.define_model()
        # models_all.fit()
        # models_all = [models_all]
        models_lstm_attention_cv = cross_validate(LSTMModelAttention, data, embeddings)
        models_cnn_cv = cross_validate(CNNModel, data, embeddings)
        models_all = models_lstm_attention_cv + models_cnn_cv
        cleanup_models(models_all)
        ensemble_cv = Ensemble(models_all)
        train_X = [data.train_X]
        val_X = [data.val_X]
        test_X = [data.test_X]
        if data.custom_features:
            train_X += [data.train_features]
            val_X += [data.val_features]
            test_X += [data.test_features]
        pred_train_y = ensemble_cv.predict_linear_regression(train_X, data.train_y, train_X)
        thresh = find_best_threshold(pred_train_y, data.train_y)
        pred_val_y = ensemble_cv.predict_linear_regression(val_X, data.val_y, val_X)
        print_diagnostics(data.val_y, (pred_val_y > thresh).astype(int))
        pred_y_test = ensemble_cv.predict_linear_regression(val_X, data.val_y, test_X)
        write_predictions(data, pred_y_test, thresh)
    
    
    if __name__ == "__main__":
        logging.getLogger()
        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.DEBUG,
            datefmt='%Y-%m-%d %H:%M:%S')
        main()
        logging.info("Done!")
    