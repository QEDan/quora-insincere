import gc
import time

import logging
import numpy as np
import operator
import traceback
from gensim.models import KeyedVectors


class Embedding:
    def __init__(self, data):
        self.embeddings_index = None
        self.nb_words = None
        self.embeddings_index = None
        self.embed_size = None
        self.embedding_matrix = None
        self.data = data
        self.name = None
        self.lower_case = False

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
            self.lower_case = True
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
            if self.lower_case:
                word = word.lower()
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
            if self.lower_case:
                word_key = word.lower()
            else:
                word_key = word
            try:
                known_words[word] = self.embeddings_index[word_key]
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
