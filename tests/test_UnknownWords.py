import unittest
import numpy as np
import keras
import spacy

from src.UnknownWords import UnknownWords
from src.data_mappers import TextMapper


class TestEmbedding:
    def __init__(self):
        self.embeddings_index = {'aaa': np.array([1.0, 1.0, 1.0]),
                                'aab': np.array([1.0, 1.1, 1.1]),
                                'aac': np.array([1.0, 1.1, 1.2])}
        self.embedding_matrix = np.array([self.embeddings_index[k] for k in self.embeddings_index.keys()])
        self.embedding_matrix = np.vstack([self.embedding_matrix, np.zeros((2, 3))])
        self.unknown_words = ['unk', 'unc']
        self.embed_size = 3
        self.word_vocab = ['aaa', 'aab', 'aac', 'unk', 'unc']


class TestUnknownWords(unittest.TestCase):
    def setUp(self):
        self.embedding = TestEmbedding()
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])
        self.text_mapper = TextMapper(word_counts=[('aaa', 1), ('aab', 1), ('aac', 1)],
                                      char_counts=[('a', 1), ('b', 1), ('c', 1)],
                                      word_threshold=10, max_word_len=3,
                                      char_threshold=350, max_sent_len=3, nlp=self.nlp, word_lowercase=True,
                                      char_lowercase=True)
        self.unknown_words_model = UnknownWords(self.text_mapper, self.embedding, max_words=3)

    def test_define_model(self):
        model = self.unknown_words_model.define_model()

        self.assertTrue(type(model) == keras.Model)

    def test_sample_training_data(self):
        sentences = ['aaa aaa aab', 'aaa aab aab', 'aac aab aac']
        train_X, val_X, train_y, val_y = self.unknown_words_model.sample_training_data(sentences)

        self.assertGreater(len(train_X), 0)
        self.assertGreater(len(val_X), 0)
        self.assertGreater(len(train_y), 0)
        self.assertGreater(len(val_y), 0)
        self.assertEqual(train_X.shape, (2, 3))
        self.assertEqual(val_X.shape, (1, 3))
        self.assertEqual(val_y.shape, (1, 3))
        self.assertEqual(train_y.shape, (2, 3))

    def test_fit(self):
        sentences = ['aaa aaa aab', 'aaa aab aab', 'aac aab aac']
        self.unknown_words_model.define_model()
        self.unknown_words_model.fit(sentences)

    def test_predict(self):
        sentences = ['aaa aaa aab', 'aaa aab aab', 'aac aab aac']
        self.unknown_words_model.define_model()
        self.unknown_words_model.fit(sentences)
        preds = self.unknown_words_model.predict(['aad'])

        self.assertEqual(type(preds), np.ndarray)
        self.assertEqual(preds.shape, (1, 3))

    def test_improve_embedding(self):
        sentences = ['aaa aaa aab', 'aaa aab aab', 'aac aab aac', 'unk aaa unc']
        self.unknown_words_model.define_model()
        self.unknown_words_model.fit(sentences)
        self.unknown_words_model.improve_embedding()
        self.unknown_words_model.embedding.embedding_matrix
        self.assertTrue(any(self.unknown_words_model.embedding.embedding_matrix[-1] !=
                            np.zeros(self.unknown_words_model.embedding.embed_size)))

