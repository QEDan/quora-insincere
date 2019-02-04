import numpy as np
import string
from keras.preprocessing.text import text_to_word_sequence
import math


class TextMapper:
    """ Maps text into model input x """

    def __init__(self, word_counts, char_counts, nlp, max_sent_len=100, word_threshold=10, word_lowercase=False,
                 max_word_len=12, char_threshold=5, char_lowercase=False):
        """
        :param word_counts: list of tuples of (word, count)
        :param nlp: from pre-loaded spaCy
        :param max_sent_len: maximum words in a doc
        :param word_threshold: number of times a word must appear in word_counts for it to get a representation
        :param word_lowercase: boolean, whether words should be lowercased
        """
        self.max_sent_len = max_sent_len
        self.max_word_len = max_word_len
        self.word_mapper = WordMapper(word_counts=word_counts, threshold=word_threshold,
                                      word_lowercase=word_lowercase)
        self.char_mapper = CharMapper(char_counts=char_counts, threshold=char_threshold,
                                      char_lowercase=char_lowercase)

        self.max_sent_len = max_sent_len
        self.max_word_len = max_word_len
        self.nlp = nlp
        self.num_sent_feats = 4

    def text_to_x(self, text):
        """ Handles mapping one text doc into model inputs """
        words_x = np.zeros(self.max_sent_len)
        words_feats_x = np.zeros((self.max_sent_len, self.word_mapper.num_add_feats))
        chars_x = np.zeros((self.max_sent_len, self.max_word_len))
        chars_feats_x = np.zeros((self.max_sent_len, self.max_word_len, self.char_mapper.num_add_feats))
        sent_feats_x = np.zeros(self.num_sent_feats)

        tokenized_question = self.nlp(text)

        word_lens = []
        sent_len = 0
        unknown_chars = 0
        unknown_words = 0
        total_chars = 0

        for word_ind, token in enumerate(tokenized_question[:self.max_sent_len]):
            sent_len += 1
            word = token.text
            word_lens.append(len(word))
            word_mapping = self.word_mapper.get_symbol_index(word)
            if word_mapping == 1:
                unknown_words += 1
            words_x[word_ind] = word_mapping
            words_feats_x[word_ind] = self.word_mapper.get_add_features(word)
            for char_ind, char in enumerate(word[:self.max_word_len]):
                total_chars += 1
                char_mapping = self.char_mapper.get_symbol_index(char)
                chars_x[word_ind][char_ind] = char_mapping
                if char_mapping == 1:
                    unknown_chars += 1
                chars_feats_x[word_ind][char_ind] = self.char_mapper.get_add_features(char)

        log_sent_length = math.log(sent_len)
        sent_feats_x[0] = log_sent_length

        num_caps = sum(1 for c in text if c.isupper())
        proportion_caps = num_caps/total_chars
        sent_feats_x[1] = proportion_caps

        # avg_word_length = np.mean(word_lens)
        # sent_feats_x[2] = avg_word_length
        #
        # std_word_length = np.std(word_lens)
        # sent_feats_x[3] = std_word_length

        proportion_unknown_chars = unknown_chars/total_chars
        sent_feats_x[2] = proportion_unknown_chars

        proportion_unknown_words = unknown_words/sent_len
        sent_feats_x[3] = proportion_unknown_words

        # if len(tokenized_question) > sent_len:
        #     sent_feats_x[6] = 1

        return words_x, words_feats_x, chars_x, chars_feats_x, sent_feats_x

    def x_to_words(self, words_x, remove_padding=True):
        words = [self.word_mapper.ix_to_symbol[int(i)] for i in words_x]
        comment_text = " ".join(words)

        # remove padding
        if remove_padding:
            comment_text = comment_text.split(self.word_mapper.PADDING_SYMBOL)[0]

        return comment_text

    def x_to_chars(self, chars_x, remove_padding=True):
        chars = []
        for word_x in chars_x:
            word_chars = [self.char_mapper.ix_to_symbol[x] for x in word_x]
            word = "".join(word_chars)
            # remove_padding
            if remove_padding:
                word = word.split(self.char_mapper.PADDING_SYMBOL)[0]

            # don't add empty words
            if word:
                chars.append(word)

        question_chars = " ".join(chars)

        return question_chars

    def texts_to_x(self, texts):
        inputs_x = [self.text_to_x(text) for text in texts]
        words_input, words_feats_input, chars_input, chars_feats_input, sent_feats_input = map(np.array, zip(*inputs_x))
        return {"words_input": words_input, "words_feats_input": words_feats_input,
                "chars_input": chars_input, "chars_feats_input": chars_feats_input,
                "sent_feats_input": sent_feats_input}

    def set_max_sentence_len(self, max_sent_len):
        self.max_sent_len = max_sent_len

    def set_max_char_len(self, max_word_len):
        self.max_word_len = max_word_len

    def get_words_vocab(self):
        return self.word_mapper.vocab


class SymbolMapper:
    """ Handles mapping of any symbol (words or characters) into something an model an ingest """
    PADDING_SYMBOL = "<PAD>"
    UNKNOWN_SYMBOL = "<UNK>"
    BASE_ALPHABET = [PADDING_SYMBOL, UNKNOWN_SYMBOL]

    def __init__(self, symbol_counts, threshold, lowercase):
        self.symbol_counts = symbol_counts
        self.threshold = threshold
        self.lowercase = lowercase

        self.vocab = []
        self.symbol_to_ix = dict()
        self.ix_to_symbol = dict()

        symbol_counts = sorted(symbol_counts, key=lambda x: x[1], reverse=True)
        self.vocab = [symbol for symbol, count in symbol_counts if count >= self.threshold]
        self.vocab = self.BASE_ALPHABET + self.vocab

        self.init_mappings(threshold)

    def init_mappings(self, check_coverage=True):

        self.symbol_to_ix = {symbol: ix for ix, symbol in enumerate(self.vocab)}
        self.ix_to_symbol = {ix: symbol for ix, symbol in enumerate(self.vocab)}

        if check_coverage:
            self.print_coverage_statistics()

    def print_top_n_symbols(self, n=10):
        print([(symbol, count) for symbol, count in self.symbol_counts if count >= self.threshold][:n])

    def print_bot_n_symbols(self, n=10):
        print([(symbol, count) for symbol, count in self.symbol_counts if count >= self.threshold][-n:])

    def get_symbol_index(self, symbol):
        if self.lowercase:
            symbol = symbol.lower()
        try:
            num = self.symbol_to_ix[symbol]
        except KeyError:
            num = self.symbol_to_ix[self.UNKNOWN_SYMBOL]
        return num

    def get_vocab_len(self):
        return len(self.symbol_to_ix)

    def print_coverage_statistics(self, symbols_name='symbol'):
        """
        Simple metric on coverage of symbols

        :param symbols_name: str, printed to distinguish different mappers
        :param persist: bool, write stats to file rather than stdout
        """
        symbol_mappings = self.symbol_to_ix.keys()
        print("Number of unique {}: {}".format(symbols_name, len(self.symbol_counts)))
        print("Number of unique {} mapped: {}".format(symbols_name, len(symbol_mappings)))
        total_tokens = 0
        mapped_tokens = 0
        for symbol, count in self.symbol_counts:
            total_tokens += count
            if symbol in symbol_mappings:
                mapped_tokens += count
        print("Percent of unique symbols mapped: {}%".format(
            100 * len(symbol_mappings) / len(self.symbol_counts)))
        print("Percent of total symbols mapped: {}%".format(
            100 * mapped_tokens / total_tokens))


class WordMapper(SymbolMapper):

    def __init__(self, word_counts, threshold, word_lowercase):
        super().__init__(word_counts, threshold, word_lowercase)
        self.num_add_feats = 1

    def print_coverage_statistics(self, symbols_name='words'):
        super().print_coverage_statistics(symbols_name)

    def get_add_features(self, word):
        add_feats = np.zeros(self.num_add_feats)
        add_feats[0] = len(word) / 10
        # if char in self.puncutation_list:
        #     add_feats[2] = 1
        return add_feats


class CharMapper(SymbolMapper):

    def __init__(self, char_counts, threshold, char_lowercase, letters_only=False):
        super().__init__(char_counts, threshold, char_lowercase)
        self.num_add_feats = 2
        if letters_only:
            self.init_letter_mapping()

    def init_letter_mapping(self):
        self.vocab = self.BASE_ALPHABET + list('abcdefghijklmnopqrstuvwxyz')
        self.symbol_to_ix = {symbol: ix for ix, symbol in enumerate(self.vocab)}
        self.ix_to_symbol = {ix: symbol for ix, symbol in enumerate(self.vocab)}

    def print_coverage_statistics(self, symbols_name='chars'):
        super().print_coverage_statistics(symbols_name)

    def get_add_features(self, char):
        add_feats = np.zeros(self.num_add_feats)
        if char.isupper():
            add_feats[0] = 1
        if char.isdigit():
            add_feats[1] = 1
        # if char in self.puncutation_list:
        #     add_feats[2] = 1
        return add_feats
