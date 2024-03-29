import numpy as np
import string
from keras.preprocessing.text import text_to_word_sequence


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
        self.word_mapper = WordMapper(word_counts=word_counts, threshold=word_threshold, max_sent_len=max_sent_len,
                                      word_lowercase=word_lowercase)
        self.char_mapper = CharMapper(char_counts=char_counts, threshold=char_threshold, max_word_len=max_word_len,
                                      char_lowercase=char_lowercase)

        self.max_sent_len = max_sent_len
        self.max_word_len = max_word_len
        self.nlp = nlp

    def text_to_x(self, text):
        """ Handles mapping one text doc into model inputs """
        words_x = np.zeros(self.max_sent_len)
        chars_x = np.zeros((self.max_sent_len, self.max_word_len))
        chars_feats_x = np.zeros((self.max_sent_len, self.max_word_len, self.char_mapper.num_add_feats))

        tokenized_question = self.nlp(text)
        # tokenized_question = text_to_word_sequence(text, lower=False)

        for word_ind, token in enumerate(tokenized_question[:self.max_sent_len]):
            word = token.text
            # word = token
            words_x[word_ind] = self.word_mapper.get_symbol_index(word)
            for char_ind, char in enumerate(word[:self.max_word_len]):
                chars_x[word_ind][char_ind] = self.char_mapper.get_symbol_index(char)
                chars_feats_x[word_ind][char_ind] = self.char_mapper.get_add_features(char)
        return words_x, chars_x, chars_feats_x

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
        words_input, chars_input, char_feats_input = map(np.array, zip(*inputs_x))
        return {"words_input": words_input, "chars_input": chars_input, "chars_feats_input": char_feats_input}

    def set_max_sentence_len(self, max_sent_len):
        self.word_mapper.set_max_len(max_sent_len)

    def set_max_char_len(self, max_char_len):
        self.char_mapper.set_max_len(max_char_len)

    def get_words_vocab(self):
        return self.word_mapper.vocab


class SymbolMapper:
    """ Handles mapping of any symbol (words or characters) into something an model an ingest """
    PADDING_SYMBOL = "<PAD>"
    UNKNOWN_SYMBOL = "<UNK>"
    BASE_ALPHABET = [PADDING_SYMBOL, UNKNOWN_SYMBOL]

    def __init__(self, symbol_counts, threshold, max_len, lowercase):
        self.symbol_counts = symbol_counts
        self.threshold = threshold
        self.max_len = max_len
        self.lowercase = lowercase

        self.vocab = []
        self.symbol_to_ix = dict()
        self.ix_to_symbol = dict()

        self.init_mappings(threshold)

    def init_mappings(self, check_coverage=True):
        symbol_counts = sorted(self.symbol_counts, key=lambda x: x[1], reverse=True)
        self.vocab = [symbol for symbol, count in symbol_counts if count >= self.threshold]
        self.vocab = self.BASE_ALPHABET + self.vocab

        self.symbol_to_ix = {symbol: ix for ix, symbol in enumerate(self.vocab)}
        self.ix_to_symbol = {ix: symbol for ix, symbol in enumerate(self.vocab)}

        if check_coverage:
            self.print_coverage_statistics()

    def set_threshold(self, threshold, check_coverage=True):
        self.threshold = threshold
        self.init_mappings(check_coverage)

    def set_max_len(self, max_len):
        self.max_len = max_len

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

    def print_coverage_statistics(self, symbols_name='symbol', persist=True):
        """
        Simple metric on coverage of symbols

        :param symbols_name: str, printed to distinguish different mappers
        :param persist: bool, write stats to file rather than stdout
        """
        symbol_mappings = self.symbol_to_ix.keys()
        with open('coverage_stats.txt', 'w') if persist else None as f:
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

    def __init__(self, word_counts, threshold, max_sent_len, word_lowercase):
        super().__init__(word_counts, threshold, max_sent_len, word_lowercase)

    def print_coverage_statistics(self, symbols_name='words'):
        super().print_coverage_statistics(symbols_name)


class CharMapper(SymbolMapper):

    def __init__(self, char_counts, threshold, max_word_len, char_lowercase):
        super().__init__(char_counts, threshold, max_word_len, char_lowercase)
        self.num_add_feats = 2
        # self.puncutation_list = string.punctuation

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
