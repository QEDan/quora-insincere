"""Config file is in python for compatibility with stickytape and kernel submissions. Import configs as needed."""

random_state = 2018

config_data = {'preprocess': {'lower_case': False,
                              'remove_stop_words': False,
                              'remove_contractions': True,
                              'remove_specials':True,
                              'correct_spelling': True,
                              'replace_acronyms': True,
                              'replace_non_words': True,
                              'replace_numbers': False,
                              'use_custom_features': True},
               'test_size': 0.1,
               'max_feature': 120000,
               'max_seq_len': 72
               }

config_lrfinder = {}

config_attention = {}