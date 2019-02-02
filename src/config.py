"""Model and hyperparameter configurations. This file is generated by generate_config(). Do not edit!"""


random_state = 2018

config_data = {'max_feature': 50000,
 'max_seq_len': 200,
 'preprocess': {'correct_spelling': True,
                'lower_case': False,
                'remove_contractions': True,
                'remove_specials': True,
                'remove_stop_words': False,
                'replace_acronyms': True,
                'replace_non_words': True,
                'replace_numbers': False,
                'use_custom_features': True},
 'test_size': 0.1}

config_insincere_model = {'callbacks': {'checkpoint': {'mode': 'max',
                              'monitor': 'val_f1_score',
                              'save_best_only': True,
                              'verbose': True},
               'early_stopping': {'mode': 'max',
                                  'monitor': 'val_f1_score',
                                  'patience': 5,
                                  'verbose': True}},
 'fit': {'batch_size': 1000,
         'epochs': 5,
         'pseudo_labels': False,
         'save_curve': True},
 'predict': {'batch_size': 1024, 'verbose': True}}

config_lrfinder = {'loss_smoothing_beta': 0.98,
 'lr_scale': 'exp',
 'maximum_lr': 10.0,
 'minimum_lr': 1e-05,
 'save_dir': None,
 'stopping_criterion_factor': 4,
 'validation_sample_rate': 5,
 'verbose': True}

config_one_cycle = {'end_percentage': 0.1,
 'max_lr': 0.1,
 'maximum_momentum': 0.95,
 'minimum_momentum': 0.85,
 'scale_percentage': None,
 'verbose': True}

config_main = {'dev_size': None,
 'embedding_files': ['../input/embeddings/glove.840B.300d/glove.840B.300d.txt',
                     '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt']}

