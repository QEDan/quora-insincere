import os

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

config_insincere_model = {'callbacks': {'checkpoint': {'monitor': 'val_f1_score',
                                                       'mode': 'max',
                                                       'verbose': True,
                                                       'save_best_only': True},
                                        'early_stopping': {'monitor': 'val_f1_score',
                                                           'mode': 'max',
                                                           'patience': 3,
                                                           'verbose': True}
                                        },
                          'fit': {'pseudo_labels': False,
                                  'batch_size': 1536,
                                  'epochs': 10,
                                  'save_curve': True
                                  },
                          'predict': {'batch_size': 1024,
                                      'verbose': True}
                          }

config_lrfinder = {'minimum_lr': 1e-5,
                   'maximum_lr': 10.0,
                   'lr_scale': 'exp',
                   'validation_sample_rate': 5,
                   'stopping_criterion_factor': 4,
                   'loss_smoothing_beta': 0.98,
                   'save_dir': None,
                   'verbose': True
                   }

config_one_cycle = {'end_percentage': 0.1,
                    'scale_percentage': None,
                    'maximum_momentum': 0.95,
                    'minimum_momentum': 0.85,
                    'max_lr': 1.0e-1,
                    'verbose': True}

config_main = {'embedding_files': [# '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin',
                                   '../input/embeddings/glove.840B.300d/glove.840B.300d.txt',
                                   # '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec',
                                   '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
                                  ],
               'dev_size': 500
              }


def generate_config(random_state,
                    config_data,
                    config_insincere_model,
                    config_lrfinder,
                    config_one_cycle,
                    config_main,
                    path_config=os.path.join('src', 'config.py')):
    with open(path_config, 'w') as f:
        f.write('random_state = {}\n\n'.format(random_state))
        f.write('config_data = {}\n\n'.format(config_data))
        f.write('config_insincere_model = {}\n\n'.format(config_insincere_model))
        f.write('config_lrfinder = {}\n\n'.format(config_lrfinder))
        f.write('config_one_cycle = {}\n\n'.format(config_one_cycle))
        f.write('config_main = {}\n\n'.format(config_main))
    return


generate_config(random_state, config_data, config_insincere_model, config_lrfinder, config_one_cycle, config_main)