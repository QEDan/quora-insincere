import numpy as np
import os

from src.generate_config import generate_config


def random_config_data():
    max_feature = np.random.choice([50000, 100000, 120000, 200000])
    max_seq_len = np.random.choice([50, 75, 100, 200])
    config_data = {'preprocess': {'lower_case': True,
                                  'remove_stop_words': False,
                                  'remove_contractions': True,
                                  'remove_specials': True,
                                  'correct_spelling': True,
                                  'replace_acronyms': True,
                                  'replace_non_words': True,
                                  'replace_numbers': False,
                                  'use_custom_features': True},
                   'test_size': 0.1,
                   'max_feature': max_feature,
                   'max_seq_len': max_seq_len
                   }
    return config_data


def random_config_insincere_model():
    early_stopping_patience = np.random.choice([2, 3, 5])
    fit_batch_size = np.random.choice([1000, 1536, 2000])
    fit_epochs = np.random.choice([5, 10, 15])
    predict_batch_size = np.random.choice([1024, 1536, 2048])
    config_insincere_model = {'callbacks': {'checkpoint': {'monitor': 'val_f1_score',
                                                           'mode': 'max',
                                                           'verbose': True,
                                                           'save_best_only': True},
                                            'early_stopping': {'monitor': 'val_f1_score',
                                                               'mode': 'max',
                                                               'patience': early_stopping_patience,
                                                               'verbose': True}
                                            },
                              'fit': {'pseudo_labels': False,
                                      'batch_size': fit_batch_size,
                                      'epochs': fit_epochs,
                                      'save_curve': True
                                      },
                              'predict': {'batch_size': predict_batch_size,
                                          'verbose': True}
                              }
    return config_insincere_model


def random_config_lrfinder():
    minimum_lr = np.random.choice([1.0e-6, 1.0e-5, 1.0e-4])
    maximum_lr = np.random.choice([10.0])
    config_lrfinder = {'minimum_lr': minimum_lr,
                       'maximum_lr': maximum_lr,
                       'lr_scale': 'exp',
                       'validation_sample_rate': 5,
                       'stopping_criterion_factor': 4,
                       'loss_smoothing_beta': 0.98,
                       'save_dir': None,
                       'verbose': True
                       }
    return config_lrfinder


def random_config_one_cycle():
    end_percentage = np.random.choice([0.1])
    max_lr = np.random.choice([1.0e-1])
    config_one_cycle = {'end_percentage': end_percentage,
                        'scale_percentage': None,
                        'maximum_momentum': 0.95,
                        'minimum_momentum': 0.85,
                        'max_lr': max_lr,
                        'verbose': True}
    return config_one_cycle


def random_configs():
    random_state = 2018

    config_data = random_config_data()

    config_insincere_model = random_config_insincere_model()

    config_lrfinder = random_config_lrfinder()

    config_one_cycle = random_config_one_cycle()

    config_main = {
        'embedding_files': [  # '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin',
            '../input/embeddings/glove.840B.300d/glove.840B.300d.txt',
            '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec',
            # '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
        ],
        'dev_size': None,
        'models': [{'class': 'LSTMModelAttention',
                   'args': None
                    },
                   {'class': 'CNNModel',
                    'args': None}
                   ]
    }

    return random_state, config_data, config_insincere_model, \
        config_lrfinder, config_one_cycle, config_main


if __name__ == "__main__":
    search_size = 10
    config_list = [[{'args':
                         {'dense_size_1': 64, 'dense_size_2': 32,
                          'dropout_rate': 0.1, 'lstm_size': 128},
                     'class':
                         'LSTMModelAttention'}]]
    for models in config_list:
        for i in range(search_size):
            random_state, config_data, config_insincere_model, \
                config_lrfinder, config_one_cycle, config_main \
                = random_configs()
            config_main['models'] = models
            generate_config(random_state, config_data, config_insincere_model,
                            config_lrfinder, config_one_cycle, config_main)
            print('launching configuration number {}'.format(str(i)))
            os.system('./generate_script.sh')
            os.system('kaggle kernels push')