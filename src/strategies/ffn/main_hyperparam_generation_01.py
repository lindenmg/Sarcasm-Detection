import os

from src.hyperparameters.grid_search.grid_param_iterator import GridParameterIterator
from src.tools.config import Config

hyperparams_folder = os.path.join(Config.path.project_root_folder,
                                  'src', 'strategies', 'ffn', 'hyperparameter', 'session_01')

variation_list = [
    {
        'keys': ['optimizer', 'args', 'lr'],
        'val': [0.005, 0.003, 0.001]
    }
]

filename_keys = [['model', 'class_name'],
                 ['data_factory', 'args', 'pp_params', 'conversion', 'token_kind'],
                 ['data_factory', 'args', 'pp_params', 'vectorization', 'tfidf'],
                 ['data_factory', 'args', 'pp_params', 'vectorization', 'min_df'],
                 ['data_factory', 'args', 'pp_params', 'vectorization', 'analyzer'],
                 ['data_factory', 'args', 'pp_params', 'vectorization', 'ngram_range'],
                 ['data_factory', 'args', 'pp_params', 'vectorization', 'max_features'],
                 ['model', 'args', 'reply_input_size']]

default_ffn_params = {
    "learning_session": {
        "session_tag": "ffn_test_00",
        "epochs": 10,
        "mode": "train",
        "save_interval": 5,
        "cnn_mode": False,
        "cache_prefix": "fnn_fact",
        "cache_folder": Config.path.cache_folder,
        "seed": 1337
    },
    "logger": {
        "module": "src.strategies.ffn.ffn_mongo_tensorboard_logger",
        "class_name": "FFNMongoTensorboardLogger",
        "args": {
            "connectTimeoutMS": 5000,
            "serverSelectionTimeoutMS": 5000
        }
    },
    "metrics": {
        "output_activation": "log_softmax"
    },
    "model": {
        "module": "src.models.ffn_model",
        "class_name": "FFN02",
        "args": {
            "post_input_size": 15000,
            "reply_input_size": 15000,
            "post_layer_size": 50,
            "reply_layer_size": 150,
            "post_dropout": 0.3,
            "reply_dropout": 0.3,
            "output_size": 2
        }
    },
    "criterion": {
        "module": "torch.nn",
        "class_name": "NLLLoss",
        "args": {
            "size_average": True
        }
    },
    "optimizer": {
        "module": "torch.optim",
        "class_name": "ASGD",
        "args": {
            "lr": 0.01
        }
    },
    "lr_scheduler": {
        "module": "torch.optim.lr_scheduler",
        "class_name": "ReduceLROnPlateau",
        "args": {
            "mode": "min",
            "factor": 0.95,
            "patience": 10,
            "threshold_mode": "abs",
            "threshold": 0.002
        }
    },
    "cv_iterator_factory": {
        "module": "src.training.base_cv_it_factory",
        "class_name": "BaseCvIteratorFactory",
        "args": {
            "n_splits": 2,
            "split_ratio": 0.9,
            "batch_size": 200,
            "shuffle": False,
            "num_workers": 4,
            "dataset_module": "src.strategies.ffn.ffn_dataset",
            "dataset_classname": "FFNDataset"
        }
    },
    "data_factory": {
        "module": "src.strategies.ffn.ffn_data_factory",
        "class_name": "FFNDataFactory",
        "args": {
            "pp_params": {
                "sw_cut_filename": "stop_words_cut_ultra.txt",
                "sw_full_filename": "stop_words_full_ultra.txt",
                "raw_data": {
                    "n_replies": None
                },
                "tokenization": {
                    "spacy_model": "en_core_web_md"
                },
                "filter": {
                    "no_stop_words": True,
                    "no_punctuation": False
                },
                "conversion": {
                    "token_kind": "lower_",
                    "transform_specials": True
                },
                "vectorization": {
                    "tfidf": True,
                    "min_df": 10,
                    "ngram_range": [
                        2,
                        4
                    ],
                    "analyzer": "char",
                    "max_features": 15000
                }
            }
        }
    }
}

if __name__ == '__main__':
    generator = GridParameterIterator(
        default_dict=default_ffn_params, variation_list=variation_list, prefix="ffn01",
        enumerate_digits=3, keys=filename_keys, filename_path=['learning_session', 'session_tag'])
    generator.dump_hyperparameters(hyperparams_folder)
    print('finished')
