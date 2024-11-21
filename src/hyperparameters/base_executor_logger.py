import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from pprint import pprint

import pandas as pd

from src.hyperparameters.abstract_executor_logger import AbstractExecutorLogger
from src.hyperparameters.param_operation import filter_by_keys_lists, flatten_dict_auto, move_col_in_df
from src.tools.config import Config


class BaseExecutorLogger(AbstractExecutorLogger):
    def __init__(self, variation_list, iterations):
        pd.set_option('display.width', 1000)
        self.cache_dir = None
        self.set_cache_dir()
        self.variation_list = variation_list
        self.variation_keys = [d['keys'] for d in variation_list]
        self.iterations = iterations
        self.summary_list = []
        self._param_modification_list = []
        self._i = 0
        self.df_hyperparam_ranking = None

    def set_cache_dir(self, tag=None):
        p = Path(Config.path.log_folder) / 'search_executor'
        os.makedirs(str(p), exist_ok=True)
        self.cache_dir = p / (datetime.now().strftime('%Y%m%d-%H%M%S') + '_hyperparam_search')
        if tag is not None:
            self.cache_dir = p / (tag + '_' + datetime.now().strftime('%Y%m%d-%H%M%S') + '_hyperparam_search')

    def log_start_hyperparam_test(self, params):
        self._i += 1
        params_path = self.cache_dir / 'params'
        os.makedirs(str(params_path), exist_ok=True)
        file_path = params_path / ('param_%03d.json' % self._i)
        with open(str(file_path), 'w') as f:
            json.dump(params, f, indent=True)
        print("======== START HYPERPARAM TEST (%i/%i)============" % (self._i, self.iterations))
        print("The next hyperparams are: \n")
        param_modification = filter_by_keys_lists(params, self.variation_keys)
        print(pd.DataFrame(flatten_dict_auto(param_modification), index=[0]))
        print("\n")
        self._param_modification_list.append(param_modification)

    def log_end_hyperparam_test(self, summary):
        summary_path = self.cache_dir / 'session_summaries'
        os.makedirs(str(summary_path), exist_ok=True)
        file_path = summary_path / ('session_summary_%03d.json' % self._i)
        with open(str(file_path), 'w') as f:
            json.dump(summary, f, indent=True)
        print("======== FINISHED HYPERPARAM TEST (%i/%i)============" % (self._i, self.iterations))
        print("Summary: ")
        pprint(summary)
        print('\n')
        self.summary_list.append(summary)
        self.log_end_hyperparam_search()

    def log_end_hyperparam_search(self):
        search_summary_path = self.cache_dir / 'search_summary.csv'
        df = self.get_search_summary()
        print(df)
        df.to_csv(str(search_summary_path))
        self.df_hyperparam_ranking = df

    def get_search_summary(self):
        val_accs = [s['validation']['mean_last_epoch_acc'] for s in self.summary_list]
        val_loss = [s['validation']['mean_last_epoch_loss'] for s in self.summary_list]
        train_accs = [s['training']['mean_last_epoch_acc'] for s in self.summary_list]
        train_loss = [s['training']['mean_last_epoch_loss'] for s in self.summary_list]
        mod_list = deepcopy(self._param_modification_list)
        for i, mod in enumerate(mod_list):
            mod['mean_last_epoch_val_acc'] = val_accs[i]
            mod['mean_last_epoch_val_loss'] = val_loss[i]
            mod['mean_last_epoch_train_acc'] = train_accs[i]
            mod['mean_last_epoch_train_loss'] = train_loss[i]
        df = pd.DataFrame([flatten_dict_auto(m) for m in mod_list])
        df = move_col_in_df(df, 'mean_last_epoch_train_loss', 0)
        df = move_col_in_df(df, 'mean_last_epoch_train_acc', 1)
        df = move_col_in_df(df, 'mean_last_epoch_val_loss', 2)
        df = move_col_in_df(df, 'mean_last_epoch_val_acc', 3)
        return df
