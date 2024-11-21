import json
from os import listdir
from os.path import join

import pandas as pd


def query_preprocessing_params(keyslists, values, hyperparameter_directory,
                               pp_step_mapping_directory, pp_step_data_directory,
                               pp_step_params_directory):
    def get_cachefiles(h_file):
        cachefiles = []
        session_tag = params['learning_session']['session_tag']
        for m_file in listdir(pp_step_mapping_directory):
            mapping_path = join(pp_step_mapping_directory, m_file)
            with open(mapping_path) as f:
                lines = f.read().splitlines(keepends=False)
            if session_tag in lines:
                hash = m_file.split('.')[0].split('_')[-1]
                for dfile in listdir(pp_step_data_directory):
                    if dfile.find(hash) > 0:
                        for pfile in listdir(pp_step_params_directory):
                            if pfile.split('.')[0] == dfile.split('.')[0]:
                                with open(join(pp_step_params_directory, pfile)) as f:
                                    step_params = json.load(f)
                                for k in last_keys:
                                    if k in step_params.keys():
                                        cachefiles.append({'hyperparams': h_file, 'step_params': pfile, 'data': dfile})
        return cachefiles

    def _get_val(params, keyslist):
        for key in keyslist:
            params = params[key]
        if isinstance(params, dict):
            return False
        else:
            return params

    last_keys = [keys[-1] for keys in keyslists]
    df = pd.DataFrame({'hyperparams': [], 'step_params': [], 'data': []})
    for h_file in listdir(hyperparameter_directory):
        path = join(hyperparameter_directory, h_file)
        match = False
        with open(path) as f:
            params = json.load(f)
        try:
            match = all([_get_val(params, keys) == v for (v, keys) in zip(values, keyslists)])
        except Exception:
            pass
        if match:
            cachefiles = get_cachefiles(h_file)
            for cache_match in cachefiles:
                df = df.append(pd.Series(cache_match), ignore_index=True)
    return df
