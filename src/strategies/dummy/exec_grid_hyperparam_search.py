import os
import sys

project_root_path = os.path.realpath(os.path.join(__file__, "..", "..", "..", ".."))
sys.path.append(project_root_path)

from src.training.learning_session import LearningSession
from src.tools.config import Config
from os import listdir
from os.path import join
import json

params_file = None


def learn(params_f, directory):
    params_path = join(directory, params_f)
    print('TRAIN %s' % params_path)
    with open(params_path) as file:
        params_dict = json.load(file)
    ls = LearningSession(params_dict, save_interval=4,
                         cache_dir=Config.path.cache_folder,
                         logger_args={
                             'tensorboard_log_dir': Config.path.log_folder
                         })
    ls.run()


# replace this with executor

if __name__ == '__main__':
    hyperparams_dir = join(project_root_path, 'src', 'strategies', 'dummy', 'hyperparameter')

    if params_file is not None:
        learn(params_file, hyperparams_dir)
    else:
        for params_file in listdir(hyperparams_dir):
            learn(params_file, hyperparams_dir)
