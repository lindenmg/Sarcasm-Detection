import os
import sys
import traceback

project_root_path = os.path.realpath(os.path.join(__file__, "..", "..", "..", ".."))
sys.path.append(project_root_path)

from src.training.learning_session import LearningSession
from src.tools.config import Config
from os import listdir
from os.path import join
import json
import argparse


def learn(params_f, directory, save_interval, mongo_host, mongo_port):
    params_path = join(directory, params_f)
    print('TRAIN %s' % params_path)
    with open(params_path) as file:
        params_dict = json.load(file)

    params_dict['learning_session'].update({'save_interval': save_interval})
    params_dict['learning_session'].update({'mode': "train"})
    params_dict['learning_session'].update({'cnn_mode': False})
    params_dict['learning_session'].update({'cache_prefix': "fnn_fact"})
    params_dict['learning_session'].update({'session_tag': "fnn_train_std"})
    params_dict['learning_session'].update({'cache_folder': Config.path.cache_folder})
    params_dict['logger']['args'].update({
        'tensorboard_log_dir': Config.path.cache_folder
        , 'mongo_host': mongo_host
        , 'mongo_port': mongo_port})

    try:
        ls = LearningSession(params_dict)
        ls.run()
    except:
        traceback.print_exc(file=sys.stdout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tries to find sarcasm')
    parser.add_argument('-s', '--save_interval', type=int, nargs='?', default=5,
                        help='The (epoch) interval on which the network parameters shall be saved')
    parser.add_argument('-p', '--params', type=str, nargs='*',
                        help='the paths to the json files, that contain hyperparameters')
    parser.add_argument('-pd', '--params_dir', type=str, nargs=1,
                        help='The path to the directory with all hyperparameter files')
    parser.add_argument('-i', '--file_interval', type=int, nargs=2,
                        help='The files in `src/strategies/ffn/hyperparameters` are enumerated. ' +
                             'The first number that is passed in represents the number of the first file, that shall be executed. ' +
                             'The second number that is passed in represents the number of the last file, that shall be executed. ' +
                             'All files in between the interval are executed as well.')
    parser.add_argument('--mongo_host', type=str, nargs='?', default='localhost',
                        help='the host of the mongo-db server')
    parser.add_argument('--mongo_port', type=int, nargs='?', default=47213,
                        help='the port of the mongo-db server')

    args = parser.parse_args()
    param_files = args.params
    file_interval = args.file_interval
    save_interval = args.save_interval
    hyperparams_dir = args.params_dir
    if file_interval is None and param_files is None:
        param_files = listdir(hyperparams_dir[0])
    elif isinstance(file_interval, list) and len(file_interval) == 2:
        param_files = listdir(hyperparams_dir[0])[file_interval[0]:file_interval[1]]
    for param_file in param_files:
        learn(param_file, hyperparams_dir[0], save_interval, args.mongo_host, args.mongo_port)
