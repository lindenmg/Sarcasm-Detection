import os
import sys

project_root_path = os.path.realpath(os.path.join(__file__, "..", "..", "..", ".."))
sys.path.append(project_root_path)

import json
from pathlib import Path

from src.tools.config import Config
from src.training.learning_session import LearningSession


def __main__():
    train_pipe_config_path = Path(Config.path.project_root_folder) / 'src' / 'strategies'
    train_pipe_config_path = train_pipe_config_path / 'cnn' / 'attentive_cnn_config.json'

    with open(str(train_pipe_config_path)) as file:
        params = json.load(file)

    logger_args = {
        'tensorboard_log_dir': Config.path.log_folder,
        'mongo_host': 'localhost',
        'mongo_port': Config.logging.port
    }
    params['logger']['args'].update(logger_args)
    params['learning_session'].update({'cache_folder': Config.path.cache_folder})
    ls = LearningSession(params)
    result_dict = ls.run()
    print()
    print(result_dict)


if __name__ == '__main__':
    __main__()
