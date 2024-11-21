import json
from pathlib import Path

from src.tools.config import Config
from src.training.learning_session import LearningSession


def __main__():
    train_pipe_config_path = Path(Config.path.test_data_folder) / 'cnn_test_hp_config.json'

    with open(str(train_pipe_config_path)) as file:
        params = json.load(file)

    logger_args = {
        'tensorboard_log_dir': Config.path.test_log_folder,
        'mongo_host': 'localhost',
        'mongo_port': Config.logging.port
    }
    params['logger']['args'].update(logger_args)
    params['learning_session'].update({'cache_folder': Config.path.test_cache_folder})
    ls = LearningSession(params)
    ls.run()


if __name__ == '__main__':
    __main__()
