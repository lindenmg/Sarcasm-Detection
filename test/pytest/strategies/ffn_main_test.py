import json
from os.path import join

from src.tools.config import Config
from src.training.learning_session import LearningSession
from pymongo.errors import ServerSelectionTimeoutError


class TestMain:
    """
    For this test it is required to start a mongo_db with the configuration as suggested
    in 'config_json_template' under 'logging'.
    """

    def test_main(self):
        success = False
        params_path = join(Config.path.test_data_folder, 'ffn_test_hyperparameters.json')
        with open(params_path) as file:
            params = json.load(file)

        logger_args = {
            'tensorboard_log_dir': Config.path.test_log_folder,
            'mongo_host': 'localhost',
            'mongo_port': Config.logging.port
        }
        params['cv_iterator_factory']['args'].update({'n_splits': 3})
        params['logger']['args'].update(logger_args)
        params['learning_session'].update({'cache_folder': Config.path.test_cache_folder})
        ls = LearningSession(params)
        try:
            ls.run()
            success = True
        except ServerSelectionTimeoutError:
            raise ServerSelectionTimeoutError(
                "For this test it is required to start a mongo_db "
                "with the configuration as suggested \n in "
                "'config_json_template' under 'logging'. Afterwards "
                "you can use start_mongodb.sh in the project root.")
        assert success
