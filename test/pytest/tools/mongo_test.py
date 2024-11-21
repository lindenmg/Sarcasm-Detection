from src.tools.config import Config
from src.tools.mongo import *


class TestMongo:
    def test_collection_to_df(self):
        db = 'session_db'
        collection = 'log'
        port = Config.logging.port
        host = 'localhost'
        df_log = collection_to_df(db, collection, host, port)
        assert df_log.shape == (100, 9)

        collection = 'session_args'
        df_session_args = collection_to_df(db, collection, host, port)
        assert df_session_args.shape == (5, 10)
