import pandas as pd
from pymongo import MongoClient

from src.tools.config import Config
from src.tools.mongo import collection_to_df

db = 'session_db'

port = Config.logging.port
host = 'localhost'

client = MongoClient(host, port)
df_log = collection_to_df('session_db', 'log', 'localhost', Config.logging.port)
df_args = collection_to_df('session_db', 'session_args', 'localhost', Config.logging.port,
                           filter={'_id': {'$gt': '5a9067649233a2051495fea9'}},
                           flatten=True)

df_merge = pd.merge(df_args, df_log, left_on='_id', right_on='_session_id', how='inner')
df = df_merge[['data_factory.args.pp_params.vectorization.max_features',
               'data_factory.args.pp_params.vectorization.min_df',
               'data_factory.args.pp_params.vectorization.ngram_range',
               'data_factory.args.pp_params.vectorization.tfidf',
               'model.args.post_layer_size', 'model.args.reply_layer_size',
               'fold', 'epoch', 'val_acc', 'val_loss', 'train_acc', 'train_loss']]

df.columns = ['pp_max_features', 'pp_min_df', 'pp_ngram_range', 'pp_tfidf', 'post_layer_size',
              'reply_layer_size', 'fold', 'epoch', 'val_acc', 'val_loss', 'train_acc', 'train_loss']