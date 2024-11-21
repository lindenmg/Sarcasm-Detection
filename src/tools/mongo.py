import pandas as pd
from pymongo import MongoClient

from src.hyperparameters.param_operation import flatten_dict_auto


def collection_to_df(db, collection, host='localhost', port=27017, flatten=False):
    client = MongoClient(host, port)
    l = list(client[db][collection].find())
    if flatten:
        l = [flatten_dict_auto(d, sep='.') for d in l]
    d = {k: [el[k] for el in l] for k in l[0].keys()}
    return pd.DataFrame(d)
