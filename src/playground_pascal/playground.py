import os

os.chdir('/Users/pascalweiss/scripts/cdpath/dev/python/dl_project_gl_pw')

# Add this to the beginning of your main
# Now the other imports
from src.tools.queries import *
from src.tools.config import Config

dir_hyper = Config.path.tfidf_hyperparameter_folder
dir_data = join(Config.path.cache_folder, 'preprocessing', 'pp_data')
dir_params = join(Config.path.cache_folder, 'preprocessing', 'pp_step_params')
dir_mappings = join(Config.path.cache_folder, 'preprocessing', 'pp_step_mapping')

keyslists = [
    ['data_factory', 'args', 'pp_params', 'tfidf', 'analyzer']
]
values = ['char']
df = query_preprocessing_params(keyslists, values, dir_hyper, dir_mappings, dir_data, dir_params)
