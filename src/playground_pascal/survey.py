from os.path import join

import pandas as pd

from src.preprocessing.datahandler import DataHandler
from src.tools.config import Config

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

dh = DataHandler()
dh.load_train_test(Config.path.data_folder)
df = dh.get_train_df()
df_result = df
df = df[['post_id', 'post', 'reply']]
df1 = df.iloc[100:160][0::2]
df2 = df.iloc[200:260]
df3 = df.iloc[300:360]
df4 = df.iloc[400:460]
df5 = df.iloc[500:560]

base_path = '~/sarcasm_data'
df1.to_csv(join(base_path, 'df1.csv'))
df2.to_csv(join(base_path, 'df2.csv'))
df3.to_csv(join(base_path, 'df3.csv'))
df4.to_csv(join(base_path, 'df4.csv'))
df5.to_csv(join(base_path, 'df5.csv'))

df_result_1 = df_result.iloc[100:130]
df_result_2 = df_result.iloc[200:230]
df_result_3 = df_result.iloc[310:330]
df_result_4 = df_result.iloc[410:430]
df_result_5 = df_result.iloc[510:530]

df_result_1.to_csv(join(base_path, 'df_result_1.csv'))
df_result_2.to_csv(join(base_path, 'df_result_2.csv'))
df_result_3.to_csv(join(base_path, 'df_result_3.csv'))
df_result_4.to_csv(join(base_path, 'df_result_4.csv'))
df_result_5.to_csv(join(base_path, 'df_result_5.csv'))

df1_hard = df.iloc[1100:1140][0::2]
df2_hard = df.iloc[1200:1240][0::2]
df3_hard = df.iloc[1300:1340][0::2]
df4_hard = df.iloc[1400:1440][0::2]
df5_hard = df.iloc[1500:1540][0::2]
df6_hard = df.iloc[1600:1640][0::2]
df7_hard = df.iloc[1700:1740][0::2]

df1_hard.to_csv(join(base_path, 'df1_hard.csv'))
df2_hard.to_csv(join(base_path, 'df2_hard.csv'))
df3_hard.to_csv(join(base_path, 'df3_hard.csv'))
df4_hard.to_csv(join(base_path, 'df4_hard.csv'))
df5_hard.to_csv(join(base_path, 'df5_hard.csv'))
df6_hard.to_csv(join(base_path, 'df6_hard.csv'))
df7_hard.to_csv(join(base_path, 'df7_hard.csv'))

df1_hard_result = df_result.iloc[1100:1140][0::2]
df2_hard_result = df_result.iloc[1200:1240][0::2]
df3_hard_result = df_result.iloc[1300:1340][0::2]
df4_hard_result = df_result.iloc[1400:1440][0::2]
df5_hard_result = df_result.iloc[1500:1540][0::2]
df6_hard_result = df_result.iloc[1600:1640][0::2]
df7_hard_result = df_result.iloc[1700:1740][0::2]
