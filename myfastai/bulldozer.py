import pandas as pd
import os
from fastai.imports import *
from fastcore.all import *
# from fastai.structured import *
from fastai.structured import proc_df
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


CWD = os.path.dirname(__file__)
df_path = os.path.abspath(os.path.join(CWD, "../data/bd_data/Train.csv"))
df_raw = pd.read_csv(df_path, low_memory=False, parse_dates=['saledate'])



train_cats(df_raw)
m = RandomForestRegressor(n_jobs=1)



print(df_raw.saledate[:10])
# m.fit(df_raw.drop('SalePrice', axis=1), df_raw.SalePrice)



