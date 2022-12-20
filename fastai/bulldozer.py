import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


CWD = os.path.dirname(__file__)
df_path = os.path.abspath(os.path.join(CWD, "../data/bd_data/Train.csv"))
df_raw = pd.read_csv(df_path, low_memory=False, parse_dates=['saledate'])
