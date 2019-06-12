import pandas as pd
import os
from properties.config import get_config

def input_file(env):
    data_dir = get_config(env)
    df = pd.DataFrame()
    for i in os.listdir(data_dir):
        df = df.append(pd.read_csv(data_dir+i,sep='|'))
    df['INDUSTRY'] = df['INDUSTRY'].replace(['Unknown'],0)
    inds = list(df['INDUSTRY'].unique())[:-1]
    df['INDUSTRY'] = df['INDUSTRY'].replace(inds,1)

    return df