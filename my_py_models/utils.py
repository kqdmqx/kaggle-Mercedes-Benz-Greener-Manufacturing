# coding=utf-8

import pandas as pd
from sklearn.metrics import mean_squared_error
import os


def factorize_obj(train_test):
    df_numeric = train_test.select_dtypes(exclude=['object'])
    df_obj = train_test.select_dtypes(include=['object']).copy()

    for c in df_obj:
        df_obj[c] = pd.factorize(df_obj[c])[0]

    return pd.concat([df_numeric, df_obj], axis=1)


def r_square(pred, true):
    sstotal = true.std() ** 2
    ssresid = mean_squared_error(pred, true)
    return (sstotal - ssresid) / sstotal


def get_script_title(string):
    def head_upper(string):
        return string[0].upper() + string[1:] if len(string) > 0 else string

    return ''.join(map(head_upper,
                       (os.path.split(string)[1]
                               .split('.')[0]
                               .split('-'))))
