# coding=utf-8

import pandas as pd
from sklearn.metrics import mean_squared_error


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
