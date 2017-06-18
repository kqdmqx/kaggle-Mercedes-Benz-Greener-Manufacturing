# coding=utf-8

import pandas as pd
from sklearn.metrics import mean_squared_error

oof = pd.read_csv(
    '../../output/stacking/Submission-XgbBaseline-OutOfFold.csv').y
y = pd.read_csv('../../input/train.csv').y


def r_square(pred, true):
    sstotal = true.std() ** 2
    ssresid = mean_squared_error(pred, true)
    return (sstotal - ssresid) / sstotal


print r_square(oof, y)
