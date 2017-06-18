# coding=utf-8

import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

oof = pd.read_csv(
    '../../output/stacking/Submission-XgbBaseline120-OutOfFold.csv')
temp = pd.read_csv('../../input/train.csv')
y = temp.loc[temp.ID.isin(oof.ID.unique())].y


def r_square(pred, true):
    sstotal = true.std() ** 2
    ssresid = mean_squared_error(pred, true)
    return (sstotal - ssresid) / sstotal


print r_square(oof.y, y)

# sns.kdeplot(y)
# plt.show()