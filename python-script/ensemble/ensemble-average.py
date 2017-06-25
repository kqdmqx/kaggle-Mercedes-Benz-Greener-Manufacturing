# coding=utf-8

import pandas as pd
from sklearn.metrics import r2_score

origin_oof = pd.read_csv(
    '../../output/stacking/Submission-XgbBaseline-OutOfFold.csv', index_col='ID')
decomposition_oof = pd.read_csv(
    '../../output/stacking/Submission-Decomposition-OutOfFold.csv', index_col='ID')
train = pd.read_csv('../../input/train.csv', index_col='ID')

avg_oof = pd.DataFrame(
    {
        'origin': origin_oof.y,
        'decomposition': decomposition_oof.y,
        'y_true': train.y
    }
)

avg_oof['y'] = avg_oof.origin * 0.5 + avg_oof.decomposition * 0.5

for col in avg_oof.columns:
    print col, r2_score(avg_oof.y_true, avg_oof[col])


origin = pd.read_csv(
    '../../output/stacking/Submission-XgbBaseline-Test.csv', index_col='ID')
decomposition = pd.read_csv(
    '../../output/stacking/Submission-Decomposition-Test.csv', index_col='ID')

avg = pd.DataFrame(
    {
        'origin': origin.y,
        'decomposition': decomposition.y
    }
)

avg['y'] = avg.origin * 0.5 + avg.decomposition * 0.5

avg[['y']].to_csv('../../output/ensemble/Submission-avg-Test.csv')
