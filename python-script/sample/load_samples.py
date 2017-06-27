# coding=utf-8

import pandas as pd
import numpy as np
import os
import sys
sys.path.append('../..')
from my_py_models.utils import get_script_title, kfold_cv_score
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

stacking_dir = '../../output/stacking/'
oof_format = 'Submission-{}-OutOfFold.csv'
test_format = 'Submission-{}-Test.csv'


model_list = map(get_script_title, os.listdir('../../python-script/stacking/'))
index_col = 'ID'
tar_col = 'y'


def load_stacking_data(model_list,
                       index_col=index_col,
                       tar_col=tar_col,
                       path_format=oof_format,
                       stacking_dir=stacking_dir,
                       silence=False):
    oof_dict = {}
    for model_str in model_list:
        model_path = os.path.join(stacking_dir, path_format.format(model_str))
        if os.path.isfile(model_path):
            if not silence:
                print 'load file', model_path
            oof_dict[model_str] = pd.read_csv(
                model_path, index_col=index_col)[tar_col]
    return pd.DataFrame(oof_dict)


def load_stacking_predictions(model_list,
                              train_path,
                              index_col=index_col,
                              tar_col=tar_col,
                              oof_format=oof_format,
                              test_format=test_format,
                              stacking_dir=stacking_dir,
                              silence=False):
    oof_df = load_stacking_data(model_list,
                                index_col=index_col,
                                tar_col=tar_col,
                                path_format=oof_format,
                                stacking_dir=stacking_dir,
                                silence=silence)
    oof_df[tar_col] = pd.read_csv(train_path, index_col=index_col)[tar_col]
    test_df = load_stacking_data(model_list,
                                 index_col=index_col,
                                 tar_col=tar_col,
                                 path_format=test_format,
                                 stacking_dir=stacking_dir,
                                 silence=silence)
    return oof_df, test_df


oof_df, test_df = load_stacking_predictions(model_list, '../../input/train.csv')

kf = KFold(n_splits=5, shuffle=True, random_state=2016)

for col in oof_df.columns:
    try:
        split_result, metric_result = kfold_cv_score(oof_df.y.values, oof_df[col].values, kf, r2_score)
        print "{},{},{},{}".format(
            col, np.mean(metric_result), np.std(metric_result),
            r2_score(oof_df.y, oof_df[col])
        )
    except Exception, e:
        print 'column {} wrong'.format(col)
