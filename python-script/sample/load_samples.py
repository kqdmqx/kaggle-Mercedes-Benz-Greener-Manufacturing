# coding=utf-8

import pandas as pd
import os

stacking_dir = '../../output/stacking/'
oof_format = 'Submission-{}-OutOfFold.csv'
test_format = 'Submission-{}-Test.csv'

model_list = ['Lasso', 'LassoLars', 'Decomposition',
              'RandomForestRegressor', 'XgbBaseline']
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

print oof_df.head()
print test_df.head()

