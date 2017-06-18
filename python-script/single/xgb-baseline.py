# coding=utf-8

import numpy as np
import pandas as pd
import sys
sys.path.append('../..')
from my_py_models.stacking import Stacking
from my_py_models.my_xgb_classifier2 import MyXgbClassifier2
from my_py_models.config import INPUT_PATH, OUTPUT_PATH
from my_py_models.utils import factorize_obj
from os.path import join
from sklearn.metrics import mean_squared_error

# Any results you write to the current directory are saved as output.
train = pd.read_csv(join(INPUT_PATH, 'train.csv'))
test = pd.read_csv(join(INPUT_PATH, 'test.csv'))
train_ID = train.ID
test_ID = test.ID

y_train = train.y

train_test = pd.concat([train, test])
train_test.drop(["ID", "y"], axis=1, inplace=True)
train_test_p = factorize_obj(train_test)

# Convert to numpy values
X_all = train_test_p.values
print(X_all.shape)

num_train = train_ID.shape[0]
X_train = X_all[:num_train]
X_test = X_all[num_train:]


xgb_params = {
    'n_trees': 500,
    'eta': 0.005,
    'max_depth': 4,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

clf = MyXgbClassifier2(xgb_params)
stacking = Stacking(5, [clf])
pred_oof, pred_test = stacking.fit_predict(X_train, y_train, X_test)

# 0.56200717888
sstotal = y_train.values.std()
for pred_oof_single in pred_oof.T:
    print (sstotal - mean_squared_error(pred_oof_single, y_train)) / sstotal

submission = pd.DataFrame({'ID': test_ID, 'y': pred_test[:, 0]})
submission.to_csv(join(
    OUTPUT_PATH, 'stacking/Submission-XgbBaseline-Test.csv'), index=False)

oof_pred = pd.DataFrame({'ID': train_ID, 'y': pred_oof[:, 0]})
oof_pred.to_csv(join(
    OUTPUT_PATH, 'stacking/Submission-XgbBaseline-OutOfFold.csv'), index=False)
