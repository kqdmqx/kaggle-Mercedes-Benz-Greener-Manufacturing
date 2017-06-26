# coding=utf-8

import numpy as np
import pandas as pd
import sys
sys.path.append('../..')
from my_py_models.stacking import Stacking
from my_py_models.my_xgb_classifier2 import MyXgbClassifier2
from my_py_models.config import INPUT_PATH, OUTPUT_PATH
from my_py_models.utils import factorize_obj, get_script_title
from os.path import join
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Sample
title = get_script_title(__file__)
print title
train = pd.read_csv(join(INPUT_PATH, 'train.csv'))
test = pd.read_csv(join(INPUT_PATH, 'test.csv'))
train_ID = train.ID
test_ID = test.ID

# Y
y_train, labels = pd.cut(train.y, [0, 82, 96, 103, 112, 999], right=False).factorize()

# Features
train_test = pd.concat([train, test])
train_test.drop(["ID", "y"], axis=1, inplace=True)
train_test_p = factorize_obj(train_test)

X_all = train_test_p.values
print(X_all.shape)

# X_train & X_test
num_train = train_ID.shape[0]
X_train = X_all[:num_train]
X_test = X_all[num_train:]

# 5-cv Xgb
xgb_params = {
    'n_trees': 500,
    'eta': 0.25,
    'max_depth': 3,
    'subsample': 0.95,
    "num_class": 5,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'silent': 1
}

trid, valid, y_tr, y_val = train_test_split(np.arange(len(y_train)), y_train, random_state=2016)
X_tr, X_val = X_train[trid], X_train[valid]

clf = MyXgbClassifier2(xgb_params)
clf.fit(X_tr, y_tr)
y_val_pred = clf.predict(X_val)
y_val_pred = np.argmax(y_val_pred, axis=1).astype(np.float64)
y_val = y_val.astype(np.float64)
print y_val_pred.shape, y_val.shape

y_val_pred += np.random.random(len(y_val_pred)) * .4
y_val += np.random.random(len(y_val_pred)) * .4
plt.scatter(y_val_pred, y_val, alpha=.2)
plt.show()

y_val_float = train.y.values[valid]

mean_dict = {}
for i in np.unique(y_val_pred):
    mean_dict[i] = y_val_float[y_val_pred == i].mean()


for i in mean_dict:
    y_val_pred[y_val_pred == i] = mean_dict[i]

plt.scatter(y_val_pred, y_val_float, alpha=.2)
plt.show()

# stacking = Stacking(5, [clf])
# pred_oof, pred_test = stacking.fit_predict(X_train, y_train, X_test)

# # r^2 0.56200717888
# for pred_oof_single in pred_oof.T:
#     print r2_score(y_train, pred_oof_single)

# # Save test
# submission = pd.DataFrame({'ID': test_ID, 'y': pred_test[:, 0]})
# submission.to_csv(join(
#     OUTPUT_PATH, 'stacking/Submission-{}-Test.csv'.format(title)), index=False)

# # Save oof
# oof_pred = pd.DataFrame({'ID': train_ID, 'y': pred_oof[:, 0]})
# oof_pred.to_csv(join(
#     OUTPUT_PATH, 'stacking/Submission-{}-OutOfFold.csv'.format(title)), index=False)
