# coding=utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('../../input/train.csv')
oof = pd.read_csv('../../output/stacking/Submission-EnsembleLasso-OutOfFold.csv')
# sns.kdeplot(train.y)
# plt.show()

sns.kdeplot(train.loc[train.y < 200].y, bw=1)
plt.plot((82, 82), (0, .045), c='r', alpha=.5)
plt.plot((96, 96), (0, .045), c='r', alpha=.5)
plt.plot((103, 103), (0, .045), c='r', alpha=.5)
plt.plot((112, 112), (0, .045), c='r', alpha=.5)
plt.plot((117, 117), (0, .045), c='r', alpha=.5)
plt.show()

# 82, 96, 103
y = train.loc[train.y < 200].y.values
y_oof = oof.loc[oof.ID.isin(train.loc[train.y < 200].ID.unique())].y.values
# y = train.y.values
rd = np.random.random(len(y)) * y.std()
plt.scatter(y, rd, alpha=.2)
plt.scatter(y_oof, rd, alpha=.3, c='k')
plt.plot((82, 82), (0, 13), c='r', alpha=.5)
plt.plot((96, 96), (0, 13), c='r', alpha=.5)
plt.plot((103, 103), (0, 13), c='r', alpha=.5)
plt.plot((112, 112), (0, 13), c='r', alpha=.5)
plt.plot((117, 117), (0, 13), c='r', alpha=.5)
plt.show()

plt.scatter(y, y_oof, alpha=.3, c='b')
offset = 20
plt.plot((82 + offset, 82 - offset), (82 - offset, 82 + offset), c='r', alpha=.5)
plt.plot((96 + offset, 96 - offset), (96 - offset, 96 + offset), c='r', alpha=.5)
plt.plot((103 + offset, 103 - offset), (103 - offset, 103 + offset), c='r', alpha=.5)
plt.plot((112 + offset, 112 - offset), (112 - offset, 112 + offset), c='r', alpha=.5)
plt.plot((117 + offset, 117 - offset), (117 - offset, 117 + offset), c='r', alpha=.5)
plt.plot((70, 125), (70, 125), c='r', alpha=.5)
plt.show()
