import numpy as np
from sklearn import metrics

# K (number of shots)
x = np.array([1., 10., 30., 50., 100., 300., 500., 1000.])
x_log = np.log(x) / np.log(1000)
# Average Recall scores
y = np.array([0.0, 18.0, 26.5, 29.6, 33.4, 39.0, 41.5, 45.0])
y *= 0.01
auc = metrics.auc(x_log, y)
print('AUC score:', auc)