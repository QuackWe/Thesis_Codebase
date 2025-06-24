import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Orange.evaluation import compute_CD, graph_ranks
from scipy.stats import friedmanchisquare, rankdata


auc_orange = [0.57, 0.86, 0.73, 0.96, 0.98, 0.98, 0.90, 0.67, 0.61, 0.70, 0.71]
auc_xgb = [0.33, 0.85, 0.72, 0.94, 0.98, 0.98, 0.86, 0.70, 0.57, 0.69, 0.70]
auc_rf = [0.41, 0.79, 0.66, 0.94, 0.98, 0.98, 0.89, 0.69, 0.61, 0.70, 0.63]
auc_lr = [0.57, 0.86, 0.73, 0.92, 0.94, 0.96, 0.87, 0.65, 0.59, 0.69, 0.67]
auc_svm = [0.49, 0.82, 0.72, 0.87, 0.95, 0.96, 0.87, 0.63, 0.55, 0.70, 0.66]

f1_orange = [0.11, 0.50, 0.89, 0.86, 0.94, 0.88, 0.81, 0.66, 0.21, 0.46, 0.59]
f1_xgb = [0.00,	0.42, 0.35,	0.86, 0.95,	0.94, 0.78,	0.59, 0.16,	0.36, 0.59]
f1_rf = [0.00,	0.39, 0.34,	0.86, 0.95,	0.94, 0.80,	0.64, 0.17,	0.42, 0.54]
f1_lr = [0.09,	0.47, 0.37,	0.83, 0.90,	0.86, 0.75,	0.53, 0.13,	0.30, 0.58]
f1_svm = [0.00,	0.00, 0.00,	0.73, 0.91,	0.00, 0.35,	0.38, 0.00,	0.20, 0.54]

#sign test AUC
performances = pd.DataFrame({'dataset':['df1', 'df2', 'df3', 'df4', 'df5', 'df6', 'df7', 'df8', 'df9', 'df10', 'df11'],'ORANGE': auc_orange, 'XGB': auc_xgb, 'RF': auc_rf, 'SVM': auc_svm, 'LR': auc_lr})
#sign test F1-score
#performances = pd.DataFrame({'dataset':['df1', 'df2', 'df3', 'df4', 'df5', 'df6', 'df7', 'df8', 'df9', 'df10', 'df11'],'ORANGE': f1_orange, 'XGB': f1_xgb, 'RF': f1_rf, 'SVM': f1_svm, 'LR': f1_lr})

# First, we extract the algorithms names.
algorithms_names = performances.drop('dataset', axis=1).columns

print(algorithms_names)
# Then, we extract the performances as a numpy.ndarray.
performances_array = performances[algorithms_names].values
print(performances_array)
# Finally, we apply the Friedman test.
print(friedmanchisquare(*performances_array))
ranks = np.array([rankdata(-p) for p in performances_array])

# Calculating the average ranks.
average_ranks=np.mean(ranks,axis=0)
print('\n'.join('{} average rank: {}'.format(a,r) for a,r in zip(algorithms_names, average_ranks)))

cd = compute_CD(average_ranks, n=len(performances), alpha='0.05', test='nemenyi')
# This method generates the plot.
graph_ranks(average_ranks, names=algorithms_names, cd=cd, width=10, textspace=3, reverse=True)
plt.show()