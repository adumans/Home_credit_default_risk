import pandas as pd
import numpy as np
import os
import csv
import ExtractionFeatures
from sklearn import model_selection
import lightgbm as lgb
from hyperopt import STATUS_OK
from hyperopt import fmin,tpe,hp,partial
from sklearn import datasets
from lightgbm import LGBMClassifier
from sklearn.utils import column_or_1d
# File to save first results
# out_file = 'gbm_trials.csv'
# of_connection = open(out_file, 'w')
# writer = csv.writer(of_connection)
#
# # Write the headers to the file
# writer.writerow(['loss', 'params'])
# of_connection.close()

N_FOLDS = 10

# prepare data start
training_file='../data/training.csv'
testing_file ='../data/testing.csv'
# read file
if os.path.exists(training_file)==False or os.path.exists(testing_file)==False:
    ExtractionFeatures.main()

train_df=pd.read_csv(training_file)
train_features = train_df[[column for column in train_df.columns if column != 'TARGET']]
train_labels = pd.DataFrame(train_df['TARGET'])
# train_labels = column_or_1d(train_labels)
train_set = lgb.Dataset(train_features, train_labels)
# prepare data end

print ('data done')

def objective(params, n_folds=N_FOLDS):
    # params['subsample_for_bin'] = int(params['subsample_for_bin'])
    # clf = LGBMClassifier(n_estimators=500,**params)
    # score = model_selection.cross_val_score(clf, train_features, train_labels, cv=3, scoring='roc_auc')
    cv_results = lgb.cv(params, train_set, nfold=n_folds, num_boost_round=10000, early_stopping_rounds = 100, metrics = 'auc')
    score = max(cv_results['auc-mean'])
    loss = 1 - score

    # of_connection = open(out_file, 'a')
    # writer = csv.writer(of_connection)
    # writer.writerow([loss, params])
    # of_connection.close()
    return {'loss': loss, 'params': params, 'status': STATUS_OK}

# Define the search space
space = {
   'class_weight': hp.choice('class_weight', [None, 'balanced']),
   'num_leaves': hp.choice('num_leaves', np.arange(30, 150, dtype=int)),
   'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
   'subsample_for_bin': hp.choice('subsample_for_bin', np.arange(20000, 300000, 20000, dtype=int)),
   'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
   'min_data_in_leaf': hp.choice('min_data_in_leaf', np.arange(30, 150, dtype=int)),
   'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
   'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
   'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
}


algo = partial(tpe.suggest,n_startup_jobs=10)
MAX_EVALS = 500
best = fmin(fn = objective, space = space, algo = algo, max_evals = MAX_EVALS)

print (best)
i = 0
