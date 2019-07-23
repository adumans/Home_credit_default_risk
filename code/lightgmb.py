from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import gc

import ExtractionFeatures

# LightGBM GBDT with KFold or Stratified KFold
def kfold_lightgbm(training_file, testing_file, num_folds, stratified=False):
    # Divide in training/validation and test data

    train_df=pd.read_csv(training_file);
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    feature_importance_df = pd.DataFrame()
    #feature name: train data dont use 'TARGET', '*_ID_*'is just a indentification of a sample, some is optional.
    # your data may include 'ID' to identify a sample, and modify feature name in the next row code
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=4,
            n_estimators=100,
            learning_rate=0.01,
            num_leaves=40,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc', verbose=100, early_stopping_rounds=100)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        test_df=pd.read_csv(testing_file)
        sub_preds = np.zeros(test_df.shape[0])
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    test_df['TARGET'] = sub_preds
    test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index=False)
    #display_importances(feature_importance_df)
    return feature_importance_df
#def display_importances(feature_importance_df):


if __name__ == "__main__":
    submission_file_name = "submission.csv"
    importance_file_name = "feature_importance.csv"
    # your format of training and tesing data, now they are all existed
    training_file='../data/training.csv'
    testing_file ='../data/testing.csv'
    # read file, now they are all existed!
    if os.path.exists(training_file)==False or os.path.exists(testing_file)==False:
        ExtractionFeatures.main() # Extraction Features


    # train classifier
    model=kfold_lightgbm(training_file, testing_file, 3, True)

    merge_re = model.groupby('feature').sum()
    sorted_re = merge_re.sort_values(by = 'importance', ascending=False)
    sorted_re.to_csv(importance_file_name, index=True)
    print (sorted_re)


    #submission_file.to_csv(submission_file_name, index=False)
