from lightgbm import LGBMClassifier
from preprocess import build_model_input
from utils import display_importances, display_precision_recall, display_roc_curve
import gc
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from datetime import datetime

def train_model(data_, test_, y_, folds_, categorical_features_):

    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])

    feature_importance_df = pd.DataFrame()

    feats = [f for f in data_.columns if f not in ['SK_ID_CURR']]

    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_, y_)):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]

        clf = LGBMClassifier(
            n_estimators=10000,
            learning_rate=0.01,
            num_leaves=30,
            colsample_bytree=.9,
            subsample=0.5,
            max_depth=7,
            reg_alpha=.04,
            reg_lambda=.07,
            min_split_gain=.02,
            min_child_weight=39,
            silent=-1,
            verbose=-1,
            n_jobs=-1,
        )

        clf.fit(trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],
                eval_metric='auc',
                verbose=100,
                early_stopping_rounds=100  # 30
                )

        oof_preds[val_idx] = clf.predict_proba(val_x,
                                               num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_[feats],
                                       num_iteration=clf.best_iteration_)[:, 1] / folds_.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))

    test_['TARGET'] = sub_preds

    df_oof_preds = pd.DataFrame({'SK_ID_CURR':ids, 'TARGET':y, 'PREDICTION':oof_preds})
    df_oof_preds = df_oof_preds[['SK_ID_CURR', 'TARGET', 'PREDICTION']]

    return oof_preds, df_oof_preds, test_[['SK_ID_CURR', 'TARGET']], \
           feature_importance_df, roc_auc_score(y, oof_preds)


if __name__=="__main__":
    gc.enable()

    #Build model inputs
    data, test, y, ids = build_model_input()
    categorical_features = [f for f in data.columns if data[f].dtype == 'object']
    #Creat Folds
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    #Train model and get oof and test predictions
    oof_preds, df_oof_preds, test_preds, importances, score = train_model(data, test, y, folds, categorical_features)
    # Save test predictions
    now = datetime.now()
    score = str(round(score, 6)).replace('.', '')
    sub_file = '../predictions/submission_5x-average-LGB-run-01-v1_' + score + '_' + str(now.strftime('%Y-%m-%d-%H-%M')) + '.csv'
    test_preds.to_csv(sub_file, index=False)
    oof_file = '../predictions/train_5x-LGB-run-01-v1-oof_' + score + '_' + str(now.strftime('%Y-%m-%d-%H-%M')) + '.csv'
    df_oof_preds.to_csv(oof_file, index=False)
    # Display a few graphs
    folds_idx = [(trn_idx, val_idx) for trn_idx, val_idx in folds.split(data, y)]
    vis_file = '../visualization/' + score + '_' + str(now.strftime('%Y-%m-%d-%H-%M'))
    display_importances(feature_importance_df_=importances, vis_file= vis_file + "_feature_importances.png")
    display_roc_curve(y_=y, oof_preds_=oof_preds, folds_idx_=folds_idx, vis_file=vis_file + "_roc_curve.png")
    display_precision_recall(y_=y, oof_preds_=oof_preds, folds_idx_=folds_idx, vis_file=vis_file + "_precision_recall.png")
