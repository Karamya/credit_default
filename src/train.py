from lightgbm import LGBMClassifier
from utils import display_importances, display_precision_recall, display_roc_curve
import gc
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from datetime import datetime

def kfold_lightgbm(df, num_folds=5, stratified=False, debug=False):

    # Divide into train/valid and text data

    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    ids = train_df['SK_ID_CURR']

    print('Starting Lightgbm. Train shape: {}, test shape: {}'.format(train_df.shape, test_df.shape))
    del df
    gc.collect()

    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=321)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=123)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])

    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        clf = LGBMClassifier(
            n_estimators=10000,
            learning_rate=0.01,
            num_leaves=30,
            colsample_bytree=.9,
            subsample=0.5,
            max_depth=2,
            reg_alpha=.04,
            reg_lambda=.07,
            min_split_gain=.02,
            min_child_weight=39,
            silent=-1,
            verbose=-1,
            n_jobs=-1,
        )

        clf.fit(train_x, train_y,
                eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc',
                verbose=100,
                early_stopping_rounds=100  # 30
                )

        oof_preds[valid_idx] = clf.predict_proba(valid_x,
                                                 num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats],
                                       num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    score = roc_auc_score(train_df['TARGET'], oof_preds)
    print('Full AUC score %.6f' % score)

    df_oof_preds = pd.DataFrame({'SK_ID_CURR': ids, 'TARGET': train_df['TARGET'], 'PREDICTION': oof_preds})
    df_oof_preds = df_oof_preds[['SK_ID_CURR', 'TARGET', 'PREDICTION']]

    if not debug:
        test_df['TARGET'] = sub_preds

        # Save test predictions
        now = datetime.now()
        created_time = now.strftime('%Y-%m-%d-%H-%M')
        score = str(round(score, 6)).replace('.', '')

        # submission file
        sub_file = f'../predictions/{created_time}_{score}_{num_folds}_fold-average-LGBClassifier_submission.csv'
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(sub_file, index=False)

        # oof prediction file
        oof_file = f'../predictions/{created_time}_{score}_{num_folds}_fold-average-LGBClassifier_oof.csv'
        df_oof_preds.to_csv(oof_file, index=False)

        # Display a few plots
        vis_file = f'../visualization/{score}_{created_time}_'
        folds_idx = [(train_idx, valid_idx) for train_idx, valid_idx in folds.split(train_df[feats], train_df['TARGET'])]
        display_importances(feature_importance_df_=feature_importance_df,
                            vis_file=vis_file + "_feature_importances_without_ext_source.png")
        display_roc_curve(y_=train_df['TARGET'], oof_preds_=oof_preds, folds_idx_=folds_idx,
                          vis_file=vis_file + "_roc_curve_without_ext_source.png")
        display_precision_recall(y_=df['TARGET'], oof_preds_=oof_preds, folds_idx_=folds_idx,
                                 vis_file=vis_file + "_precision_recall_without_ext_source.png")



    return None


if __name__=="__main__":
    gc.enable()
