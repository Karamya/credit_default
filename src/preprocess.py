# https://www.kaggle.com/ogrellier/good-fun-with-ligthgbm/code

import pandas as pd
import numpy as np
import gc

gc.enable()


def build_model_input():
    ## Bureau and Bureau_balance
    bureau = pd.read_csv("../data/bureau.csv")
    bureau_balance = pd.read_csv("../data/bureau_balance.csv")

    print("Preprocessing Bureau_balance.....")
    buro_counts = bureau_balance.groupby('SK_ID_BUREAU')['STATUS'].value_counts(normalize=False)
    buro_counts_unstacked = buro_counts.unstack('STATUS', fill_value=0)
    buro_counts_unstacked.columns = ["STATUS_" + column for column in buro_counts_unstacked.columns]

    buro_counts_unstacked['MONTHS_COUNT'] = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].size()
    buro_counts_unstacked['MONTHS_MAX'] = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].max()
    buro_counts_unstacked['MONTHS_MIN'] = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].min()
    buro_counts_unstacked['MONTHS_MEAN'] = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].mean()
    buro_counts = bureau_balance[['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').count()
    buro_counts_unstacked['bureau_count'] = bureau_balance['SK_ID_BUREAU'].map(buro_counts['MONTHS_BALANCE'])

    bureau = bureau.join(buro_counts_unstacked, on="SK_ID_BUREAU", how='left')
    del buro_counts, buro_counts_unstacked, bureau_balance
    gc.collect()

    print("one-hot encoding of credits from bureau")
    bureau = pd.get_dummies(bureau, columns=['CREDIT_ACTIVE'], prefix='ca_')
    bureau = pd.get_dummies(bureau, columns=['CREDIT_CURRENCY'], prefix='cur_')
    bureau = pd.get_dummies(bureau, columns=['CREDIT_TYPE'], prefix='cty_')

    print('Counting buro per SK_ID_CURR')
    nb_bureau_per_curr = bureau[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()
    bureau['SK_ID_BUREAU'] = bureau['SK_ID_CURR'].map(nb_bureau_per_curr['SK_ID_BUREAU'])

    print("Averaging bureau")
    avg_bureau = bureau.groupby('SK_ID_CURR').mean()

    del bureau

    print('Loading previous application...')
    previous_application = pd.read_csv("../data/previous_application.csv")

    print("Preprocessing previous_application")

    print("one-hot encoding of categorical features")
    prev_cat_features = [pcol for pcol in previous_application.columns if previous_application[pcol].dtype == "object"]
    previous_application = pd.get_dummies(previous_application, columns=prev_cat_features)

    avg_prev = previous_application.groupby('SK_ID_CURR').mean()
    cnt_prev = previous_application[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    avg_prev['nb_app'] = cnt_prev['SK_ID_PREV']
    del avg_prev['SK_ID_PREV']

    del previous_application, cnt_prev
    gc.collect()

    print('Reading POS_CASH...')
    pos = pd.read_csv("../data/POS_CASH_balance.csv")

    print('One-hot encoding of categorical feature')
    pos = pd.get_dummies(pos, columns=["NAME_CONTRACT_STATUS"])

    print('compute number of prevs per curr')
    nb_prevs = pos[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    pos['SK_ID_PREV'] = pos['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

    print('Go to averages')
    avg_pos = pos.groupby('SK_ID_CURR').mean()

    del pos, nb_prevs
    gc.collect()

    print('Reading credit card balance...')
    cc_bal = pd.read_csv('../data/credit_card_balance.csv')

    print('One-hot encoding of categorical feature')
    cc_bal = pd.get_dummies(cc_bal, columns=["NAME_CONTRACT_STATUS"])

    nb_prevs = cc_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    cc_bal['SK_ID_PREV'] = cc_bal['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

    print('Compute average')
    avg_cc_bal = cc_bal.groupby('SK_ID_CURR').mean()
    avg_cc_bal.columns = ['cc_bal_' + f_ for f_ in avg_cc_bal.columns]

    del cc_bal, nb_prevs
    gc.collect()

    print('Reading intallments payments')
    inst = pd.read_csv("../data/installments_payments.csv")

    nb_prevs = inst[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    inst['SK_ID_PREV'] = inst['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

    print('Compute average')
    avg_inst = inst.groupby('SK_ID_CURR').mean()
    avg_inst.columns = ['inst_' + f_ for f_ in avg_inst.columns]

    del inst, nb_prevs
    gc.collect()

    print("Reading application train and test data")
    data = pd.read_csv("../data/application_train.csv")
    test = pd.read_csv("../data/application_test.csv")

    y = data['TARGET']
    ids = data['SK_ID_CURR']
    del data['TARGET']

    categorical_features = [f for f in data.columns if data[f].dtype == 'object']
    print("Categorical Features :", categorical_features)
    # one-hot encoding of categorical features
    data = pd.get_dummies(data, columns=categorical_features)
    test = pd.get_dummies(test, columns=categorical_features)

    del data['CODE_GENDER_XNA'], data['NAME_INCOME_TYPE_Maternity leave'], data['NAME_FAMILY_STATUS_Unknown']

    # correlated_features = ["AMT_GOODS_PRICE", "APARTMENTS_MEDI", "APARTMENTS_MODE",
    #                        "BASEMENTAREA_MEDI", "BASEMENTAREA_MODE", "COMMONAREA_MEDI",
    #                        "COMMONAREA_MODE", "ELEVATORS_MEDI", "ELEVATORS_MODE",
    #                        "ENTRANCES_MEDI", "ENTRANCES_MODE", "FLOORSMAX_MEDI",
    #                        "FLOORSMAX_MODE", "FLOORSMIN_MEDI", "FLOORSMIN_MODE",
    #                        "LANDAREA_MEDI", "LANDAREA_MODE", "LIVINGAPARTMENTS_AVG",
    #                        "LIVINGAPARTMENTS_MEDI", "LIVINGAPARTMENTS_MODE", "LIVINGAREA_AVG",
    #                        "LIVINGAREA_MEDI", "LIVINGAREA_MODE", "NONLIVINGAPARTMENTS_MEDI",
    #                        "NONLIVINGAPARTMENTS_MODE", "NONLIVINGAREA_MEDI", "NONLIVINGAREA_MODE",
    #                        "OBS_60_CNT_SOCIAL_CIRCLE", "REGION_RATING_CLIENT_W_CITY", "TOTALAREA_MODE",
    #                        "YEARS_BEGINEXPLUATATION_MEDI", "YEARS_BEGINEXPLUATATION_MODE",
    #                        "YEARS_BUILD_MEDI", "YEARS_BUILD_MODE",
    #                        # Test data constant values of 0
    #                        "FLAG_DOCUMENT_10", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13",
    #                        "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16",
    #                        "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_19", "FLAG_DOCUMENT_2",
    #                        "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21",
    #                        ]
    #
    # for f_ in correlated_features:
    #     del data[f_], test[f_]

    print("Shapes: ", data.shape, test.shape)

    print('Merging all datasets...')

    data = data.merge(right=avg_bureau.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_bureau.reset_index(), how='left', on='SK_ID_CURR')

    data = data.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')

    data = data.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')

    data = data.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')

    data = data.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')

    del avg_bureau, avg_prev, avg_pos, avg_cc_bal, avg_inst
    gc.collect()

    return data, test, y, ids

