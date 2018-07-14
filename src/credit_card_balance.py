import pandas as pd
import gc

gc.enable()


def f_dpd(DPD):
    # DPD is a series of values of SK_DPD for each of the groupby combination
    # We convert it to a list to get the number of SK_DPD values NOT EQUALS ZERO
    x = DPD.tolist()
    c = 0
    for i, j in enumerate(x):
        if j != 0:
            c += 1

    return c


# % of minimum payments missed
def f_pay(min_pay, total_pay): 
    M = min_pay.tolist()
    T = total_pay.tolist()
    P = len(M)
    c = 0
    # Find the count of transactions when Payment made is less than Minimum Payment
    for i in range(len(M)):
        if T[i] < M[i]:
            c += 1
    return (100 * c) / P


def get_credit_balance_features(num_rows=None, nan_as_category=True):
    print('Reading credit card balance...')
    cc_bal = pd.read_csv('../data/credit_card_balance.csv', nrows=num_rows)

    # https://www.kaggle.com/shanth84/credit-card-balance-feature-engineering
    # No of loans per customer
    nb_loans = cc_bal.groupby(by=['SK_ID_CURR'])['SK_ID_PREV'].nunique().reset_index().rename(index=str, columns={
        'SK_ID_PREV': 'NO_LOANS'})
    cc_bal = cc_bal.merge(nb_loans, on=['SK_ID_CURR'], how='left')
    nb_loans1 = cc_bal.groupby(by=['SK_ID_CURR'])['SK_ID_PREV'].count().reset_index().rename(index=str, columns={
        'SK_ID_PREV': 'NO_LOANS1'})
    cc_bal = cc_bal.merge(nb_loans1, on=['SK_ID_CURR'], how='left')
    del nb_loans, nb_loans1

    # Rate of installments paid per loan per customer
    no_installments = cc_bal.groupby(by=['SK_ID_CURR', 'SK_ID_PREV'])[
        'CNT_INSTALMENT_MATURE_CUM'].max().reset_index().rename(index=str,
                                                                columns={'CNT_INSTALMENT_MATURE_CUM': 'NO_INSTALMENTS'})
    no_installments1 = no_installments.groupby(by=['SK_ID_CURR'])['NO_INSTALMENTS'].sum().reset_index().rename(
        index=str, columns={'NO_INSTALMENTS': 'TOTAL_INSTALMENTS'})
    cc_bal = cc_bal.merge(no_installments1, on=['SK_ID_CURR'], how='left')
    del no_installments, no_installments1
    gc.collect()

    # Average number of installments paid per loan
    cc_bal['INSTALMENTS_PER_LOAN'] = (cc_bal['TOTAL_INSTALMENTS'] / cc_bal['NO_LOANS']).astype('uint32')
    del cc_bal['TOTAL_INSTALMENTS'], cc_bal['NO_LOANS']
    gc.collect()

    # Number of times days past due occurred
    grp = cc_bal.groupby(by=['SK_ID_CURR', 'SK_ID_PREV']).apply(lambda x: f_dpd(x.SK_DPD)).reset_index().rename(
        index=str, columns={0: 'NO_DPD'})
    grp1 = grp.groupby(by=['SK_ID_CURR'])['NO_DPD'].mean().reset_index().rename(index=str,
                                                                                columns={'NO_DPD': 'DPD_COUNT'})

    cc_bal = cc_bal.merge(grp1, on=['SK_ID_CURR'], how='left')
    del grp, grp1
    gc.collect()

    # Average of days past due per customer
    grp = cc_bal.groupby(by=['SK_ID_CURR'])['SK_DPD'].mean().reset_index().rename(index=str,
                                                                                  columns={'SK_DPD': 'AVG_DPD'})
    cc_bal = cc_bal.merge(grp, on=['SK_ID_CURR'], how='left')
    del grp
    gc.collect()

    grp = cc_bal.groupby(by=['SK_ID_CURR']).apply(
        lambda x: f_pay(x.AMT_INST_MIN_REGULARITY, x.AMT_PAYMENT_CURRENT)).reset_index().rename(index=str, columns={
        0: 'PERCENTAGE_MISSED_PAYMENTS'})
    cc_bal = cc_bal.merge(grp, on=['SK_ID_CURR'], how='left')
    del grp
    gc.collect()

    grp = cc_bal.groupby(by=['SK_ID_CURR'])['AMT_DRAWINGS_ATM_CURRENT'].sum().reset_index().rename(index=str, columns={
        'AMT_DRAWINGS_ATM_CURRENT': 'DRAWINGS_ATM'})
    cc_bal = cc_bal.merge(grp, on=['SK_ID_CURR'], how='left')
    del grp
    gc.collect()

    grp = cc_bal.groupby(by=['SK_ID_CURR'])['AMT_DRAWINGS_CURRENT'].sum().reset_index().rename(index=str, columns={
        'AMT_DRAWINGS_CURRENT': 'DRAWINGS_TOTAL'})
    cc_bal = cc_bal.merge(grp, on=['SK_ID_CURR'], how='left')
    del grp
    gc.collect()

    cc_bal['CASH_CARD_RATIO1'] = (cc_bal['DRAWINGS_ATM'] / cc_bal['DRAWINGS_TOTAL']) * 100
    del cc_bal['DRAWINGS_ATM']
    del cc_bal['DRAWINGS_TOTAL']
    gc.collect()

    grp = cc_bal.groupby(by=['SK_ID_CURR'])['CASH_CARD_RATIO1'].mean().reset_index().rename(index=str, columns={
        'CASH_CARD_RATIO1': 'CASH_CARD_RATIO'})
    cc_bal = cc_bal.merge(grp, on=['SK_ID_CURR'], how='left')
    del grp
    gc.collect()

    del cc_bal['CASH_CARD_RATIO1']
    gc.collect()

    grp = cc_bal.groupby(by=['SK_ID_CURR'])['AMT_DRAWINGS_CURRENT'].sum().reset_index().rename(index=str, columns={
        'AMT_DRAWINGS_CURRENT': 'TOTAL_DRAWINGS'})
    cc_bal = cc_bal.merge(grp, on=['SK_ID_CURR'], how='left')
    del grp
    gc.collect()

    grp = cc_bal.groupby(by=['SK_ID_CURR'])['CNT_DRAWINGS_CURRENT'].sum().reset_index().rename(index=str, columns={
        'CNT_DRAWINGS_CURRENT': 'NO_DRAWINGS'})
    cc_bal = cc_bal.merge(grp, on=['SK_ID_CURR'], how='left')
    del grp
    gc.collect()

    cc_bal['DRAWINGS_RATIO1'] = (cc_bal['TOTAL_DRAWINGS'] / cc_bal['NO_DRAWINGS']) * 100
    del cc_bal['TOTAL_DRAWINGS']
    del cc_bal['NO_DRAWINGS']
    gc.collect()

    grp = cc_bal.groupby(by=['SK_ID_CURR'])['DRAWINGS_RATIO1'].mean().reset_index().rename(index=str, columns={
        'DRAWINGS_RATIO1': 'DRAWINGS_RATIO'})
    cc_bal = cc_bal.merge(grp, on=['SK_ID_CURR'], how='left')
    del grp
    gc.collect()

    del cc_bal['DRAWINGS_RATIO1']

    print('One-hot encoding of categorical feature')
    cc_bal = pd.get_dummies(cc_bal, columns=["NAME_CONTRACT_STATUS"])

    nb_prevs = cc_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    cc_bal['SK_ID_PREV'] = cc_bal['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

    print('Compute average')
    avg_cc_bal = cc_bal.groupby('SK_ID_CURR').mean()
    avg_cc_bal.columns = ['cc_bal_' + f_ for f_ in avg_cc_bal.columns]

    del cc_bal, nb_prevs
    gc.collect()
    return avg_cc_bal