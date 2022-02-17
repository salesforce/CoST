import pandas as pd
import numpy as np

calendar = pd.read_csv('calendar.csv', index_col='date', parse_dates=True)
train_validation = pd.read_csv('sales_train_validation.csv')
train_evaluation = pd.read_csv('sales_train_evaluation.csv')
test_validation = pd.read_csv('sales_test_validation.csv')
test_evaluation = pd.read_csv('sales_test_evaluation.csv')

all_data = pd.merge(
    train_evaluation,
    test_evaluation,
    how="inner",
    on=None,
    left_on=['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
    right_on=['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
    sort=False,
    suffixes=("_x", "_y"),
    copy=True,
    indicator=False,
    validate=None,
)

groups = {
    'l1': ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
    'l2': ['state_id'],
    'l3': ['store_id'],
    'l4': ['cat_id'],
    'l5': ['dept_id'],
    'l6': ['state_id', 'cat_id'],
    'l7': ['state_id', 'dept_id'],
    'l8': ['store_id', 'cat_id'],
    'l9': ['store_id', 'dept_id'],
    'l10': ['item_id'],
}

for k, v in groups.items():
    if k == 'l1':
        grouped_data = all_data.drop(columns=v).sum().to_frame(name='total')
    else:
        grouped_data = all_data.groupby(v).sum().transpose()
    grouped_data['date'] = calendar.index
    grouped_data = grouped_data.set_index('date')

    if isinstance(grouped_data.columns, pd.MultiIndex):
        grouped_data.columns = [c[0] + "_" + c[1] for c in grouped_data.columns]

    grouped_data.to_csv(f'M5-{k}.csv', index=True)
