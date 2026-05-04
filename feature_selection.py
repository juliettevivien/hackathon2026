import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

N_FINGERS = 5
TOP_K     = 100
SUBJECTS  = ['sub1', 'sub2', 'sub3']

for subj in SUBJECTS:
    print(f'\n── {subj} ──────────────────────────────────────')
    X_train = pd.read_parquet(f'X_train_{subj}.parquet')
    X_test  = pd.read_parquet(f'X_test_{subj}.parquet')
    y_train = np.load(f'y_train_{subj}.npy')

    X_train.dropna(axis=1, inplace=True)
    X_test = X_test[X_train.columns]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    mask = np.zeros(X_scaled.shape[1], dtype=bool)
    for finger in range(N_FINGERS):
        sel = SelectKBest(f_regression, k=TOP_K)
        sel.fit(X_scaled, y_train[:, finger])
        mask |= sel.get_support()

    print(f'  {mask.sum()} features selected after union')
    selected_cols = X_train.columns[mask].tolist()

    X_train_sel = pd.DataFrame(X_scaled[:, mask], columns=selected_cols)
    X_test_sel  = pd.DataFrame(scaler.transform(X_test)[:, mask], columns=selected_cols)

    X_train_sel.to_parquet(f'X_train_{subj}_selected.parquet')
    X_test_sel.to_parquet(f'X_test_{subj}_selected.parquet')
    np.save(f'selected_features_{subj}.npy', np.array(selected_cols))
    print(f'  saved X_train_{subj}_selected.parquet, X_test_{subj}_selected.parquet')
