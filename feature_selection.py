import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

N_FINGERS = 5
TOP_K     = 100  # features selected per finger (union across fingers)

# ── Load ────────────────────────────────────────────────────────────────────
X_train = pd.read_parquet('X_train.parquet')
X_test  = pd.read_parquet('X_test.parquet')
y_train = np.load('y_train.npy')   # (n_epochs, 5)

X_train.dropna(axis=1, inplace=True)
X_test = X_test[X_train.columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# ── SelectKBest per finger → union ──────────────────────────────────────────
print(f'Selecting top {TOP_K} features per finger …')
mask = np.zeros(X_scaled.shape[1], dtype=bool)
for finger in range(N_FINGERS):
    sel = SelectKBest(f_regression, k=TOP_K)
    sel.fit(X_scaled, y_train[:, finger])
    mask |= sel.get_support()

print(f'  {mask.sum()} features selected after union')

# ── Apply and save ───────────────────────────────────────────────────────────
selected_cols = X_train.columns[mask].tolist()

X_train_sel = pd.DataFrame(X_scaled[:, mask], columns=selected_cols)
X_test_sel  = pd.DataFrame(scaler.transform(X_test)[:, mask], columns=selected_cols)

X_train_sel.to_parquet('X_train_selected.parquet')
X_test_sel.to_parquet('X_test_selected.parquet')
np.save('selected_features.npy', np.array(selected_cols))
print('Saved X_train_selected.parquet, X_test_selected.parquet, selected_features.npy')
