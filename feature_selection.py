import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV, SelectKBest, f_regression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

N_FINGERS = 5
CV_FOLDS  = 5
TOP_K     = 100  # features kept per finger before RFECV (union across fingers)

# ── Load ────────────────────────────────────────────────────────────────────
X_train = pd.read_parquet('X_train.parquet')
X_test  = pd.read_parquet('X_test.parquet')
y_train = np.load('y_train.npy')          # shape: (n_epochs, 5)

# Drop any columns that tsfel couldn't compute (NaN from short windows etc.)
X_train.dropna(axis=1, inplace=True)
X_test  = X_test[X_train.columns]

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# ── Estimator ───────────────────────────────────────────────────────────────
# define `estimator` — the base model used inside RFECV.
#
# This choice controls:
#   - Speed   : Ridge/Lasso run in seconds; RandomForest takes minutes
#   - Linearity: Ridge selects linearly predictive features;
#                RandomForest captures non-linear interactions
#   - Scoring : 'r2' is used below (equivalent to r² ≈ correlation² for regression)
#
# Examples:
#   from sklearn.linear_model import Ridge
#   estimator = Ridge(alpha=1.0)
#
#   from sklearn.ensemble import RandomForestRegressor
#   estimator = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)

estimator = Ridge(alpha=1.0)


# ── Step 1: univariate pre-filter (top 10% of features per finger, union) ───
print(f'Pre-filtering with SelectKBest (top {TOP_K} per finger) …')
pre_mask = np.zeros(X_scaled.shape[1], dtype=bool)
for finger in range(N_FINGERS):
    pre = SelectKBest(f_regression, k=TOP_K)
    pre.fit(X_scaled, y_train[:, finger])
    pre_mask |= pre.get_support()

X_pre = X_scaled[:, pre_mask]
print(f'  {pre_mask.sum()} features retained after pre-filter')

# ── Step 2: RFECV on the reduced set ────────────────────────────────────────
selected_mask_pre = np.zeros(X_pre.shape[1], dtype=bool)
cv = KFold(n_splits=CV_FOLDS, shuffle=False)

for finger in range(N_FINGERS):
    print(f'  RFECV finger {finger + 1}/{N_FINGERS} …')
    selector = RFECV(estimator, step=0.2, cv=cv, scoring='r2', n_jobs=-1)
    selector.fit(X_pre, y_train[:, finger])
    selected_mask_pre |= selector.support_
    print(f'    → {selector.support_.sum()} features selected')

# Map back to original feature indices
pre_indices = np.where(pre_mask)[0]
selected_mask = np.zeros(X_scaled.shape[1], dtype=bool)
selected_mask[pre_indices[selected_mask_pre]] = True

print(f'\nTotal features after union: {selected_mask.sum()} / {len(selected_mask)}')

# ── Apply mask and save ─────────────────────────────────────────────────────
selected_cols = X_train.columns[selected_mask].tolist()

X_train_sel = pd.DataFrame(X_scaled[:, selected_mask], columns=selected_cols)
X_test_sel  = pd.DataFrame(
    scaler.transform(X_test)[: , selected_mask], columns=selected_cols
)

X_train_sel.to_parquet('X_train_selected.parquet')
X_test_sel.to_parquet('X_test_selected.parquet')
np.save('selected_features.npy', np.array(selected_cols))
print('Saved X_train_selected.parquet, X_test_selected.parquet, selected_features.npy')
