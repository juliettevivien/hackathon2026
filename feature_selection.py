import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

N_FINGERS = 5
CV_FOLDS  = 5

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

from xgboost import XGBRegressor
estimator = XGBRegressor(n_estimators=100, tree_method='hist', n_jobs=-1, random_state=42)


# ── RFECV per finger → union of selected features ───────────────────────────
selected_mask = np.zeros(X_scaled.shape[1], dtype=bool)
cv = KFold(n_splits=CV_FOLDS, shuffle=False)

for finger in range(N_FINGERS):
    print(f'  RFECV finger {finger + 1}/{N_FINGERS} …')
    selector = RFECV(estimator, step=0.2, cv=cv, scoring='r2', n_jobs=-1)
    selector.fit(X_scaled, y_train[:, finger])
    selected_mask |= selector.support_
    print(f'    → {selector.support_.sum()} features selected')

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
