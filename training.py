import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

N_FINGERS = 5
CV_FOLDS  = 5

# ── Load ────────────────────────────────────────────────────────────────────
X_train = pd.read_parquet('X_train_selected.parquet').values
X_test  = pd.read_parquet('X_test_selected.parquet').values
y_train = np.load('y_train.npy')   # (n_epochs, 5)

print(f'X_train : {X_train.shape}')
print(f'X_test  : {X_test.shape}')
print(f'y_train : {y_train.shape}')

# ── Cross-validated Pearson r ────────────────────────────────────────────────
cv = KFold(n_splits=CV_FOLDS, shuffle=False)
cv_scores = np.zeros((CV_FOLDS, N_FINGERS))

print('\nCross-validation …')
for fold, (tr, va) in enumerate(cv.split(X_train)):
    for finger in range(N_FINGERS):
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train[tr], y_train[tr, finger])
        pred = model.predict(X_train[va])
        r, _ = pearsonr(y_train[va, finger], pred)
        cv_scores[fold, finger] = r

    fold_mean = cv_scores[fold].mean()
    print(f'  fold {fold + 1}: r = {cv_scores[fold]} → mean {fold_mean:.4f}')

print(f'\nCV mean r per finger : {cv_scores.mean(axis=0).round(4)}')
print(f'Overall CV mean r    : {cv_scores.mean():.4f}')

# ── Retrain on full train set and predict test ───────────────────────────────
print('\nRetraining on full dataset …')
y_pred = np.zeros((X_test.shape[0], N_FINGERS))

for finger in range(N_FINGERS):
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train[:, finger])
    y_pred[:, finger] = model.predict(X_test)
    print(f'  finger {finger + 1} done')

# ── Save predictions ─────────────────────────────────────────────────────────
np.save('y_pred.npy', y_pred)
pd.DataFrame(y_pred, columns=[f'finger_{i+1}' for i in range(N_FINGERS)]).to_csv('predictions.csv', index=False)
print('\nSaved y_pred.npy, predictions.csv')
