import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

N_FINGERS = 5
CV_FOLDS  = 5
SUBJECTS  = ['sub1', 'sub2', 'sub3']

all_cv_r = []

for subj in SUBJECTS:
    print(f'\n── {subj} ──────────────────────────────────────')
    X_train = pd.read_parquet(f'X_train_{subj}_selected.parquet').values
    X_test  = pd.read_parquet(f'X_test_{subj}_selected.parquet').values
    y_train = np.load(f'y_train_{subj}.npy')

    print(f'  X_train {X_train.shape}  y_train {y_train.shape}')

    # ── Cross-validated Pearson r ──────────────────────────────────────────
    cv = KFold(n_splits=CV_FOLDS, shuffle=False)
    cv_scores = np.zeros((CV_FOLDS, N_FINGERS))

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

        print(f'  fold {fold + 1}: r = {cv_scores[fold].round(4)} → mean {cv_scores[fold].mean():.4f}')

    mean_r = cv_scores.mean(axis=0)
    print(f'  CV mean r per finger : {mean_r.round(4)}')
    print(f'  CV mean r overall    : {cv_scores.mean():.4f}')
    all_cv_r.append(cv_scores.mean())

    # ── Retrain on full data and predict test ──────────────────────────────
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

    np.save(f'y_pred_{subj}.npy', y_pred)
    pd.DataFrame(y_pred, columns=[f'finger_{i+1}' for i in range(N_FINGERS)]).to_csv(
        f'predictions_{subj}.csv', index=False
    )
    print(f'  saved y_pred_{subj}.npy, predictions_{subj}.csv')

print(f'\n══ Overall mean CV r across subjects: {np.mean(all_cv_r):.4f} ══')
