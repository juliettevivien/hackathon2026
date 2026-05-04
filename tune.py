import numpy as np
import pandas as pd
import optuna
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

TRUE_LABELS_PATH = '/Users/tanmoysil/Downloads/true_labels'
N_FINGERS  = 5
CV_FOLDS   = 5
N_TRIALS   = 50
SUBJECTS   = ['sub1', 'sub2', 'sub3']
DECIMATE_Q = 4
EPOCH_SAMPLES = 250  # 1s at 250 Hz


def load_true_labels(subj: str) -> np.ndarray:
    import scipy.io
    mat = scipy.io.loadmat(f'{TRUE_LABELS_PATH}/{subj}_testlabels.mat')
    dg = mat['test_dg'].astype(np.float64)
    # stride downsample then window — same as training labels
    dg = dg[::DECIMATE_Q]
    n_epochs = len(dg) // EPOCH_SAMPLES
    return dg[: n_epochs * EPOCH_SAMPLES : EPOCH_SAMPLES]


def mean_pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean([pearsonr(y_true[:, f], y_pred[:, f])[0] for f in range(N_FINGERS)])


def make_objective(X_train, y_train):
    cv = KFold(n_splits=CV_FOLDS, shuffle=False)

    def objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 100, 600),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth':         trial.suggest_int('max_depth', 3, 8),
            'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight':  trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'tree_method': 'hist',
            'n_jobs': -1,
            'random_state': 42,
        }
        scores = []
        for tr, va in cv.split(X_train):
            fold_r = []
            for finger in range(N_FINGERS):
                model = XGBRegressor(**params)
                model.fit(X_train[tr], y_train[tr, finger])
                pred = model.predict(X_train[va])
                fold_r.append(pearsonr(y_train[va, finger], pred)[0])
            scores.append(np.mean(fold_r))
        return np.mean(scores)

    return objective


# ── Per-subject tuning ───────────────────────────────────────────────────────
results = {}

for subj in SUBJECTS:
    print(f'\n── {subj} ──────────────────────────────────────')
    X_train = pd.read_parquet(f'X_train_{subj}_selected.parquet').values
    X_test  = pd.read_parquet(f'X_test_{subj}_selected.parquet').values
    y_train = np.load(f'y_train_{subj}.npy')
    y_test  = load_true_labels(subj)

    print(f'  X_train {X_train.shape}  X_test {X_test.shape}  y_test {y_test.shape}')

    study = optuna.create_study(direction='maximize')
    study.optimize(make_objective(X_train, y_train), n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_params
    print(f'  Best CV r : {study.best_value:.4f}')
    print(f'  Best params: {best}')

    # Retrain on full training set with best params
    y_pred = np.zeros((X_test.shape[0], N_FINGERS))
    for finger in range(N_FINGERS):
        model = XGBRegressor(**best, tree_method='hist', n_jobs=-1, random_state=42)
        model.fit(X_train, y_train[:, finger])
        y_pred[:, finger] = model.predict(X_test)

    test_r = mean_pearson_r(y_test, y_pred)
    print(f'  Test r (true labels): {test_r:.4f}')
    results[subj] = {'cv_r': study.best_value, 'test_r': test_r, 'params': best}

    np.save(f'y_pred_{subj}.npy', y_pred)
    pd.DataFrame(y_pred, columns=[f'finger_{i+1}' for i in range(N_FINGERS)]).to_csv(
        f'predictions_{subj}.csv', index=False
    )

print('\n══ Summary ══')
for subj, res in results.items():
    print(f'  {subj}  CV r={res["cv_r"]:.4f}  Test r={res["test_r"]:.4f}')
print(f'  Overall test r: {np.mean([r["test_r"] for r in results.values()]):.4f}')
