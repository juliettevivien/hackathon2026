import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import scipy.io
import scipy.signal
import tsfel
import pandas as pd
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────
DATA_PATH = Path('/Users/tanmoysil/Downloads/BCICIV_4_mat')
FS_RAW = 1000
FS = 250                          # target sampling rate after decimation
DECIMATE_Q = FS_RAW // FS         # integer decimation factor (4)
EPOCH_SEC = 1.0
EPOCH_SAMPLES = int(FS * EPOCH_SEC)  # 250 samples at 250 Hz → 1 s windows

BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 100),
}

SUBJECTS = ['sub1_comp.mat', 'sub2_comp.mat', 'sub3_comp.mat']
N_WORKERS = multiprocessing.cpu_count()


# ── Downsample ─────────────────────────────────────────────────────────────
def downsample(data: np.ndarray, q: int = DECIMATE_Q) -> np.ndarray:
    # scipy.signal.decimate applies an anti-aliasing filter before subsampling
    return scipy.signal.decimate(data, q, axis=0, zero_phase=True)


# ── Bandpass filter ─────────────────────────────────────────────────────────
def bandpass_filter(data: np.ndarray, low: float, high: float, fs: int = FS) -> np.ndarray:
    nyq = fs / 2.0
    sos = scipy.signal.butter(4, [low / nyq, high / nyq], btype='bandpass', output='sos')
    return scipy.signal.sosfiltfilt(sos, data, axis=0)


# ── Label alignment ─────────────────────────────────────────────────────────
def window_labels(dg: np.ndarray, epoch_samples: int = EPOCH_SAMPLES) -> np.ndarray:
    n_epochs = len(dg) // epoch_samples
    return dg[: n_epochs * epoch_samples : epoch_samples]


# ── Worker (module-level so it is picklable) ────────────────────────────────
def _band_worker(args: tuple) -> tuple[str, pd.DataFrame]:
    """Extract tsfel features for one band's full signal, letting tsfel window it."""
    signal, band_name, fs = args
    cfg = tsfel.get_features_by_domain("spectral")
    keep = {
        "Spectral centroid", "Spectral entropy", "Spectral roll-off", "Spectral roll-on",
        "Spectral spread", "Spectral skewness", "Spectral kurtosis", "Spectral slope",
        "Power bandwidth", "Median frequency", "Max power spectrum",
    }
    for feat_name in list(cfg.get("spectral", {})):
        if feat_name not in keep:
            cfg["spectral"][feat_name]["use"] = "no"
    n_ch = signal.shape[1]
    signal_df = pd.DataFrame(signal, columns=[f'ch{i}' for i in range(n_ch)])
    feat = tsfel.time_series_features_extractor(
        cfg, signal_df, fs=fs, window_size=EPOCH_SAMPLES, overlap=0, verbose=0,
    )
    feat.columns = [f'{band_name}__{c}' for c in feat.columns]
    return band_name, feat


# ── Parallel feature extraction — one task per band ────────────────────────
def extract_all_features(filtered: dict[str, np.ndarray], fs: int = FS) -> pd.DataFrame:
    tasks = [(sig, band, fs) for band, sig in filtered.items()]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(_band_worker, tasks))
    ordered = {band: df for band, df in results}
    return pd.concat([ordered[b] for b in filtered], axis=1)


# ── Per-subject pipeline ────────────────────────────────────────────────────
def process_subject(mat_file: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    mat = scipy.io.loadmat(DATA_PATH / mat_file)
    train_raw = downsample(mat['train_data'].astype(np.float64))
    test_raw  = downsample(mat['test_data'].astype(np.float64))
    dg        = mat['train_dg'].astype(np.float64)[::DECIMATE_Q]  # stride — no filter needed for slow finger signal

    y_tr = window_labels(dg)

    # Step 1: filter all bands in parallel (threads — scipy releases the GIL)
    print(f'  [{mat_file}] filtering …')
    def _filter_band(args):
        raw, band, low, high = args
        return band, bandpass_filter(raw, low, high)

    filter_tasks = [
        (train_raw, band, low, high) for band, (low, high) in BANDS.items()
    ] + [
        (test_raw, band, low, high) for band, (low, high) in BANDS.items()
    ]
    with ThreadPoolExecutor(max_workers=len(filter_tasks)) as pool:
        filter_results = list(pool.map(_filter_band, filter_tasks))

    n_bands = len(BANDS)
    train_filtered = {band: sig for band, sig in filter_results[:n_bands]}
    test_filtered  = {band: sig for band, sig in filter_results[n_bands:]}

    # Step 2: extract features — one process per band, tsfel handles windowing
    print(f'  [{mat_file}] extracting features (train) …')
    x_tr = extract_all_features(train_filtered)
    print(f'  [{mat_file}] extracting features (test) …')
    x_te = extract_all_features(test_filtered)

    return x_tr, x_te, y_tr


# ── Lagged feature construction ────────────────────────────────────────────
# Brain activity precedes finger movement by ~100–300ms; including past epochs
# as additional features lets the model exploit that causal lead.
def make_lagged_features(
    X: np.ndarray,
    y: np.ndarray | None = None,
    lags: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 10, 15),
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    max_lag = max(lags)

    rows = []
    for t in range(max_lag, len(X)):
        row = np.concatenate([X[t - lag] for lag in lags])
        rows.append(row)

    X_lagged = np.asarray(rows)

    if y is None:
        return X_lagged

    y_lagged = y[max_lag:]
    return X_lagged, y_lagged


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    train_frames, test_frames, label_frames = [], [], []

    for subj in SUBJECTS:
        print(f'\nProcessing {subj} …')
        x_tr, x_te, y_tr = process_subject(subj)
        train_frames.append(x_tr)
        test_frames.append(x_te)
        label_frames.append(y_tr)

        # Save per-subject files
        subj_id = subj.replace('_comp.mat', '')
        x_tr.to_parquet(f'X_train_{subj_id}.parquet')
        x_te.to_parquet(f'X_test_{subj_id}.parquet')
        np.save(f'y_train_{subj_id}.npy', y_tr)
        print(f'  saved {subj_id}: train {x_tr.shape}, test {x_te.shape}, labels {y_tr.shape}')

    x_train = pd.concat(train_frames, ignore_index=True)
    x_test  = pd.concat(test_frames,  ignore_index=True)
    y_train = np.vstack(label_frames)

    print(f'\nX_train : {x_train.shape}')
    print(f'X_test  : {x_test.shape}')
    print(f'y_train : {y_train.shape}')

    x_train.to_parquet('X_train.parquet')
    x_test.to_parquet('X_test.parquet')
    np.save('y_train.npy', y_train)
    print('Saved X_train.parquet, X_test.parquet, y_train.npy')
