"""Feature 2: classical machine-learning baseline.

Images / frames : HOG features [Dalal & Triggs 2005] -> SVM
Audio           : MFCC statistics (mean + std over time) -> SVM
Shows ML and DL side by side and why deep learning is needed.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm.auto import tqdm

from .classifier import metrics_from
from .config import NUM_CLASSES
from .utils import read_rgb


def hog_features(path, size=128):
    g = cv2.cvtColor(read_rgb(path), cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (size, size))
    return hog(g, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))


def mfcc_features(npy_path, n_mfcc=20):
    import librosa
    S_db = np.load(npy_path).astype(np.float32)
    m = librosa.feature.mfcc(S=S_db, n_mfcc=n_mfcc)
    d = librosa.feature.delta(m) if m.shape[1] >= 9 else np.zeros_like(m)
    return np.concatenate([m.mean(1), m.std(1), d.mean(1), d.std(1)])


def _features(df):
    feats = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="features", leave=False):
        feats.append(mfcc_features(r["npy_path"]) if r["modality"] == "audio" else hog_features(r["path"]))
    return np.array(feats)


def run_svm(train_df, test_df, C=10.0):
    if len(train_df) == 0 or len(test_df) == 0:
        return None
    Xtr, Xte = _features(train_df), _features(test_df)
    ytr, yte = train_df["label_idx"].values, test_df["label_idx"].values
    n_comp = min(100, Xtr.shape[1], len(Xtr) - 1)
    clf = make_pipeline(StandardScaler(), PCA(n_components=n_comp, random_state=0),
                        SVC(C=C, kernel="rbf", probability=False))
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    probs = np.eye(NUM_CLASSES)[pred]
    return metrics_from(yte, probs)
