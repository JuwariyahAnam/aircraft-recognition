"""Step 8 + feature 9: learn from the unlabelled 40% (clustering + pseudo-labels).

1. Pass unlabelled samples through the trained CNN, take the penultimate features.
2. Group them with K-means (k = 4); compare with DBSCAN using the silhouette score.
3. Keep model predictions with confidence > 0.90 as pseudo-labels.
4. Add them to the training set and retrain.
5. Report accuracy before/after and draw a t-SNE plot.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .classifier import make_loader
from .config import CLASSES, CLASS_TO_IDX, NUM_CLASSES, PRETTY, Config


@torch.no_grad()
def extract_features(model, df, cfg: Config, mask_mode="none"):
    """Return (features, probabilities) for every row of df."""
    device = next(model.parameters()).device
    model.eval()
    loader = make_loader(df, cfg, False, 64, "label_idx", mask_mode)
    feats, probs = [], []
    for x, _ in loader:
        x = x.to(device)
        f = model.embed(x)
        feats.append(f.float().cpu().numpy())
        probs.append(torch.softmax(model.head(f).float(), 1).cpu().numpy())
    if not feats:
        return np.zeros((0, 1)), np.zeros((0, NUM_CLASSES))
    return np.concatenate(feats), np.concatenate(probs)


def _purity(clusters, labels):
    tot = 0
    for c in np.unique(clusters):
        m = clusters == c
        tot += np.bincount(labels[m]).max()
    return tot / len(labels)


def cluster_analysis(feats, true_idx=None, n_clusters=4, seed=42):
    """K-means vs DBSCAN. `true_idx` (hidden labels) is used ONLY to report purity."""
    out = {}
    if len(feats) < max(n_clusters + 1, 5):
        return out, np.zeros(len(feats), int)
    X = StandardScaler().fit_transform(feats)
    X = PCA(n_components=min(50, X.shape[1], len(X) - 1), random_state=seed).fit_transform(X)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(X)
    out["kmeans"] = {"silhouette": float(silhouette_score(X, km.labels_)), "n_clusters": n_clusters}
    # DBSCAN: eps from the "knee" of the 5-nearest-neighbour distances (90th percentile)
    k = min(5, len(X) - 1)
    d = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)[0][:, -1]
    eps = float(np.percentile(d, 90))
    db = DBSCAN(eps=eps, min_samples=k).fit(X)
    core = db.labels_ != -1
    n_db = len(set(db.labels_[core]))
    out["dbscan"] = {"eps": eps, "n_clusters": int(n_db), "noise_fraction": float(1 - core.mean()),
                     "silhouette": float(silhouette_score(X[core], db.labels_[core]))
                     if n_db >= 2 and core.sum() > n_db else float("nan")}
    if true_idx is not None:
        out["kmeans"]["purity"] = float(_purity(km.labels_, true_idx))
        out["kmeans"]["ARI"] = float(adjusted_rand_score(true_idx, km.labels_))
        if core.any():
            out["dbscan"]["purity_non_noise"] = float(_purity(db.labels_[core], true_idx[core]))
    return out, km.labels_


def make_pseudo_labels(unl_df, probs, clusters, threshold=0.9, require_cluster_agreement=False):
    """Confident predictions become labels. Optionally also require the prediction to
    match its K-means cluster's majority prediction (a stricter filter)."""
    pred = probs.argmax(1)
    conf = probs.max(1)
    keep = conf >= threshold
    agree = np.ones(len(pred), bool)
    if len(clusters):
        cluster_major = {c: np.bincount(pred[clusters == c], minlength=NUM_CLASSES).argmax()
                         for c in np.unique(clusters)}
        agree = np.array([pred[i] == cluster_major[clusters[i]] for i in range(len(pred))])
        if require_cluster_agreement:
            keep &= agree
    df = unl_df.reset_index(drop=True).copy()
    df["pseudo_label_idx"] = pred
    df["confidence"] = conf
    df["cluster"] = clusters if len(clusters) else -1
    df["cluster_agrees"] = agree
    kept = df[keep].copy()
    kept["label_idx"] = kept["pseudo_label_idx"]       # train on the pseudo label
    report = {"n_unlabelled": int(len(df)), "n_pseudo": int(keep.sum()),
              "kept_fraction": float(keep.mean()) if len(df) else 0.0,
              "cluster_agreement": float(agree.mean()) if len(df) else 0.0}
    if "true_label" in df and len(kept):   # evaluation only
        truth = kept["true_label"].map(CLASS_TO_IDX).values
        report["pseudo_label_accuracy"] = float((truth == kept["pseudo_label_idx"].values).mean())
    return kept, df, report


def plot_tsne(feats, clusters, true_idx, path, seed=42, title=""):
    if len(feats) < 10:
        return
    X = PCA(n_components=min(50, feats.shape[1], len(feats) - 1), random_state=seed).fit_transform(
        StandardScaler().fit_transform(feats))
    Z = TSNE(n_components=2, perplexity=min(30, max(5, len(X) // 4)), random_state=seed,
             init="pca").fit_transform(X)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for c in np.unique(clusters):
        m = clusters == c
        ax[0].scatter(Z[m, 0], Z[m, 1], s=8, label=f"cluster {c}")
    ax[0].set_title("K-means clusters (no labels used)")
    ax[0].legend(fontsize=7, markerscale=2)
    if true_idx is not None:
        for i, c in enumerate(CLASSES):
            m = true_idx == i
            if m.any():
                ax[1].scatter(Z[m, 0], Z[m, 1], s=8, label=PRETTY[c])
        ax[1].set_title("Hidden true labels (for checking only)")
        ax[1].legend(fontsize=7, markerscale=2)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title or "t-SNE of CNN features for unlabelled samples")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
