"""Section 3.1: standardize the macro series, run PCA, and keep enough
components to explain >= PCA_VARIANCE_TARGET of the variance."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import PCA_VARIANCE_TARGET, PCA_MAX_COMPONENTS


def run_pca(macro_returns):
    """Returns (component scores, explained-variance table, n_keep)."""
    X = StandardScaler().fit_transform(macro_returns.values)

    pca_full = PCA(n_components=min(PCA_MAX_COMPONENTS, X.shape[1]))
    pca_full.fit(X)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cum_var, PCA_VARIANCE_TARGET) + 1)
    n_keep = min(n_keep, PCA_MAX_COMPONENTS)

    pca = PCA(n_components=n_keep)
    scores = pca.fit_transform(X)
    components = pd.DataFrame(
        scores, index=macro_returns.index,
        columns=[f"PC{i+1}" for i in range(n_keep)],
    )

    variance = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(cum_var))],
        "explained_variance_ratio": pca_full.explained_variance_ratio_,
        "cumulative": cum_var,
        "retained": [i < n_keep for i in range(len(cum_var))],
    })
    print(f"PCA: kept {n_keep} components explaining {cum_var[n_keep-1]*100:.1f}% of variance")
    return components, variance, n_keep
