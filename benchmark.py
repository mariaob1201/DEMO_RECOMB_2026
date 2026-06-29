"""
benchmark.py — compare ADAPRE (baseline) against alternative causal discovery methods.

Methods compared
----------------
ADAPRE          : adaptive-lambda inspre (this repo) — the baseline
inspre_baseline : uniform-lambda inspre (this repo, adaptive_lambda=False)
DAGMA_control   : DAGMA on control cells only (pure observational)
DAGMA_all       : DAGMA on all cells, ignoring intervention labels
SEN             : Semantic Expression Network (new method, this file)

Usage
-----
1. Run run_adapre_demo.R to generate output/
2. pip install -r requirements.txt
3. python benchmark.py

Outputs
-------
output/benchmark_results.csv     best-F1 metrics per method
output/benchmark_f1_paths.csv    F1 / precision / recall vs hyperparameter
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = "output"
EDGE_THR = 0.001  # matches the threshold used in run_adapre_demo.R


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(output_dir=OUTPUT_DIR):
    X            = pd.read_csv(os.path.join(output_dir, "X.csv")).values
    targets      = pd.read_csv(os.path.join(output_dir, "targets.csv"))["target"].values
    R_hat_tce    = pd.read_csv(os.path.join(output_dir, "R_hat_tce.csv"), index_col=0).values
    G_true       = pd.read_csv(os.path.join(output_dir, "G_true.csv")).values
    G_hat_adapre = pd.read_csv(os.path.join(output_dir, "G_hat_adapre.csv")).values
    gene_names   = [f"V{i+1}" for i in range(G_true.shape[0])]
    return X, targets, R_hat_tce, G_true, G_hat_adapre, gene_names


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(G_hat, G_true, thr=EDGE_THR):
    """Precision, recall, F1, AUROC, SHD on off-diagonal entries."""
    D    = G_true.shape[0]
    mask = ~np.eye(D, dtype=bool)

    g_true = G_true[mask]
    g_hat  = G_hat[mask]

    true_pos = np.abs(g_true) > thr
    pred_pos = np.abs(g_hat)  > thr
    scores   = np.abs(g_hat)

    tp = np.sum(pred_pos & true_pos)
    fp = np.sum(pred_pos & ~true_pos)
    fn = np.sum(~pred_pos & true_pos)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    try:
        auroc = roc_auc_score(true_pos.astype(int), scores)
    except Exception:
        auroc = float("nan")

    shd = int(np.sum(pred_pos != true_pos))
    return dict(precision=precision, recall=recall, F1=f1, AUROC=auroc, SHD=shd)


def best_over_sweep(G_hats_dict, G_true, thr=EDGE_THR):
    """Return metrics at the hyperparameter value that maximises F1."""
    rows = []
    for param, G_hat in G_hats_dict.items():
        m = compute_metrics(G_hat, G_true, thr)
        m["param"] = param
        rows.append(m)
    path_df = pd.DataFrame(rows).sort_values("param")
    best    = path_df.loc[path_df["F1"].idxmax()].to_dict()
    return best, path_df


# ---------------------------------------------------------------------------
# Method: Semantic Expression Network  (SEN)
# ---------------------------------------------------------------------------

def compute_fingerprints(X, targets, gene_names):
    """
    Perturbation fingerprint for each gene.

    delta[i, j] = mean(X_j | gene_i perturbed) - mean(X_j | control)

    Row i encodes "what does the whole transcriptome look like when gene i
    is knocked down?"  This is the raw, unnormalized TCE signal.
    """
    D        = len(gene_names)
    ctrl_idx = targets == "control"
    ctrl_mu  = X[ctrl_idx].mean(axis=0)          # (D,)

    delta = np.zeros((D, D))
    for i, gene in enumerate(gene_names):
        mask = targets == gene
        if mask.sum() > 0:
            delta[i] = X[mask].mean(axis=0) - ctrl_mu

    return delta                                   # (D, D)


def normalize_fingerprints_to_tce(delta):
    """
    Convert raw fingerprints to a naive TCE estimate by normalising
    each row by its own diagonal (self-effect), matching the
    get_tce() formula used in the R simulation code.

    R_naive[i, j] = delta[i, j] / delta[i, i]
    """
    diag_vals  = np.diag(delta)
    diag_safe  = np.where(np.abs(diag_vals) > 1e-8, diag_vals, 1.0)
    R_naive    = delta / diag_safe[:, np.newaxis]
    np.fill_diagonal(R_naive, 1.0)
    return R_naive


def semantic_similarity(delta, n_components):
    """
    Embed gene perturbation fingerprints via PCA and compute pairwise
    cosine similarity in the embedding space.

    Returns
    -------
    S : (D, D) float array, values in [0, 1]
        S[i, j] high  →  gene i and gene j perturb the transcriptome
                          in similar directions (semantically related)
    embeddings : (D, n_components) PCA coordinates
    """
    D            = delta.shape[0]
    n_components = min(n_components, D - 1)

    scaler      = StandardScaler()
    delta_sc    = scaler.fit_transform(delta)

    pca         = PCA(n_components=n_components, random_state=0)
    embeddings  = pca.fit_transform(delta_sc)    # (D, n_components)

    norms       = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    emb_unit    = embeddings / norms
    S           = np.clip(emb_unit @ emb_unit.T, 0, 1)   # (D, D)
    return S, embeddings


def semantic_smooth(R_naive, S):
    """
    Smooth the naive TCE matrix by borrowing signal from semantically
    similar genes.

    R_smooth[i, :] = weighted average of R_naive[k, :] for genes k
                     with high similarity to gene i.

    This denoises the TCE estimate using the global structure of
    perturbation responses rather than per-entry standard errors.
    """
    row_sums  = S.sum(axis=1, keepdims=True) + 1e-8
    S_norm    = S / row_sums                           # row-stochastic (D, D)
    return S_norm @ R_naive                            # (D, D)


def tce_to_direct_effects(R_smooth):
    """
    Recover direct effects from a TCE matrix.

    Matches get_direct() in R:  G = I - R_inv / diag(R_inv)  (column-wise)
    """
    D = R_smooth.shape[0]
    try:
        R_inv = np.linalg.solve(R_smooth, np.eye(D))
    except np.linalg.LinAlgError:
        R_inv = np.linalg.pinv(R_smooth)

    diag_R_inv = np.diag(R_inv)
    diag_safe  = np.where(np.abs(diag_R_inv) > 1e-8, diag_R_inv, 1.0)

    G = np.eye(D) - R_inv / diag_safe[np.newaxis, :]   # column-wise divide
    np.fill_diagonal(G, 0.0)
    return G


def run_semantic_expression_network(X, targets, gene_names,
                                    n_components=10, thr_grid=None):
    """
    Semantic Expression Network (SEN) — causal graph inference from
    perturbation fingerprints.

    Algorithm
    ---------
    1. Compute perturbation fingerprints Δ from raw expression means.
    2. Normalise Δ → naive TCE matrix R_naive (row ÷ self-effect).
    3. Embed genes in PCA space; compute cosine similarity S.
    4. Semantic smoothing: R_smooth = S_row_norm @ R_naive
       — each gene's TCE row is a weighted average of similar genes' rows,
         denoising by exploiting community structure in the network.
    5. Recover direct effects: G = I - R_smooth^{-1} / diag(R_smooth^{-1})
    6. Sweep a threshold on |G| to generate a family of graphs for evaluation.

    Parameters
    ----------
    X            : (N, D) expression matrix
    targets      : (N,) array of intervention labels or "control"
    gene_names   : list of length D
    n_components : PCA components for semantic embedding
    thr_grid     : threshold values to sweep; if None, 30 values are used

    Returns
    -------
    G_hats : dict {threshold: (D, D) G_hat array}
    meta   : dict with intermediate quantities (S, embeddings, R_smooth)
    """
    delta    = compute_fingerprints(X, targets, gene_names)
    R_naive  = normalize_fingerprints_to_tce(delta)
    S, embs  = semantic_similarity(delta, n_components)
    R_smooth = semantic_smooth(R_naive, S)
    G_full   = tce_to_direct_effects(R_smooth)

    if thr_grid is None:
        off_diag    = np.abs(G_full[~np.eye(G_full.shape[0], dtype=bool)])
        thr_max     = np.percentile(off_diag, 99)
        thr_grid    = np.linspace(0.0, thr_max, 40)

    G_hats = {}
    for thr in thr_grid:
        G_t = G_full.copy()
        G_t[np.abs(G_t) < thr] = 0.0
        G_hats[float(thr)] = G_t

    meta = dict(S=S, embeddings=embs, R_smooth=R_smooth, G_full=G_full,
                delta=delta, R_naive=R_naive)
    return G_hats, meta


# ---------------------------------------------------------------------------
# Method: DAGMA variants
# ---------------------------------------------------------------------------

def run_dagma(X_fit, lambda_grid=None, w_threshold=EDGE_THR, label=""):
    try:
        from dagma.linear import DagmaLinear
    except ImportError:
        print(f"[DAGMA{label}] dagma not installed — skipping. pip install dagma")
        return None

    if lambda_grid is None:
        lambda_grid = np.logspace(-2, 0, 12)

    G_hats = {}
    for lam in lambda_grid:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = DagmaLinear(loss_type="l2")
            W     = model.fit(X_fit, lambda1=lam, w_threshold=w_threshold)
        G_hats[float(lam)] = W

    return G_hats


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    print("Loading data from", OUTPUT_DIR)
    X, targets, R_hat_tce, G_true, G_hat_adapre, gene_names = load_data()

    all_results = {}
    all_paths   = {}

    # -- ADAPRE (baseline, pre-fit in R) -------------------------------------
    m = compute_metrics(G_hat_adapre, G_true)
    m["param"] = float("nan")
    all_results["ADAPRE"] = m
    print(f"[ADAPRE]           F1={m['F1']:.4f}  AUROC={m['AUROC']:.4f}  SHD={m['SHD']}")

    # -- SEN: Semantic Expression Network ------------------------------------
    print("\nFitting Semantic Expression Network (SEN)...")
    for n_comp in [5, 10, 20]:
        label = f"SEN_pca{n_comp}"
        G_hats_sen, sen_meta = run_semantic_expression_network(
            X, targets, gene_names, n_components=n_comp
        )
        best_m, path_df = best_over_sweep(G_hats_sen, G_true)
        all_results[label] = best_m
        path_df["method"] = label
        all_paths[label]  = path_df
        print(f"[{label}]    F1={best_m['F1']:.4f}  AUROC={best_m['AUROC']:.4f}  SHD={best_m['SHD']}")

    # Save similarity matrix from best SEN variant for inspection
    np.savetxt(os.path.join(OUTPUT_DIR, "sen_similarity_matrix.csv"),
               sen_meta["S"], delimiter=",")

    # -- DAGMA: control cells only -------------------------------------------
    print("\nFitting DAGMA (control cells only)...")
    X_ctrl = X[targets == "control"]
    G_hats_dagma_ctrl = run_dagma(X_ctrl, label="_control")
    if G_hats_dagma_ctrl is not None:
        best_m, path_df = best_over_sweep(G_hats_dagma_ctrl, G_true)
        all_results["DAGMA_control"] = best_m
        path_df["method"] = "DAGMA_control"
        all_paths["DAGMA_control"] = path_df
        print(f"[DAGMA_control]    F1={best_m['F1']:.4f}  AUROC={best_m['AUROC']:.4f}  SHD={best_m['SHD']}")

    # -- DAGMA: all observations (ignores intervention labels) ---------------
    print("\nFitting DAGMA (all observations, no target labels)...")
    G_hats_dagma_all = run_dagma(X, label="_all")
    if G_hats_dagma_all is not None:
        best_m, path_df = best_over_sweep(G_hats_dagma_all, G_true)
        all_results["DAGMA_all"] = best_m
        path_df["method"] = "DAGMA_all"
        all_paths["DAGMA_all"] = path_df
        print(f"[DAGMA_all]        F1={best_m['F1']:.4f}  AUROC={best_m['AUROC']:.4f}  SHD={best_m['SHD']}")

    # -- Save results --------------------------------------------------------
    cols       = ["precision", "recall", "F1", "AUROC", "SHD"]
    results_df = pd.DataFrame(all_results).T[cols].round(4)

    print("\n=== Benchmark Results (best F1 per method) ===")
    print(results_df.to_string())

    out_path = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
    results_df.to_csv(out_path)
    print(f"\nSaved to {out_path}")

    if all_paths:
        paths_df  = pd.concat(all_paths.values(), ignore_index=True)
        paths_out = os.path.join(OUTPUT_DIR, "benchmark_f1_paths.csv")
        paths_df.to_csv(paths_out, index=False)
        print(f"Saved F1 paths to {paths_out}")


if __name__ == "__main__":
    main()
