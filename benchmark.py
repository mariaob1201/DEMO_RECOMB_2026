"""
benchmark.py — compare ADAPRE against alternative causal discovery methods.

Methods
-------
ADAPRE          baseline (this repo): IV-regression TCE → ADMM (adaptive λ)
SEN_ADMM        Semantic Expression Network TCE → same ADMM (uniform λ)
SEN_ADAPRE      Semantic Expression Network TCE → same ADMM (adaptive λ)
DAGMA_control   DAGMA on control cells only  (pure observational)
DAGMA_all       DAGMA on all cells, ignoring intervention labels

Key design: SEN and ADAPRE share the exact same ADMM solver (R/run_inspre_on_tce.R
called via subprocess). The only variable between them is the TCE estimation
strategy — IV regression (ADAPRE) vs semantic smoothing (SEN).

Usage
-----
1.  Rscript run_adapre_demo.R     # generates output/ CSVs
2.  pip install -r requirements.txt
3.  python benchmark.py

Outputs
-------
output/benchmark_results.csv      best-F1 metrics per method
output/benchmark_f1_paths.csv     metrics across hyperparameter sweep
output/sen_similarity_matrix.csv  gene-gene semantic similarity (D×D)
"""

import os
import subprocess
import tempfile
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = "output"
EDGE_THR   = 0.001   # matches run_adapre_demo.R
R_HELPER   = os.path.join("R", "run_inspre_on_tce.R")


# ---------------------------------------------------------------------------
# ADMM via subprocess  (calls R/run_inspre_on_tce.R)
# ---------------------------------------------------------------------------

def admm_on_tce(R_tce_np, adaptive_lambda=False, beta_obs_np=None, nlambda=20):
    """
    Run fit_inspre_from_R on a (D,D) TCE matrix by calling R as a subprocess.
    Returns dict {lambda_value: (D,D) G_hat}, or None on failure.
    """
    D = R_tce_np.shape[0]
    with tempfile.TemporaryDirectory() as tmp:
        tce_path = os.path.join(tmp, "R_tce.csv")
        out_path = os.path.join(tmp, "G_hat_long.csv")

        gene_cols = [f"V{i+1}" for i in range(D)]
        pd.DataFrame(R_tce_np, index=gene_cols, columns=gene_cols).to_csv(tce_path)

        beta_path = ""
        if beta_obs_np is not None:
            beta_path = os.path.join(tmp, "beta_obs.csv")
            pd.DataFrame({"beta_obs": beta_obs_np}).to_csv(beta_path, index=False)

        cmd = ["Rscript", "--vanilla", R_HELPER,
               tce_path, out_path, str(adaptive_lambda),
               beta_path, str(nlambda)]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[ADMM] R subprocess failed:\n{proc.stderr[-800:]}")
            return None

        long_df = pd.read_csv(out_path)

    # Reconstruct dict {lambda: D×D matrix}
    G_hats = {}
    for lam, grp in long_df.groupby("lambda"):
        G = np.zeros((D, D))
        G[grp["i"].values, grp["j"].values] = grp["G_hat"].values
        G_hats[float(lam)] = G
    return G_hats


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
    D    = G_true.shape[0]
    mask = ~np.eye(D, dtype=bool)

    g_true   = G_true[mask]
    g_hat    = G_hat[mask]
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
    """Return metrics at the hyperparameter that maximises F1."""
    rows = []
    for param, G_hat in G_hats_dict.items():
        m = compute_metrics(G_hat, G_true, thr)
        m["param"] = param
        rows.append(m)
    path_df = pd.DataFrame(rows).sort_values("param")
    best    = path_df.loc[path_df["F1"].idxmax()].to_dict()
    return best, path_df


# ---------------------------------------------------------------------------
# Semantic Expression Network (SEN)
# ---------------------------------------------------------------------------

def compute_fingerprints(X, targets, gene_names):
    """
    Perturbation fingerprint matrix.

    delta[i, j] = mean(X_j | gene_i perturbed) - mean(X_j | control)

    Row i: the whole-transcriptome signature of perturbing gene i.
    This is a noisy, unscaled estimate of the Total Causal Effect row.
    """
    D        = len(gene_names)
    ctrl_mu  = X[targets == "control"].mean(axis=0)
    delta    = np.zeros((D, D))
    for i, gene in enumerate(gene_names):
        mask = targets == gene
        if mask.sum() > 0:
            delta[i] = X[mask].mean(axis=0) - ctrl_mu
    return delta


def fingerprints_to_tce(delta):
    """
    Naive TCE estimate: normalise each row by its own diagonal (self-effect).

    R_naive[i, j] = delta[i, j] / delta[i, i]

    Matches get_tce() in the R simulation code.
    The self-effect delta[i,i] plays the same role as beta_obs in IV regression
    but uses the raw observed mean instead of the IV-corrected estimate.
    """
    diag_vals = np.diag(delta)
    diag_safe = np.where(np.abs(diag_vals) > 1e-8, diag_vals, 1.0)
    R_naive   = delta / diag_safe[:, np.newaxis]
    np.fill_diagonal(R_naive, 1.0)
    return R_naive


def build_semantic_similarity(delta, n_components):
    """
    Embed gene perturbation fingerprints in PCA space and return pairwise
    cosine similarity.

    S[i, j] ≈ 1  →  genes i and j perturb the transcriptome in the same
                     direction (likely share regulators or are co-regulated)
    S[i, j] ≈ 0  →  genes i and j have orthogonal causal signatures
    """
    n_components = min(n_components, delta.shape[0] - 1)
    scaler       = StandardScaler()
    delta_sc     = scaler.fit_transform(delta)

    pca          = PCA(n_components=n_components, random_state=0)
    embeddings   = pca.fit_transform(delta_sc)                   # (D, k)

    norms        = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    emb_unit     = embeddings / norms
    S            = np.clip(emb_unit @ emb_unit.T, 0.0, 1.0)     # (D, D)
    return S, embeddings


def semantic_smooth(R_naive, S):
    """
    Denoise the naive TCE by borrowing signal from semantically similar genes.

    R_smooth[i, :] = weighted-average of R_naive[k, :] over genes k,
                     weighted by S[i, k].

    Intuition: if gene i has a similar perturbation fingerprint to gene j,
    their TCE rows should look similar — pooling reduces estimation noise
    from small per-gene sample sizes.
    """
    row_sums = S.sum(axis=1, keepdims=True) + 1e-8
    S_norm   = S / row_sums          # row-stochastic
    return S_norm @ R_naive          # (D, D)


def sen_tce(X, targets, gene_names, n_components=10):
    """
    Compute the Semantic Expression Network TCE matrix.

    Returns R_sem (D×D) and auxiliary quantities for inspection.
    """
    delta    = compute_fingerprints(X, targets, gene_names)
    R_naive  = fingerprints_to_tce(delta)
    S, embs  = build_semantic_similarity(delta, n_components)
    R_sem    = semantic_smooth(R_naive, S)
    # Self-effect on diagonal: keep =1 to match TCE convention
    np.fill_diagonal(R_sem, 1.0)
    meta = dict(delta=delta, R_naive=R_naive, S=S, embeddings=embs)
    return R_sem, meta


def run_sen_admm(X, targets, gene_names, n_components=10,
                 adaptive_lambda=False, nlambda=20):
    """
    SEN + ADMM: feed the semantic TCE into the same ADMM solver used by ADAPRE.

    When adaptive_lambda=False  →  SEN_ADMM   (uniform λ, different TCE)
    When adaptive_lambda=True   →  SEN_ADAPRE (adaptive λ, different TCE)

    The self-effect magnitudes |delta[i,i]| are used as beta_obs when
    adaptive_lambda=True, analogous to ADAPRE's observed instrument strengths.
    """
    R_sem, meta = sen_tce(X, targets, gene_names, n_components)

    beta_obs_np = None
    if adaptive_lambda:
        # Use |self-effect| as the instrument strength proxy
        beta_obs_np = np.abs(np.diag(meta['delta']))

    G_hats = admm_on_tce(R_sem, adaptive_lambda=adaptive_lambda,
                          beta_obs_np=beta_obs_np, nlambda=nlambda)
    return G_hats, meta


# ---------------------------------------------------------------------------
# DAGMA variants
# ---------------------------------------------------------------------------

def run_dagma(X_fit, lambda_grid=None, w_threshold=EDGE_THR, label=""):
    try:
        from dagma.linear import DagmaLinear
    except ImportError:
        print(f"[DAGMA{label}] not installed — skipping.  pip install dagma")
        return None

    if lambda_grid is None:
        lambda_grid = np.logspace(-2, 0, 12)

    G_hats = {}
    for lam in lambda_grid:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            W = DagmaLinear(loss_type="l2").fit(
                X_fit, lambda1=lam, w_threshold=w_threshold)
        G_hats[float(lam)] = W
    return G_hats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data from", OUTPUT_DIR)
    X, targets, R_hat_tce, G_true, G_hat_adapre, gene_names = load_data()

    all_results = {}
    all_paths   = {}

    def record(label, G_hats_dict):
        best_m, path_df = best_over_sweep(G_hats_dict, G_true)
        all_results[label] = best_m
        path_df["method"]  = label
        all_paths[label]   = path_df
        print(f"[{label:<20}]  "
              f"F1={best_m['F1']:.4f}  "
              f"AUROC={best_m['AUROC']:.4f}  "
              f"SHD={int(best_m['SHD'])}")

    # -- ADAPRE (baseline, pre-fit in R) ------------------------------------
    m = compute_metrics(G_hat_adapre, G_true)
    m["param"] = float("nan")
    all_results["ADAPRE"] = m
    print(f"[{'ADAPRE':<20}]  "
          f"F1={m['F1']:.4f}  AUROC={m['AUROC']:.4f}  SHD={m['SHD']}")

    # -- SEN + ADMM (uniform λ) — same solver as ADAPRE, semantic TCE -------
    n_pca = 10   # number of PCA components for semantic embedding
    print(f"\nFitting SEN_ADMM  (semantic TCE, uniform λ,   pca={n_pca})...")
    G_hats_sen, sen_meta = run_sen_admm(X, targets, gene_names,
                                        n_components=n_pca,
                                        adaptive_lambda=False)
    if G_hats_sen:
        record("SEN_ADMM", G_hats_sen)
        np.savetxt(os.path.join(OUTPUT_DIR, "sen_similarity_matrix.csv"),
                   sen_meta["S"], delimiter=",")

    # -- SEN + ADAPRE (adaptive λ) — semantic TCE + adaptive λ ---------------
    print(f"\nFitting SEN_ADAPRE (semantic TCE, adaptive λ, pca={n_pca})...")
    G_hats_sen_adap, _ = run_sen_admm(X, targets, gene_names,
                                       n_components=n_pca,
                                       adaptive_lambda=True)
    if G_hats_sen_adap:
        record("SEN_ADAPRE", G_hats_sen_adap)

    # -- DAGMA: control cells only ------------------------------------------
    print("\nFitting DAGMA (control cells only)...")
    G_hats_dagma_ctrl = run_dagma(X[targets == "control"], label="_control")
    if G_hats_dagma_ctrl:
        record("DAGMA_control", G_hats_dagma_ctrl)

    # -- DAGMA: all observations (ignores labels) ----------------------------
    print("\nFitting DAGMA (all observations, no labels)...")
    G_hats_dagma_all = run_dagma(X, label="_all")
    if G_hats_dagma_all:
        record("DAGMA_all", G_hats_dagma_all)

    # -- Summary table -------------------------------------------------------
    cols       = ["precision", "recall", "F1", "AUROC", "SHD"]
    results_df = pd.DataFrame(all_results).T[cols].round(4)

    print("\n=== Benchmark Results (best F1 per method) ===")
    print(results_df.to_string())

    out = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
    results_df.to_csv(out)
    print(f"\nSaved {out}")

    if all_paths:
        paths_df = pd.concat(all_paths.values(), ignore_index=True)
        paths_out = os.path.join(OUTPUT_DIR, "benchmark_f1_paths.csv")
        paths_df.to_csv(paths_out, index=False)
        print(f"Saved {paths_out}")


if __name__ == "__main__":
    main()
