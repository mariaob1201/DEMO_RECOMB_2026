"""
tensor_regression.py — Population-scale Perturb-seq: tensor regression of the
TCE tensor on SNP covariates.

Model
-----
For each edge (i, j) independently across K donors:

    R_hat[k, i, j] = mu[i, j]  +  sum_m  B[m, i, j] * snps[k, m]  +  eps[k, i, j]

where
    R_hat  : K × D × D  total-causal-effect tensor  (one IV-regression matrix per donor)
    snps   : K × M      standardised genotype matrix (values in {0, 1, 2} before scaling)
    B      : M × D × D  coefficient tensor           (target: which SNPs shift which edge)
    mu     : D × D      population-mean edge weights

Per-edge OLS is fit by least-squares across the K donor axis.  A Benjamini-
Hochberg FDR correction is applied across all M × D(D-1) (SNP, edge) tests.

Outputs  (all written to output/population/)
-------
tensor_B_hat.csv        — long-format  (snp, i, j, B_hat)
tensor_pvals.csv        — long-format  (snp, i, j, p_val, q_val)
tensor_detection.csv    — ground-truth heritable (edge, SNP) pairs with p/q
tensor_summary.csv      — scalar performance: precision, recall, F1 at q<0.1
tensor_edge_variation.csv — per-heritable-edge: R_hat[:,i,j] and genotype
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

OUTPUT_DIR = os.path.join("output", "population")
FDR_THR = 0.1


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_tce_tensor(output_dir, K, D):
    """
    Read R_hat_donor{k}.csv files → (K, D, D) numpy array.
    Missing genes filled with NaN (filter_tce can drop different genes per donor).
    """
    gene_names = [f"V{i+1}" for i in range(D)]
    R_tensor = np.full((K, D, D), np.nan)

    for k in range(K):
        path = os.path.join(output_dir, f"R_hat_donor{k+1}.csv")
        if not os.path.exists(path):
            print(f"  [warn] R_hat_donor{k+1}.csv not found — donor {k+1} skipped")
            continue
        df = pd.read_csv(path, index_col=0)
        for i, gi in enumerate(gene_names):
            for j, gj in enumerate(gene_names):
                if gi in df.index and gj in df.columns:
                    val = df.loc[gi, gj]
                    R_tensor[k, i, j] = float(val) if pd.notna(val) else np.nan
    return R_tensor, gene_names


def load_snps(output_dir):
    df = pd.read_csv(os.path.join(output_dir, "donor_snps.csv"), index_col=0)
    return df.values.astype(float), list(df.columns), list(df.index)


def load_heritable_edges(output_dir):
    return pd.read_csv(os.path.join(output_dir, "heritable_edges_truth.csv"))


# ---------------------------------------------------------------------------
# 2. Tensor regression: per-edge OLS across donors
# ---------------------------------------------------------------------------

def run_tensor_regression(R_tensor, snps_raw):
    """
    Marginal (edge-QTL) regression: for every (i,j,m) triple, fit a simple
    univariate slope across K donors.

        R_hat[k, i, j]  =  mu[i,j]  +  B[m,i,j] * snp_m[k]  +  eps

    Testing all M SNPs against all D(D-1) edges margially is the standard
    edge-QTL / GWAS scan approach.  With K=8 and M=20, joint OLS would be
    underdetermined (20 regressors, 8 observations); marginal regression
    uses only 2 parameters per test and gives df=K-2 degrees of freedom.

    Returns
    -------
    B_hat  : (M, D, D)  OLS slope for each SNP × edge
    P_vals : (M, D, D)  two-sided t-test p-values
    SE_hat : (M, D, D)  standard errors of B_hat
    T_hat  : (M, D, D)  t-statistics
    mu_hat : (D, D)     per-edge intercepts (grand mean across donors)
    """
    K, D, _ = R_tensor.shape
    M = snps_raw.shape[1]

    # Standardise each SNP column across donors  (mean=0, sd=1)
    snp_mean = snps_raw.mean(axis=0)
    snp_std  = snps_raw.std(axis=0) + 1e-8
    snps_std = (snps_raw - snp_mean) / snp_std   # (K, M)

    B_hat  = np.zeros((M, D, D))
    SE_hat = np.full((M, D, D), np.nan)
    T_hat  = np.zeros((M, D, D))
    P_vals = np.ones((M, D, D))
    mu_hat = np.zeros((D, D))

    for i in range(D):
        for j in range(D):
            if i == j:
                continue

            y = R_tensor[:, i, j]        # K-vector, may contain NaN
            valid = ~np.isnan(y)
            n_valid = valid.sum()

            if n_valid < 3:              # need at least 3 obs for any slope
                continue

            y_v   = y[valid]
            mu_hat[i, j] = y_v.mean()

            # --- marginal loop over each SNP ---
            for m in range(M):
                x = snps_std[valid, m]   # scalar predictor across valid donors
                X = np.column_stack([np.ones(n_valid), x])   # (n, 2)

                beta, _, _, _ = np.linalg.lstsq(X, y_v, rcond=None)
                B_hat[m, i, j] = beta[1]

                resid  = y_v - X @ beta
                df_res = n_valid - 2
                if df_res < 1:
                    continue
                sigma2 = np.sum(resid ** 2) / df_res
                XtXinv = np.linalg.pinv(X.T @ X)
                se_beta = np.sqrt(sigma2 * XtXinv[1, 1])

                SE_hat[m, i, j] = se_beta
                t_stat = beta[1] / (se_beta + 1e-12)
                T_hat[m, i, j]  = t_stat
                P_vals[m, i, j] = 2 * stats.t.sf(abs(t_stat), df=df_res)

    return B_hat, P_vals, SE_hat, T_hat, mu_hat, snp_mean, snp_std


# ---------------------------------------------------------------------------
# 3. FDR correction across all (SNP, edge) tests
# ---------------------------------------------------------------------------

def fdr_correct(P_vals, D, M):
    """BH correction over all M × D(D-1) off-diagonal tests."""
    off_mask = ~np.eye(D, dtype=bool)
    # Gather all off-diagonal p-values: shape (M, D*(D-1))
    p_flat = P_vals[:, off_mask].flatten()      # M × D(D-1)
    _, q_flat, _, _ = multipletests(p_flat, method="fdr_bh")

    Q_vals = np.ones((M, D, D))
    Q_vals[:, off_mask] = q_flat.reshape(M, -1)
    return Q_vals


# ---------------------------------------------------------------------------
# 4. Evaluate against ground truth
# ---------------------------------------------------------------------------

def evaluate(B_hat, P_vals, Q_vals, heritable_df, D, M, fdr_thr=FDR_THR):
    """
    For every ground-truth (edge_i, edge_j, linked_snp) triple, collect
    the estimated B_hat and q-value.  Compute precision/recall/F1 at fdr_thr.
    """
    rows = []
    for _, row in heritable_df.iterrows():
        i0 = int(row["i"]) - 1   # 0-indexed
        j0 = int(row["j"]) - 1
        m0 = int(row["snp"]) - 1
        rows.append({
            "i": i0 + 1, "j": j0 + 1, "snp": m0 + 1,
            "true_weight": float(row["true_weight"]),
            "B_hat":   float(B_hat[m0, i0, j0]),
            "p_val":   float(P_vals[m0, i0, j0]),
            "q_val":   float(Q_vals[m0, i0, j0]),
            "detected": float(Q_vals[m0, i0, j0]) < fdr_thr,
        })
    det_df = pd.DataFrame(rows)

    # True positives at fdr_thr
    off_mask = ~np.eye(D, dtype=bool)
    n_tests  = M * int(off_mask.sum())
    n_sig    = int((Q_vals < fdr_thr).sum())          # all significant (edge, SNP) pairs
    n_true   = len(heritable_df)
    n_tp     = int(det_df["detected"].sum())
    n_fp     = max(0, n_sig - n_tp)
    n_fn     = n_true - n_tp

    prec = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
    rec  = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    summary = {
        "n_heritable_edges":   n_true,
        "n_tests":             n_tests,
        "n_significant_q01":   n_sig,
        "TP": n_tp, "FP": n_fp, "FN": n_fn,
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "F1":        round(f1, 4),
    }
    return det_df, summary


# ---------------------------------------------------------------------------
# 5. Build long-format export DataFrames
# ---------------------------------------------------------------------------

def build_long_B(B_hat, P_vals, Q_vals, SE_hat, T_hat, D, M):
    """All off-diagonal (snp, i, j) entries in long format."""
    rows = []
    for m in range(M):
        for i in range(D):
            for j in range(D):
                if i == j:
                    continue
                rows.append({
                    "snp":   m + 1,
                    "i":     i + 1,
                    "j":     j + 1,
                    "B_hat": float(B_hat[m, i, j]),
                    "SE":    float(SE_hat[m, i, j]) if not np.isnan(SE_hat[m, i, j]) else np.nan,
                    "t_stat": float(T_hat[m, i, j]),
                    "p_val": float(P_vals[m, i, j]),
                    "q_val": float(Q_vals[m, i, j]),
                })
    return pd.DataFrame(rows)


def build_edge_variation(R_tensor, snps_raw, heritable_df, donor_names, D, K):
    """
    For each heritable (i,j) edge: R_hat across donors + linked SNP genotype.
    Used to draw scatter plots of edge weight vs genotype.
    """
    rows = []
    for _, row in heritable_df.iterrows():
        i0 = int(row["i"]) - 1
        j0 = int(row["j"]) - 1
        m0 = int(row["snp"]) - 1
        for k in range(K):
            rows.append({
                "donor":      donor_names[k],
                "edge":       f"({int(row['i'])},{int(row['j'])})",
                "i": i0 + 1, "j": j0 + 1, "snp": m0 + 1,
                "genotype":   int(snps_raw[k, m0]),
                "R_hat_ij":   float(R_tensor[k, i0, j0]),
                "true_weight_base": float(row["true_weight"]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    print("=== Tensor regression: TCE[K,D,D] ~ SNPs[K,M] ===\n")

    K, D = 8, 20
    print(f"Loading TCE tensor ({K} donors × {D} × {D})...")
    R_tensor, gene_names = load_tce_tensor(OUTPUT_DIR, K, D)
    donors_with_data = np.sum(~np.all(np.isnan(R_tensor), axis=(1, 2)))
    print(f"  Donors with data: {donors_with_data}/{K}")
    print(f"  NaN fraction: {np.isnan(R_tensor).mean():.2%}")

    print("\nLoading SNP matrix...")
    snps_raw, snp_names, donor_names = load_snps(OUTPUT_DIR)
    M = snps_raw.shape[1]
    print(f"  Shape: {snps_raw.shape}  (K={K} donors × M={M} SNPs)")
    print(f"  MAF range: [{snps_raw.mean(axis=0).min():.2f}, {snps_raw.mean(axis=0).max():.2f}]")
    print(f"  Allele counts (0/1/2): "
          f"{(snps_raw==0).sum()}/{(snps_raw==1).sum()}/{(snps_raw==2).sum()}")

    print("\nLoading heritable edge ground truth...")
    heritable_df = load_heritable_edges(OUTPUT_DIR)
    print(f"  {len(heritable_df)} heritable (edge, SNP) pairs")
    print(heritable_df.to_string(index=False))

    print("\nRunning per-edge OLS tensor regression...")
    B_hat, P_vals, SE_hat, T_hat, mu_hat, snp_mean, snp_std = \
        run_tensor_regression(R_tensor, snps_raw)
    print(f"  B_hat range: [{B_hat.min():.3f}, {B_hat.max():.3f}]")

    print("\nApplying BH FDR correction...")
    Q_vals = fdr_correct(P_vals, D, M)
    n_sig_01 = int((Q_vals[~np.eye(D, dtype=bool)[np.newaxis, :, :].repeat(M, 0)] < 0.1).sum())
    print(f"  Significant (edge, SNP) pairs at q<0.1: {n_sig_01} / {M * D * (D-1)}")

    print("\nEvaluating against ground truth heritable edges...")
    det_df, summary = evaluate(B_hat, P_vals, Q_vals, heritable_df, D, M)
    print(f"  TP={summary['TP']}  FP={summary['FP']}  FN={summary['FN']}")
    print(f"  Precision={summary['precision']}  Recall={summary['recall']}  F1={summary['F1']}")
    print("\n  Per-heritable-edge results:")
    print(det_df[["edge", "snp", "true_weight", "B_hat", "p_val", "q_val", "detected"]].rename(
        columns={"edge": "(i,j)"}
    ).to_string(index=False) if "edge" in det_df.columns else det_df.to_string(index=False))

    # Build edge labels for det_df
    det_df["edge"] = det_df.apply(lambda r: f"({int(r.i)},{int(r.j)})", axis=1)

    print("\nBuilding long-format outputs...")
    long_df  = build_long_B(B_hat, P_vals, Q_vals, SE_hat, T_hat, D, M)
    edge_var = build_edge_variation(R_tensor, snps_raw, heritable_df, donor_names, D, K)

    # Save outputs
    long_df.to_csv(os.path.join(OUTPUT_DIR, "tensor_pvals.csv"), index=False)
    det_df.to_csv(os.path.join(OUTPUT_DIR, "tensor_detection.csv"), index=False)
    pd.DataFrame([summary]).to_csv(os.path.join(OUTPUT_DIR, "tensor_summary.csv"), index=False)
    edge_var.to_csv(os.path.join(OUTPUT_DIR, "tensor_edge_variation.csv"), index=False)

    # Save B_hat as (M*D*D) long format for visualisation
    long_df.to_csv(os.path.join(OUTPUT_DIR, "tensor_B_hat.csv"), index=False)

    # Save the raw TCE tensor slices for visualisation
    for k in range(K):
        slice_df = pd.DataFrame(R_tensor[k], index=gene_names, columns=gene_names)
        slice_df.to_csv(os.path.join(OUTPUT_DIR, f"tce_slice_donor{k+1}.csv"))

    # Save SNP matrix with standardised values too
    snps_std = (snps_raw - snp_mean) / snp_std
    pd.DataFrame(snps_std, index=donor_names, columns=snp_names).to_csv(
        os.path.join(OUTPUT_DIR, "donor_snps_standardised.csv"))

    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print("  tensor_pvals.csv          — all (snp, edge) p-values and q-values")
    print("  tensor_detection.csv      — ground-truth heritable pairs with B_hat")
    print("  tensor_summary.csv        — precision / recall / F1")
    print("  tensor_edge_variation.csv — per-donor R_hat[i,j] vs genotype (scatter data)")
    print("  tce_slice_donor{k}.csv    — per-donor TCE matrix slices")
    print("  donor_snps_standardised.csv")

    return {
        "R_tensor": R_tensor, "snps_raw": snps_raw, "B_hat": B_hat,
        "Q_vals": Q_vals, "det_df": det_df, "summary": summary,
        "edge_var": edge_var, "gene_names": gene_names,
        "snp_names": snp_names, "donor_names": donor_names,
    }


if __name__ == "__main__":
    main()
