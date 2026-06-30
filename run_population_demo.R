#!/usr/bin/env Rscript

## Population-scale Perturb-seq demo: simulate K donors with genotype-linked
## heritable edges, then fit ADAPRE independently per donor as a naive
## baseline. This demonstrates why independent per-donor fits break down
## at realistic population-scale cell counts (motivating a fused/joint
## estimator as the next step).

suppressPackageStartupMessages({
  library(Rcpp)
  library(dplyr)
  library(purrr)
})

source("R/adapre_inspre_core.R")
source("R/simulate_two_interventions.R")
source("R/simulate_population_perturb_seq.R")
Rcpp::sourceCpp("src/adapre_inspre_cpp.cpp")

output_dir <- "output/population"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

set.seed(2)

D                <- 20
K_donors         <- 8
N_cont_per_donor <- 200
N_int_per_donor  <- 15   # realistic population-scale budget (vs N=300 single-pop)
M_snps           <- 20

cat("Simulating population-scale Perturb-seq:\n")
cat(sprintf("  D=%d genes, K=%d donors, N_int=%d cells/perturbation/donor, M=%d SNPs\n",
            D, K_donors, N_int_per_donor, M_snps))

pop <- simulate_population_perturb_seq(
  D = D, K_donors = K_donors,
  N_cont_per_donor = N_cont_per_donor, N_int_per_donor = N_int_per_donor,
  M_snps = M_snps, heritable_edge_frac = 0.15, beta_qtl = 1.5,
  graph = "random", v = 0.3, p = 0.1, DAG = FALSE
)

save(pop, file = file.path(output_dir, "population_sim.RData"))
cat(sprintf("  %d heritable edges (out of %d nonzero edges in G_pop)\n",
            nrow(pop$heritable_edges), sum(pop$G_pop != 0 & !diag(D))))

## ---------------------------------------------------------------------
## Naive baseline: fit ADAPRE independently per donor
## ---------------------------------------------------------------------

f1_at_best <- function(G_hat_arr, lambda_vec, G_true_full, gene_names, thr = 0.05) {
  G_true <- G_true_full[gene_names, gene_names]
  Dg <- nrow(G_true)
  mask <- !diag(Dg)
  true_pos <- abs(G_true[mask]) > thr
  best_f1 <- 0
  for (k in seq_along(lambda_vec)) {
    pred_vals <- G_hat_arr[, , k][mask]
    if (any(is.na(pred_vals)) || any(is.infinite(pred_vals))) next
    pred_pos <- abs(pred_vals) > thr
    tp <- sum(pred_pos & true_pos)
    fp <- sum(pred_pos & !true_pos)
    fn <- sum(!pred_pos & true_pos)
    prec <- if (tp + fp > 0) tp / (tp + fp) else 0
    rec  <- if (tp + fn > 0) tp / (tp + fn) else 0
    f1   <- if (prec + rec > 0) 2 * prec * rec / (prec + rec) else 0
    if (f1 > best_f1) best_f1 <- f1
  }
  best_f1
}

cat("\nFitting ADAPRE independently per donor (naive baseline)...\n")
results <- data.frame()
for (k in seq_len(K_donors)) {
  d <- pop$donor_data[[k]]
  G_true_named <- pop$donor_graphs[[k]]
  rownames(G_true_named) <- colnames(G_true_named) <- paste0("V", 1:D)

  res <- tryCatch(
    fit_inspre_from_X(d$Y, d$targets, weighted = TRUE, verbose = 0, adaptive_lambda = TRUE),
    error = function(e) NULL
  )
  if (is.null(res)) {
    cat(sprintf("  Donor %d: FAILED entirely\n", k))
    next
  }
  genes_kept   <- colnames(res$R_hat)
  n_unstable_k <- sum(apply(res$G_hat, 3, function(m) any(is.na(m)) || any(is.infinite(m))))
  f1 <- f1_at_best(res$G_hat, res$lambda, G_true_named, genes_kept)

  cat(sprintf("  Donor %d: F1=%.3f  (genes retained: %d/%d, unstable lambdas: %d/%d)\n",
              k, f1, length(genes_kept), D, n_unstable_k, length(res$lambda)))

  results <- rbind(results, data.frame(
    donor = k, F1 = f1, genes_retained = length(genes_kept),
    n_unstable_lambdas = n_unstable_k, n_lambdas = length(res$lambda)
  ))
}

cat(sprintf("\n=== Summary: independent per-donor ADAPRE ===\n"))
cat(sprintf("Mean F1 across donors: %.3f\n", mean(results$F1)))
cat(sprintf("Gene retention range: %d-%d out of %d\n",
            min(results$genes_retained), max(results$genes_retained), D))
cat(sprintf("Donors with >=1 numerically unstable lambda: %d/%d\n",
            sum(results$n_unstable_lambdas > 0), nrow(results)))

write.csv(results, file.path(output_dir, "independent_donor_baseline.csv"), row.names = FALSE)
write.csv(pop$heritable_edges, file.path(output_dir, "heritable_edges_truth.csv"), row.names = FALSE)
write.csv(pop$snps, file.path(output_dir, "donor_snps.csv"))

cat("\nOutputs saved to", output_dir, "\n")
cat("  - population_sim.RData (full simulation + ground truth)\n")
cat("  - independent_donor_baseline.csv (per-donor F1, naive baseline)\n")
cat("  - heritable_edges_truth.csv (ground truth: which edges vary by genotype)\n")
cat("  - donor_snps.csv (genotype matrix)\n")
cat("\nThis demonstrates the motivation for a fused/joint multi-donor estimator:\n")
cat("independent per-donor fits lose statistical power and even drop genes\n")
cat("entirely at realistic population-scale cell counts.\n")
