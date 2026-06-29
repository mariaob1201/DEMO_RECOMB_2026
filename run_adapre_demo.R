#!/usr/bin/env Rscript

## Toy example comparing baseline inspre vs ADAPRE (adaptive row-wise penalty)
## using the local implementation in this folder.

suppressPackageStartupMessages({
  library(Rcpp)
  library(dplyr)
  library(purrr)
  library(ggplot2)
})


## Assume working directory is this folder. If run via Rscript, this is usually true.
## If running interactively, setwd() into ADAPRE_RECOMB first.

source("R/adapre_inspre_core.R")
source("R/simulate_two_interventions.R")
Rcpp::sourceCpp("src/adapre_inspre_cpp.cpp")

## Create output directory
output_dir <- "output"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

set.seed(1)

## Data generation using simulate_two_interventions.R (generate_dataset_two_int_v2.R)
## (two intervention strength settings; we use set2 with heterogeneous int_beta)

D <- 50
N_int <- 300
N_cont <- D * N_int

int_beta_base <- -1
int_beta_hetero <- c(rep(int_beta_base * 2, D / 2), rep(int_beta_base, D - D / 2))

graph <- "random" # random / scale-free
v <- 0.3 # edge weight scale
p <- 0.04 # edge density
DAG <- FALSE # cyclic graph
C <- 0 # no confounding

cat("Generating dataset...\n")

sim <- generate_dataset_two_int(
  D = D,
  N_cont = N_cont,
  N_int = N_int,
  int_beta_list = list(int_beta_base, int_beta_hetero),
  graph = graph,
  v = v,
  p = p,
  DAG = DAG,
  C = C
)

save(sim, file = file.path(output_dir, "sim_data.RData"))
cat("Dataset saved to ", file.path(output_dir, "sim_data.RData"), "\n", sep = "")

set <- sim$set2
Y <- set$Y
targets <- set$targets
G <- set$G
int_beta <- as.numeric(set$int_beta)

cat("Data shape: ", nrow(Y), " x ", ncol(Y), "\n", sep = "")

## Helper: flatten off-diagonal entries
off_diag_vec <- function(M) {
  D <- nrow(M)
  M[!diag(D)]
}

## Helper to pick best lambda by training error and compute off-diagonal correlation
evaluate_fit <- function(res, G_true) {
  # full_res$G_hat stores the path of inferred direct-effect matrices (D x D x nlambda)
  if (length(dim(res$G_hat)) == 3L) {
    best_idx <- which.min(res$train_error)
    R_hat <- res$G_hat[, , best_idx]
    best_lambda <- res$lambda[best_idx]
  } else {
    # Fallback: if only a single matrix is present
    R_hat <- res$G_hat
    best_lambda <- if (!is.null(res$lambda)) res$lambda[1] else NA_real_
  }
  list(
    best_lambda = best_lambda,
    corr = cor(off_diag_vec(G_true), off_diag_vec(R_hat))
  )
}

## Helper to build degree vs true-degree data for each lambda
## Degrees are out-degrees (row-wise) after thresholding small edges.
build_degree_df <- function(res, G_true, int_beta, method_label, thr = 0.001) {
  D <- nrow(G_true)
  G_true_nodiag <- G_true
  diag(G_true_nodiag) <- 0
  true_deg <- apply(G_true_nodiag, 1, function(row) sum(abs(row) > thr))

  if (length(dim(res$G_hat)) == 3L) {
    nlambda <- dim(res$G_hat)[3]
    lambda_vec <- res$lambda
    out <- lapply(seq_len(nlambda), function(k) {
      Gh <- res$G_hat[, , k]
      diag(Gh) <- 0
      est_deg <- apply(Gh, 1, function(row) sum(abs(row) > thr))
      data.frame(
        gene = paste0("V", 1:D),
        lambda = lambda_vec[k],
        method = method_label,
        true_deg = true_deg,
        est_deg = est_deg,
        strength = abs(int_beta)
      )
    })
    do.call(rbind, out)
  } else {
    Gh <- res$G_hat
    diag(Gh) <- 0
    est_deg <- apply(Gh, 1, function(row) sum(abs(row) > thr))
    data.frame(
      gene = paste0("V", 1:D),
      lambda = if (!is.null(res$lambda)) res$lambda[1] else NA_real_,
      method = method_label,
      true_deg = true_deg,
      est_deg = est_deg,
      strength = abs(int_beta)
    )
  }
}

## Helper: compute precision/recall/F1 along lambda path against true G
compute_f1_path <- function(res, G_true, thr = 0.001) {
  D <- nrow(G_true)
  mask <- !diag(D)
  g_true_vec <- G_true[mask]
  true_pos <- abs(g_true_vec) > thr

  if (length(dim(res$G_hat)) == 3L) {
    nlambda <- dim(res$G_hat)[3]
    lambda_vec <- res$lambda
    out <- lapply(seq_len(nlambda), function(k) {
      Gh <- res$G_hat[, , k]
      est_vec <- Gh[mask]
      est_pos <- abs(est_vec) > thr
      tp <- sum(est_pos & true_pos)
      fp <- sum(est_pos & !true_pos)
      fn <- sum(!est_pos & true_pos)
      precision <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
      recall <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
      f1 <- if (is.na(precision) || is.na(recall) || (precision + recall) == 0) {
        NA_real_
      } else {
        2 * precision * recall / (precision + recall)
      }
      data.frame(
        lambda = lambda_vec[k],
        precision = precision,
        recall = recall,
        F1 = f1
      )
    })
    do.call(rbind, out)
  } else {
    Gh <- res$G_hat
    est_vec <- Gh[mask]
    est_pos <- abs(est_vec) > thr
    tp <- sum(est_pos & true_pos)
    fp <- sum(est_pos & !true_pos)
    fn <- sum(!est_pos & true_pos)
    precision <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
    recall <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
    f1 <- if (is.na(precision) || is.na(recall) || (precision + recall) == 0) {
      NA_real_
    } else {
      2 * precision * recall / (precision + recall)
    }
    data.frame(
      lambda = if (!is.null(res$lambda)) res$lambda[1] else NA_real_,
      precision = precision,
      recall = recall,
      F1 = f1
    )
  }
}

select_best_by_f1 <- function(res, G_true, thr = 0.001) {
  path <- compute_f1_path(res, G_true, thr)
  best_idx <- which.max(path$F1)
  list(
    index = best_idx,
    lambda = path$lambda[best_idx],
    metrics = path[best_idx, ],
    G_hat = if (length(dim(res$G_hat)) == 3L) res$G_hat[, , best_idx] else res$G_hat
  )
}

## Baseline: local fit_inspre_from_X, no adaptive lambda (standard inspre behaviour)
cat("\nFitting baseline (local inspre, adaptive_lambda = FALSE)...\n")
fit_inspre_pkg <- fit_inspre_from_X(
  X = Y,
  targets = targets,
  weighted = TRUE,
  verbose = 0,
  constraint = "UV",
  adaptive_lambda = FALSE
)
eval_inspre <- evaluate_fit(fit_inspre_pkg, G)

## ADAPRE, UV constraint (our implementation with adaptive lambda)
cat("\nFitting ADAPRE (UV, adaptive lambda)...\n")
fit_adapre <- fit_inspre_from_X(
  X = Y,
  targets = targets,
  weighted = TRUE,
  verbose = 0,
  constraint = "UV",
  adaptive_lambda = TRUE
)
eval_adapre <- evaluate_fit(fit_adapre, G)

## CHECK RESULTS (UV only)

edge_thr <- 0.001

best_inspre <- select_best_by_f1(fit_inspre_pkg, G, thr = edge_thr)
best_adapre <- select_best_by_f1(fit_adapre,     G, thr = edge_thr)

cat("\n[Summary] Best F1 per method (edge_thr =", edge_thr, "):\n")
summary_table <- data.frame(
  method = c("inspre", "ADAPRE"),
  lambda = c(best_inspre$lambda, best_adapre$lambda),
  precision = c(best_inspre$metrics$precision, best_adapre$metrics$precision),
  recall = c(best_inspre$metrics$recall, best_adapre$metrics$recall),
  F1 = c(best_inspre$metrics$F1, best_adapre$metrics$F1)
)
print(summary_table)

## Save summary table
write.csv(summary_table, file.path(output_dir, "summary_best_f1.csv"), row.names = FALSE)

## Export data for Python benchmark harness
write.csv(Y,
          file.path(output_dir, "X.csv"), row.names = FALSE)
write.csv(data.frame(target = targets),
          file.path(output_dir, "targets.csv"), row.names = FALSE)
write.csv(fit_adapre$R_hat,
          file.path(output_dir, "R_hat_tce.csv"))
write.csv(as.data.frame(G),
          file.path(output_dir, "G_true.csv"), row.names = FALSE)
write.csv(data.frame(gene = paste0("V", seq_along(int_beta)), beta_obs = int_beta),
          file.path(output_dir, "beta_obs.csv"), row.names = FALSE)
write.csv(as.data.frame(best_adapre$G_hat),
          file.path(output_dir, "G_hat_adapre.csv"), row.names = FALSE)
cat("  - X.csv, targets.csv, R_hat_tce.csv, G_true.csv, beta_obs.csv, G_hat_adapre.csv\n")

## Compare adaptive vs baseline networks (best-F1 lambda)
# Removed correlation print to reduce output clutter
# corr_uv_adap <- cor(off_diag_vec(best_inspre$G_hat), off_diag_vec(best_adapre$G_hat))

## Degree vs true-degree plot for both (best-F1 lambda)
deg_inspre_all <- build_degree_df(fit_inspre_pkg, G, int_beta, method_label = "inspre", thr = edge_thr)
deg_adapre_all <- build_degree_df(fit_adapre,     G, int_beta, method_label = "ADAPRE", thr = edge_thr)

deg_best <- dplyr::bind_rows(
  dplyr::filter(deg_inspre_all, lambda == best_inspre$lambda),
  dplyr::filter(deg_adapre_all, lambda == best_adapre$lambda)
) %>%
  mutate(stronger_half = ifelse(abs(strength) > abs(median(strength)), "stronger", "weaker"))

p_deg_best <- ggplot(deg_best, aes(x = true_deg, y = est_deg, color = stronger_half)) +
  geom_jitter(size = 2, width = 0.2, height = 0.2) +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~ method, ncol = 2) +
  labs(
    title = "Estimated vs true out-degree (best F1 per method, UV)",
    x = "True out-degree",
    y = "Estimated out-degree",
    color = "intervention strength"
  ) +
  theme_bw() + 
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "bottom"
  )

ggsave(file.path(output_dir, "degree_vs_true_degree_best.png"), p_deg_best, width = 6, height = 3.6, dpi = 300)

## Save detailed metrics for all lambdas
f1_path_inspre <- compute_f1_path(fit_inspre_pkg, G, thr = edge_thr)
f1_path_adapre <- compute_f1_path(fit_adapre, G, thr = edge_thr)
f1_path_inspre$method <- "inspre"
f1_path_adapre$method <- "ADAPRE"
f1_combined <- rbind(f1_path_inspre, f1_path_adapre)
write.csv(f1_combined, file.path(output_dir, "f1_metrics_all_lambdas.csv"), row.names = FALSE)

cat("\nOutputs saved to:", output_dir, "\n")
cat("  - summary_best_f1.csv\n")
cat("  - f1_metrics_all_lambdas.csv\n")
cat("  - degree_vs_true_degree_best.png\n")
cat("\nDone.\n")
