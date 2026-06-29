#!/usr/bin/env Rscript
# Thin wrapper: read a TCE matrix CSV → run fit_inspre_from_R → write results.
# Called by benchmark.py via subprocess. Not intended for interactive use.
#
# Usage:
#   Rscript R/run_inspre_on_tce.R <tce_csv> <out_csv> [adaptive_lambda] [beta_obs_csv] [nlambda]
#
# Outputs a CSV with columns: lambda, i, j, G_hat  (long format, all lambdas)

suppressPackageStartupMessages({
  library(Rcpp)
  library(dplyr)
  library(purrr)
})
source("R/adapre_inspre_core.R")
Rcpp::sourceCpp("src/adapre_inspre_cpp.cpp")

args            <- commandArgs(trailingOnly = TRUE)
tce_path        <- args[1]
out_csv         <- args[2]
adaptive_lambda <- isTRUE(as.logical(args[3]))
beta_obs_path   <- if (length(args) >= 4 && nchar(args[4]) > 0) args[4] else NULL
nlambda         <- if (length(args) >= 5) as.integer(args[5]) else 20L

R_tce    <- data.matrix(read.csv(tce_path, row.names = 1, check.names = FALSE))
D        <- nrow(R_tce)
beta_obs <- NULL
if (!is.null(beta_obs_path)) {
  beta_obs <- read.csv(beta_obs_path)$beta_obs
}

res <- fit_inspre_from_R(R_tce, verbose = 0, nlambda = nlambda,
                         adaptive_lambda = adaptive_lambda,
                         beta_obs = beta_obs)

# Flatten D×D×nlambda into long-format data.frame
G_arr      <- res$R_hat          # D x D x nlambda
lambda_vec <- res$lambda
rows       <- vector("list", length(lambda_vec))

for (k in seq_along(lambda_vec)) {
  G_k   <- G_arr[, , k]
  idx   <- which(!diag(D), arr.ind = TRUE)
  rows[[k]] <- data.frame(
    lambda = lambda_vec[k],
    i      = idx[, 1] - 1L,     # 0-indexed for Python
    j      = idx[, 2] - 1L,
    G_hat  = G_k[idx]
  )
}

write.csv(do.call(rbind, rows), out_csv, row.names = FALSE)
cat(sprintf("Wrote %d lambda × %d off-diagonal entries to %s\n",
            length(lambda_vec), nrow(idx), out_csv))
