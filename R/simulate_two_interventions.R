# The following function generates a dataset with two intervention variables.
# It is used for the inspre simulation.
# Functions are taken from the inspre package and modified to allow for different intervention strengths.

# Source of original function:
# Commits on Sep 4, 2025
# https://github.com/brielin/inspre/commits/master/R/simulate.R

# ---------------------------------------------------------------------------
# Network generation helpers (copied from inspre::simulate.R so the repo has
# no dependency on the inspre package at runtime).
# ---------------------------------------------------------------------------

scale_free <- function(D, p = 0.5, DAG = FALSE, a_in = 1, a_out = 1) {
  A <- matrix(c(0, 0, 1, 0), nrow = 2)
  while (nrow(A) < D) {
    d_in  <- colSums(A)
    d_out <- rowSums(A)
    choice   <- stats::runif(1)
    D_curr   <- nrow(A)
    if (choice < p) {
      v      <- sample(D_curr, 1, prob = d_out + a_out)
      a_next <- rep(0, D_curr)
      a_next[v] <- 1
      A <- rbind(cbind(A, a_next, deparse.level = 0), rep(0, D_curr + 1))
    } else {
      repeat {
        v <- sample(D_curr, 1, prob = d_out + a_out)
        w <- sample(D_curr, 1, prob = log(d_in + a_in))
        if (DAG) {
          if (v < w) break
        } else if (v != w && A[w, v] == 0) break
      }
      A[v, w] <- 1
    }
  }
  return(A)
}

random_graph <- function(D, p = 3 / D, DAG = FALSE) {
  A <- matrix(as.integer(stats::runif(D * D) < (DAG + 1) * p), nrow = D)
  diag(A) <- 0
  if (DAG) {
    perm <- sample.int(D)
    A <- A[perm, perm]
    A <- A * upper.tri(A)
  }
  return(A)
}

#' Simulates a random weighted graph.
#'
#' Adapted from inspre::generate_network (brielin/inspre, Sep 2025).
#'
#' @param D Integer. Number of nodes.
#' @param graph String. "scalefree" or "random".
#' @param p Float. Edge probability parameter.
#' @param v Float. Mode of PERT distribution for edge weights.
#' @param DAG Logical. TRUE to restrict to a DAG.
#' @param max_eig Float. Maximum spectral radius (for stability).
generate_network <- function(D, graph = "scalefree", p = 0.4, v = 0.2,
                             DAG = FALSE, max_eig = 0.9) {
  if (graph == "scalefree") {
    A <- scale_free(D, p, DAG)
  } else if (graph == "random") {
    A <- random_graph(D, p, DAG)
  } else {
    stop("graph must be 'scalefree' or 'random'.")
  }
  G <- A
  G[A != 0] <- sample(c(1, -1), sum(A), replace = TRUE) *
    mc2d::rpert(sum(A), min = v / 2, mode = v, max = 2 * v)
  rg <- Mod(eigen(G, only.values = TRUE)$values)[1]
  if (rg > max_eig) G <- G * (max_eig / rg)
  rownames(G) <- paste0("V", 1:D)
  colnames(G) <- paste0("V", 1:D)
  return(G)
}

generate_data_knockout <- function(G, N_cont, N_int, noise = "gaussian") {
  stop("generate_data_knockout is not used in this demo (model='inhibition' only).")
}

# --- helper: generate_data_inhibition_two_int ---
# Supports N_int of length 1, 2, or D
generate_data_inhibition_two_int <- function(
  G, N_cont, N_int, int_beta_list = list(-2, -2), noise = "gaussian"
){
  D <- nrow(G)

  # per-gene intervention sizes
  if (length(N_int) == 1) {
    int_sizes <- rep(N_int, D)
  } else if (length(N_int) == 2) {
    int_sizes <- rep(N_int, length.out = D)  # alternate a,b,a,b,...
  } else if (length(N_int) == D) {
    int_sizes <- N_int
  } else {
    stop("N_int must be length 1, 2, or D.")
  }

  Ncs <- cumsum(c(N_cont, int_sizes))
  N   <- tail(Ncs, 1L)

  # intervention design (controls = first N_cont rows)
  XB <- matrix(0, nrow = N, ncol = D)
  for (d in 1:D) {
    start <- Ncs[d]
    end   <- Ncs[d + 1]
    if (end > start) XB[(start + 1):end, d] <- 1
  }

  # two different intervention magnitudes
  XB1 <- t(t(XB) * int_beta_list[[1]])
  XB2 <- t(t(XB) * int_beta_list[[2]])

  # variance equalization & noise
  net_vars <- colSums(G^2)
  eps_vars <- max(0.9, max(net_vars)) - net_vars + 0.1

  if (noise == "gaussian") {
    eps <- t(matrix(rnorm(D * N, sd = sqrt(eps_vars)), nrow = D, ncol = N))
  } else {
    stop("NotImplementedError")
  }

  Ainv <- solve(diag(D) - G)
  Y1   <- (XB1 + eps) %*% Ainv
  Y2   <- (XB2 + eps) %*% Ainv

  # control-based normalization per dataset
  mu_cont1 <- colMeans(Y1[1:N_cont, , drop = FALSE])
  sd_cont1 <- apply(Y1[1:N_cont, , drop = FALSE], 2, sd)
  Y1       <- t((t(Y1) - mu_cont1) / sd_cont1)

  mu_cont2 <- colMeans(Y2[1:N_cont, , drop = FALSE])
  sd_cont2 <- apply(Y2[1:N_cont, , drop = FALSE], 2, sd)
  Y2       <- t((t(Y2) - mu_cont2) / sd_cont2)

  # total/direct effects under each dataset's scaling
  R1 <- get_tce(get_observed(G), normalize = sd_cont1)
  G1 <- get_direct(R1)$G
  int_beta1 <- int_beta_list[[1]] / sd_cont1

  R2 <- get_tce(get_observed(G), normalize = sd_cont2)
  G2 <- get_direct(R2)$G
  int_beta2 <- int_beta_list[[2]] / sd_cont2

  colnames(Y1) <- paste0("V", 1:D)
  colnames(Y2) <- paste0("V", 1:D)
  targets <- c(rep("control", N_cont), paste0("V", rep(1:D, times = int_sizes)))

  list(
    Y1 = Y1, Y2 = Y2, targets = targets,
    G1 = G1, G2 = G2, R1 = R1, R2 = R2,
    int_beta1 = int_beta1, int_beta2 = int_beta2,
    int_sizes = int_sizes  # expose sizes for downstream slicing if needed
  )
}

# --- caller: generate_dataset_two_int ---
# Accepts N_int of length 1 or 2 (or D), and slices by actual total size.
generate_dataset_two_int <- function(
  D, N_cont, N_int, int_beta_list = list(-2, -2), graph = "scalefree",
  v = 0.2, p = 0.4, DAG = FALSE, C = floor(0.1 * D), noise = "gaussian",
  model = "inhibition"
){
  G <- generate_network(D, graph, p, v, DAG)
  if (C > 0) {
    if (graph == "scalefree") {
      new_vars <- matrix(mc2d::rpert(D * C, 0.01, v^(log(D) / log(log(D))), 2 * v), nrow = C)
    } else if (graph == "random") {
      new_vars <- matrix(mc2d::rpert(D * C, 0.01, v^(log(D)), 2 * v), nrow = C)
    }
    G <- cbind(rbind(G, new_vars), matrix(0, nrow = D + C, ncol = C))
  }
  G_full <- G

  if (model == "inhibition") {
    data <- generate_data_inhibition_two_int(G, N_cont, N_int, int_beta_list, noise)
  } else if (model == "knockout") {
    data <- generate_data_knockout(G, N_cont, N_int, noise)
  } else {
    stop("Unknown model.")
  }

  # actual total rows to keep = controls + sum per-gene intervention sizes
  if (is.null(data$int_sizes)) {
    # fallback: compute sizes from N_int if the generator doesn't expose it
    if (length(N_int) == 1) {
      int_sizes <- rep(N_int, D)
    } else if (length(N_int) == 2) {
      int_sizes <- rep(N_int, length.out = D)
    } else if (length(N_int) == D) {
      int_sizes <- N_int
    } else {
      stop("N_int must be length 1, 2, or D.")
    }
  } else {
    int_sizes <- data$int_sizes
  }
  N_tot <- N_cont + sum(int_sizes)

  Y1 <- data$Y1
  Y2 <- data$Y2
  G1 <- data$G1
  G2 <- data$G2

  # variance decompositions (use the available rows)
  var_all1 <- apply(Y1 %*% G1, 2, var)[1:D]
  var_obs1 <- apply(Y1[, 1:D, drop = FALSE] %*% G1[1:D, 1:D, drop = FALSE], 2, var)
  if (C > 0) {
    var_conf1 <- apply(
      Y1[, (D + 1):(D + C), drop = FALSE] %*% G_full[(D + 1):(D + C), 1:D, drop = FALSE],
      2, var
    )
  } else {
    var_conf1 <- rep(0, D)
  }
  var_eps1 <- 1 - var_all1

  var_all2 <- apply(Y2 %*% G2, 2, var)[1:D]
  var_obs2 <- apply(Y2[, 1:D, drop = FALSE] %*% G2[1:D, 1:D, drop = FALSE], 2, var)
  if (C > 0) {
    var_conf2 <- apply(
      Y2[, (D + 1):(D + C), drop = FALSE] %*% G_full[(D + 1):(D + C), 1:D, drop = FALSE],
      2, var
    )
  } else {
    var_conf2 <- rep(0, D)
  }
  var_eps2 <- 1 - var_all2

  set1 <- list(
    Y        = Y1[1:N_tot, 1:D, drop = FALSE],
    targets  = data$targets[1:N_tot],
    R        = data$R1[1:D, 1:D, drop = FALSE],
    G        = G1[1:D, 1:D, drop = FALSE],
    var_all  = var_all1,
    var_obs  = var_obs1,
    var_conf = var_conf1,
    var_eps  = var_eps1,
    int_beta = data$int_beta1
  )

  set2 <- list(
    Y        = Y2[1:N_tot, 1:D, drop = FALSE],
    targets  = data$targets[1:N_tot],
    R        = data$R2[1:D, 1:D, drop = FALSE],
    G        = G2[1:D, 1:D, drop = FALSE],
    var_all  = var_all2,
    var_obs  = var_obs2,
    var_conf = var_conf2,
    var_eps  = var_eps2,
    int_beta = data$int_beta2
  )

  list(set1 = set1, set2 = set2, G_orig = G)
}



# --- required functions ---


#' Gets matrix of TCE from observed network.
#'
#' @param R_obs D x D matrix of observed effects.
#' @param normalize A length D vector which is used to convert R_tce from the
#'   per-allele to the per-variance scale. Each entry should be the
#'   std dev of the corresponding phenotype. Set to NULL for no normalization.
#' @return D x D matrix of total causal effects.
get_tce <- function(R_obs, normalize = NULL) {
  diag_R_obs <- diag(R_obs)
  R_tce <- (1 / (1 + diag_R_obs)) * R_obs
  diag(R_tce) <- 1
  if (!is.null(normalize)) {
    R_tce <- R_tce * outer(normalize, 1 / normalize)
  }
  return(R_tce)
}


#' Gets observed network from direct effects.
#'
#' @param G D x D matrix of direct effects.
#' @return D x D matrix of observed effects.
get_observed <- function(G) {
  D <- dim(G)[1]
  return(solve(diag(D) - G, G))
}

#' Fits exact model to data.
#'
#' @param R D x D matrix of "total causal effects".
#' @return D x D matrix with zero diagonal of deconvoluted direct effects.
get_direct <- function(R) {
  D <- dim(R)[1]
  R[is.na(R)] <- 0

  R_inv <- solve(R)
  G <- diag(D) - t(t(R_inv) / diag(R_inv))
  return(list("G" = G, "R_inv" = R_inv))
}
