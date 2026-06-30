# Population-scale Perturb-seq simulator.
#
# Extends generate_network()/generate_data_inhibition_two_int() (see
# simulate_two_interventions.R) from a single shared network to K donors,
# each with their own causal graph G_k. A subset of edges are "heritable":
# their weight depends on the donor's genotype at a linked SNP, mirroring
# the trans-eQTL / heritable-perturbation-effect design in Feng et al.
# (PMC12903452). Non-heritable edges are identical across donors (shared
# population-level architecture, as in Lu & Keleş's personalized-network
# framing, PMID 37295843).
#
# Ground truth returned includes G_pop, each G_k, the genotype matrix, and
# which (edge, SNP) pairs are truly heritable — needed to validate any
# downstream joint/fused estimator and edge-QTL test.

generate_donor_data <- function(G, N_cont, N_int, int_beta = -1,
                                noise = "gaussian") {
  D  <- nrow(G)
  Ncs <- cumsum(c(N_cont, rep(N_int, D)))
  N   <- tail(Ncs, 1L)

  XB <- matrix(0, nrow = N, ncol = D)
  for (d in 1:D) {
    start <- Ncs[d]
    end   <- Ncs[d + 1]
    if (end > start) XB[(start + 1):end, d] <- int_beta
  }

  net_vars <- colSums(G^2)
  eps_vars <- max(0.9, max(net_vars)) - net_vars + 0.1
  if (noise != "gaussian") stop("NotImplementedError")
  eps <- t(matrix(stats::rnorm(D * N, sd = sqrt(eps_vars)), nrow = D, ncol = N))

  Ainv <- solve(diag(D) - G)
  Y    <- (XB + eps) %*% Ainv

  mu_cont <- colMeans(Y[1:N_cont, , drop = FALSE])
  sd_cont <- apply(Y[1:N_cont, , drop = FALSE], 2, sd)
  Y <- t((t(Y) - mu_cont) / sd_cont)

  colnames(Y) <- paste0("V", 1:D)
  targets <- c(rep("control", N_cont), paste0("V", rep(1:D, each = N_int)))

  list(Y = Y, targets = targets)
}


#' Simulates population-scale Perturb-seq across K donors with genotype-linked
#' heritable edges.
#'
#' @param D Integer. Number of genes.
#' @param K_donors Integer. Number of individuals.
#' @param N_cont_per_donor Integer. Control cells per donor.
#' @param N_int_per_donor Integer. Cells per perturbed gene, per donor.
#'   Realistically much smaller than the single-population case (population
#'   designs split the same total cell budget across many donors).
#' @param M_snps Integer. Number of SNPs simulated per donor.
#' @param heritable_edge_frac Float in [0,1]. Fraction of G_pop's nonzero
#'   edges whose weight varies with a linked SNP across donors.
#' @param beta_qtl Float. Effect size of genotype on linked edge weight
#'   (multiplicative, per standardized genotype unit).
#' @param graph,v,p,DAG Passed to generate_network() for the population graph.
#' @param int_beta Float. Intervention (knockdown) strength, shared across donors.
#' @param maf_range Length-2 vector. Range to draw each SNP's minor allele
#'   frequency from (uniform).
#' @export
simulate_population_perturb_seq <- function(D, K_donors,
                                            N_cont_per_donor, N_int_per_donor,
                                            M_snps = D, heritable_edge_frac = 0.1,
                                            beta_qtl = 1.0,
                                            graph = "random", v = 0.3, p = 0.04,
                                            DAG = FALSE, int_beta = -1,
                                            maf_range = c(0.1, 0.5)) {
  G_pop <- generate_network(D, graph = graph, p = p, v = v, DAG = DAG)

  off_diag_mask <- !diag(D)
  edge_idx      <- which(G_pop != 0 & off_diag_mask, arr.ind = TRUE)
  n_edges       <- nrow(edge_idx)
  n_heritable   <- max(1, round(heritable_edge_frac * n_edges))
  heritable_rows <- sample(n_edges, n_heritable)

  maf  <- stats::runif(M_snps, maf_range[1], maf_range[2])
  snps <- sapply(maf, function(f) stats::rbinom(K_donors, 2, f))
  dim(snps) <- c(K_donors, M_snps)
  colnames(snps) <- paste0("SNP", 1:M_snps)
  rownames(snps) <- paste0("Donor", 1:K_donors)

  linked_snp <- sample(1:M_snps, n_heritable, replace = TRUE)
  heritable_edges <- data.frame(
    i           = edge_idx[heritable_rows, 1],
    j           = edge_idx[heritable_rows, 2],
    snp         = linked_snp,
    true_weight = G_pop[edge_idx[heritable_rows, , drop = FALSE]]
  )

  donor_graphs <- vector("list", K_donors)
  donor_data   <- vector("list", K_donors)

  for (k in 1:K_donors) {
    G_k <- G_pop
    for (e in seq_len(n_heritable)) {
      i <- heritable_edges$i[e]
      j <- heritable_edges$j[e]
      m <- heritable_edges$snp[e]
      geno_centered <- (snps[k, m] - mean(snps[, m])) / (stats::sd(snps[, m]) + 1e-8)
      G_k[i, j] <- G_pop[i, j] * (1 + beta_qtl * geno_centered)
    }
    rg <- Mod(eigen(G_k, only.values = TRUE)$values)[1]
    if (rg > 0.9) G_k <- G_k * (0.9 / rg)

    donor_graphs[[k]] <- G_k
    donor_data[[k]]   <- generate_donor_data(
      G_k, N_cont_per_donor, N_int_per_donor, int_beta
    )
  }

  list(
    G_pop           = G_pop,
    donor_graphs    = donor_graphs,
    donor_data      = donor_data,
    snps            = snps,
    heritable_edges = heritable_edges,
    params = list(
      D = D, K_donors = K_donors, N_cont_per_donor = N_cont_per_donor,
      N_int_per_donor = N_int_per_donor, M_snps = M_snps,
      heritable_edge_frac = heritable_edge_frac, beta_qtl = beta_qtl
    )
  )
}
