# ADAPRE: ADAptive Penalized inverse REgression

This repository contains a demonstration of **ADAPRE** (ADAptive Penalized
inverse REgression), a method for inferring gene regulatory networks from perturbation experiments with heterogeneous intervention strengths.

## Overview

Real perturbation experiments often exhibit variable intervention strengths—some genes are perturbed more strongly than others. ADAPRE addresses this by introducing **adaptive row-wise penalties** that scale with estimated intervention strength, enabling more accurate network reconstruction.

## Repository Structure

```
.
├── run_adapre_demo.R              # Main demo script
├── R/
│   ├── adapre_inspre_core.R       # Core R implementation with ADAPRE extensions
│   └── simulate_two_interventions.R  # Synthetic data generator
├── src/
│   └── adapre_inspre_cpp.cpp      # C++ ADMM solver with adaptive penalties
└── output/
    ├── summary_best_f1.csv        # Best F1 scores per method
    ├── f1_metrics_all_lambdas.csv # Full regularization path metrics
    ├── degree_vs_true_degree_best.png  # Out-degree comparison plot
    └── sim_data.RData             # Cached simulation data
```

## Requirements

### System Requirements
- **R** ≥ 4.0.0
- **C++ compiler** with C++11 support (for Rcpp)

### R Packages

Core dependencies:
```r
install.packages(c("Rcpp", "RcppEigen", "dplyr", "purrr", "ggplot2"))
```

Optional dependencies:
```r
install.packages(c("expm", "igraph", "nnls", "caret"))
```

inspre package (for network simulation and baseline comparison):
```r
# Install devtools if needed
install.packages("devtools")

# Install inspre from GitHub
devtools::install_github("brielin/inspre", INSTALL_opts = "--install-tests")
```

## Running the Demo

From the repository root directory:

```bash
Rscript run_adapre_demo.R
```

### Expected Output

```
Generating dataset...
Dataset saved to output/sim_data.RData
Data shape: 30000 x 50

Fitting baseline inspre (package)...

Fitting ADAPRE (UV, adaptive lambda)...

[Summary] Best F1 per method (edge_thr = 0.001 ):
  method    lambda precision    recall        F1
1 inspre 0.2470706 0.7722772 0.7289720 0.7500000
2 ADAPRE 0.2470706 0.9056604 0.8971963 0.9014085
`geom_smooth()` using formula = 'y ~ x'

Outputs saved to: output 
  - summary_best_f1.csv
  - f1_metrics_all_lambdas.csv
  - degree_vs_true_degree_best.png

Done.
```

## Method Details

### Adaptive Penalty Formula

For each gene (row) *i* with estimated intervention strength β_i:

```
λ_i = λ × (|β_i| / β̄)
```

where:
- λ is the base regularization parameter
- β̄ is the mean absolute intervention strength
- Genes with weaker interventions (smaller |β_i|) get smaller penalties
- Genes with stronger interventions (larger |β_i|) get larger penalties

This is implemented in the C++ ADMM solver ([src/adapre_inspre_cpp.cpp](src/adapre_inspre_cpp.cpp)) through the `adaptive_lambda` parameter in functions `fit_V_VU_const` and `fit_V_UV_const`.

## Interpreting Results

### F1 Metrics
Higher F1 scores indicate better edge detection. ADAPRE typically achieves higher F1 by reducing false positives for weakly-perturbed genes.

### Out-Degree Plot
The scatter plot ([output/degree_vs_true_degree_best.png](output/degree_vs_true_degree_best.png)) shows:
- **x-axis**: True out-degree (number of regulated targets)
- **y-axis**: Estimated out-degree
- **Color**: Intervention strength group (stronger/weaker)
- **Panels**: Comparison between inspre and ADAPRE

Points closer to the diagonal (y=x) indicate better degree estimation. ADAPRE often shows tighter agreement, especially for genes with varying intervention strengths.

## Contact

For questions or issues, please open an issue on this repository.
