# Fit the Bayesian xG model in Stan (cmdstanr). Run this in RStudio.
# install.packages("cmdstanr", repos = c("https://stan-dev.r-universe.dev"))
library(cmdstanr)
library(posterior)

shots <- read.csv("data/model_features.csv")

# choose predictors: start simple with distance + angle
X <- as.matrix(shots[, c("distance", "angle_deg")])
X <- scale(X)                       # standardising helps sampling
y <- shots$goal

stan_data <- list(N = nrow(X), K = ncol(X), X = X, y = y)

mod <- cmdstan_model("stan/xg_logistic.stan")
fit <- mod$sample(
  data = stan_data,
  chains = 4, parallel_chains = 4,
  iter_warmup = 500, iter_sampling = 500,
  refresh = 0
)

# coefficient summaries with 95% credible intervals
print(fit$summary(variables = c("alpha", "beta")))

# example: posterior mean xG for the first few shots
xg_draws <- fit$draws("xg", format = "draws_matrix")
cat("first 5 shots, posterior mean xG:\n")
print(round(colMeans(xg_draws[, 1:5]), 3))
