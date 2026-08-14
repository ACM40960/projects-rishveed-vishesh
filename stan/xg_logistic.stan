// Bayesian logistic regression for xG.
// Works for any number of predictors K (pass distance+angle, or all features).
data {
  int<lower=1> N;              // number of shots
  int<lower=1> K;              // number of predictors
  matrix[N, K] X;              // predictor matrix (standardised is fine)
  array[N] int<lower=0, upper=1> y;   // 1 = goal, 0 = no goal
}
parameters {
  real alpha;                  // intercept
  vector[K] beta;              // coefficients
}
model {
  // weakly-informative priors; with tens of thousands of shots the data dominate
  alpha ~ normal(0, 5);
  beta  ~ normal(0, 5);
  y ~ bernoulli_logit(alpha + X * beta);   // vectorised likelihood
}
generated quantities {
  vector[N] xg = inv_logit(alpha + X * beta);  // posterior xG per shot
}
