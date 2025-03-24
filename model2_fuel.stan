
data {
  int<lower=0> N;
  vector[N] y;
  vector[N] x;
  int<lower=0, upper=1> engine[N];
}

parameters {
  real beta0_0;
  real beta1_0;
  real beta0_1;
  real beta1_1;
  real<lower=0> sigma0;
  real<lower=0> sigma1;
}

model {
  beta0_0 ~ normal(0, 100);
  beta1_0 ~ normal(0, 100);
  beta0_1 ~ normal(0, 100);
  beta1_1 ~ normal(0, 100);
  sigma0 ~ cauchy(1.0, 10.0);
  sigma1 ~ cauchy(1.0, 10.0);
  
  for (i in 1:N) {
    if (engine[i] == 0)
      y[i] ~ normal(beta0_0 + beta1_0 * x[i], sigma0);
    else
      y[i] ~ normal(beta0_1 + beta1_1 * x[i], sigma1);
  }
}

