# Load required libraries
# Helper function to install packages if not already installed
install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

# List of required packages
required_packages <- c("dplyr", "readr", "ggplot2", "rstan", "bayesplot")

# Install missing packages
lapply(required_packages, install_if_missing)

# Load the libraries
lapply(required_packages, library, character.only = TRUE)



setwd("/home/kerupakaran/Desktop/Course")
# 1.1
# Read the data file
fuel_data <- read_csv("MATH501_2024_25_fuel_consumption_data.csv")

# Print the first few rows of the data to see its structure
head(fuel_data)

# Check the structure of the data
str(fuel_data)

# Transform the Engine variable into a factor with levels 0 and 1
fuel_data <- fuel_data %>%
  mutate(Engine = as.factor(Engine))

# Verify the transformation
str(fuel_data)
summary(fuel_data)

# 1.2
# Create the plot
ggplot(fuel_data, aes(x = Ambient_temperature, y = Running_time, color = Engine)) +
  geom_point(alpha = 0.6) +  # Scatter plot with transparency
  geom_smooth(method = "lm", se = FALSE) +  # Add trend lines
  labs(
    title = "Effect of Ambient Temperature on Running Time",
    x = "Ambient Temperature (°C)",
    y = "Running Time (minutes)",
    color = "Engine Type"
  ) +
  theme_minimal()

# 1.3

# Fit the ANCOVA model
model <- lm(Running_time ~ Ambient_temperature * Engine, data = fuel_data)

# Summary of the model
summary(model)

confint(model,level = 0.95)  # Provides 95% confidence intervals for all coefficients
#Extract just δ₀ and δ₁:
confint(model)[c("Engine1", "Ambient_temperature:Engine1"), ]

#δ₀: the difference in intercept between Engine 1 and Engine 0 (i.e., the amount Engine 1’s base fuel time differs from Engine 0 when temperature is 0°C).
#δ₁: the difference in slope (rate of change of running time with temperature) between Engine 1 and Engine 0.


coef(model)
beta00 <- coef(model)["(Intercept)"]
beta10 <- coef(model)["Ambient_temperature"]
delta0 <- coef(model)["Engine1"]
delta1 <- coef(model)["Ambient_temperature:Engine1"]

# β01 and β11
beta01 <- beta00 + delta0
beta11 <- beta10 + delta1

cat("β01 (intercept for Engine 1):", beta01, "\n")
cat("β11 (slope for Engine 1):", beta11, "\n")


sigma_hat <- summary(model)$sigma  # Residual standard deviation
sigma_hat

# Add studentized residuals to the dataset
fuel_data$resid_studentized <- rstudent(model)

# Histogram to check normality
ggplot(fuel_data, aes(x = resid_studentized)) +
  geom_histogram(bins = 20, color = "black", fill = "skyblue") +
  labs(title = "Histogram of Studentized Residuals")

# Q-Q plot for normality
qqnorm(fuel_data$resid_studentized)
qqline(fuel_data$resid_studentized)

# Residuals vs Fitted values for homoscedasticity
plot(model$fitted.values, fuel_data$resid_studentized,
     xlab = "Fitted Values", ylab = "Studentized Residuals",
     main = "Residuals vs Fitted Values")
abline(h = 0, col = "red")

#Histogram & Q-Q Plot: If residuals appear symmetric and mostly follow the Q-Q line, that supports normality.

#Residuals vs Fitted Plot: If residuals are randomly scattered around zero with no clear pattern or funnel shape, that supports constant variance (homoscedasticity).

# Outliers: Points with large studentized residuals (e.g., > 3 or < -3) may be outliers and should be investigated.


#1.5
# Create new data for prediction
new_data <- data.frame(
  Ambient_temperature = rep(c(18, 28), each = 2),
  Engine = factor(c("0", "1", "0", "1"))  # ensure Engine is a factor
)

# Get confidence intervals for mean running time
conf_pred <- predict(model, newdata = new_data, interval = "confidence")

# Get prediction intervals for individual running times
pred_interval <- predict(model, newdata = new_data, interval = "prediction")

# Combine results
results <- cbind(new_data, conf_pred, pred_interval[ , 2:3])
colnames(results)[4:7] <- c("Mean_Fit", "CI_Lower", "CI_Upper", "PI_Lower", "PI_Upper")

# View results
print(results)

#1.7

fuel_data_1 <- read_csv("MATH501_2024_25_fuel_consumption_data.csv") %>%
  mutate(Engine = ifelse(Engine == "1", 1L, 0L))  # Force it to 0 or 1 (as integers)


stan_data <- list(
  N = nrow(fuel_data_1),
  y = fuel_data_1$Running_time,
  x = fuel_data_1$Ambient_temperature,
  engine = fuel_data_1$Engine
)

# Define Stan model code as a string
stan_code <- "
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
"

stan_file <- "model2_fuel.stan"
writeLines(stan_code, con = stan_file)

# Run Stan model

fit <- stan(
  file = stan_file,
  data = stan_data,
  chains = 2,
  iter = 5000,
  warmup = 2500,
  thin = 1,
  seed = 1234
)

# Print summary of posterior estimates
print(fit, pars = c("beta0_0", "beta1_0", "beta0_1", "beta1_1", "sigma0", "sigma1"))

# Convert to data frame for bayesplot and ggplot2
posterior <- as.data.frame(fit)


# Parameters of interest
params <- c("beta0_0", "beta1_0", "beta0_1", "beta1_1", "sigma0", "sigma1")

# Extract draws
posterior_array <- as.array(fit)

# Traceplot
mcmc_trace(posterior_array, pars = params)
# The traceplots for all six model parameters (intercepts, slopes, and standard deviations) show good mixing and no visible trends or drift across iterations. The two chains overlap well, indicating that the MCMC has converged and is exploring the posterior distributions effectively. Therefore, the posterior estimates can be considered reliable.

mcmc_dens(posterior_array, pars = c("beta0_0", "beta0_1"))
mcmc_dens(posterior_array, pars = c("beta1_0", "beta1_1"))
mcmc_dens(posterior_array, pars = c("sigma0", "sigma1"))

# The posterior density plots for σ₀ and σ₁ show that Engine 1 has slightly higher residual variability than Engine 0, with minimal overlap between the distributions. This suggests it would not be appropriate to assume equal variances, and the model's assumption of separate σ values is justified.

#1.7
# Compute the difference in regression lines at x = 23 and x = 29
delta_23 <- (posterior$beta0_1 + posterior$beta1_1 * 23) - 
  (posterior$beta0_0 + posterior$beta1_0 * 23)

delta_29 <- (posterior$beta0_1 + posterior$beta1_1 * 29) - 
  (posterior$beta0_0 + posterior$beta1_0 * 29)

# Compute 90% credible intervals
ci_23 <- quantile(delta_23, probs = c(0.05, 0.95))
ci_29 <- quantile(delta_29, probs = c(0.05, 0.95))

# Print the results
cat("90% Credible Interval for difference at x = 23°C:\n")
print(ci_23)
# At an ambient temperature of 23°C, the 90% credible interval for the difference in mean running time between Engine 1 and Engine 0 is approximately [0.74, 0.97] minutes. This means that, given the data and model, we are 90% confident that Engine 1 runs 0.74 to 0.97 minutes longer than Engine 0 at 23°C.
#Since the entire interval is positive, this provides strong evidence that Engine 1 is more fuel efficient than Engine 0 at this temperature.

cat("\n90% Credible Interval for difference at x = 29°C:\n")
print(ci_29)
#At 29°C, the 90% credible interval for the difference is approximately [-0.70, -0.37] minutes. This means we are 90% confident that Engine 1 runs 0.37 to 0.70 minutes less than Engine 0 at this temperature.
# Since the interval is entirely negative, it suggests that Engine 0 outperforms Engine 1 at higher temperatures — a reversal of the earlier result.

# Extract posterior samples
posterior <- as.data.frame(fit)

# Calculate x_intersect for each posterior draw
x_intersect <- (posterior$beta0_1 - posterior$beta0_0) / 
  (posterior$beta1_0 - posterior$beta1_1)


ci_intersect <- quantile(x_intersect, probs = c(0.025, 0.975))
print(ci_intersect)

library(ggplot2)

# Convert to data frame for ggplot
df <- data.frame(x_intersect = x_intersect)

# Plot
ggplot(df, aes(x = x_intersect)) +
  geom_density(fill = "red", alpha = 0.6) +
  geom_vline(xintercept = c(27, 27.25), color = "black", linetype = "dashed", size = 1) +
  labs(title = expression(paste("Posterior Density of ", x[intersect])),
       x = expression(x[intersect]), y = "Density")

prob <- mean(x_intersect > 27 & x_intersect < 27.25)
cat("Posterior probability that 27 < x_intersect < 27.25 is:", prob)
