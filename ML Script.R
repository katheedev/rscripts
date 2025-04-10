if (!require("MASS")) install.packages("MASS")
if (!require("class")) install.packages("class")
if (!require("gbm")) install.packages("gbm")
if (!require("randomForest")) install.packages("randomForest")
if (!require("pROC")) install.packages("pROC")
if (!require("dplyr")) install.packages("dplyr")
if (!require("tidyverse")) install.packages("tidyverse")
library(dplyr)
library(MASS)           # for LDA
library(class)          # for KNN
library(gbm)            # for Boosting
library(randomForest)   # for Random Forest
library(pROC)           # for ROC/AUC
library(tidyverse)

# Task 2.1
wef <- read_csv("Wind_Energy_Failure.csv")

wef <- wef %>%
  mutate(Component_Failure = factor(Component_Failure, levels = c(0, 1)))

# Step 3: View a summary of the data
summary(wef)
# 900 observations available for failed and not failed 

# Step 4: Check class balance
table(wef$Component_Failure)

# Step 5: Visualize pairwise sensor relationships colored by failure
pairs(dplyr::select(wef, starts_with("Sensor_")), 
      col = as.numeric(wef$Component_Failure),
      pch = 19,
      main = "Pairwise Plots of Sensor Data by Component Failure")


# Step 6: Boxplots of each sensor by Component_Failure
wef_long <- wef %>%
  pivot_longer(cols = starts_with("Sensor_"), 
               names_to = "Sensor", 
               values_to = "Reading")

ggplot(wef_long, aes(x = Component_Failure, y = Reading, fill = Component_Failure)) +
  geom_boxplot() +
  facet_wrap(~ Sensor, scales = "free") +
  labs(title = "Sensor Readings by Component Failure",
       x = "Component Failure", y = "Sensor Reading") +
  theme_minimal()

# Step 7: Correlation matrix of sensor readings
sensor_data <- wef %>% dplyr::select(starts_with("Sensor_"))
cor_matrix <- round(cor(sensor_data), 2)

# Visualize correlation matrix
ggcorrplot::ggcorrplot(cor_matrix, lab = TRUE, title = "Sensor Correlation Matrix")

# Task 2.2

# Set a seed for reproducibility
set.seed(168)  # You can change this number if needed

# Step 8: Split the data into Training, Validation, and Test sets
n <- nrow(wef)

which_set <- sample(c("Training", "Validation", "Test"),
                    size = n,
                    replace = TRUE)

# Create subsets
wef_training <- wef[which_set == "Training", ]
wef_validation <- wef[which_set == "Validation", ]
wef_test <- wef[which_set == "Test", ]

# Step 9: Check the sizes of the splits
cat("Training set size:", nrow(wef_training), "\n")
cat("Validation set size:", nrow(wef_validation), "\n")
cat("Test set size:", nrow(wef_test), "\n")

# Step 10: Optionally, check class balance in each set
table(wef_training$Component_Failure)
table(wef_validation$Component_Failure)
table(wef_test$Component_Failure)

# Task 2.3
# ------------------------
# Logistic Regression (with backward elimination)
# ------------------------
logit_model <- glm(Component_Failure ~ ., data = wef_training, family = binomial)

# Backward elimination manually
step_logit <- step(logit_model, direction = "backward")

# Predict on test set
logit_probs <- predict(step_logit, newdata = wef_test, type = "response")
logit_pred <- ifelse(logit_probs > 0.5, "1", "0")
logit_error <- mean(logit_pred != wef_test$Component_Failure)

# ROC and AUC for Logistic Regression
roc_logit <- roc(wef_test$Component_Failure, logit_probs)



# ------------------------
# K-Nearest Neighbours (optimize K)
# ------------------------
train_x <- wef_training %>% dplyr::select(starts_with("Sensor_")) %>% as.matrix()
train_y <- wef_training$Component_Failure
valid_x <- wef_validation %>% dplyr::select(starts_with("Sensor_")) %>% as.matrix()
valid_y <- wef_validation$Component_Failure
test_x <- wef_test %>% dplyr::select(starts_with("Sensor_")) %>% as.matrix()
test_y <- wef_test$Component_Failure


# Tune K
k_vals <- 1:20
validation_errors <- sapply(k_vals, function(k) {
  pred <- knn(train = train_x, test = valid_x, cl = train_y, k = k)
  mean(pred != valid_y)
})
best_k <- which.min(validation_errors)

# Apply KNN to test set
knn_pred <- knn(train = train_x, test = test_x, cl = train_y, k = best_k)
knn_error <- mean(knn_pred != test_y)

# ------------------------
# Boosting (with shrinkage parameter)
# ------------------------
boost_model <- gbm(as.numeric(as.character(Component_Failure)) ~ ., 
                   data = wef_training, 
                   distribution = "bernoulli", 
                   n.trees = 5000,
                   interaction.depth = 3,
                   shrinkage = 0.01,
                   verbose = FALSE)

boost_probs <- predict(boost_model, newdata = wef_test, n.trees = 5000, type = "response")
boost_pred <- ifelse(boost_probs > 0.5, "1", "0")
boost_error <- mean(boost_pred != wef_test$Component_Failure)

# ROC and AUC for Boosting
roc_boost <- roc(wef_test$Component_Failure, boost_probs)

# ------------------------
# Linear Discriminant Analysis
# ------------------------
lda_model <- lda(Component_Failure ~ ., data = wef_training)
lda_pred <- predict(lda_model, newdata = wef_test)$class
lda_error <- mean(lda_pred != wef_test$Component_Failure)

# ------------------------
# Random Forest
# ------------------------
rf_model <- randomForest(Component_Failure ~ ., data = wef_training)
rf_pred <- predict(rf_model, newdata = wef_test)
rf_error <- mean(rf_pred != wef_test$Component_Failure)

# ------------------------
# Print All Error Rates
# ------------------------
cat("Classification Error Rates (Test Set):\n")
cat("Logistic Regression: ", round(logit_error, 3), "\n")
cat("K-Nearest Neighbours (k =", best_k, "): ", round(knn_error, 3), "\n")
cat("Boosting: ", round(boost_error, 3), "\n")
cat("LDA: ", round(lda_error, 3), "\n")
cat("Random Forest: ", round(rf_error, 3), "\n")

# ------------------------
# Plot ROC Curves
# ------------------------

# Calculate AUCs
auc_logit <- round(auc(roc_logit), 4)
auc_boost <- round(auc(roc_boost), 4)

# Plot logistic regression ROC
plot(roc_logit, col = "black", lwd = 2, main = "ROC curves",
     legacy.axes = TRUE, print.auc = FALSE)

# Add boosting ROC curve
lines(roc_boost, col = "red", lwd = 2)

# Add legend with AUC values
legend("bottomright",
       legend = c(paste("Binary Logistic Regression,", auc_logit),
                  paste("Boosting,", auc_boost)),
       col = c("black", "red"),
       lwd = 2)

# Task 2.4

# In addition to test classification error rate, we also computed sensitivity (true positive rate) as an important performance measure.
# In this application, missing a true failure (i.e., a false negative) can lead to major operational risks and costs, while a false alarm (false positive) simply causes an inspection.
# Therefore, a high sensitivity is crucial to ensure that failing components are caught early and preventative actions are taken.
# This makes sensitivity the most relevant secondary performance measure in this predictive maintenance scenario.
# 

# Confusion Matrices
confusion_logit <- table(Predicted = logit_pred, Actual = wef_test$Component_Failure)
confusion_knn <- table(Predicted = knn_pred, Actual = wef_test$Component_Failure)
confusion_boost <- table(Predicted = boost_pred, Actual = wef_test$Component_Failure)
confusion_lda <- table(Predicted = lda_pred, Actual = wef_test$Component_Failure)
confusion_rf <- table(Predicted = rf_pred, Actual = wef_test$Component_Failure)

# Function to calculate sensitivity
calc_sensitivity <- function(conf_matrix) {
  TP <- conf_matrix["1", "1"]
  FN <- conf_matrix["0", "1"]
  sensitivity <- TP / (TP + FN)
  return(round(sensitivity, 3))
}

# Sensitivities
sens_logit <- calc_sensitivity(confusion_logit)
sens_knn <- calc_sensitivity(confusion_knn)
sens_boost <- calc_sensitivity(confusion_boost)
sens_lda <- calc_sensitivity(confusion_lda)
sens_rf <- calc_sensitivity(confusion_rf)

# Print results
cat("\nSensitivity (True Positive Rate):\n")
cat("Logistic Regression:", sens_logit, "\n")
cat("K-Nearest Neighbours:", sens_knn, "\n")
cat("Boosting:", sens_boost, "\n")
cat("LDA:", sens_lda, "\n")
cat("Random Forest:", sens_rf, "\n")


# Task 2.5