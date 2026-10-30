# =============================================================================
# WEEK 2 | SIMPLE CLASSROOM DEMONSTRATION
# Prediction in Economic Analysis: Supervised Machine Learning
# Mirrors R demonstrations in slides 53, 54, 56, 57, 58 and 59 of Week 2 v3.
# Uses the SAME synthetic data and OLS specification as Week 1.
# =============================================================================
# HOW TO RUN
# 1. Extract the zip. Place this script with ML1_Economic_Panel_Synthetic.xlsx.
# 2. Set the RStudio working directory to that folder.
# 3. Source or run this script sequentially from the top.
# 4. Tables and graphs will appear in ML2_Simple_Outputs.
# =============================================================================

# 0. SETUP: ONLY NEEDED WHEN RUNNING WEEK 2 AS A STANDALONE SCRIPT ------------
packages <- c("readxl", "dplyr", "ggplot2", "glmnet", "openxlsx")
for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}
library(readxl)
library(dplyr)
library(ggplot2)

input_file <- "ML1_Economic_Panel_Synthetic.xlsx"
out_dir <- "ML2_Simple_Outputs"
if (!file.exists(input_file)) {
  stop("Excel workbook not found. Set the working directory to the extracted folder.")
}
if (!dir.exists(out_dir)) dir.create(out_dir)

# 0A. RECREATE THE WEEK 1 PANEL, TRAIN/TEST SPLIT AND OLS ---------------------
raw <- read_excel(input_file, sheet = "Economic_Panel")

predictors <- c("grdp_pc_million_idr", "unemployment_pct", "schooling_years",
                "internet_access_pct", "manufacturing_share_pct",
                "investment_pc_million_idr")

panel <- raw %>%
  arrange(region_id, year) %>%
  group_by(region_id) %>%
  mutate(across(all_of(predictors), dplyr::lag, .names = "lag_{.col}")) %>%
  ungroup() %>%
  filter(year >= 2021)

lag_predictors <- paste0("lag_", predictors)
train <- panel %>% filter(year <= 2024)
test  <- panel %>% filter(year == 2025)
stopifnot(nrow(train) == 480, nrow(test) == 120)

# Week 1: learn missing-value medians ONLY from training observations.
medians <- vapply(lag_predictors, function(v)
  median(train[[v]], na.rm = TRUE), numeric(1))
for (v in lag_predictors) {
  train[[v]][is.na(train[[v]])] <- medians[[v]]
  test[[v]][is.na(test[[v]])]   <- medians[[v]]
}

# This is exactly the six-predictor multiple OLS specification from Week 1.
ols_model <- lm(reformulate(lag_predictors, response = "poverty_rate_pct"),
                data = train)
print(summary(ols_model))

# =============================================================================
# SLIDE DEMO 1: Matrices
# =============================================================================
# Run Week 1 first: train, test, lag_predictors and ols_model exist.
# install.packages("glmnet")  # only if needed (already checked in setup)
library(glmnet)

x_train <- as.matrix(train[, lag_predictors])
x_test  <- as.matrix(test[, lag_predictors])
y_train <- train$poverty_rate_pct
y_test  <- test$poverty_rate_pct

stopifnot(ncol(x_train) == 6,
          nrow(x_train) == 480,
          nrow(x_test) == 120)

# =============================================================================
# SLIDE DEMO 2: Ridge / LASSO paths
# =============================================================================
lambda_grid <- 10^seq(2, -4, length.out = 75)

ridge_path <- glmnet(x_train, y_train, alpha = 0,
                     lambda = lambda_grid, standardize = TRUE)
lasso_path <- glmnet(x_train, y_train, alpha = 1,
                     lambda = lambda_grid, standardize = TRUE)

plot(ridge_path, xvar = "lambda", main = "Ridge paths")
plot(lasso_path, xvar = "lambda", main = "LASSO paths")

# =============================================================================
# SLIDE DEMO 4: 2024 validation set
# =============================================================================
# panel contains the lagged variables BEFORE imputation.
subtrain <- subset(panel, year <= 2023)
valid    <- subset(panel, year == 2024)

# Use only subtrain to learn missing-data replacements.
med <- vapply(lag_predictors, function(v)
  median(subtrain[[v]], na.rm = TRUE), numeric(1))

for (v in lag_predictors) {
  subtrain[[v]][is.na(subtrain[[v]])] <- med[[v]]
  valid[[v]][is.na(valid[[v]])]       <- med[[v]]
}

# =============================================================================
# SLIDE DEMO 5: Validation RMSE
# =============================================================================
xx <- as.matrix(subtrain[, lag_predictors])
xv <- as.matrix(valid[, lag_predictors])
yy <- subtrain$poverty_rate_pct

tune_once <- function(alpha) {
  fit <- glmnet(xx, yy, alpha = alpha,
                lambda = lambda_grid, standardize = TRUE)
  predictions <- predict(fit, newx = xv, s = lambda_grid)
  squared <- sweep(predictions, 1, valid$poverty_rate_pct, "-")^2
  scores <- sqrt(colMeans(squared))
  lambda_grid[which.min(scores)]
}

# =============================================================================
# SLIDE DEMO 6: 2025 forecasts
# =============================================================================
lam_ridge <- tune_once(alpha = 0)
lam_lasso <- tune_once(alpha = 1)

ridge_final <- glmnet(x_train, y_train, alpha = 0,
                      lambda = lambda_grid, standardize = TRUE)
lasso_final <- glmnet(x_train, y_train, alpha = 1,
                      lambda = lambda_grid, standardize = TRUE)

pred_ridge <- as.numeric(predict(ridge_final, x_test, s = lam_ridge))
pred_lasso <- as.numeric(predict(lasso_final, x_test, s = lam_lasso))
pred_ols <- as.numeric(predict(ols_model, newdata = test))

# =============================================================================
# SLIDE DEMO 7: Evaluation metrics
# =============================================================================
mae  <- function(y, p) mean(abs(y - p))
rmse <- function(y, p) sqrt(mean((y - p)^2))

comparison <- data.frame(
  model = c("Multiple OLS", "Ridge", "LASSO"),
  MAE  = c(mae(y_test, pred_ols),  mae(y_test, pred_ridge),
           mae(y_test, pred_lasso)),
  RMSE = c(rmse(y_test, pred_ols), rmse(y_test, pred_ridge),
           rmse(y_test, pred_lasso))
)
print(comparison)

# =============================================================================
# 8. SIMPLE OUTPUTS: THE SAME 2025 RESULTS AS THE SLIDE
# These are short additional commands for saving workshop results.
# =============================================================================
print(data.frame(model = c("Ridge", "LASSO"),
                 selected_lambda = c(lam_ridge, lam_lasso)))

# Include the Week 1 benchmark and Simple OLS for context.
ols_simple <- lm(poverty_rate_pct ~ lag_schooling_years, data = train)
pred_simple <- as.numeric(predict(ols_simple, newdata = test))
pred_mean <- rep(mean(y_train), length(y_test))

full_comparison <- rbind(
  data.frame(model = "Training mean", MAE = mae(y_test, pred_mean),
             RMSE = rmse(y_test, pred_mean)),
  data.frame(model = "Simple OLS", MAE = mae(y_test, pred_simple),
             RMSE = rmse(y_test, pred_simple)),
  comparison
)
print(full_comparison)

# Every row is one of the SAME 120 fictitious regions from 2025.
predictions_2025 <- data.frame(
  region_id = test$region_id,
  economic_zone = test$economic_zone,
  actual = y_test,
  OLS = pred_ols,
  Ridge = pred_ridge,
  LASSO = pred_lasso
)
predictions_2025$error_OLS   <- predictions_2025$OLS - y_test
predictions_2025$error_Ridge <- predictions_2025$Ridge - y_test
predictions_2025$error_LASSO <- predictions_2025$LASSO - y_test

# OLS baseline must match the published Week 1 holdout results.
stopifnot(abs(rmse(y_test, pred_ols) - 1.977169951) < 0.0001)
stopifnot(abs(mae(y_test, pred_ols) - 1.522911029) < 0.0001)
cat("[PASS] Week 1 multiple-OLS baseline reproduced.\n")

# Save tables as CSV and one easy-to-browse Excel workbook.
write.csv(full_comparison, file.path(out_dir, "01_model_comparison_2025.csv"),
          row.names = FALSE)
write.csv(predictions_2025, file.path(out_dir, "02_predictions_2025.csv"),
          row.names = FALSE)
openxlsx::write.xlsx(
  list(Model_Comparison = full_comparison,
       Selected_Lambda = data.frame(model = c("Ridge", "LASSO"),
                                    lambda = c(lam_ridge, lam_lasso)),
       Predictions_2025 = predictions_2025),
  file = file.path(out_dir, "03_OLS_Ridge_LASSO_Results.xlsx"),
  overwrite = TRUE
)

# Plot: actual versus predicted poverty, using identical 2025 observations.
plot_df <- rbind(
  data.frame(model = "OLS", actual = y_test, predicted = pred_ols),
  data.frame(model = "Ridge", actual = y_test, predicted = pred_ridge),
  data.frame(model = "LASSO", actual = y_test, predicted = pred_lasso)
)
p_prediction <- ggplot(plot_df, aes(actual, predicted)) +
  geom_point(color = "#407958", alpha = 0.65) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  facet_wrap(~model) +
  coord_equal() +
  labs(title = "OLS vs Ridge vs LASSO: 2025 predictions",
       x = "Actual poverty rate (%)", y = "Predicted poverty rate (%)") +
  theme_minimal(base_size = 12)
print(p_prediction)
ggsave(file.path(out_dir, "04_actual_vs_predicted_2025.png"),
       p_prediction, width = 10.5, height = 4.7, dpi = 180)

# Plot: error bars for the SAME sample, model by model.
plot_errors <- rbind(
  data.frame(model = "OLS", error = predictions_2025$error_OLS),
  data.frame(model = "Ridge", error = predictions_2025$error_Ridge),
  data.frame(model = "LASSO", error = predictions_2025$error_LASSO)
)
p_errors <- ggplot(plot_errors, aes(model, error, fill = model)) +
  geom_boxplot(alpha = 0.75) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "2025 signed prediction errors",
       subtitle = "Predicted minus observed (positive = over-prediction)",
       x = NULL, y = "Prediction error (percentage points)") +
  theme_minimal(base_size = 12) + theme(legend.position = "none")
print(p_errors)
ggsave(file.path(out_dir, "05_prediction_errors_2025.png"),
       p_errors, width = 8, height = 5, dpi = 180)

cat("\nFinished. Results saved to:", normalizePath(out_dir), "\n")
cat("All input data are synthetic. Results are teaching examples, not Indonesia evidence.\n")
