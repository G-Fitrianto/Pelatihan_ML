# =============================================================================
# WEEK 2 | PREDICTION IN ECONOMIC ANALYSIS: SUPERVISED MACHINE LEARNING
# OLS versus Ridge versus LASSO on the identical Week 1 synthetic panel.
# Version 1.1 (reissued download); synthetic regional outcomes, NOT real data.
# =============================================================================
# How to run:
# 1. Set RStudio's working directory to the folder with this script.
# 2. Put ML1_Economic_Panel_Synthetic.xlsx in that folder, OR edit input_file.
# 3. Install packages listed below; then source() this file.
# 4. Results are written to ML2_outputs (not to the Week 1 output folder).
# =============================================================================

# 0. SETUP --------------------------------------------------------------------
required_packages <- c('readxl', 'dplyr', 'ggplot2', 'glmnet', 'stargazer', 'openxlsx')
not_installed <- required_packages[!vapply(required_packages, requireNamespace,
                                            logical(1), quietly = TRUE)]
if (length(not_installed)) {
  stop('Install packages first: install.packages(c("',
       paste(not_installed, collapse = '", "'), '"))')
}
suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(ggplot2)
})

# Edit this if you keep your Week 1 Excel workbook elsewhere.
input_candidates <- c(
  file.path(getwd(), 'ML1_Economic_Panel_Synthetic.xlsx'),
  file.path(getwd(), 'ML1_OLS_Prediction_Enhanced_v1_1',
            'ML1_Economic_Panel_Synthetic.xlsx'),
  'F:/MACHINE LEARNING WORKSHOP/Week 1/ML1_OLS_Prediction_Enhanced_v1_1/ML1_Economic_Panel_Synthetic.xlsx'
)
found <- input_candidates[file.exists(input_candidates)]
if (!length(found)) {
  stop('ML1_Economic_Panel_Synthetic.xlsx was not found. ',
       'Place the Week 1 workbook in getwd() or edit input_candidates.\n',
       'Working directory: ', getwd())
}
input_file <- found[1]
out_dir <- file.path(getwd(), 'ML2_outputs')
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
reference_file <- file.path(getwd(), 'Week1_reference_performance.csv')
if (!file.exists(reference_file)) {
  reference_file <- file.path(dirname(input_file), 'Week1_reference_performance.csv')
}
set.seed(20261029)
cat('Reading input:', normalizePath(input_file), '\n')

save_plot <- function(p, filename, width = 9.4, height = 5.5) {
  ggplot2::ggsave(file.path(out_dir, filename), p, width = width,
                  height = height, dpi = 170, bg = 'white')
  print(p)
}
write_out <- function(data, filename) {
  write.csv(data, file.path(out_dir, filename), row.names = FALSE, na = '')
}
score <- function(actual, predicted) {
  stopifnot(length(actual) == length(predicted),
            all(is.finite(actual)), all(is.finite(predicted)))
  e <- as.numeric(predicted - actual)
  sst <- sum((actual - mean(actual))^2)
  data.frame(n = length(e), MAE = mean(abs(e)), RMSE = sqrt(mean(e^2)),
             mean_prediction_error = mean(e),
             R2_relative_to_sample_mean = if (sst > 0) 1 - sum(e^2)/sst else NA_real_)
}

# 1. REPLICATE WEEK 1 DATA CONSTRUCTION ---------------------------------------
raw <- readxl::read_excel(input_file, sheet = 'Economic_Panel')
predictors <- c('grdp_pc_million_idr', 'unemployment_pct', 'schooling_years',
                'internet_access_pct', 'manufacturing_share_pct',
                'investment_pc_million_idr')
required_columns <- c('region_id', 'economic_zone', 'year',
                      'poverty_rate_pct', predictors)
stopifnot(all(required_columns %in% names(raw)), nrow(raw) == 720L,
          dplyr::n_distinct(raw$region_id) == 120L,
          !anyNA(raw$poverty_rate_pct))
panel <- raw %>%
  arrange(region_id, year) %>%
  group_by(region_id) %>%
  mutate(across(all_of(predictors), dplyr::lag, .names = 'lag_{.col}')) %>%
  ungroup() %>%
  filter(year >= 2021)
lag_predictors <- paste0('lag_', predictors)
train_raw <- panel %>% filter(year <= 2024)
test_raw  <- panel %>% filter(year == 2025)
stopifnot(nrow(panel) == 600L, nrow(train_raw) == 480L,
          nrow(test_raw) == 120L, max(train_raw$year) < min(test_raw$year))

# Impute using only the learning sample (within each validation fold as well).
learn_imputation <- function(learning, applying) {
  meds <- vapply(lag_predictors,
                 function(v) median(learning[[v]], na.rm = TRUE), numeric(1))
  if (!all(is.finite(meds))) stop('A training predictor has no finite median.')
  a <- learning; b <- applying
  for (nm in lag_predictors) {
    a[[nm]][is.na(a[[nm]])] <- meds[[nm]]
    b[[nm]][is.na(b[[nm]])] <- meds[[nm]]
  }
  stopifnot(!anyNA(a[, lag_predictors, drop = FALSE]),
            !anyNA(b[, lag_predictors, drop = FALSE]))
  list(learning = a, applying = b, medians = meds)
}
full_preprocess <- learn_imputation(train_raw, test_raw)
train <- full_preprocess$learning
test  <- full_preprocess$applying
write_out(data.frame(variable = names(full_preprocess$medians),
                     median = unname(full_preprocess$medians)),
          '01_training_medians.csv')

# 2. RE-ESTIMATE OLS FROM WEEK 1 EXACTLY --------------------------------------
mean_training <- mean(train$poverty_rate_pct)
ols_simple <- lm(poverty_rate_pct ~ lag_schooling_years, data = train)
ols_multiple <- lm(reformulate(lag_predictors, response = 'poverty_rate_pct'),
                   data = train)
for (dname in c('train', 'test')) {
  d <- get(dname)
  d$pred_mean <- rep(mean_training, nrow(d))
  d$pred_simple <- as.numeric(predict(ols_simple, newdata = d))
  d$pred_ols <- as.numeric(predict(ols_multiple, newdata = d))
  assign(dname, d)
}
model_column <- c('Training-mean benchmark' = 'pred_mean',
                  'Simple OLS: schooling' = 'pred_simple',
                  'Multiple OLS: six lagged indicators' = 'pred_ols')
measure_models <- function(d, split_label, models) {
  do.call(rbind, lapply(names(models), function(nm) {
    cbind(data.frame(split = split_label, model = nm),
          score(d$poverty_rate_pct, d[[models[[nm]]]]))
  }))
}
week1_reproduced <- rbind(
  measure_models(train, 'Training: 2021-2024', model_column),
  measure_models(test, 'Holdout: 2025', model_column))
rownames(week1_reproduced) <- NULL
write_out(week1_reproduced, '02_week1_baselines_reproduced.csv')

parity_status <- 'NOT CHECKED'
if (file.exists(reference_file)) {
  reference <- read.csv(reference_file, stringsAsFactors = FALSE)
  refmatch <- merge(week1_reproduced, reference, by = c('split', 'model'),
                    suffixes = c('_week2', '_week1'))
  stopifnot(nrow(refmatch) == nrow(week1_reproduced))
  for (nm in c('n', 'MAE', 'RMSE', 'mean_prediction_error',
               'R2_relative_to_sample_mean')) {
    diff <- abs(refmatch[[paste0(nm, '_week2')]] -
                refmatch[[paste0(nm, '_week1')]])
    if (anyNA(diff) || max(diff) > 1e-6)
      stop('Week 1 parity check failed for ', nm, '. Check input/preprocessing.')
  }
  parity_status <- 'PASS'
  cat('[PASS] All Week 1 baseline metrics match the supplied reference.\n')
} else {
  warning('Week 1 reference CSV not found. Baseline computed, parity not checked.')
}

# Stargazer for the unpenalized OLS coefficient table ONLY.
stargazer::stargazer(ols_simple, ols_multiple, type = 'text',
   title = 'Week 1 OLS models: training outcomes 2021-2024',
   column.labels = c('Simple OLS', 'Multiple OLS'),
   digits = 3, no.space = TRUE, header = FALSE)
for (fmt in c('text', 'html')) {
  stargazer::stargazer(ols_simple, ols_multiple, type = fmt,
    title = 'Week 1 OLS models: training outcomes 2021-2024',
    column.labels = c('Simple OLS', 'Multiple OLS'),
    digits = 3, no.space = TRUE, header = FALSE,
    out = file.path(out_dir, paste0('03_ols_stargazer.',
                                   if (fmt == 'text') 'txt' else 'html')))
}

# 3. TEACHING PLOT: LOSS + PENALTY -------------------------------------------
xz <- as.numeric(scale(train$lag_schooling_years))
y <- train$poverty_rate_pct
slope0 <- sum(xz * (y - mean(y))) / sum(xz^2)
beta_grid <- seq(slope0 - 4, slope0 + 4, length.out = 201)
illustrative_lambda <- 2 # For teaching only; never used for model selection.
loss_tbl <- do.call(rbind, lapply(c('OLS', 'Ridge', 'LASSO'), function(method) {
  data.frame(model = method, beta = beta_grid,
    objective = vapply(beta_grid, function(beta) {
      base <- mean((y - mean(y) - beta*xz)^2)/2
      if (method == 'Ridge') base <- base + illustrative_lambda*beta^2/2
      if (method == 'LASSO') base <- base + illustrative_lambda*abs(beta)
      base
    }, numeric(1)))
}))
write_out(loss_tbl, '04_loss_penalty_teaching_values.csv')
p_loss <- ggplot(loss_tbl, aes(beta, objective, color = model)) +
  geom_line(linewidth = 1) +
  labs(title = 'Same squared loss, different penalties',
       subtitle = 'One standardized predictor; lambda = 2 is illustrative',
       x = 'Schooling coefficient', y = 'Training objective') +
  theme_minimal(base_size = 12)
save_plot(p_loss, '05_loss_and_penalty_illustration.png')

# 4. EXPANDING-WINDOW VALIDATION ----------------------------------------------
# Exact 2025 outcomes are excluded from parameter tuning.
folds <- list(
  list(id = 'V1', learning = 2021L, validation = 2022L),
  list(id = 'V2', learning = 2021:2022, validation = 2023L),
  list(id = 'V3', learning = 2021:2023, validation = 2024L))
fold_design <- do.call(rbind, lapply(folds, function(f) data.frame(
  fold = f$id, training_years = paste(f$learning, collapse = ', '),
  validation_year = f$validation,
  n_training = sum(train_raw$year %in% f$learning),
  n_validation = sum(train_raw$year == f$validation))))
write_out(fold_design, '06_expanding_window_design.csv')

lambda_grid <- exp(seq(log(100), log(.001), length.out = 81))
alpha_by_model <- c(Ridge = 0, LASSO = 1)
x_matrix <- function(data) {
  x <- as.matrix(data[, lag_predictors, drop = FALSE])
  storage.mode(x) <- 'double'
  stopifnot(all(is.finite(x)), ncol(x) == length(lag_predictors))
  x
}
cv_rows <- list(); cv_ols_rows <- list(); ii <- 0L
for (f in folds) {
  pre <- learn_imputation(
    train_raw[train_raw$year %in% f$learning, ],
    train_raw[train_raw$year == f$validation, ])
  a <- pre$learning; b <- pre$applying
  X <- x_matrix(a); Xnew <- x_matrix(b)
  Y <- a$poverty_rate_pct; Ynew <- b$poverty_rate_pct

  # Give unpenalized OLS the very same historical validation observations.
  for (nm in names(model_column)) {
    if (nm == 'Training-mean benchmark') {
      this_pred <- rep(mean(Y), length(Ynew))
    } else if (nm == 'Simple OLS: schooling') {
      simple_fold <- lm(poverty_rate_pct ~ lag_schooling_years, data = a)
      this_pred <- as.numeric(predict(simple_fold, newdata = b))
    } else {
      ols_fold <- lm(reformulate(lag_predictors, response = 'poverty_rate_pct'),
                     data = a)
      this_pred <- as.numeric(predict(ols_fold, newdata = b))
    }
    cv_ols_rows[[length(cv_ols_rows) + 1L]] <-
      cbind(data.frame(fold = f$id, valid_year = f$validation, model = nm),
            score(Ynew, this_pred))
  }

  for (nm in names(alpha_by_model)) {
    fit <- glmnet::glmnet(x = X, y = Y, alpha = alpha_by_model[[nm]],
                          family = 'gaussian', lambda = lambda_grid,
                          standardize = TRUE, intercept = TRUE)
    pp <- as.matrix(predict(fit, newx = Xnew, s = lambda_grid))
    stopifnot(nrow(pp) == length(Ynew), ncol(pp) == length(lambda_grid),
              all(is.finite(pp)))
    errors <- sweep(pp, 1, Ynew, FUN = '-')
    ii <- ii + 1L
    cv_rows[[ii]] <- data.frame(
      fold = f$id, valid_year = f$validation, model = nm,
      lambda = lambda_grid, n = length(Ynew),
      MSE = colMeans(errors^2), RMSE = sqrt(colMeans(errors^2)),
      MAE = colMeans(abs(errors)))
  }
}
cv_fold_scores <- do.call(rbind, cv_rows)
ols_fold_scores <- do.call(rbind, cv_ols_rows)
write_out(cv_fold_scores, '07_penalized_validation_by_fold.csv')
write_out(ols_fold_scores, '08_OLS_validation_by_fold.csv')

# Sum over equal-sized validation years (120 regions each) and select by MSE.
cv_summary <- cv_fold_scores %>%
  group_by(model, lambda) %>%
  summarise(validation_n = sum(n), folds = n(),
            validation_MSE = weighted.mean(MSE, n),
            validation_RMSE = sqrt(validation_MSE),
            validation_MAE = weighted.mean(MAE, n), .groups = 'drop')
selected <- cv_summary %>%
  group_by(model) %>%
  arrange(validation_MSE, desc(lambda), .by_group = TRUE) %>%
  slice_head(n = 1) %>% ungroup() %>%
  rename(selected_lambda = lambda,
         selected_validation_MSE = validation_MSE,
         selected_validation_RMSE = validation_RMSE)
selected_lambda <- stats::setNames(selected$selected_lambda, selected$model)
stopifnot(all(c('Ridge', 'LASSO') %in% names(selected_lambda)))
write_out(cv_summary, '09_validation_curve.csv')
write_out(selected, '10_selected_lambda.csv')
ols_cv <- ols_fold_scores %>% group_by(model) %>%
  summarise(validation_n = sum(n),
            validation_RMSE = sqrt(weighted.mean(RMSE^2, n)),
            validation_MAE = weighted.mean(MAE, n), .groups = 'drop')
write_out(ols_cv, '11_OLS_validation_summary.csv')

p_cv <- ggplot(cv_summary, aes(lambda, validation_RMSE, color = model)) +
  geom_line(linewidth = 1.1) +
  geom_point(data = selected,
             aes(x = selected_lambda, y = selected_validation_RMSE), size = 2.6) +
  geom_hline(yintercept = ols_cv$validation_RMSE[ols_cv$model ==
             'Multiple OLS: six lagged indicators'],
             linetype = 'dashed', color = 'grey45') +
  scale_x_log10() +
  labs(title = 'Lambda selected before looking at the 2025 outcome',
       subtitle = '2022-2024 expanding validation; dashed = Multiple OLS benchmark',
       x = 'Penalty lambda (log scale)', y = 'Pooled validation RMSE') +
  theme_minimal(base_size = 12)
save_plot(p_cv, '12_validation_RMSE_vs_lambda.png')

# 5. FINAL MODELS: TRAIN 2021-2024, TEST 2025 -------------------------------
xtrain <- x_matrix(train); xtest <- x_matrix(test)
ytrain <- train$poverty_rate_pct; ytest <- test$poverty_rate_pct
final_fits <- lapply(names(alpha_by_model), function(nm) {
  glmnet::glmnet(xtrain, ytrain, family = 'gaussian',
    alpha = alpha_by_model[[nm]], lambda = lambda_grid,
    standardize = TRUE, intercept = TRUE)
})
names(final_fits) <- names(alpha_by_model)
for (nm in names(final_fits)) {
  key <- paste0('pred_', tolower(nm))
  train[[key]] <- as.numeric(predict(final_fits[[nm]],
                                    newx = xtrain, s = selected_lambda[[nm]]))
  test[[key]] <- as.numeric(predict(final_fits[[nm]],
                                   newx = xtest, s = selected_lambda[[nm]]))
}
all_columns <- c(model_column,
                 'Ridge: tuned (six predictors)' = 'pred_ridge',
                 'LASSO: tuned (six predictors)' = 'pred_lasso')
all_metrics <- rbind(measure_models(train, 'Training: 2021-2024', all_columns),
                     measure_models(test, 'Holdout: 2025', all_columns))
rownames(all_metrics) <- NULL
test_metrics <- subset(all_metrics, split == 'Holdout: 2025')
write_out(all_metrics, '13_training_and_holdout_comparison.csv')
write_out(test_metrics, '14_2025_model_comparison.csv')
cat('\n===== FINAL 2025 MODEL COMPARISON =====\n')
print(test_metrics[, c('model', 'MAE', 'RMSE', 'mean_prediction_error')],
      row.names = FALSE, digits = 5)

# 6. PREDICTION DATA, COEFFICIENTS, FIGURES ----------------------------------
predictions_2025 <- test %>%
  transmute(region_id, economic_zone, year,
    actual_poverty_pct = poverty_rate_pct,
    predicted_mean_pct = pred_mean,
    predicted_simple_OLS_pct = pred_simple,
    predicted_multiple_OLS_pct = pred_ols,
    predicted_Ridge_pct = pred_ridge,
    predicted_LASSO_pct = pred_lasso,
    signed_error_OLS_pp = pred_ols - poverty_rate_pct,
    signed_error_Ridge_pp = pred_ridge - poverty_rate_pct,
    signed_error_LASSO_pp = pred_lasso - poverty_rate_pct,
    abs_error_OLS_pp = abs(signed_error_OLS_pp),
    abs_error_Ridge_pp = abs(signed_error_Ridge_pp),
    abs_error_LASSO_pp = abs(signed_error_LASSO_pp))
stopifnot(nrow(predictions_2025) == 120L,
          dplyr::n_distinct(predictions_2025$region_id) == 120L)
write_out(predictions_2025, '15_predictions_and_errors_2025.csv')

sd_x <- vapply(seq_len(ncol(xtrain)), function(j) sd(xtrain[, j]), numeric(1))
names(sd_x) <- lag_predictors
ols_coefs <- coef(ols_multiple)
coef_ridge <- as.numeric(coef(final_fits$Ridge, s = selected_lambda[['Ridge']]))
coef_lasso <- as.numeric(coef(final_fits$LASSO, s = selected_lambda[['LASSO']]))
coef_tbl <- do.call(rbind, lapply(seq_along(lag_predictors), function(j) {
  vals <- c(unname(ols_coefs[lag_predictors[j]]),
            coef_ridge[j + 1], coef_lasso[j + 1])
  data.frame(predictor = lag_predictors[j],
    model = c('Multiple OLS', 'Ridge', 'LASSO'),
    coefficient_original_scale = vals,
    coefficient_per_training_SD = vals * sd_x[j])
}))
write_out(coef_tbl, '16_coefficients_OLS_Ridge_LASSO.csv')

paths <- do.call(rbind, lapply(names(final_fits), function(nm) {
  beta <- as.matrix(coef(final_fits[[nm]], s = lambda_grid))
  do.call(rbind, lapply(seq_along(lag_predictors), function(j) {
    data.frame(model = nm, lambda = lambda_grid,
               predictor = lag_predictors[j],
               coefficient_per_training_SD = as.numeric(beta[j + 1, ]) * sd_x[j])
  }))
}))
write_out(paths, '17_coefficient_paths.csv')

p_paths <- ggplot(paths, aes(lambda, coefficient_per_training_SD,
                             color = predictor)) +
  geom_line(linewidth = .9) +
  geom_hline(yintercept = 0, color = 'grey65') +
  geom_vline(data = data.frame(model = names(selected_lambda),
              selected_lambda = as.numeric(selected_lambda)),
              aes(xintercept = selected_lambda), linetype = 'dashed',
              inherit.aes = FALSE, show.legend = FALSE) +
  facet_wrap(~model, nrow = 2) + scale_x_log10() +
  labs(title = 'Coefficient shrinkage along the penalty path',
       subtitle = 'Lines = predictors; dashed = selected lambda',
       x = 'Penalty lambda (log scale)',
       y = 'Coefficient per 1-SD predictor change', color = 'Predictor') +
  theme_minimal(base_size = 11) + theme(legend.position = 'bottom')
save_plot(p_paths, '18_coefficient_paths.png', 10.5, 7)

p_coefs <- ggplot(coef_tbl,
                  aes(model, coefficient_per_training_SD, color = model)) +
  geom_point(size = 2.7) + facet_wrap(~predictor, scales = 'free_y') +
  geom_hline(yintercept = 0, linetype = 'dotted', color = 'grey65') +
  labs(title = 'OLS and penalized-regression coefficients',
       subtitle = 'Per 1-SD predictor increase; not causal effects',
       x = NULL, y = 'Coefficient (poverty points)') +
  theme_minimal(base_size = 11) + theme(legend.position = 'none',
      axis.text.x = element_text(angle = 28, hjust = 1))
save_plot(p_coefs, '19_coefficient_comparison.png', 10.5, 6.1)

scatter_models <- c('Multiple OLS' = 'predicted_multiple_OLS_pct',
                    'Ridge' = 'predicted_Ridge_pct',
                    'LASSO' = 'predicted_LASSO_pct')
scatter_df <- do.call(rbind, lapply(names(scatter_models), function(nm) {
  data.frame(model = nm, actual = predictions_2025$actual_poverty_pct,
             predicted = predictions_2025[[scatter_models[[nm]]]])
}))
p_scatter <- ggplot(scatter_df, aes(actual, predicted)) +
  geom_point(color = '#347052', alpha = .7) +
  geom_abline(slope = 1, intercept = 0, linetype = 'dashed',
              color = '#B58A25') + facet_wrap(~model) + coord_equal() +
  labs(title = '2025 out-of-sample: actual versus predicted',
       subtitle = 'Same 120 fictitious regions in each panel',
       x = 'Actual poverty (%)', y = 'Predicted poverty (%)') +
  theme_minimal(base_size = 11)
save_plot(p_scatter, '20_actual_vs_predicted_2025.png', 11.4, 4.8)

ordered <- predictions_2025 %>% arrange(actual_poverty_pct) %>%
  mutate(region_rank = row_number())
line_models <- c('Observed' = 'actual_poverty_pct',
                 'Multiple OLS' = 'predicted_multiple_OLS_pct',
                 'Ridge' = 'predicted_Ridge_pct',
                 'LASSO' = 'predicted_LASSO_pct')
lines_df <- do.call(rbind, lapply(names(line_models), function(nm) {
  data.frame(model = nm, rank = ordered$region_rank,
             poverty = ordered[[line_models[[nm]]]])
}))
p_lines <- ggplot(lines_df, aes(rank, poverty, color = model)) +
  geom_line(linewidth = .85) +
  labs(title = 'Observed and predicted poverty by region (2025)',
       subtitle = 'Regions sorted by observed poverty for display only',
       x = 'Region rank', y = 'Poverty (%)', color = NULL) +
  theme_minimal(base_size = 11) + theme(legend.position = 'bottom')
save_plot(p_lines, '21_prediction_profiles_2025.png', 10.3, 5.5)

p_perf <- ggplot(all_metrics, aes(model, RMSE, fill = split)) +
  geom_col(position = 'dodge') + coord_flip() +
  labs(title = 'Training versus testing accuracy',
       subtitle = 'Holdout = 2025, after historical lambda selection',
       x = NULL, y = 'RMSE (poverty percentage points)', fill = NULL) +
  theme_minimal(base_size = 11) + theme(legend.position = 'bottom')
save_plot(p_perf, '22_training_vs_test_RMSE.png', 10, 5.5)

# Report group-level errors for all 3 models on the same holdout.
zone_models <- c('Multiple OLS' = 'predicted_multiple_OLS_pct',
                 'Ridge' = 'predicted_Ridge_pct',
                 'LASSO' = 'predicted_LASSO_pct')
zone_metrics <- do.call(rbind, lapply(
  sort(unique(predictions_2025$economic_zone)), function(zone) {
    d <- predictions_2025[predictions_2025$economic_zone == zone, ]
    do.call(rbind, lapply(names(zone_models), function(nm) {
      cbind(data.frame(economic_zone = zone, model = nm),
            score(d$actual_poverty_pct, d[[zone_models[[nm]]]]))
    }))
  }))
write_out(zone_metrics, '23_zone_diagnostics.csv')

# 7. EXPORT RESULTS WORKBOOK --------------------------------------------------
wb <- openxlsx::createWorkbook()
readme <- data.frame(
  item = c('Data', 'Status', 'Target', 'Training', 'Test', 'Features',
           'Validation', 'Selected lambda', 'Error convention',
           'Parity check', 'Important caveat'),
  value = c(basename(input_file), 'Synthetic / fictitious regions',
            'Regional poverty rate (%)', '2021-2024 (n=480)',
            '2025 (n=120)', paste(lag_predictors, collapse = ', '),
            'Expanding window (2022, 2023, 2024)',
            paste(names(selected_lambda), signif(selected_lambda, 5),
                  collapse = '; '),
            'Predicted minus observed; units are percentage points',
            parity_status,
            'OLS is supervised; coefficient estimates are not causal effects'))
sheets <- list(
  Overview = readme,
  Week1_Benchmarks = week1_reproduced,
  Validation_Folds = fold_design,
  OLS_Validation = ols_cv,
  Penalized_Validation = cv_summary,
  Selected_Lambda = selected,
  Training_Test_Metrics = all_metrics,
  Test_Comparison = test_metrics,
  Predictions_2025 = predictions_2025,
  Coefficients = coef_tbl,
  Coefficient_Paths = paths,
  Zone_Diagnostics = zone_metrics)
for (nm in names(sheets)) {
  openxlsx::addWorksheet(wb, nm)
  openxlsx::writeDataTable(wb, nm, sheets[[nm]], tableStyle = 'TableStyleMedium2')
  openxlsx::freezePane(wb, nm, firstRow = TRUE)
  openxlsx::setColWidths(wb, nm, cols = seq_len(ncol(sheets[[nm]])), widths = 20)
}
openxlsx::setColWidths(wb, 'Overview', cols = 2, widths = 85)
openxlsx::setColWidths(wb, 'Test_Comparison', cols = 2, widths = 45)
openxlsx::addWorksheet(wb, 'Figures')
openxlsx::writeData(wb, 'Figures',
   'Figures are included for classroom interpretation; PNG files also saved separately.')
for (item in list(
  list(file = '12_validation_RMSE_vs_lambda.png', row = 3, h = 4.7),
  list(file = '18_coefficient_paths.png', row = 30, h = 6.0),
  list(file = '20_actual_vs_predicted_2025.png', row = 65, h = 4.7))) {
  openxlsx::insertImage(wb, 'Figures', file = file.path(out_dir, item$file),
                        startRow = item$row, startCol = 1,
                        width = 9.5, height = item$h, units = 'in')
}
output_xlsx <- file.path(out_dir, '24_ML2_OLS_Ridge_LASSO_Results.xlsx')
openxlsx::saveWorkbook(wb, output_xlsx, overwrite = TRUE)
writeLines(capture.output(sessionInfo()),
           con = file.path(out_dir, '25_session_info.txt'))

stopifnot(all(is.finite(test_metrics$RMSE)), file.exists(output_xlsx),
          nrow(predictions_2025) == 120L)
cat('\n[FINISHED] Results folder:', normalizePath(out_dir), '\n')
cat('[INFO] Week 1 baseline audit:', parity_status, '\n')
cat('[INFO] Ridge lambda:', selected_lambda[['Ridge']], '\n')
cat('[INFO] LASSO lambda:', selected_lambda[['LASSO']], '\n')
cat('[INFO] Workbook:', output_xlsx, '\n')
cat('[NOTE] Ridge/LASSO might outperform or underperform OLS on this holdout.\n')
# =============================================================================
