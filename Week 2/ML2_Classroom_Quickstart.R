# WEEK 2 | CLASSROOM WALKTHROUGH
# Source ML2_Supervised_OLS_Ridge_LASSO.R in the same RStudio session FIRST.
# This file presents results and figures without retuning models using 2025.

stopifnot(exists('week1_reproduced'), exists('cv_summary'),
          exists('selected'), exists('selected_lambda'),
          exists('test_metrics'), exists('coef_tbl'),
          exists('predictions_2025'))

cat('\n========= 1. WHAT IS SUPERVISED? =========\n')
cat('The observed training poverty Y supervises estimation of OLS, Ridge and LASSO.\n')
cat('All three learn to predict Y from the same lagged predictors X.\n')
cat('Ridge/LASSO add a coefficient penalty. OLS does not.\n')

cat('\n========= 2. WEEK 1 BENCHMARK =========\n')
print(subset(week1_reproduced, split == 'Holdout: 2025')[,
  c('model', 'MAE', 'RMSE')], row.names = FALSE)

cat('\n========= 3. PENALTY STRENGTH: LEARNING WITHOUT LEAKAGE =========\n')
print(fold_design, row.names = FALSE)
print(selected[, c('model', 'selected_lambda',
                  'selected_validation_RMSE')], row.names = FALSE)
print(p_cv)

cat('\n========= 4. COEFFICIENT SHRINKAGE =========\n')
print(coef_tbl, row.names = FALSE)
print(p_paths)
print(p_coefs)

cat('\n========= 5. OUT-OF-SAMPLE 2025: FAIR COMPARISON =========\n')
print(test_metrics[, c('model', 'MAE', 'RMSE',
                       'mean_prediction_error')], row.names = FALSE)
print(p_scatter)
print(p_lines)
print(p_perf)

cat('\n========= 6. ECONOMIC-ZONE DIAGNOSTICS =========\n')
print(zone_metrics[, c('economic_zone', 'model', 'MAE', 'RMSE')],
      row.names = FALSE)

cat('\n========= 7. EXPORTS =========\n')
cat('Output directory:', normalizePath(out_dir), '\n')
cat('Excel workbook: 24_ML2_OLS_Ridge_LASSO_Results.xlsx\n')

cat('\nQUESTIONS TO DISCUSS\n')
cat('1. Why does the observed outcome Y make this supervised learning?\n')
cat('2. Which Ridge/LASSO lambda minimized historical validation RMSE?\n')
cat('3. Are any LASSO coefficients exactly zero?\n')
cat('4. Which model has the lowest 2025 RMSE? Is the gap large?\n')
cat('5. Why is coefficient precision not the same as prediction accuracy?\n')
cat('6. Why is a LASSO-selected coefficient not a causal effect?\n')
