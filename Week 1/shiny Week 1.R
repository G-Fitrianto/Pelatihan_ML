# ================================================================
# ECONOMIC REGRESSION LAB: OLS, RIDGE, LASSO, AND GENERALIZATION
# Session 1 - Introduction to Machine Learning for Economic Analysis
# An interactive teaching app in R Shiny.
# Version 1.1.0 (2026-10-22): QR namespace, single-X penalty, and ggplot fixes.
#
# Install once:
# install.packages(c("shiny", "ggplot2", "glmnet"))
# Run: shiny::runApp("path/to/folder", launch.browser = TRUE)
# Or open app.R in RStudio and click "Run App".
# ================================================================

required_packages <- c("shiny", "ggplot2", "glmnet")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0L) {
  stop("Install packages first: install.packages(c(",
       paste(sprintf('"%s"', missing_packages), collapse = ", "), "))",
       call. = FALSE)
}

library(shiny)
library(ggplot2)

# ----------------------------------------------------------------
# Shared helpers and reproducible synthetic data generators
# ----------------------------------------------------------------
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))
fmt <- function(x, digits = 3) formatC(x, digits = digits, format = "f")

plot_theme <- theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 13),
    legend.position = "bottom",
    legend.title = element_blank(),
    strip.text = element_text(face = "bold")
  )

true_linear <- function(x) 2 + 1.8 * x
true_nonlinear <- function(x) 1 + 1.15 * sin(2.2 * x) + 0.30 * x

make_simple_data <- function(n, sigma, seed, refresh) {
  set.seed(as.integer(seed + 101L * refresh))
  x <- sort(runif(n, min = 0, max = 5))
  data.frame(x = x, y = true_linear(x) + rnorm(n, sd = sigma))
}

make_multi_data <- function(n, rho, sigma, seed, refresh, p = 12L) {
  set.seed(as.integer(seed + 179L * refresh))
  n_test <- 350L
  create_x <- function(nobs) {
    shared <- rnorm(nobs)
    idiosyncratic <- matrix(rnorm(nobs * p), nrow = nobs, ncol = p)
    sqrt(rho) * matrix(shared, nrow = nobs, ncol = p) +
      sqrt(1 - rho) * idiosyncratic
  }
  x_train <- create_x(n)
  x_test <- create_x(n_test)
  colnames(x_train) <- colnames(x_test) <- paste0("X", seq_len(p))

  # Sparse ground truth; 9 additional predictors have coefficients zero.
  truth_beta <- c(2.4, -1.7, 1.2, rep(0, p - 3L))
  truth_intercept <- 3
  y_train <- as.numeric(truth_intercept + x_train %*% truth_beta +
                          rnorm(n, sd = sigma))
  y_test <- as.numeric(truth_intercept + x_test %*% truth_beta +
                         rnorm(n_test, sd = sigma))
  list(x_train = x_train, y_train = y_train,
       x_test = x_test, y_test = y_test,
       truth_beta = truth_beta)
}

make_poly_data <- function(n, sigma, seed, refresh) {
  set.seed(as.integer(seed + 251L * refresh))
  x_train <- sort(runif(n, -2, 2))
  x_test <- runif(350, -2, 2)
  list(
    train = data.frame(x = x_train,
                       y = true_nonlinear(x_train) + rnorm(n, sd = sigma)),
    test = data.frame(x = x_test,
                      y = true_nonlinear(x_test) + rnorm(350, sd = sigma))
  )
}

fit_orthogonal_poly <- function(x, y, degree) {
  # Orthogonal polynomial basis avoids numerical problems with raw x^12.
  basis <- stats::poly(x, degree = degree)
  design <- cbind(Intercept = 1, basis)
  coeff <- base::qr.coef(qr(design), y)
  list(basis = basis, coeff = as.numeric(coeff),
       train_fitted = as.numeric(design %*% coeff))
}

predict_orthogonal_poly <- function(fit, x_new) {
  basis_new <- stats::predict(fit$basis, newdata = as.numeric(x_new))
  as.numeric(cbind(Intercept = 1, basis_new) %*% fit$coeff)
}


# Single-predictor penalized fits (Gaussian squared-error objective).
# glmnet requires >= 2 predictor columns, so this transparent classroom
# example uses the exact one-feature closed-form solutions instead.
# We standardize X with sqrt(mean((x - mean(x))^2)), leave the intercept
# unpenalized, and penalize the slope on standardized X:
#   mean((y - a - beta_z*z)^2)/2 + lambda * P(beta_z)
# where P = beta_z^2/2 for Ridge or abs(beta_z) for LASSO.
fit_one_x_penalty <- function(x, y, lambda, method = c("ridge", "lasso")) {
  method <- match.arg(method)
  stopifnot(length(x) == length(y), all(is.finite(x)),
            all(is.finite(y)), length(lambda) == 1L,
            is.finite(lambda), lambda >= 0)
  mx <- mean(x)
  my <- mean(y)
  sx <- sqrt(mean((x - mx)^2))
  if (!is.finite(sx) || sx <= 0) stop("X must vary for this demo.")
  z <- (x - mx) / sx
  score <- mean(z * (y - my))
  beta_z <- if (method == "ridge") {
    score / (1 + lambda)
  } else {
    sign(score) * pmax(abs(score) - lambda, 0)
  }
  slope <- beta_z / sx
  c(intercept = my - slope * mx, slope = slope)
}

# ----------------------------------------------------------------
# User interface
# ----------------------------------------------------------------
ui <- navbarPage(
  title = "Economic Regression Lab",
  id = "active_module",
  header = tags$head(tags$style(HTML("\n    body { background: #f8fafc; }\n    .navbar-default { background-color: #113257; border: none; }\n    .navbar-default .navbar-brand, .navbar-default .navbar-nav > li > a { color: #fff !important; }\n    .navbar-default .navbar-nav > .active > a { background-color: #24618b !important; }\n    .well { background: #f1f5f9; border: 1px solid #dbe4ef; border-radius: 9px; }\n    .teaching-note { padding: 12px 16px; margin: 12px 0; background: #edf4fc;\n                     border-left: 4px solid #2e6eaa; border-radius: 4px; }\n    .small-note { font-size: 12px; color: #4b5563; }\n    .container-fluid { max-width: 1550px; }\n    h3 { color: #143858; }\n  "))),

  tabPanel(
    title = "1. Fit the OLS Line",
    fluidPage(
      h3("How OLS finds the best intercept and slope"),
      p("Move a and b, then compare your line against the least-squares optimum."),
      sidebarLayout(
        sidebarPanel(
          h4("Your model: y = a + b x"),
          sliderInput("manual_a", "Intercept (a)", -5, 10, value = 0, step = 0.1),
          sliderInput("manual_b", "Slope (b)", -2, 5, value = 0, step = 0.1),
          checkboxInput("show_residuals", "Show residuals from your line", TRUE),
          checkboxInput("show_truth", "Show true simulation line", FALSE),
          checkboxInput("show_penalized_simple", "Overlay Ridge and LASSO (one predictor)", FALSE),
          div(class = "small-note", "One-predictor penalty curves use exact closed-form solutions."),
          conditionalPanel(
            condition = "input.show_penalized_simple == true",
            sliderInput("simple_loglambda", "log10(lambda) for Ridge/LASSO",
                        -2, 1, value = -0.2, step = 0.1)
          ),
          actionButton("use_ols", "Move sliders to OLS optimum", class = "btn-primary"),
          tags$hr(),
          sliderInput("simple_n", "Number of observations", 15, 100, value = 45, step = 5),
          sliderInput("simple_noise", "Noise (standard deviation)", 0.2, 3, value = 1, step = 0.1),
          numericInput("simple_seed", "Random seed", value = 123, min = 1, max = 100000),
          actionButton("simple_refresh", "Generate another sample"),
          div(class = "small-note", "Data are simulated; true process: Y = 2 + 1.8 X + error.")
        ),
        mainPanel(
          fluidRow(
            column(8, plotOutput("simple_fit", height = "400px")),
            column(4, h4("Fit diagnostics"), tableOutput("simple_metrics"),
                   conditionalPanel(condition = "input.show_penalized_simple == true",
                                    h4("Penalized one-X comparison"),
                                    tableOutput("simple_penalty_table")),
                   div(class = "teaching-note",
                       strong("Teaching point"),
                       p("OLS minimizes the sum of squared residuals in the training sample. The intercept and slope are chosen jointly.")))
          ),
          h4("Loss landscape: your (a, b) versus the OLS minimum"),
          plotOutput("simple_loss", height = "360px"),
          p(class = "small-note", "The heatmap uses log(1 + sum of squared residuals) for readability. The OLS minimum is computed analytically with lm().")
        )
      )
    )
  ),

  tabPanel(
    title = "2. OLS vs Ridge & LASSO",
    fluidPage(
      h3("What changes when we penalize regression coefficients?"),
      p("Twelve predictors, three genuine signals, correlated predictors, and a genuinely held-out test sample."),
      sidebarLayout(
        sidebarPanel(
          sliderInput("multi_n", "Training sample size", 30, 180, value = 45, step = 5),
          sliderInput("multi_rho", "Correlation among predictors (rho)", 0, 0.97, value = 0.85, step = 0.01),
          sliderInput("multi_noise", "Outcome noise", 0.3, 5, value = 2.5, step = 0.1),
          selectInput("lambda_mode", "How to choose lambda?",
                      choices = c("5-fold CV on training only" = "cv",
                                  "Manual exploration" = "manual")),
          conditionalPanel(
            condition = "input.lambda_mode == 'manual'",
            sliderInput("log_lambda", "log10(lambda)", -3, 1.7, value = -0.4, step = 0.1)
          ),
          numericInput("multi_seed", "Random seed", value = 123, min = 1, max = 100000),
          actionButton("multi_refresh", "Generate another sample"),
          div(class = "teaching-note",
              strong("Important distinction"),
              p("OLS is also supervised learning. Ridge and LASSO are penalized linear regression models."),
              p("Ridge uses an L2 penalty; LASSO uses an L1 penalty and can set coefficients to zero."))
        ),
        mainPanel(
          h4("Training error versus held-out test error"),
          plotOutput("multi_rmse", height = "300px"),
          tableOutput("multi_metrics"),
          fluidRow(
            column(6, h4("Predictions on unseen test data"),
                   plotOutput("multi_predictions", height = "365px")),
            column(6, h4("Estimated coefficients"),
                   plotOutput("multi_coefficients", height = "365px"))
          ),
          h4("Choose lambda using training-fold cross-validation"),
          plotOutput("multi_cv", height = "305px"),
          div(class = "teaching-note",
              strong("Correct evaluation workflow"),
              p("Cross-validation selects lambda using training data. The test set evaluates the selected models. Manually optimizing lambda against displayed test error is test-set leakage and should be treated as a demonstration only."),
              p("Standardization for regularized models is estimated inside glmnet using the training sample. Coefficients are returned on the original predictor scale."))
        )
      )
    )
  ),

  tabPanel(
    title = "3. Complexity & Overfitting",
    fluidPage(
      h3("Can a more flexible model make worse predictions?"),
      p("Fit polynomial OLS models of varying complexity to one nonlinear economic relationship."),
      sidebarLayout(
        sidebarPanel(
          sliderInput("degree", "Polynomial degree", 1, 12, value = 3, step = 1),
          sliderInput("poly_n", "Training sample size", 25, 120, value = 35, step = 5),
          sliderInput("poly_noise", "Outcome noise", 0.15, 1.8, value = 0.8, step = 0.05),
          numericInput("poly_seed", "Random seed", value = 123, min = 1, max = 100000),
          actionButton("poly_refresh", "Generate another sample"),
          div(class = "teaching-note",
              strong("Teaching point"),
              p("Increasing the polynomial degree cannot increase training SSE under a nested least-squares fit. Test error, however, can rise because of overfitting."))
        ),
        mainPanel(
          plotOutput("poly_fit", height = "385px"),
          h4("Train and test RMSE across degrees"),
          plotOutput("poly_errors", height = "315px"),
          tableOutput("poly_metrics"),
          p(class = "small-note", "Both training and test X values are generated from the same support. The true curve is known only because this is simulated data.")
        )
      )
    )
  ),

  tabPanel(
    title = "4. Bias-Variance Simulation",
    fluidPage(
      h3("Where does expected prediction error come from?"),
      p("Repeat the same experiment many times at a fixed evaluation point x0."),
      sidebarLayout(
        sidebarPanel(
          sliderInput("bv_n", "Observations in each training sample", 25, 100, value = 35, step = 5),
          sliderInput("bv_sigma", "Noise standard deviation", 0.2, 2, value = 0.9, step = 0.1),
          sliderInput("bv_x0", "Evaluation point x0", -1.5, 1.5, value = 0.8, step = 0.1),
          sliderInput("bv_rep", "Number of repeated samples", 30, 250, value = 100, step = 10),
          numericInput("bv_seed", "Random seed", value = 123, min = 1, max = 100000),
          div(class = "teaching-note",
              strong("Precise interpretation"),
              p("Bias here means the bias of predictions at x0, not coefficient bias or causal identification bias."),
              p("We hold the design X fixed and re-draw the noise. The decomposition is pointwise and Monte Carlo approximate."))
        ),
        mainPanel(
          h4("Prediction distribution across repeated samples"),
          plotOutput("bv_distribution", height = "350px"),
          h4("Squared bias + variance + irreducible error"),
          plotOutput("bv_decomposition", height = "325px"),
          tableOutput("bv_metrics"),
          p(class = "small-note", "E[(Y_new - fhat(x0))^2] = Bias[fhat(x0)]^2 + Var[fhat(x0)] + sigma^2, assuming independent zero-mean noise and squared-error loss.")
        )
      )
    )
  ),

  tabPanel(
    title = "Instructor Guide",
    fluidPage(
      h3("Suggested in-class teaching route (approximately 35-50 minutes)"),
      tags$ol(
        tags$li(strong("7-10 min - Fit the OLS Line:"),
                " Let students move a and b and observe the residuals and loss landscape."),
        tags$li(strong("12-15 min - OLS vs Ridge & LASSO:"),
                " Compare correlated predictors, coefficients, CV lambda, and untouched test error."),
        tags$li(strong("8-10 min - Complexity:"),
                " Increase polynomial degree until training and test performance diverge."),
        tags$li(strong("8-10 min - Bias-Variance:"),
                " Show why low training error and low prediction error are different objectives.")
      ),
      h4("Concepts to emphasize"),
      tags$ul(
        tags$li("OLS minimizes in-sample squared residuals; this alone does not establish future predictive accuracy."),
        tags$li("Ridge and LASSO change the optimization criterion by adding a penalty to non-intercept coefficients."),
        tags$li("Regularization may introduce prediction bias yet lower variance; it does not guarantee better test RMSE."),
        tags$li("LASSO selection among highly correlated predictors can be unstable."),
        tags$li("A test set should not be repeatedly used to select models or hyperparameters."),
        tags$li("Predictive importance is not equivalent to a causal effect.")
      ),
      h4("Equations for whiteboard"),
      tags$p(tags$code("OLS: minimize sum_i (y_i - a - X_i b)^2")),
      tags$p(tags$code("glmnet Gaussian: minimize (1/(2n))*SSE + lambda * [((1-alpha)/2)*sum(b_j^2) + alpha*sum(abs(b_j))]")),
      tags$p(tags$code("Ridge: alpha = 0; LASSO: alpha = 1; OLS: no penalty")),
      tags$p("glmnet standardizes predictors internally by default and does not penalize the intercept."),
      h4("Dependencies and launch"),
      tags$pre('install.packages(c("shiny", "ggplot2", "glmnet"))\nshiny::runApp("folder_containing_app.R", launch.browser = TRUE)'),
      h4("References"),
      tags$p("James, Witten, Hastie, Tibshirani & Taylor, An Introduction to Statistical Learning (R edition)."),
      tags$p("Friedman, Hastie & Tibshirani (2010), Regularization Paths for Generalized Linear Models via Coordinate Descent, Journal of Statistical Software."),
      tags$p("R package docs: shiny.posit.co; glmnet on CRAN.")
    )
  )
)

# ----------------------------------------------------------------
# Reactive server logic
# ----------------------------------------------------------------
server <- function(input, output, session) {

  # MODULE 1 - OLS manually adjusted line
  simple_data <- reactive({
    make_simple_data(input$simple_n, input$simple_noise,
                     input$simple_seed, input$simple_refresh)
  })

  simple_fit <- reactive({
    dat <- simple_data()
    mod <- lm(y ~ x, data = dat)
    list(data = dat, coef = coef(mod), y_ols = fitted(mod),
         y_manual = input$manual_a + input$manual_b * dat$x)
  })

  observeEvent(input$use_ols, {
    ab <- simple_fit()$coef
    updateSliderInput(session, "manual_a", value = unname(ab[1]))
    updateSliderInput(session, "manual_b", value = unname(ab[2]))
  })

  simple_penalized <- reactive({
    req(input$show_penalized_simple)
    dat <- simple_data()
    lam <- 10^input$simple_loglambda
    list(ridge = fit_one_x_penalty(dat$x, dat$y, lam, "ridge"),
         lasso = fit_one_x_penalty(dat$x, dat$y, lam, "lasso"),
         lambda = lam)
  })

  output$simple_fit <- renderPlot({
    v <- simple_fit()
    dd <- v$data
    xx <- seq(0, 5, length.out = 200)
    line_data <- rbind(
      data.frame(x = xx, y = input$manual_a + input$manual_b * xx,
                 Model = "Your manual line"),
      data.frame(x = xx, y = v$coef[1] + v$coef[2] * xx,
                 Model = "OLS optimum")
    )
    if (input$show_truth) {
      line_data <- rbind(line_data,
                         data.frame(x = xx, y = true_linear(xx), Model = "True process"))
    }
    if (isTRUE(input$show_penalized_simple)) {
      pen <- simple_penalized()
      line_data <- rbind(
        line_data,
        data.frame(x = xx, y = pen$ridge[1] + pen$ridge[2] * xx,
                   Model = "Ridge (one X)"),
        data.frame(x = xx, y = pen$lasso[1] + pen$lasso[2] * xx,
                   Model = "LASSO (one X)")
      )
    }
    g <- ggplot(dd, aes(x = x, y = y)) +
      geom_point(color = "#374151", size = 2.1, alpha = 0.85) +
      geom_line(data = line_data, aes(x = x, y = y, color = Model),
                linewidth = 1.2, inherit.aes = FALSE) +
      scale_color_manual(values = c("Your manual line" = "#d95f02",
                                    "OLS optimum" = "#1b73b8",
                                    "True process" = "#278057",
                                    "Ridge (one X)" = "#8257a6",
                                    "LASSO (one X)" = "#138687")) +
      labs(x = "X", y = "Y", title = "Drag a and b to change the fitted line") +
      plot_theme
    if (isTRUE(input$show_residuals)) {
      g <- g + geom_segment(data = dd, aes(x = x, xend = x,
                             y = y, yend = v$y_manual),
                            color = "#d95f02", alpha = 0.42, linewidth = 0.45,
                            inherit.aes = FALSE)
    }
    g
  })

  output$simple_metrics <- renderTable({
    v <- simple_fit()
    dat <- v$data
    manual_sse <- sum((dat$y - v$y_manual)^2)
    optimal_sse <- sum((dat$y - v$y_ols)^2)
    data.frame(
      Statistic = c("a (intercept)", "b (slope)", "SSE", "RMSE"),
      `Your line` = c(fmt(input$manual_a), fmt(input$manual_b),
                      fmt(manual_sse), fmt(sqrt(manual_sse / nrow(dat)))),
      `OLS minimum` = c(fmt(v$coef[1]), fmt(v$coef[2]),
                        fmt(optimal_sse), fmt(sqrt(optimal_sse / nrow(dat)))),
      check.names = FALSE
    )
  }, striped = TRUE, spacing = "s", rownames = FALSE)

  output$simple_penalty_table <- renderTable({
    req(input$show_penalized_simple)
    dat <- simple_data()
    pen <- simple_penalized()
    ols <- coef(lm(y ~ x, data = dat))
    bs <- list(OLS = unname(ols), Ridge = pen$ridge, LASSO = pen$lasso)
    do.call(rbind, lapply(names(bs), function(nm) {
      b <- bs[[nm]]
      data.frame(Model = nm, Slope = fmt(b[2]),
                 `Training SSE` = fmt(sum((dat$y - b[1] - b[2] * dat$x)^2)),
                 check.names = FALSE)
    }))
  }, striped = TRUE, spacing = "xs", rownames = FALSE)

  output$simple_loss <- renderPlot({
    dat <- simple_data()
    ab <- coef(lm(y ~ x, data = dat))
    ax <- seq(-5, 10, length.out = 95)
    bx <- seq(-2, 5, length.out = 95)
    grid <- expand.grid(a = ax, b = bx)
    # Vectorized residual-sum-of-squares evaluation for all (a,b).
    # SSE = sum(y^2) - 2a sum(y) - 2b sum(xy) + n a^2 + 2ab sum(x) + b^2 sum(x^2)
    n <- nrow(dat)
    grid$SSE <- sum(dat$y^2) - 2*grid$a*sum(dat$y) -
      2*grid$b*sum(dat$x * dat$y) + n*grid$a^2 +
      2*grid$a*grid$b*sum(dat$x) + grid$b^2*sum(dat$x^2)
    grid$log_sse <- log1p(pmax(grid$SSE, 0))
    ggplot(grid, aes(x = a, y = b)) +
      geom_raster(aes(fill = log_sse), interpolate = TRUE) +
      geom_contour(aes(z = log_sse), color = "#ffffff", alpha = 0.60,
                   bins = 13, linewidth = 0.3) +
      geom_point(data = data.frame(a = input$manual_a, b = input$manual_b),
                 aes(x = a, y = b), inherit.aes = FALSE,
                 shape = 21, size = 4.5,
                 fill = "#e77e23", color = "white", stroke = 1.2) +
      geom_point(data = data.frame(a = unname(ab[1]), b = unname(ab[2])),
                 aes(x = a, y = b), inherit.aes = FALSE,
                 shape = 8, size = 5, color = "#e52737") +
      scale_fill_gradient(low = "#edf6ff", high = "#163d6f") +
      labs(x = "Intercept (a)", y = "Slope (b)",
           title = "Orange: your line     Red star: OLS minimum",
           fill = "log(1+SSE)") +
      plot_theme
  })

  # MODULE 2 - high-dimensional correlated linear regression
  multi_data <- reactive({
    make_multi_data(input$multi_n, input$multi_rho, input$multi_noise,
                    input$multi_seed, input$multi_refresh)
  })

  multi_fits <- reactive({
    dd <- multi_data()
    xtr <- dd$x_train
    ytr <- dd$y_train
    # OLS on the same 12 predictors; no penalty.
    ols <- lm.fit(x = cbind(Intercept = 1, xtr), y = ytr)
    b_ols <- as.numeric(ols$coefficients)
    if (any(!is.finite(b_ols))) stop("OLS coefficients were not identified.")

    # Entire lambda paths are trained without consulting the held-out test set.
    lambda_grid <- exp(seq(log(50), log(0.001), length.out = 110))
    ridge <- glmnet::glmnet(xtr, ytr, family = "gaussian", alpha = 0,
                            lambda = lambda_grid, standardize = TRUE)
    lasso <- glmnet::glmnet(xtr, ytr, family = "gaussian", alpha = 1,
                            lambda = lambda_grid, standardize = TRUE)
    set.seed(as.integer(input$multi_seed + input$multi_refresh + 771L))
    fold_id <- sample(rep(seq_len(5), length.out = nrow(xtr)))
    cv_ridge <- glmnet::cv.glmnet(xtr, ytr, family = "gaussian", alpha = 0,
                                  lambda = lambda_grid, foldid = fold_id,
                                  type.measure = "mse", standardize = TRUE)
    cv_lasso <- glmnet::cv.glmnet(xtr, ytr, family = "gaussian", alpha = 1,
                                  lambda = lambda_grid, foldid = fold_id,
                                  type.measure = "mse", standardize = TRUE)
    list(ols = b_ols, ridge = ridge, lasso = lasso,
         cv_ridge = cv_ridge, cv_lasso = cv_lasso, data = dd)
  })

  multi_eval <- reactive({
    ff <- multi_fits()
    dd <- ff$data
    lam_r <- if (identical(input$lambda_mode, "cv")) ff$cv_ridge$lambda.min else 10^input$log_lambda
    lam_l <- if (identical(input$lambda_mode, "cv")) ff$cv_lasso$lambda.min else 10^input$log_lambda
    b_ridge <- as.numeric(coef(ff$ridge, s = lam_r))
    b_lasso <- as.numeric(coef(ff$lasso, s = lam_l))
    bb <- list(OLS = ff$ols, Ridge = b_ridge, LASSO = b_lasso)
    all_preds <- lapply(bb, function(b) {
      list(train = as.numeric(cbind(1, dd$x_train) %*% b),
           test = as.numeric(cbind(1, dd$x_test) %*% b))
    })
    perf <- do.call(rbind, lapply(names(bb), function(nm) {
      data.frame(Model = nm, `Training RMSE` = rmse(dd$y_train, all_preds[[nm]]$train),
                 `Test RMSE` = rmse(dd$y_test, all_preds[[nm]]$test),
                 `Nonzero slopes` = sum(abs(bb[[nm]][-1]) > 1e-5),
                 `Lambda` = if (nm == "OLS") 0 else if (nm == "Ridge") lam_r else lam_l,
                 check.names = FALSE)
    }))
    list(perf = perf, betas = bb, preds = all_preds,
         lambda_ridge = lam_r, lambda_lasso = lam_l)
  })

  output$multi_metrics <- renderTable({
    ev <- multi_eval()
    tab <- ev$perf
    tab$`Training RMSE` <- fmt(tab$`Training RMSE`)
    tab$`Test RMSE` <- fmt(tab$`Test RMSE`)
    tab$Lambda <- formatC(tab$Lambda, digits = 4, format = "fg", flag = "#")
    tab
  }, striped = TRUE, bordered = TRUE, spacing = "s", rownames = FALSE)

  output$multi_rmse <- renderPlot({
    perf <- multi_eval()$perf
    chart_data <- rbind(
      data.frame(Model = perf$Model, Error = perf$`Training RMSE`, Sample = "Training"),
      data.frame(Model = perf$Model, Error = perf$`Test RMSE`, Sample = "Held-out test")
    )
    ggplot(chart_data, aes(x = Model, y = Error, fill = Sample)) +
      geom_col(position = position_dodge(width = 0.72), width = 0.60) +
      scale_fill_manual(values = c("Training" = "#3c80b9", "Held-out test" = "#e69d3c")) +
      labs(title = "Lower test RMSE indicates better performance on this held-out sample",
           x = NULL, y = "RMSE") + plot_theme
  })

  output$multi_predictions <- renderPlot({
    dd <- multi_data()
    pp <- multi_eval()$preds
    pts <- do.call(rbind, lapply(names(pp), function(nm) {
      data.frame(Observed = dd$y_test, Predicted = pp[[nm]]$test, Model = nm)
    }))
    lim <- range(c(pts$Observed, pts$Predicted), finite = TRUE)
    ggplot(pts, aes(x = Observed, y = Predicted)) +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "#818d9b") +
      geom_point(alpha = 0.37, color = "#276da0", size = 1.0) +
      facet_wrap(~Model, ncol = 3) +
      coord_equal(xlim = lim, ylim = lim) +
      labs(x = "Observed test outcome", y = "Predicted test outcome") + plot_theme +
      theme(axis.text.x = element_text(size = 8), axis.text.y = element_text(size = 8))
  })

  output$multi_coefficients <- renderPlot({
    bb <- multi_eval()$betas
    dd <- multi_data()
    cf <- rbind(
      data.frame(Feature = paste0("X", seq_along(dd$truth_beta)),
                 Estimate = dd$truth_beta, Model = "True coefficient"),
      do.call(rbind, lapply(names(bb), function(nm) {
        data.frame(Feature = paste0("X", seq_along(dd$truth_beta)),
                   Estimate = bb[[nm]][-1], Model = nm)
      }))
    )
    cf$Feature <- factor(cf$Feature, levels = rev(paste0("X", seq_along(dd$truth_beta))))
    ggplot(cf, aes(x = Feature, y = Estimate, color = Model)) +
      geom_hline(yintercept = 0, color = "#9aa5b2") +
      geom_point(position = position_dodge(width = 0.65), size = 1.9) +
      coord_flip() +
      scale_color_manual(values = c("True coefficient" = "#20252b",
                                    "OLS" = "#a24fba", "Ridge" = "#2878b9",
                                    "LASSO" = "#dc8634")) +
      labs(x = "Predictors", y = "Estimated coefficients",
           title = "LASSO may set some coefficients exactly to zero") + plot_theme +
      theme(legend.text = element_text(size = 8))
  })

  output$multi_cv <- renderPlot({
    ff <- multi_fits()
    ee <- multi_eval()
    curve_data <- rbind(
      data.frame(lambda = ff$cv_ridge$lambda,
                 error = sqrt(ff$cv_ridge$cvm), Model = "Ridge"),
      data.frame(lambda = ff$cv_lasso$lambda,
                 error = sqrt(ff$cv_lasso$cvm), Model = "LASSO")
    )
    selected <- data.frame(
      loglam = log10(c(ee$lambda_ridge, ee$lambda_lasso)),
      Model = c("Ridge", "LASSO"),
      err = c(sqrt(ff$cv_ridge$cvm[which.min(abs(log(ff$cv_ridge$lambda) - log(ee$lambda_ridge)))]),
              sqrt(ff$cv_lasso$cvm[which.min(abs(log(ff$cv_lasso$lambda) - log(ee$lambda_lasso)))]))
    )
    ggplot(curve_data, aes(x = log10(lambda), y = error, color = Model)) +
      geom_line(linewidth = 1.0) +
      geom_point(data = selected, aes(x = loglam, y = err),
                 inherit.aes = FALSE, shape = 21, fill = "white", size = 3,
                 color = "#343434") +
      geom_vline(data = selected, aes(xintercept = loglam, color = Model),
                 linetype = "dotted", linewidth = 0.75) +
      scale_color_manual(values = c("Ridge" = "#2878b9", "LASSO" = "#dc8634")) +
      labs(x = "log10(lambda)", y = "5-fold CV RMSE (training folds)",
           title = "Dots indicate active lambda values") + plot_theme
  })

  # MODULE 3 - polynomial model flexibility and generalization
  poly_data <- reactive({
    make_poly_data(input$poly_n, input$poly_noise,
                   input$poly_seed, input$poly_refresh)
  })

  poly_results <- reactive({
    dat <- poly_data()
    models <- lapply(1:12, function(d) {
      model <- fit_orthogonal_poly(dat$train$x, dat$train$y, degree = d)
      c(train = rmse(dat$train$y, model$train_fitted),
        test = rmse(dat$test$y,
                    predict_orthogonal_poly(model, dat$test$x)))
    })
    er <- do.call(rbind, models)
    data.frame(Degree = 1:12, Training = er[, "train"], Test = er[, "test"])
  })

  output$poly_fit <- renderPlot({
    dat <- poly_data()
    model <- fit_orthogonal_poly(dat$train$x, dat$train$y, input$degree)
    xx <- seq(-2, 2, length.out = 300)
    curves <- rbind(
      data.frame(x = xx, y = true_nonlinear(xx), Model = "True relationship"),
      data.frame(x = xx, y = predict_orthogonal_poly(model, xx),
                 Model = paste("Degree", input$degree))
    )
    ggplot(dat$train, aes(x = x, y = y)) +
      geom_point(alpha = 0.55, color = "#484f58", size = 1.9) +
      geom_line(data = curves, aes(x = x, y = y, color = Model),
                linewidth = 1.15, inherit.aes = FALSE) +
      scale_color_manual(values = c("True relationship" = "#169a7a",
                                    setNames("#d97927", paste("Degree", input$degree)))) +
      labs(title = "Higher complexity can fit noise as well as signal",
           x = "Economic predictor X", y = "Outcome Y") + plot_theme
  })

  output$poly_errors <- renderPlot({
    results <- poly_results()
    chart_data <- rbind(
      data.frame(Degree = results$Degree, RMSE = results$Training, Sample = "Training"),
      data.frame(Degree = results$Degree, RMSE = results$Test, Sample = "Held-out test")
    )
    ggplot(chart_data, aes(x = Degree, y = RMSE, color = Sample)) +
      geom_line(linewidth = 1.1) + geom_point(size = 1.9) +
      geom_vline(xintercept = input$degree, linetype = "dotted", color = "#475569") +
      scale_color_manual(values = c("Training" = "#347eb5", "Held-out test" = "#d98135")) +
      scale_x_continuous(breaks = 1:12) +
      labs(x = "Polynomial degree", y = "RMSE", title = "Dotted line: selected complexity") +
      plot_theme
  })

  output$poly_metrics <- renderTable({
    dd <- poly_results()
    chosen <- dd[dd$Degree == input$degree, ]
    data.frame(Statistic = c("Selected polynomial degree", "Training RMSE",
                             "Held-out test RMSE", "Degree with lowest test RMSE (illustration only)"),
               Value = c(input$degree, fmt(chosen$Training), fmt(chosen$Test),
                         dd$Degree[which.min(dd$Test)]))
  }, striped = TRUE, rownames = FALSE)

  # MODULE 4 - pointwise bias / variance simulation
  bv_results <- reactive({
    n <- input$bv_n
    repetitions <- input$bv_rep
    x <- seq(-2, 2, length.out = n)
    x0 <- input$bv_x0
    sig <- input$bv_sigma
    set.seed(as.integer(input$bv_seed))
    noise_matrix <- matrix(rnorm(n * repetitions, sd = sig),
                           nrow = n, ncol = repetitions)
    yy <- matrix(true_nonlinear(x), nrow = n, ncol = repetitions) + noise_matrix
    degrees <- c(1L, 4L, 10L)
    actual <- true_nonlinear(x0)
    draws <- lapply(degrees, function(d) {
      p <- stats::poly(x, degree = d)
      X <- cbind(Intercept = 1, p)
      # A QR fit with multiple Y columns is efficient and replicable.
      coefficients <- base::qr.coef(qr(X), yy)
      evaluation_row <- matrix(c(1, as.numeric(predict(p, newdata = x0))), nrow = 1)
      as.numeric(evaluation_row %*% coefficients)
    })
    names(draws) <- paste0("Degree ", degrees)
    parts <- do.call(rbind, lapply(names(draws), function(nm) {
      pred <- draws[[nm]]
      bias_squared <- (mean(pred) - actual)^2
      variance <- var(pred)
      data.frame(Model = nm, `Squared bias` = bias_squared,
                 Variance = variance, `Irreducible noise` = sig^2,
                 `Expected squared error` = bias_squared + variance + sig^2,
                 check.names = FALSE)
    }))
    list(draws = draws, parts = parts, actual = actual)
  })

  output$bv_distribution <- renderPlot({
    rv <- bv_results()
    dd <- do.call(rbind, lapply(names(rv$draws), function(nm) {
      data.frame(Prediction = rv$draws[[nm]], Model = nm)
    }))
    ggplot(dd, aes(x = Prediction, fill = Model, color = Model)) +
      geom_density(alpha = 0.16, linewidth = 0.85) +
      geom_vline(xintercept = rv$actual, linetype = "dashed",
                 color = "#242424", linewidth = 1.0) +
      scale_color_manual(values = c("Degree 1" = "#d97927",
                                    "Degree 4" = "#277cb5", "Degree 10" = "#138566")) +
      scale_fill_manual(values = c("Degree 1" = "#d97927",
                                   "Degree 4" = "#277cb5", "Degree 10" = "#138566")) +
      labs(x = "Prediction at x0 across repeated training samples", y = "Density",
           title = "Dashed line: true expected outcome f(x0)") + plot_theme
  })

  output$bv_decomposition <- renderPlot({
    dd <- bv_results()$parts
    chart_data <- rbind(
      data.frame(Model = dd$Model, Value = dd$`Squared bias`, Component = "Squared bias"),
      data.frame(Model = dd$Model, Value = dd$Variance, Component = "Variance"),
      data.frame(Model = dd$Model, Value = dd$`Irreducible noise`, Component = "Irreducible noise")
    )
    ggplot(chart_data, aes(x = Model, y = Value, fill = Component)) +
      geom_col(width = 0.58) +
      scale_fill_manual(values = c("Squared bias" = "#e19a46",
                                   "Variance" = "#367db4",
                                   "Irreducible noise" = "#a6b5c9")) +
      labs(x = "Polynomial model", y = "Estimated error component",
           title = "Pointwise Monte Carlo error decomposition") + plot_theme
  })

  output$bv_metrics <- renderTable({
    dd <- bv_results()$parts
    num_cols <- setdiff(names(dd), "Model")
    dd[num_cols] <- lapply(dd[num_cols], fmt)
    dd
  }, striped = TRUE, rownames = FALSE)
}

shinyApp(ui = ui, server = server)
