# install.packages(c("RLT", "MASS", "reshape2", "gridExtra")) # If not installed
library(RLT)
library(MASS)
library(reshape2)
library(gridExtra)
library(grid)

# ==============================================================================
# 1. DATA GENERATION FUNCTIONS (Exact Implementation of Section 4.2)
# ==============================================================================

generate_data <- function(scenario, n, p) {
  
  # Helper to create Covariance Matrices
  get_sigma <- function(p, rho, type = "AR1") {
    idx <- 1:p
    dist_mat <- abs(outer(idx, idx, "-"))
    
    if (type == "AR1") {
      # Scenario 3: 0.9^|i-j|
      return(rho^dist_mat)
    } else if (type == "Linear") {
      # Scenario 4: 0.5^|i-j| + 0.2 * I(i != j)
      Sigma <- rho^dist_mat
      Sigma[row(Sigma) != col(Sigma)] <- Sigma[row(Sigma) != col(Sigma)] + 0.2
      return(Sigma)
    }
  }
  
  X <- NULL
  y <- NULL
  
  epsilon <- rnorm(n)
  
  if (scenario == 1) {
    # --- Scenario 1: Classification, Independent ---
    X <- matrix(runif(n * p), nrow = n, ncol = p)
    term <- 10 * (X[, 1] - 1) + 20 * abs(X[, 2] - 0.5)
    mu <- pnorm(term)
    y <- as.factor(rbinom(n, 1, mu))
    
  } else if (scenario == 2) {
    # --- Scenario 2: Non-linear, Independent ---
    X <- matrix(runif(n * p), nrow = n, ncol = p)
    term2 <- pmax(0, X[, 2] - 0.25)
    y <- 100 * (X[, 1] - 0.5)^2 * term2 + epsilon
    
  } else if (scenario == 3) {
    # --- Scenario 3: Checkerboard, Strong Correlation ---
    Sigma <- get_sigma(p, 0.9, "AR1")
    X <- mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
    y <- 2 * X[, 50] * X[, 100] + 2 * X[, 150] * X[, 200] + epsilon
    
  } else if (scenario == 4) {
    # --- Scenario 4: Linear, Correlation ---
    Sigma <- get_sigma(p, 0.5, "Linear")
    X <- mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
    y <- 2 * X[, 50] + 2 * X[, 100] + 4 * X[, 150] + epsilon
  }
  
  return(list(X = X, y = y))
}

# ==============================================================================
# 2. EXPERIMENT RUNNER (Matching Paper's Parameters)
# ==============================================================================

run_rlt_experiment <- function(target_scenario, target_p, n_repeats = 200) {
  
  # Determine N from Section 4.2
  n_train <- switch(target_scenario, 100, 100, 300, 200)
  model_type <- if (target_scenario == 1) "classification" else "regression"
  
  # RLT Hyperparameters (Table 3 from paper)
  k_values <- c(1, 2, 5)
  muting_configs <- list(
    "None" = list(method = 0, percent = 0), # pa = 0
    "Moderate" = list(method = -1, percent = 0.50), # pa = 50%
    "Aggressive" = list(method = -1, percent = 0.80) # pa = 80%
  )
  
  results <- data.frame()
  
  cat(sprintf("\n=== Running Scenario %d | P = %d | N = %d ===\n", target_scenario, target_p, n_train))
  
  for (m_name in names(muting_configs)) {
    m_conf <- muting_configs[[m_name]]
    
    for (k in k_values) {
      errors <- numeric(n_repeats)
      
      for (rep in 1:n_repeats) {
        set.seed(target_scenario * 10000 + target_p * 100 + rep)
        
        train_data <- generate_data(target_scenario, n_train, target_p)
        test_data <- generate_data(target_scenario, 1000, target_p)
        
        fit <- RLT(
          x = train_data$X, 
          y = train_data$y,
          model = model_type,
          ntrees = 100,
          nmin = floor(n_train^(1/3)),
          reinforcement = TRUE,
          combsplit = k,
          muting = m_conf$method,
          muting.percent = m_conf$percent,
          protect = ceiling(log(target_p)),
          combsplit.th = 0.25,
          embed.ntrees = 100,
          use.cores = parallel::detectCores(),
          print.summary = 0
        )
        
        preds <- predict(fit, test_data$X)
        
        errors[rep] <- if (model_type == "classification") {
          1 - mean(preds$Prediction == test_data$y)
        } else {
          mean((preds$Prediction - test_data$y)^2)
        }
      }
      
      mean_err <- mean(errors)
      sd_err <- sd(errors)
      
      cat(sprintf("%-12s | K=%d | Error: %.4f (%.4f)\n", m_name, k, mean_err, sd_err))
      
      results <- rbind(results, data.frame(
        Scenario = target_scenario,
        P = target_p,
        Muting = m_name,
        K = k,
        Mean_Error = mean_err,
        SD_Error = sd_err
      ))
    }
  }
  return(results)
}

# ==============================================================================
# 3. FORMATTING & SAVING FUNCTIONS
# ==============================================================================

format_dso1_table <- function(df_results, target_p) {
  df_p <- df_results[df_results$P == target_p, ]
  
  df_p$Formatted_Result <- apply(df_p, 1, function(row) {
    mean_val <- as.numeric(row["Mean_Error"])
    sd_val <- as.numeric(row["SD_Error"])
    if (as.numeric(row["Scenario"]) == 1) {
      sprintf("%.1f%% (%.1f%%)", mean_val * 100, sd_val * 100)
    } else {
      sprintf("%.2f (%.2f)", mean_val, sd_val)
    }
  })
  
  table_wide <- dcast(df_p, Muting + K ~ Scenario, value.var = "Formatted_Result")
  
  # Reorder Muting factor levels
  table_wide$Muting <- factor(table_wide$Muting, levels = c("None", "Moderate", "Aggressive"))
  table_wide <- table_wide[order(table_wide$Muting), ]
  
  # Set index names for display
  rownames(table_wide) <- NULL
  colnames(table_wide)[2] <- "Linear Comb."
  colnames(table_wide)[3:6] <- paste("Scenario", 1:4)
  
  return(table_wide)
}

save_table_as_image <- function(df_table, filename, title) {
  png(filename, height = 400, width = 800, res = 100)
  
  # Use grid.table to draw the data frame
  tbl_grob <- tableGrob(
    df_table, 
    rows = NULL, 
    theme = ttheme_default(
      core = list(bg_params = list(fill = c("white"), col = "grey50")),
      colhead = list(bg_params = list(fill = "lightgrey", col = "black"))
    )
  )
  
  # Add a title
  title_grob <- textGrob(title, gp = gpar(fontsize = 14, fontface = "bold"))
  
  # Arrange the title and table
  g <- grid.arrange(title_grob, tbl_grob, nrow = 2, heights = unit(c(0.2, 0.8), "npc"))
  
  dev.off()
  cat(sprintf("Saved table image to '%s'\n", filename))
}


# ==============================================================================
# 4. EXECUTION
# ==============================================================================

# WARNING: This is computationally intensive.
N_REPEATS <- 5 # Set to 200 for full replication

# Create an output directory
output_dir <- "dso1_results"
dir.create(output_dir, showWarnings = FALSE)

all_results_dso1 <- data.frame()

for (s in 1:4) {
  for (p in c(200, 500, 1000)) {
    res <- run_rlt_experiment(s, p, n_repeats = N_REPEATS)
    all_results_dso1 <- rbind(all_results_dso1, res)
  }
}

# Save results to CSV for record-keeping
write.csv(all_results_dso1, file.path(output_dir, "dso1_raw_results.csv"), row.names = FALSE)

# Generate and save each table as an image
table4 <- format_dso1_table(all_results_dso1, 200)
save_table_as_image(
  table4,
  file.path(output_dir, "table4_p200.png"),
  "Table 4: Classification/prediction error (SD), p = 200"
)

table5 <- format_dso1_table(all_results_dso1, 500)
save_table_as_image(
  table5,
  file.path(output_dir, "table5_p500.png"),
  "Table 5: Classification/prediction error (SD), p = 500"
)

table6 <- format_dso1_table(all_results_dso1, 1000)
save_table_as_image(
  table6,
  file.path(output_dir, "table6_p1000.png"),
  "Table 6: Classification/prediction error (SD), p = 1000"
)

cat("\nAll experiments complete and tables saved to the 'dso1_results' folder.\n")

