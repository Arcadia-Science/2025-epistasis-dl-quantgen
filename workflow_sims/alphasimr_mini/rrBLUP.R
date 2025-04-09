library(rrBLUP)

# Set seed for reproducibility
set.seed(42)

# Number of train/test split iterations
n_iterations <- 10  # You can adjust this number based on your needs

# Base file name - adjust if needed
base_file <- 'test_sim_qhaplo_10k_100sites_Ve0'

#######################
# Read data
cat("Processing input files\n")
df <- read.table(paste0(base_file, '_g.txt'), header = TRUE)
df <- df[,-1] # Remove row (QTL) names

pheno <- read.table(paste0(base_file, '_p.txt'), header = TRUE)
pheno <- pheno[,-1]

# Full dataset preparation
markers <- as.matrix(t(df))
markers[markers == 0] <- -1
cat("Marker matrix dimensions:", dim(markers), "\n")

# Initialize dataframe to store results
results <- data.frame(
  trait = integer(),
  iteration = integer(),
  pearson_r = numeric()
)

cat("Running rrBLUP with", n_iterations, "random train/test splits\n")

# Loop through iterations
for(iter in 1:n_iterations) {
  cat(paste0("Iteration ", iter, "/", n_iterations, "\n"))

  # Calculate train/test split
  n_total <- nrow(markers)
  n_train <- round(0.85 * n_total)

  # Create random indices for training
  train_indices <- sample(1:n_total, n_train, replace = FALSE)

  # Test indices are the remaining ones
  test_indices <- setdiff(1:n_total, train_indices)

  # Loop through each trait
  for(trait_index in 1:ncol(pheno)) {
    cat(paste0("  Trait ", trait_index, "\n"))

    # Split the data
    markers_train <- markers[train_indices, ]
    markers_test <- markers[test_indices, ]

    y_train <- pheno[train_indices, trait_index]
    y_test <- pheno[test_indices, trait_index]

    # Fit rrBLUP model
    mod_rrBLUP_train <- mixed.solve(y_train, Z = markers_train, K = NULL)

    # Predict on test set
    pred_values <- markers_test %*% mod_rrBLUP_train$u

    # Calculate Pearson's r
    pearson_r <- cor(pred_values, y_test, use = 'complete.obs')

    # Store results
    results <- rbind(results, data.frame(
      trait = trait_index,
      iteration = iter,
      pearson_r = pearson_r
    ))
  }
}

# Calculate summary statistics for each trait
summary_stats <- aggregate(
  pearson_r ~ trait,
  data = results,
  FUN = function(x) c(
    mean = mean(x),
    sd = sd(x),
    min = min(x),
    max = max(x),
    n = length(x)
  )
)

# Print results
cat("\n=== SUMMARY STATISTICS ===\n")
for(i in 1:nrow(summary_stats)) {
  trait <- summary_stats$trait[i]
  cat(paste0("\nTrait ", trait, ":\n"))
  cat(paste0("  Mean Pearson r: ", format(summary_stats$pearson_r[i, "mean"], digits=4), "\n"))
  cat(paste0("  Std Dev: ", format(summary_stats$pearson_r[i, "sd"], digits=4), "\n"))
  cat(paste0("  Range: [", format(summary_stats$pearson_r[i, "min"], digits=4),
             " to ", format(summary_stats$pearson_r[i, "max"], digits=4), "]\n"))
}

# Save results to CSV (uncomment if needed)
# write.csv(results, paste0(base_file, "_all_iterations.csv"), row.names = FALSE)
