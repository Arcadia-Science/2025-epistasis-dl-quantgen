library(rrBLUP)

set.seed(42)

#######################
df <- read.table('BYxRM_GenoData.txt', header = T)
df <- df[,-1] #remove row (QTL) names

pheno <- read.table('BYxRM_PhenoData.txt', header = T)
pheno<- as.data.frame(scale(pheno))

#full dataset
markers = as.matrix(t(df))
markers[markers == 'R'] <- -1 #replace to inbred alt value for rrBLUP
markers[markers == 'B'] <- 1
mode(markers) <- "numeric"
cat("Marker matrix dimensions:", dim(markers), "\n")

A = A.mat(markers, shrink=FALSE)/2

#test train split
# Calculate number of samples for training (85%)
n_total = nrow(markers)
n_train = round(0.85 * n_total)
n_test = n_total - n_train

# Create random indices for training
train_indices = sample(1:n_total, n_train, replace = FALSE)

# Test indices are the remaining ones
test_indices = setdiff(1:n_total, train_indices)

#######################
#dataframe for output
results_df <- data.frame()
results_df_breeding_vals <- data.frame()
# Inside your for loop
for(trait_index in 1:ncol(pheno)) {
print("Model fits phenotype:")
print(trait_index)
#######################

#full data rrBLUP model
mod_rrBLUP <- mixed.solve(pheno[,trait_index], Z = markers, K=NULL)

#full data GBLUP
mod_GBLUP = mixed.solve(pheno[,trait_index], K=A, method='REML')

# heritabilities
h2_rrBLUP_v <- mod_rrBLUP$Vu / (mod_rrBLUP$Vu + mod_rrBLUP$Ve)  # Narrow-sense heritability

h2_GBLUP_v <- mod_GBLUP$Vu / (mod_GBLUP$Vu + mod_GBLUP$Ve)

#extract breeding values from full data GBLUP model
#breeding_values = mod_GBLUP$u
#beta_value = as.numeric(mod_GBLUP$beta)
#fitted_values = breeding_values + beta_value
#cor(fitted_values, pheno[,trait_index],use = 'complete.obs')^2

#rrBLUP breeding values
marker_effects <- mod_rrBLUP$u
beta_value <- as.numeric(mod_rrBLUP$beta)
fitted_values <- beta_value + markers %*% marker_effects
fitted_values <- data.frame(fitted_values)
#cor(fitted_values, pheno[,trait_index],use = 'complete.obs')^2

#names(fitted_values) <- names(pheno[trait_index])

#output breeding values
if (!exists("results_df_breeding_vals") || ncol(results_df_breeding_vals) == 0) {
  # First iteration - create the dataframe
  results_df_breeding_vals <- data.frame(fitted_values)
  colnames(results_df_breeding_vals)[1] <- colnames(pheno)[trait_index]
} else {
  # Subsequent iterations - add column to existing dataframe
  results_df_breeding_vals <- cbind(results_df_breeding_vals, fitted_values)
  colnames(results_df_breeding_vals)[ncol(results_df_breeding_vals)] <- colnames(pheno)[trait_index]
}
############################################################
############################################################
# Split the data
markers_train = markers[train_indices, ]
markers_test = markers[test_indices, ]

pheno_train = pheno[train_indices, ]
pheno_test = pheno[test_indices, ]

#######
G_train <- A[train_indices, train_indices]
G_test_train <- A[test_indices, train_indices]

mod_rrBLUP_train <- mixed.solve(pheno_train[,trait_index], Z = markers_train, K=NULL)

mod_GBLUP_train = mixed.solve(pheno_train[,trait_index], K=G_train, method='REML')

rrBLUP_pred <- markers_test %*% mod_rrBLUP_train$u

gblup_pred <- G_test_train %*% solve(G_train) %*% mod_GBLUP_train$u

h2_rrBLUP_pc <- cor(rrBLUP_pred, pheno_test[,trait_index], use = 'complete.obs')^2

h2_GBLUP_pc <- cor(gblup_pred, pheno_test[,trait_index], use = 'complete.obs')^2


# Create row of results
h2_estimates <- data.frame(trait = colnames(pheno)[trait_index],
                            h2_rrBLUP_v = h2_rrBLUP_v,
                            h2_GBLUP_v = h2_GBLUP_v,
                            h2_rrBLUP_pc = h2_rrBLUP_pc,
                            h2_GBLUP_pc = h2_GBLUP_pc)

# Append to results dataframe
results_df <- rbind(results_df, h2_estimates)
}


head(results_df)
head(results_df_breeding_vals)

write.table(results_df, 'BYxRM_herits_2scaled.csv', row.names = F, col.names = T, quote = F, sep=',')
write.table(results_df_breeding_vals, 'BYxRM_breeding_vals_scaled.csv', row.names = F, col.names = T, quote = F, sep=',')