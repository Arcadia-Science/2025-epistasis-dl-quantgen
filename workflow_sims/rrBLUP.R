library(rrBLUP)
library(ggplot2)

#######################
input_pheno <- snakemake@input[["input_pheno"]]
input_geno <- snakemake@input[["input_geno"]]
loci_effects <- snakemake@input[["loci_effects"]]

outpoutput_rrblup_corr <- snakemake@output[["correlation_summary"]]
output_pheno_pred_plot <- snakemake@output[["rrBLUP_pheno_pred_plot"]]

#######################
df <- read.table('../alphasimr/WF_Ne100k_samples_10k_H2_05_qtl_10k_corA_02_25traits_g.txt', header = T)
df <- df[,-1] #remove row (QTL) names

pheno <- read.table('../alphasimr/WF_Ne100k_samples_10k_H2_05_qtl_10k_corA_02_25traits_p.txt', header = T)
pheno = pheno[,-1] #remove ind identifiers

loci_eff <- read.table('../alphasimr/WF_Ne100k_samples_10k_H2_05_qtl_10k_corA_02_25traits_eff.txt', header = T)

#######################

#df2 <- df[,1:50]
#pheno2 <- pheno[1:50,]
#loci_eff2 <- loci_eff[1:1000,]

markers = as.matrix(t(df))
markers[markers == 0] <- -1 #replace to inbred alt value for rrBLUP

#######################

n_train = round(0.9*nrow(markers))
n_test = nrow(markers) - n_train

markers_train = markers[1:n_train,]
markers_test = markers[1:n_test,]

pheno_train = pheno[1:n_train,]
pheno_test = pheno[1:n_test,]

#######################

pheno_output <- data.frame(trait = numeric(0),
                            true_pheno = numeric(0),
                            pred_pheno = numeric(0))


mod_cors <- data.frame(trait = numeric(0),
                       pred_accuracy = numeric(0),
                       add_eff_cor = numeric(0))

for(i in c(1:ncol(pheno))){

#fit rrBLUP model on train set
mod_trait_train <- mixed.solve(pheno_train[,i], Z = markers_train, K=NULL)

#predict trait from test set
pred_trait <- markers_test %*% mod_trait_train$u

#data for plotting predicted vs actual pheno
pheno_output <- rbind(pheno_output, data.frame(trait = i,
                                               true_pheno = c(pheno_test[,i]),
                                               pred_pheno = c(pred_trait)))

#calculate correlations of phenotypes, and true additive marker effects in test set
cor_pheno <- cor(pred_trait, pheno_test[,i])
cor_add_eff <- cor(mod_trait_train$u,  loci_eff[loci_eff$trait == i,]$add_eff)

mod_cors <- rbind(mod_cors, data.frame(trait = i,
                                        pred_accuracy = cor_pheno,
                                        add_eff_cor = cor_add_eff))
}

pheno_output$epistasis_bin <- cut(pheno_output$trait,
                        breaks = seq(0, 25, by = 5),  # Breaks: 0-5, 5-10, 10-15, etc.
                        labels = c("relAA_0", "relAA_0.25", "relAA_0.5", "relAA_1", "relAA_3"),
                        right = TRUE,  # Ensures right endpoint is included
                        include.lowest = TRUE)  # Ensures the first bin includes "1"

write.table(pheno_output, outpoutput_rrblup_corr, quote = F, col.names = T, row.names = F)


pheno_output$rep <- ((pheno_output$trait - 1) %% 5) + 1

pl_pheno_cor <- ggplot(pheno_output, aes(x=true_pheno, y=pred_pheno, colour = as.factor(rep))) + geom_point() +
                facet_wrap(~epistasis_bin, scales = 'free') + theme_bw() + theme(legend.position="none")

ggsave(output_pheno_pred_plot, pl_pheno_cor)
