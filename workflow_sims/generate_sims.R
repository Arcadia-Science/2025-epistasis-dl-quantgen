library(AlphaSimR)
library(dplyr)
library(ggplot2)
library(reshape2)
library(gridExtra)

set.seed(1)

#########################################

pop_sim_scenario <- snakemake@params[['sim_scenario']]
bp_len <- as.numeric(snakemake@params[['bp_len']])
sample_size <- as.numeric(snakemake@params[['sample_size']])

output_file_pheno <- snakemake@output[["output_pheno"]]
output_file_geno <- snakemake@output[["output_geno"]]
output_file_eff <- snakemake@output[["loci_effects"]]
output_file_summary <- snakemake@output[["sim_summary"]]
output_file_TA<- snakemake@output[["trait_architecture"]]
output_file_ld_plot<- snakemake@output[["sim_LD_plot"]]

#accessory_output_prefix = paste0('alphasimr_output/',pop_sim_scenario, "_",  )

print(pop_sim_scenario)
print(bp_len)
print(sample_size)
#########################################
#coalescent simulation setup here
source('population_setup.R')

#initialisize sim from script storing various coalescent scenarios
if (exists(pop_sim_scenario)) {

  run_founder_sims <- get(pop_sim_scenario)
  founders <- run_founder_sims(sample_size, bp_len) #n_ind, n_bp

} else {
  stop(paste("Function", pop_sim_scenario, "does not exist in population setup script."))
}

SP = SimParam$new(founders)


#########################################
#add Additive*Epistatic traits
#correlation matrix for trait effects (0.2 correlation for Va effects)
G = 0.8*diag(5)+0.2

#add 5 traits for each level of relative epistasis (Vaa/Va at 0.5 AF)
for(relAA in c(0, 0.25, 0.5, 1, 3)){
SP$addTraitAE(nQtlPerChr = 10000,
             mean = c(0, 0, 0, 0, 0),
             var = c(3, 3, 3, 3, 3),
             relAA = c(relAA, relAA, relAA, relAA, relAA),
             useVarA = F,
             corA = G)

}

#initialize pop to be
pop = newPop(founders)

#set one broad sense heritability for all traits
#TODO add snakemake param to tune H2
pop = setPheno(pop,H2 = c(0.5))

#save variance components
trait_architecture <- list("varA" = varA(pop), "varAA" = varAA(pop), "varG" = varG(pop), "varP" = varP(pop),
                          "relAA" = varAA(pop)/varA(pop), "h2" = varA(pop)/varP(pop), "H2" = varG(pop)/varP(pop))

saveRDS(trait_architecture, file = output_file_TA)

#extract phenotypes
pheno = pheno(pop)

#extract all segregating sites
df <- as.data.frame(pullSegSiteHaplo(pop))
print(paste("Number segsites", ncol(df)))
#############################################################
#############################################################
#extract locus effects for each trait
loci_all = NULL
for(i in c(1:SP$nTraits)) {


#extract trait effects from sim
eff_epi <- as.data.frame(SP$traits[[i]]@epiEff) %>%
  rename(epi_locus_1 = V1,
         epi_locus_2 = V2,
         epi_eff = V3)
eff_add <- data.frame('add_eff' = SP$traits[[i]]@addEff)
eff_loci <- data.frame('locus' = SP$traits[[i]]@lociLoc)

#bind additive effects and locus position
eff <- cbind(eff_loci, eff_add)


#generate full list of seg sites in pop, and merge all qtl effects into it
#simplify epistatic effects to show the interactor locus and epi effect at each qtl
loci <- data.frame('trait' = i, 'locus' = c(1:ncol(pullSegSiteHaplo(pop)))) %>%
  left_join(.,eff, by = 'locus') %>%
  left_join(.,eff_epi, by = c ('locus' = 'epi_locus_1')) %>%
  left_join(.,eff_epi, by = c ('locus' = 'epi_locus_2')) %>%
  mutate(epi_loc = ifelse(is.na(epi_locus_1), epi_locus_2, epi_locus_1 )) %>%
  mutate(epi_eff = ifelse(is.na(epi_eff.x), epi_eff.y, epi_eff.x )) %>%
  select(trait, locus, add_eff, epi_loc, epi_eff)

loci$add_eff[is.na(loci$add_eff)] <- 0 #add missing QTL additive effect of 0
loci$sfs = colSums(df)/nrow(df)
loci_all = rbind(loci_all, loci)

}

#############################################################
#############################################################
#summary stats
segsites = ncol(df)

sfs <- as.data.frame(colSums(df))
median_AF = median(sfs$`colSums(df)`)/nrow(df)

#LD summary
compute_ld_matrix <- function(genotype_matrix) {
  ld_matrix <- cor(genotype_matrix)
  return(ld_matrix)
}

#subsample first 100 sites and calculate LD
subsample_df <- as.matrix(df[,1:100])
ld_matrix_subsample<- compute_ld_matrix(subsample_df)
melted_cormat <- melt(ld_matrix_subsample)

mean_ld <- mean(abs(melted_cormat$value))
median_ld <- median(abs(melted_cormat$value))


sumstats <- data.frame(segsites, median_AF,  mean_ld, median_ld)

#write effect sizes
write.table(sumstats, output_file_summary, quote = F, col.names = T, row.names = F)

#########
#sfs plot

sfs$af = sfs$`colSums(df)`/nrow(df)

pl_sfs <- ggplot(sfs, aes(x=af)) + geom_histogram(color="black", fill="grey") + theme_bw()


#########
#LD plots
pl_ld_heatmap <- ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + geom_tile() +
        scale_fill_gradientn(colors = hcl.colors(20, "RdBu"), limits=c(-1, 1))


melted_cormat$Var1_clean <- as.numeric(sub("1_", "", melted_cormat$Var1))
melted_cormat$Var2_clean <- as.numeric(sub("1_", "", melted_cormat$Var2))
melted_cormat$dist <- abs(melted_cormat$Var1_clean  - melted_cormat$Var2_clean)


pl_ld_decay <- ggplot(melted_cormat, aes(dist, abs(value))) + geom_point(alpha = 0.5) + geom_smooth(method = "loess", se=F)


#########
#grm

compute_grm <- function(genotype_matrix) {
  # Validate input
  if (!is.matrix(genotype_matrix) || !all(genotype_matrix %in% c(0, 1))) {
    stop("genotype_matrix must be a numeric matrix with values 0 or 1.")
  }

  # Dimensions of the genotype matrix
  n <- nrow(genotype_matrix)  # Number of individuals
  L <- ncol(genotype_matrix)  # Number of markers

  # Step 1: Compute allele frequencies (p) for each marker
  p <- colMeans(genotype_matrix)  # Allele frequency

  # Step 2: Center the genotype matrix by subtracting p for each marker
  Z <- sweep(genotype_matrix, 2, p)  # Centered genotype matrix

  # Step 3: Compute the genomic relationship matrix (GRM)
  GRM <- Z %*% t(Z) / L  # Normalize by the number of markers

  # Step 4: Standardize the GRM by dividing by the mean of the diagonal elements
  mean_diag <- mean(diag(GRM))
  GRM <- GRM / mean_diag

  # Ensure symmetry
  if (!all.equal(GRM, t(GRM))) {
    stop("GRM is not symmetric. Check the genotype matrix for errors.")
  }

  return(GRM)
}


subsample_df <- as.matrix(df[1:100,])
grm <- compute_grm(as.matrix(subsample_df))

#perform hierarchical clustering to order samples in plot
row_dend <- hclust(dist(grm))  # Cluster rows
col_dend <- hclust(dist(t(grm)))  # Cluster columns

melted_cormat_grm <- melt(grm)
melted_cormat_grm$Var1 <- factor(melted_cormat_grm$Var1, levels = rownames(grm)[row_dend$order])
melted_cormat_grm$Var2 <- factor(melted_cormat_grm$Var2, levels = colnames(grm)[col_dend$order])

pl_grm <- ggplot(data = melted_cormat_grm, aes(x=Var1, y=Var2, fill=value)) + geom_tile() +
          scale_fill_gradientn(colors = hcl.colors(10, "OrRd"), trans = 'reverse')


#########

summary_plot  <- grid.arrange(pl_grm, pl_sfs, pl_ld_heatmap, pl_ld_decay,  ncol = 4, nrow = 1)

#output figure
ggsave(output_file_ld_plot, summary_plot,  width = 15, height = 5)
#############################################################
#############################################################
#output files for GPatlas

GP_g <- as.data.frame(t(df))
GP_g$qtl <- row.names(GP_g)
GP_g <- GP_g %>% select(qtl, everything())

#write genetic info
write.table(GP_g, output_file_geno, quote = F, col.names = T, row.names = F)


GP_p <- as.data.frame(pheno(pop))
GP_p$ind <- row.names(GP_p)
GP_p <- GP_p %>% select(ind, everything())

#write phenotypes
write.table(GP_p, output_file_pheno, quote = F, col.names = T, row.names = F)

#write effect sizes
write.table(loci_all, output_file_eff, quote = F, col.names = T, row.names = F)
