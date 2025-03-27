library(AlphaSimR)
library(dplyr)
library(ggplot2)
library(reshape2)

set.seed(1)
#founders coalescent sim
sim_WF_1kbt <- function(n_ind, bp_len) {
  runMacs2(
    n_ind,
    nChr = 1,
    segSites = 100000,
    Ne = 100000,
    bp = bp_len,
    genLen = 10,
    mutRate = 2.5e-07,
    histNe = c(1000, 100000, 100000, 100000),
    histGen = c(1, 1000, 300000, 400000),
    inbred = FALSE,
    #split = 0,
    ploidy = 1,
    returnCommand = FALSE,
    nThreads = NULL
  )
}


founderGenomes = sim_WF_1kbt(10000,5e+06)


#############################################################
#############################################################
#set up first generation and phenotypes

#initialisize sim
SP = SimParam$new(founderGenomes)

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
pop = newPop(founderGenomes)
pop = setPheno(pop,H2 = c(0.5))

varA(pop)
varAA(pop)
varG(pop)/varP(pop)
varP(pop)
varAA(pop)/varA(pop)
varA(pop)/varP(pop)

#extract phenotypes
#pheno_f1 = pheno(pop)


df_f1 <- as.data.frame(pullSegSiteHaplo(pop))
#############################################################
#############################################################
#upsample through random crossing

#double the ploidy of parents to allow for crosses
popd = doubleGenome(pop, keepParents = F)
#calculate starting mean of parental gen
genMean = meanG(popd)

#select individuals based on stabilising selection, for 20 generations
#select 1000 individuals closest to mean and ranodmly cross to make 3 offspring
for(generation in 1:1){
  popd = randCross(popd, 100000, simParam=SP)
}
#reduce back down to haploid
poprd <- makeDH(popd, nDH = 1, keepParents = F, simParam = NULL)
poprd <- reduceGenome(popd, nProgeny = 1, keepParents = F)

poprd = setPheno(poprd,H2 = c(0.5))

pheno = pheno(poprd)

#############################################################
#############################################################
#extract SNPs
#writeRecords(poprd, useQtl=T, dir="sandler_gpatlas_data/workflow_sims/alphasimr_upsample")

#writeRecords(pop, useQtl=T, dir="sandler_gpatlas_data/workflow_sims/alphasimr_upsample")


#############################################################
#############################################################
GP_p <- as.data.frame(pheno(poprd))
GP_p$ind <- row.names(GP_p)
GP_p <- GP_p %>% select(ind, everything())

write.table(GP_p, "sandler_gpatlas_data/workflow_sims/alphasimr_upsample/test_sim_WF_1kbt_100kups_5mb_p.txt", quote = F, col.names = T, row.names = F)



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
loci$sfs = colSums(df_f1)/nrow(df_f1)
loci_all = rbind(loci_all, loci)

}

write.table(loci_all, 'sandler_gpatlas_data/workflow_sims/alphasimr_upsample/test_sim_WF_1kbt_100kups_5mb_eff.txt', quote = F, col.names = T, row.names = F)

#############################################################
#############################################################










# First create the output file with headers (if needed)
# Adjust column names as needed for your specific data
file_path <- "test_sim_WF_1kbt_100kups_5mb_g.txt"
write.table(data.frame(), file = file_path, col.names = TRUE, row.names = FALSE)

# Process in batches and append to file
batch_size <- 5000
total_inds <- length(poprd)
num_batches <- ceiling(total_inds / batch_size)

for(i in 0:(num_batches-1)) {
  # Get proper indices for this batch
  start_idx <- i * batch_size + 1
  end_idx <- min((i + 1) * batch_size, total_inds)

  # Extract subset of population
  subset_pop <- poprd[start_idx:end_idx]

  # Get segregating sites
  batch <- pullSegSiteHaplo(subset_pop)

  # Append to file (append=TRUE after first batch)
  write.table(batch, file = file_path,
              append = (i > 0),
              col.names = (i == 0),
              row.names = T,
              sep = ",",
              quote = FALSE)

  # Clear memory
  rm(batch, subset_pop)
  gc() # Force garbage collection

  # Print progress
  cat("Completed batch", i+1, "of", num_batches, "\n")
}
