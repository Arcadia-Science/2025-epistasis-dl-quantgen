library(AlphaSimR)
library(dplyr)
library(ggplot2)
library(reshape2)


n_qtl <- as.numeric(snakemake@params[['qtl_n']])
n_ind <- as.numeric(snakemake@params[['sample_size']])
rep <- as.numeric(snakemake@params[['rep']])
pleio_strength <- as.numeric(snakemake@params[['pleio_strength']])
trait_n <- as.numeric(snakemake@params[['trait_n']])

output_file_pheno <- snakemake@output[["output_pheno"]]
output_file_geno <- snakemake@output[["output_geno"]]
output_file_eff <- snakemake@output[["loci_effects"]]

set.seed(rep)

founderGenomes = quickHaplo(nInd=n_ind, nChr=1, segSites=n_qtl, ploidy=1)



#############################################################
#############################################################
#set up first generation and phenotypes

#initialisize sim
SP = SimParam$new(founderGenomes)

# Generate experimental traits based on pleio_strength
if (pleio_strength == 0) {
  # For pleio_strength=0, generate each trait independently through a loop
  for (i in 1:trait_n) {
    SP$addTraitAE(
      nQtlPerChr=n_qtl,
      mean=0,
      var=1,
      relAA=1, # epistatic effects
      useVarA = F
    )
  }
} else {
  # For pleio_strength>0, use the correlation matrix method
  G = (1-pleio_strength)*diag(trait_n)+pleio_strength
  SP$addTraitAE(nQtlPerChr = n_qtl,
               mean = rep(0, trait_n),
               var = rep(1, trait_n),
               relAA = rep(1, trait_n),
               useVarA = F,
               corA=G,
               corAA = G)
}


#initialize pop to be
pop = newPop(founderGenomes)
pop = setPheno(pop,H2 = c(0.999))

#varA(pop)
#varAA(pop)
#varG(pop)/varP(pop)
#varP(pop)
#varAA(pop)/varA(pop)
#varA(pop)/varP(pop)

#extract phenotypes
pheno = pheno(pop)
df <- as.data.frame(pullSegSiteHaplo(pop))

print('Done generating sims, params:')
print(pleio_strength)
print(n_qtl)
print(n_ind)
print(trait_n)
#############################################################
#############################################################

GP_p <- as.data.frame(pheno(pop))
GP_p$ind <- row.names(GP_p)
GP_p <- GP_p %>% select(ind, everything())

write.table(GP_p, output_file_pheno, quote = F, col.names = T, row.names = F)

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

write.table(loci_all, output_file_eff, quote = F, col.names = T, row.names = F)

#############################################################
#############################################################


GP_g <- as.data.frame(t(df))
GP_g$qtl <- row.names(GP_g)
GP_g <- GP_g %>% select(qtl, everything())


write.table(GP_g, output_file_geno, quote = F, col.names = T, row.names = F)
print('Done writing files')
