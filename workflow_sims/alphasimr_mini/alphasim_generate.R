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
    segSites = 100,
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


founderGenomes = sim_WF_1kbt(100000,1e+05)


#############################################################
#############################################################
#set up first generation and phenotypes

#initialisize sim
SP = SimParam$new(founderGenomes)

#add 1 traits for each level of relative epistasis (Vaa/Va at 0.5 AF)
SP$addTraitAE(
  nQtlPerChr=100, # Carefully tuned with meanDD
  mean=0, # ~1981 value from Eras data
  var=3, # Chosen for appropriate rate of gain
  relAA=0, # relative Vaa in a diploid organism with 50/50 allele freq
  useVarA = F
)

SP$addTraitAE(
  nQtlPerChr=100, # Carefully tuned with meanDD
  mean=0, # ~1981 value from Eras data
  var=3, # Chosen for appropriate rate of gain
  relAA=3, # relative Vaa in a diploid organism with 50/50 allele freq
  useVarA = F
)

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
pheno = pheno(pop)
df <- as.data.frame(pullSegSiteHaplo(pop))

#############################################################
#############################################################

GP_p <- as.data.frame(pheno(pop))
GP_p$ind <- row.names(GP_p)
GP_p <- GP_p %>% select(ind, everything())

write.table(GP_p, "test_sim_WF_1kbt_100k_100sites_p.txt", quote = F, col.names = T, row.names = F)

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

write.table(loci_all, "test_sim_WF_1kbt_100k_100sites_eff.txt", quote = F, col.names = T, row.names = F)

#############################################################
#############################################################


GP_g <- as.data.frame(t(df))
GP_g$qtl <- row.names(GP_g)
GP_g <- GP_g %>% select(qtl, everything())


write.table(GP_g, "test_sim_WF_1kbt_100k_100sites_g.txt", quote = F, col.names = T, row.names = F)
