library(rrBLUP)

#######################
df <- read.table('alphasimr/WF_Ne100k_samples_10k_H2_05_qtl_10k_corA_02_25traits_g.txt', header = T)
df <- df[,-1] #remove row (QTL) names

pheno <- read.table('alphasimr/WF_Ne100k_samples_10k_H2_05_qtl_10k_corA_02_25traits_p.txt', header = T)

#######################
pheno = pheno[,-1]

markers = as.matrix(t(df))
markers[markers == 0] <- -1 #replace to inbred alt value for rrBLUP

#pheno = pheno(pop)



mod_trait <- mixed.solve(pheno[,1], Z = markers, K=NULL)



#plot predicted vs actual qtl effects
plot(mod_trait$u, loci$add_eff)
cor(mod_trait$u, loci$add_eff, use = 'complete.obs')


#compare allele freq of predicted/actual QTL effects
par(mfrow = c(1, 2))
plot( mod_trait$u, sfs$`colSums(df)`)
plot(loci$add_eff,sfs$`colSums(df)`)
par(mfrow = c(1, 1))

############################################################
############################################################
#test train split

n_train = round(0.9*nrow(markers))
n_test = nrow(markers) - n_train

markers_train = markers[1:n_train,]
markers_test = markers[1:n_test,]


pheno_train = pheno[1:n_train,]
pheno_test = pheno[1:n_test,]


mod_trait_train <- mixed.solve(pheno_train[,2], Z = markers_train, K=NULL)


#predict trait from test set
pred_trait <- markers_test %*% mod_trait_train$u

#plot test Y vs Ypred
plot(pred_trait, pheno_test[,2])
cor(pred_trait, pheno_test[,2])