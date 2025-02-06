library(dplyr)

files = c('test_sim_WF_1kbt_10000n_5000000bp_trait_architecture',
            'test_sim_WF_10kbt_10000n_5000000bp_trait_architecture',
            'test_sim_WF_null_10000n_5000000bp_trait_architecture')

for(sim in c(files)){

data <- readRDS(paste0('alphasimr_output/',sim, '.RDS'))
print(head(data))




df <- data.frame(row_id = seq_len(nrow(data[[1]])))

for(name in names(data)){
df[[name]] <- as.vector(diag(data[[name]]))
write.table(df,paste0('alphasimr_output/',sim,'.txt'), row.names =F, quote = F )


}
}