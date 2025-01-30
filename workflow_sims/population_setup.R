library(AlphaSimR)
library(dplyr)
library(ggplot2)



set.seed(1)

#################################
#################################
#coalescent simulations for WF founders

#null WF pop
sim_WF_null <- function(n_ind, bp_len) {
    runMacs2(
    n_ind,
    nChr = 1,
    segSites = 100000,
    Ne = 100000,
    bp = bp_len,
    genLen = 5,
    mutRate = 2.5e-08,
    histNe = c(100000, 100000, 100000, 100000),
    histGen = c(1, 1000, 300000, 400000),
    inbred = FALSE,
    #split = 0,
    ploidy = 1,
    returnCommand = FALSE,
    nThreads = NULL
    )
}


#minor bottleneck
sim_WF_10kbt <- function(n_ind, bp_len) {
    runMacs2(
    n_ind,
    nChr = 1,
    segSites = 100000,
    Ne = 100000,
    bp = bp_len,
    genLen = 7,
    mutRate = 5.0e-08,
    histNe = c(10000, 100000, 100000, 100000),
    histGen = c(1, 1000, 300000, 400000),
    inbred = FALSE,
    #split = 0,
    ploidy = 1,
    returnCommand = FALSE,
    nThreads = NULL
    )
}

#major bottleneck
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