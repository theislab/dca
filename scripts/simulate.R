# Warning! R 3.4 and Bioconductor 3.5 are required for splatter!
# library(BiocInstaller)
# biocLite('splatter')
library(splatter) # requires splatter >= 1.2.0

save.sim <- function(sim, dir) {
  counts     <- counts(sim)
  truecounts <- assays(sim)$TrueCounts
  drp <- 'Dropout' %in% names(assays(sim))
  if (drp) {
    dropout    <- assays(sim)$Dropout
    mode(dropout) <- 'integer'
  }
  cellinfo   <- colData(sim)
  geneinfo   <- rowData(sim)

  # save count matrices
  write.table(counts, paste0(dir, '/counts.tsv'),
              sep='\t', row.names=T, col.names=T, quote=F)
  write.table(truecounts, paste0(dir, '/info_truecounts.tsv'),
              sep='\t', row.names=T, col.names=T, quote=F)

  if (drp) {
    # save ground truth dropout labels
    write.table(dropout, paste0(dir, '/info_droupout.tsv'),
                sep='\t', row.names=T, col.names=T, quote=F)
  }

  # save metadata
  write.table(cellinfo, paste0(dir, '/info_cellinfo.tsv'), sep='\t',
              row.names=F, quote=F)
  write.table(geneinfo, paste0(dir, '/info_geneinfo.tsv'), sep='\t',
              row.names=F, quote=F)

  saveRDS(sim, paste0(dir, '/sce.rds'))
}


for (dropout in c(0, 1, 3, 5)) {
  for (ngroup in c(1, 2, 3, 6)) {
    for(swap in c(F, T)) {

      nGenes <- 200
      batchCells <- 2000

      if (swap) {
        tmp <- nGenes
        nGenes <- batchCells
        batchCells <- tmp
      }

      # split nCells into roughly ngroup groups
      if(ngroup==1) {
        group.prob <- 1
      } else {
        group.prob <- rep(1, ngroup)/ngroup
      }
      method <- ifelse(ngroup == 1, 'single', 'groups')

      dirname <- paste0('real/group', ngroup, '/dropout', dropout, ifelse(swap, '/swap', ''))
      if (!dir.exists(dirname))
        dir.create(dirname, showWarnings=F, recursive=T)

      #### Estimate parameters from the real dataset
      data(sc_example_counts)
      params <- splatEstimate(sc_example_counts)

      # simulate scRNA data
      sim <- splatSimulate(params, group.prob=group.prob, nGenes=nGenes,
                           dropout.present=(dropout!=0), dropout.shape=-1,
                           dropout.mid=dropout, seed=42, method=method,
                           bcv.common=1) # limit disp to get fewer true zeros
      save.sim(sim, dirname)


      dirname <- paste0('sim/group', ngroup, '/dropout', dropout, ifelse(swap, '/swap', ''))
      if (!dir.exists(dirname))
        dir.create(dirname, showWarnings=F, recursive=T)

      #### Simulate data without using real data
      sim <- splatSimulate(group.prob=group.prob, nGenes=nGenes, batchCells=batchCells,
                           dropout.present=(dropout!=0), method=method,
                           seed=42, dropout.shape=-1, dropout.mid=dropout)
      save.sim(sim, dirname)
    }
  }
}
