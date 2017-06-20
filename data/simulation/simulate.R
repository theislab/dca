# Warning! R 3.4 and Bioconductor 3.5 are required for splatter!
# library(BiocInstaller)
# biocLite('splatter')
library(splatter)

save.sim <- function(sim, dir) {
  counts     <- counts(sim)
  truecounts <- get_exprs(sim, 'TrueCounts')
  drp <- 'Dropout' %in% names(assayData(sim))
  if (drp) {
    dropout    <- get_exprs(sim, 'Dropout')
    mode(dropout) <- 'integer'
  }
  lognorm <- get_exprs(sim, 'exprs')
  cellinfo   <- pData(sim)
  geneinfo   <- fData(sim)

  # save count matrices
  write.table(counts, paste0(dir, '/counts.tsv'),
              sep='\t', row.names=F, col.names=F, quote=F)
  write.table(truecounts, paste0(dir, '/info_truecounts.tsv'),
              sep='\t', row.names=F, col.names=F, quote=F)
  write.table(format(lognorm, digits=6), paste0(dir, '/counts_lognorm.tsv'),
              sep='\t', row.names=F, col.names=F, quote=F)

  if (drp) {
    # save ground truth dropout labels
    write.table(dropout, paste0(dir, '/info_droupout.tsv'),
                sep='\t', row.names=F, col.names=F, quote=F)
  }

  # save metadata
  write.table(cellinfo, paste0(dir, '/info_cellinfo.tsv'), sep='\t',
              row.names=F, quote=F)
  write.table(geneinfo, paste0(dir, '/info_geneinfo.tsv'), sep='\t',
              row.names=F, quote=F)

  saveRDS(sim, paste0(dir, '/sce.rds'))
}

nGenes <- 200
nCells <- 2000

for (dropout in c(0, 1, 3, 5)) {
  for (ngroup in c(1, 2, 3, 6)) {

    # split nCells into roughly ngroup groups
    groupCells <- ifelse(ngroup==1, nCells, as.vector(table(as.integer(cut(seq_len(nCells), ngroup)))))
    method <- ifelse(ngroup == 1, 'single', 'groups')

    dirname <- paste0('real/group', ngroup, '/dropout', dropout)
    if (!dir.exists(dirname))
      dir.create(dirname, showWarnings=F, recursive=T)

    #### Estimate parameters from the real dataset
    data(sc_example_counts)
    params <- splatEstimate(sc_example_counts)

    # simulate scRNA data
    sim <- splatSimulate(params, groupCells=groupCells, nGenes=nGenes,
                         dropout.present=(dropout!=0), dropout.shape=-1,
                         dropout.mid=dropout, seed=42,
                         bcv.common=1) # limit disp to get
                                       # fewer true zeros
    save.sim(sim, dirname)


    dirname <- paste0('sim/group', ngroup, '/dropout', dropout)
    if (!dir.exists(dirname))
      dir.create(dirname, showWarnings=F, recursive=T)

    #### Simulate data without using real data
    sim <- splatSimulate(groupCells=groupCells, nGenes=nGenes,
                         dropout.present=(dropout!=0), method=method,
                         seed=42, dropout.shape=-1, dropout.mid=dropout)
    save.sim(sim, dirname)
  }
}
