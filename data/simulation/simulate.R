# Warning! R 3.4 and Bioconductor 3.5 are required for splatter!
# library(BiocInstaller)
# biocLite('splatter')
library(splatter)

save.sim <- function(sim, dir) {
  counts     <- counts(sim)
  truecounts <- get_exprs(sim, 'TrueCounts')
  dropout    <- get_exprs(sim, 'Dropout')
  mode(dropout) <- 'integer'
  cellinfo   <- pData(sim)
  geneinfo   <- fData(sim)

  id <- deparse(substitute(sim))

  # save count matrices
  write.table(counts, paste0(dir, '/counts_', id, '.tsv'),
              sep='\t', row.names=F, col.names=F, quote=F)
  write.table(truecounts, paste0(dir, '/info_', id, '_truecounts.tsv'),
              sep='\t', row.names=F, col.names=F, quote=F)

  # save ground truth dropout labels
  write.table(dropout, paste0(dir, '/info_', id, '_droupout.tsv'),
              sep='\t', row.names=F, col.names=F, quote=F)

  # save metadata
  write.table(cellinfo, paste0(dir, '/info_', id, '_cellinfo.tsv'), sep='\t',
              row.names=F, quote=F)
  write.table(geneinfo, paste0(dir, '/info_', id, '_geneinfo.tsv'), sep='\t',
              row.names=F, quote=F)
}

if (!dir.exists('real/single')) dir.create('real/single', showWarnings=F, recursive=T)
if (!dir.exists('real/group')) dir.create('real/group', showWarnings=F, recursive=T)
if (!dir.exists('sim/single')) dir.create('sim/single', showWarnings=F, recursive=T)
if (!dir.exists('sim/group')) dir.create('sim/group', showWarnings=F, recursive=T)

#### Estimate parameters from the real dataset
data(sc_example_counts)
params <- splatEstimate(sc_example_counts)

# simulate scRNA data with default parameters
sample_real_single <- splatSimulateSingle(params, groupCells=2000, nGenes=500,
                                          dropout.present=T, seed=42,
                                          bcv.common=2) # limit disp to get
                                                        # fewer true zeros
save.sim(sample_real_single, 'real/single')

# simulate data, two groups
sample_real_group <- splatSimulateGroups(params, groupCells=c(1000, 1000),
                                         nGenes=500, dropout.present=T, seed=42,
                                         bcv.common=2)
save.sim(sample_real_group, 'real/group')


#### Simulate data with default params
sample_sim_single <- splatSimulateSingle(groupCells=2000, nGenes=500,
                                         dropout.present=T, seed=42,
                                         dropout.shape=-0.5, dropout.mid=4)
save.sim(sample_sim_single, 'sim/single')

# simulate data, two groups
sample_sim_group <- splatSimulateGroups(groupCells=c(1000, 1000),
                                        nGenes=500, dropout.present=T,
                                        dropout.shape=-0.5, dropout.mid=4,
                                        seed=42)
save.sim(sample_sim_group, 'sim/group')
