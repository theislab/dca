suppressMessages(library(Seurat, quietly = T))
suppressMessages(library(ggplot2, quietly = T))
suppressMessages(library(Rtsne, quietly = T))

normalize <- function(x) {
  sf <- rowSums(x)
  sf <- sf / median(sf)
  x <- x / sf
  x <- log(x+1)
  scale(x, center = T, scale = T)
}


`%+%` <- paste0
args <- commandArgs(trailingOnly = T)
stopifnot(length(args) == 1)
arg <- args[[1]]

if (!dir.exists(arg)) {
  files <- arg
} else {
  files <- list.files(arg, recursive = T, pattern = '^counts\\..sv', full.names = T)
}

for (cnt.file in files) {
  print('Visualizing ' %+% cnt.file)

  output.dir <- dirname(cnt.file)
  tbl <- read.table(cnt.file, header = T)

# Load labels if available ------------------------------------------------

  if (file.exists(output.dir %+% '/info_cellinfo.tsv')) {
    labels <- read.table(output.dir %+% '/info_cellinfo.tsv', header=T)$Group
  } else if (file.exists(output.dir %+% '/../info_cellinfo.tsv')) {
    labels <- read.table(output.dir %+% '/../info_cellinfo.tsv', header=T)$Group
  }
  else labels <- NULL

# Seurat PCA and tSNE -----------------------------------------------------

  s <- CreateSeuratObject(tbl, min.cells = 1, min.genes = 1)
  print(s)
  s <- NormalizeData(s, display.progress = F)
  s <- ScaleData(s, display.progress = F)

  s <- RunPCA(s, pc.genes = rownames(s@data), do.print = F)
  s <- RunTSNE(s)
  s <- FindClusters(s, reduction.type = "pca", dims.use=1:5, save.SNN = T, print.output = 0)
  print('Number of clusters: ' %+% length(levels(s@ident)))

  DimPlot(s)
  ggsave(output.dir %+% '/seurat_PCA_all_CL.png')
  DimPlot(s, reduction.use = 'tsne')
  ggsave(output.dir %+% '/seurat_tSNE_all_CL.png')
  if (!is.null(labels)) {
    s@meta.data$ground.truth <- labels
    DimPlot(s, group.by='ground.truth')
    ggsave(output.dir %+% '/seurat_PCA_all_GT.png')
    DimPlot(s, reduction.use = 'tsne', group.by='ground.truth')
    ggsave(output.dir %+% '/seurat_tSNE_all_GT.png')
  }

  s <- FindVariableGenes(s, do.plot = F, display.progress = F)
  print('Number of variable genes: ' %+% length(s@var.genes))
  s <- RunPCA(s, do.print = F) # use variable genes by default
  s <- RunTSNE(s)
  s <- FindClusters(s, reduction.type = "pca", dims.use = 1:5, save.SNN = T, print.output = 0, force.recalc = T)
  print('Number of clusters: ' %+% length(levels(s@ident)))

  DimPlot(s)
  ggsave(output.dir %+% '/seurat_PCA_var_CL.png')
  DimPlot(s, reduction.use = 'tsne')
  ggsave(output.dir %+% '/seurat_tSNE_var_CL.png')
  if (!is.null(labels)) {
    DimPlot(s, group.by='ground.truth')
    ggsave(output.dir %+% '/seurat_PCA_var_GT.png')
    DimPlot(s, reduction.use = 'tsne', group.by='ground.truth')
    ggsave(output.dir %+% '/seurat_tSNE_var_GT.png')
  }

  write.table(data.frame(label=unname(s@ident), cell=names(s@ident)),
              output.dir %+% '/seurat_cluster_labels.tsv',
              row.names = F, quote = F)

  saveRDS(s, output.dir %+% '/seurat.Rds')

  # PCA and tSNE with sf and lognorm ----------------------------------------

  if (!is.null(labels)) {
    counts <- t(tbl)
    counts <- counts[, colSums(counts)>0]
    norm.counts <- normalize(counts)

    pca.counts <- prcomp(norm.counts, rank. = 2)$x
    qplot(pca.counts[,1], pca.counts[,2], color=labels, xlab='PC1', ylab='PC2')
    ggsave(output.dir %+% '/seurat_PCA_all_simplepre_GT.png')

    tsne.counts <- Rtsne(norm.counts)$Y
    qplot(tsne.counts[,1], tsne.counts[,2], color=labels, xlab='tsne1', ylab='tsne2')
    ggsave(output.dir %+% '/seurat_tSNE_all_simplepre_GT.png')

    if (file.exists(output.dir %+% '/info_truecounts.tsv')) {

      tr <- t(read.table(output.dir %+% '/info_truecounts.tsv'))
      tr<- tr[, colSums(tr)>0]
      tr.norm <- normalize(tr)
      pca.tr <- prcomp(tr.norm, rank. = 2)$x
      qplot(pca.tr[,1], pca.tr[,2], color=labels, xlab='pca1', ylab='pca2')
      ggsave(output.dir %+% '/seurat_TRUECOUNT_PCA_all_simplepre_GT.png')

      tsne.tr <- Rtsne(tr.norm)$Y
      qplot(tsne.tr[,1], tsne.tr[,2], color=labels, xlab='tsne1', ylab='tsne2')
      ggsave(output.dir %+% '/seurat_TRUECOUNT_tSNE_all_simplepre_GT.png')

    }
  }

}