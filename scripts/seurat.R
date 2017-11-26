suppressMessages(library(Seurat, quietly = T))

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
  tbl <- read.table(cnt.file, header = T, sep = '\t')

  s <- CreateSeuratObject(tbl, min.cells = 1, min.genes = 1)
  print(s)
  s <- NormalizeData(s, display.progress = F)
  s <- ScaleData(s, display.progress = F)

  s <- RunPCA(s, pc.genes = rownames(s@data), do.print = F)
  s <- RunTSNE(s)
  s <- FindClusters(s, reduction.type = "pca", dims.use=1:5, save.SNN = T, print.output = 0)
  print('Number of clusters: ' %+% length(levels(s@ident)))

  DimPlot(s)
  ggsave(output.dir %+% '/seurat_plot_pca_allgenes.png')
  DimPlot(s, reduction.use = 'tsne')
  ggsave(output.dir %+% '/seurat_plot_tsne_allgenes.png')

  s <- FindVariableGenes(s, do.plot = F, display.progress = F)
  print('Number of variable genes: ' %+% length(s@var.genes))
  s <- RunPCA(s, do.print = F) # use variable genes by default
  s <- RunTSNE(s)
  s <- FindClusters(s, reduction.type = "pca", dims.use = 1:5, save.SNN = T, print.output = 0, force.recalc = T)
  print('Number of clusters: ' %+% length(levels(s@ident)))

  DimPlot(s)
  ggsave(output.dir %+% '/seurat_plot_pca_vargenes.png')
  DimPlot(s, reduction.use = 'tsne')
  ggsave(output.dir %+% '/seurat_plot_tsne_vargenes.png')

  write.table(data.frame(label=unname(s@ident), cell=names(s@ident)),
              output.dir %+% '/seurat_cluster_labels.tsv',
              row.names = F, quote = F)
}