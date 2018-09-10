# Load pre-calculated Seurat object ####
library(Seurat)
load("../data/stoeckius/CBMC.seurat.RData")

# Generate tSNE visualization showing celltype clustering (Fig Panel A) ####
panelA <- TSNEPlot(cbmc, do.label = TRUE, pt.size = 0.5)
panelA

# Load imputed data ####
dca <- read.csv("../data/stoeckius/stoeckius_dca.csv", row.names = 1)
magic <- read.csv("../data/stoeckius/stoeckius_magic.csv", row.names = 1)
saver <- read.csv("../data/stoeckius/stoeckius_saver.csv", row.names = 1)
scimpute <- read.csv("../data/stoeckius/stoeckius_scimpute.csv", row.names = 1)

# Define protein-mRNA pairs ####
protein <- c("CD3", "CD19", "CD4", "CD8", "CD56", "CD16", "CD11c", "CD14")
rna <- c("CD3E", "CD19", "CD4", "CD8A", "NCAM1", "FCGR3A", "ITGAX", "CD14")

# Add imputed RNA levels to Seurat object ####
tmp <- dca
rownames(tmp) <- gsub("HUMAN", "IMPUTED", rownames(tmp))
cbmc <- SetAssayData(cbmc, assay.type = "IMPUTED", slot = "raw.data", new.data = data.matrix(tmp))
cbmc <- NormalizeData(cbmc, assay.type = "IMPUTED")
cbmc <- ScaleData(cbmc, assay.type = "IMPUTED", display.progress = FALSE)

tmp <- magic
rownames(tmp) <- gsub("HUMAN", "MAGIC", rownames(tmp))
cbmc <- SetAssayData(cbmc, assay.type = "MAGIC", slot = "raw.data", new.data = data.matrix(tmp))
cbmc <- NormalizeData(cbmc, assay.type = "MAGIC")
cbmc <- ScaleData(cbmc, assay.type = "MAGIC", display.progress = FALSE)

tmp <- saver
rownames(tmp) <- gsub("HUMAN", "SAVER", rownames(tmp))
cbmc <- SetAssayData(cbmc, assay.type = "SAVER", slot = "raw.data", new.data = data.matrix(tmp))
cbmc <- NormalizeData(cbmc, assay.type = "SAVER")
cbmc <- ScaleData(cbmc, assay.type = "SAVER", display.progress = FALSE)

tmp <- scimpute
rownames(tmp) <- gsub("HUMAN", "SCIMPUTE", rownames(tmp))
cbmc <- SetAssayData(cbmc, assay.type = "SCIMPUTE", slot = "raw.data", new.data = data.matrix(tmp))
cbmc <- NormalizeData(cbmc, assay.type = "SCIMPUTE")
cbmc <- ScaleData(cbmc, assay.type = "SCIMPUTE", display.progress = FALSE)

# tSNE colored by imputed and original RNA expression (Fig Panel B) ####
panelB1 <- FeaturePlot(cbmc, features.plot = c(paste0("CITE_", protein[1:4]), paste0("HUMAN_", rna[1:4]), paste0("IMPUTED_", rna[1:4])),
                       min.cutoff = "q05", max.cutoff = "q95", nCol = 4, cols.use = c("lightgrey", "blue"), pt.size = 0.5, do.return = T)
panelB2 <- FeaturePlot(cbmc, features.plot = c(paste0("CITE_", protein[5:8]), paste0("HUMAN_", rna[5:8]), paste0("IMPUTED_", rna[5:8])),
                       min.cutoff = "q05", max.cutoff = "q95", nCol = 4, cols.use = c("lightgrey", "blue"), pt.size = 0.5, do.return = T)

# Example plot of CD3 expression in T cells (Fig Panel C) ####
tmp <- SubsetData(cbmc, ident.use = c(0, 5))
rna.raw <- tmp@data["HUMAN_CD3E",]
protein <- tmp@assay$CITE@scale.data["CITE_CD3",]
rna.imputed <- tmp@assay$IMPUTED@scale.data["IMPUTED_CD3E",]
table(rna.raw == 0)[["TRUE"]]/length(rna.raw)
scale01 <- function(x){
  x <- (x-min(x)) / (max(x) - min(x))
  x - median(x)
}
aframe <- data.frame(Relative.expresion = c(scale01(protein), scale01(rna.raw), scale01(rna.imputed)), type = c(rep("Protein", length(protein)), rep("Original", length(protein)), rep("Denoised", length(protein))))
panelC <- ggplot(aframe, aes(Relative.expresion, colour = type)) + geom_density() + ggtitle("CD3 in T cells")
panelC

# Calculate likelihoods of co-occurrence (Fig Panel D) ####
protein <- c("CD3", "CD19", "CD4", "CD8", "CD56", "CD16", "CD11c", "CD14")
rna <- c("CD3E", "CD19", "CD4", "CD8A", "NCAM1", "FCGR3A", "ITGAX", "CD14")=
l <- list(cor(t(cbmc@scale.data[paste0("HUMAN_", rna),]), t(cbmc@assay$CITE@scale.data[paste0("CITE_", protein),]), method = "spearman"),
          cor(t(cbmc@assay$IMPUTED@scale.data[paste0("IMPUTED_", rna),]), t(cbmc@assay$CITE@scale.data[paste0("CITE_", protein),]), method = "spearman"),
          cor(t(cbmc@assay$MAGIC@scale.data[paste0("MAGIC_", rna),]), t(cbmc@assay$CITE@scale.data[paste0("CITE_", protein),]), method = "spearman"),
          cor(t(cbmc@assay$SAVER@scale.data[paste0("SAVER_", rna),]), t(cbmc@assay$CITE@scale.data[paste0("CITE_", protein),]), method = "spearman"),
          cor(t(cbmc@assay$SCIMPUTE@scale.data[paste0("SCIMPUTE_", rna),]), t(cbmc@assay$CITE@scale.data[paste0("CITE_", protein),]), method = "spearman"))
l <- lapply(l, diag)
boxplot(l, ylab = "Spearman Correlation", names = c("Original", "DCA", "MAGIC", "SAVER", "scImpute"), las = 2)



