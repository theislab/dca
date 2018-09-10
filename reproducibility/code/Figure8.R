# Load pre-calculated Seurat object ####
library(Seurat)
load("../data/stoeckius/CBMC.seurat.RData")

# Load DCA denoised data ####
dca <- read.csv("../data/stoeckius/stoeckius_dca.csv", row.names = 1)

# Add imputed data to Seurat object ####
tmp <- dca
rownames(tmp) <- gsub("HUMAN", "IMPUTED", rownames(tmp))
cbmc <- SetAssayData(cbmc, assay.type = "IMPUTED", slot = "raw.data", new.data = data.matrix(tmp))
cbmc <- NormalizeData(cbmc, assay.type = "IMPUTED")
cbmc <- ScaleData(cbmc, assay.type = "IMPUTED", display.progress = FALSE)

# Subset to NK cells ####
sub <- SubsetData(cbmc, ident.use = 3)
sub <- ScaleData(sub, assay.type = "CITE", display.progress = FALSE)
sub <- ScaleData(sub, display.progress = FALSE, vars.to.regress = "nUMI")

# Generate tSNEs colored by protein levels ####
FeaturePlot(sub, c("CITE_CD56", "CITE_CD16"), min.cutoff = "q01", max.cutoff = "q99", cols.use = c("grey", "blue")) # Panel A & B

# Generate scatterplot of expression levels ####
par(mfrow = c(1,3))
library(mclust)
tmp <- sub@assay$CITE@scale.data[c('CITE_CD56', 'CITE_CD16'),]
m_prot <- Mclust(t(tmp), G = 2)
plot(t(tmp), col = m_prot$classification, main = 'Protein', pch = 16) # Panel C

tmp <- data.matrix(sub@data[c('HUMAN_NCAM1', 'HUMAN_FCGR3A'),])
m_orig <- Mclust(t(tmp), G = 2)
plot(t(tmp), col = m_prot$classification, main = 'Original RNA', pch = 16)  # Panel D

tmp <- data.matrix(sub@assay$IMPUTED@data[c('IMPUTED_NCAM1', 'IMPUTED_FCGR3A'),])
m_imp <- Mclust(t(tmp), G = 2)
plot(t(tmp), col = m_prot$classification, main = 'Original RNA', pch = 16)  # Panel E

fisher.test(table(m_prot$classification==1, m_imp$classification==2))
fisher.test(table(m_prot$classification==1, m_orig$classification==2))
