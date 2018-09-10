# Load libraries ####
library(DESeq2)
library(plotrix)
library(ggplot2)
library(beeswarm)

# Load DESeq2 results ####
load("../data/chu/chu_deseq2_results.RData")

# Generate plots foldchange plots ####
# Panels A and B
pdf(useDingbats = F, "../figs/Fig5_A_B.pdf", width = 8, height = 4.5)
par(mfrow = c(1, 2))
diffs <- list(abs(res_original$log2FoldChange - res_bulk$log2FoldChange),
                abs(res_dca$log2FoldChange - res_bulk$log2FoldChange))
farben <- color.scale(unlist(diffs), alpha = 0.8, extremes = c("darkblue", "darkred"))
plot(res_original$log2FoldChange, res_bulk$log2FoldChange, main = "Original", ylab = "Bulk", xlab = "Esimtated fold change", ylim = c(-5, 15), xlim = c(-30, 30), col = farben[1:1000], pch = 16)
abline(0, 1)
abline(v = 0, h = 0, col = "grey", lty = 2)
legend("bottomright", paste("Rho:", signif(cor(res_original$log2FoldChange, res_bulk$log2FoldChange), 2)), bty = "n")
plot(res_dca$log2FoldChange, res_bulk$log2FoldChange, main = "DCA denoised", ylab = "Bulk", xlab = "Esimtated fold change", ylim = c(-5, 15), xlim = c(-30, 30), col = farben[1001:2000], pch = 16)
abline(0, 1)
abline(v = 0, h = 0, col = "grey", lty = 2)
legend("bottomright", paste("Rho:", signif(cor(res_dca$log2FoldChange, res_bulk$log2FoldChange), 2)), bty = "n")
dev.off()

# Load expression tables ####
bulk <- data.matrix(read.csv("../data/chu/chu_bulk.csv", row.names = 1))
treat_bulk <- colnames(bulk)
treat_bulk <- unlist(lapply(treat_bulk, function(x) strsplit(x, "_", fixed = T)[[1]][1]))
ok <- which(treat_bulk %in% c("H1", "DEC"))
treat_bulk <- treat_bulk[ok]
bulk <- bulk[, ok]

counts <- read.csv("../data/chu/chu_original.csv", row.names = 1)
counts <- round(counts)
treat <- unlist(lapply(colnames(counts), function(x) strsplit(x, "_", fixed = T)[[1]][1]))
farben <- c("black", "yellow", "blue", "purple", "green", "red", "grey")
names(farben) <- c("H1", "H9", "EC", "NPC", "DEC", "HFF", "TB")
ok <- which(treat %in% c("H1", "DEC"))
counts <- counts[, ok]
treat <- treat[ok]
dca <- data.matrix(read.csv("../data/chu/chu_dca.csv", row.names = 1))
original <- data.matrix(counts[rownames(dca),])
bulk <- data.matrix(bulk[rownames(dca),])

# Generate single gene plots ####
# Panels C, D and E
pdf(useDingbats = F, "../figs/Fig5_C_D_E.pdf", width = 9, height = 3.5)
par(mfrow = c(1, 3))
gene <- "LEFTY1"
boxplot(split(original[gene,], treat)[c("H1", "DEC")], outline = FALSE, main = "Original", ylim = c(0, 5000), ylab = gene)
#beeswarm(split(original[gene,], treat)[c("H1", "DEC")], pch = 16, add = TRUE, cex = 0.8)
boxplot(split(dca[gene,], treat)[c("H1", "DEC")], outline = FALSE, main = "DCA denoised", ylim = c(0, 5000), ylab = gene)
#beeswarm(split(dca[gene,], treat)[c("H1", "DEC")], pch = 16, add = TRUE, cex = 0.8)
boxplot(split(bulk[gene,], treat_bulk)[c("H1", "DEC")], outline = FALSE, main = "Bulk", ylab = gene)
dev.off()

# Generate boxplot ####
# Panel F
load("../data/chu/HundredTimes_20cells.RData")
load("../data/chu/chu_deseq2_results.RData")

res_bulk <- res_bulk[rownames(res_original), ]
res_bulk$log2FoldChange <- res_bulk$log2FoldChange*(-1)
tmp <- lapply(1:5, function(y) unlist(lapply(1:100, function(x) cor(res_bulk$log2FoldChange, hundredTimes[[x]][[y]]$log2FoldChange, use = "complete"))))

colors <- list(c(192, 81, 158), c(73, 93, 115), c(152, 201, 125), c(117, 90, 36))
colors <- c("white", unlist(lapply(colors, function(x) rgb((x/sum(x))[1], (x/sum(x))[2], (x/sum(x))[3]))))

pdf(useDingbats = F, "../figs/Fig5_F.pdf", height = 4, width = 3.5)
boxplot(tmp, names = c("original", "DCA", "SAVER", "MAGIC", "scImpute"), ylab = "Pearson correlation", las = 2, col = colors, outline = F)
dev.off()


