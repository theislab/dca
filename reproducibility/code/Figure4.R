# Load matrices ####
withoutDropout <- read.csv("../data/francesconi/francesconi_withoutDropout.csv", row.names = 1)
withDropout <- read.csv("../data/francesconi/francesconi_withDropout.csv", row.names = 1)
dca <- read.csv("../data/francesconi/francesconi_dca.csv", row.names = 1)
magic <- read.csv("../data/francesconi/francesconi_magic.csv", row.names = 1)
scimpute  <- read.csv("../data/francesconi/francesconi_scimpute.csv", row.names = 1)
saver <- read.csv("../data/francesconi/francesconi_saver.csv", row.names = 1)

# Generate heatmaps ####
cors <- apply(withoutDropout,1,function(x) cor.test(method = "pearson",x,1:ncol(withoutDropout)))
pvals <- unlist(lapply(cors, function(x) x$p.value))
coefs <- unlist(lapply(cors, function(x) x$estimate))
genes.up <- names(head(sort(pvals[coefs > 0]), 100))
genes.down <- names(head(sort(pvals[coefs < 0]), 100))

genHeatmap <- function(matr){
  library(gplots)
  load("../data/BlueYellowColormaps_V1.RData")
  genes <- c(genes.up, genes.down )
  rowOrd <- order(unlist(lapply(cors[genes], function(x) x$estimate)))
  matr <- matr[match(genes, rownames(withoutDropout)),]
  tmp <- data.matrix(matr)
  tmp <- t(apply(tmp, 1, function(x) (x - mean(x)) / sd(x)))
  tmp[which(tmp > 2)] <- 2
  tmp[which(tmp < (-2))] <- (-2)
  tmp[which(matr == 0)] <- NA
  heatmap.2(tmp[rowOrd,], Rowv = NA, Colv = NA, density.info = "none", trace = "none", col = yellow2blue, na.color = "grey", scale = "none", labRow = "", labCol = "")
}

genHeatmap(withoutDropout) # Panel A
genHeatmap(withDropout) # Panel B
genHeatmap(dca) # Panel C

# Generate boxplot ####
cors <- apply(withoutDropout, 1, function(x) cor.test(method = "pearson",x,1:ncol(withoutDropout)))
pvals <- unlist(lapply(cors, function(x) x$p.value))
coefs <- unlist(lapply(cors, function(x) x$estimate))

genes <- names(head(sort(unlist(lapply(cors, function(x) x$p.value))), 500))

calc.cor <- function(x){
  ok <- match(genes, rownames(withoutDropout))
  C <- apply(x[ok,],1,function(y) cor(y,1:ncol(x)))
  return(abs(C))
}

imputed <- list(withoutDropout, withDropout, dca, saver, scimpute, magic)
imputed <- lapply(imputed, data.matrix)

imp <- lapply(imputed, calc.cor)
boxplot(imp, main = "Correlation with Time", ylab = "Pearson Correlation", names = c("Without noise", "With noise", "AE","SAVER", "scImpute", "MAGIC"),
        cex.main = 2, cex.lab = 1.5, cex.axis = 1.5, outline = F)


# Generate correlation plots ####
genes <- c("tbx-36", "his-8")

cors <- apply(withoutDropout,1,function(x) cor.test(method = "pearson", x, 1:ncol(withoutDropout)))
pvals <- unlist(lapply(cors, function(x) x$p.value))
coefs <- unlist(lapply(cors, function(x) x$estimate))

genes.up <- names(head(sort(pvals[coefs > 0]), 100))
genes.down <- names(head(sort(pvals[coefs < 0]), 100))

scale01 <- function(x) (x - min(x)) / (max(x) - min(x))
genCorPlot <- function(gene1, gene2, matr){
  par(mfrow = c(1, 3))
  library(plotrix)
  farben <- color.scale(1:206, extremes = c("blue", "red"), alpha = 0.8)
  plot(scale01(exp(unlist(withoutDropout[gene1,]))), scale01(exp(unlist(withoutDropout[gene2,]))), col = farben, pch = 16, main = "Original", ylab = gene2, xlab = gene1)
  correl <- signif(cor(method = "spearman", scale01(exp(unlist(withoutDropout[gene1,]))), scale01(exp(unlist(withoutDropout[gene2,])))), 2)
  legend("topright", paste("Spearman Rho", correl))
  
  plot(scale01(unlist(withDropout[gene1,])), scale01(unlist(withDropout[gene2,])), col = farben, pch = 16, main = "Dropout", ylab = gene2, xlab = gene1)
  correl <- signif(cor(method = "spearman", scale01(unlist(withDropout[gene1,])), scale01(unlist(withDropout[gene2,]))), 2)
  legend("topright", paste("Spearman Rho", correl))
  
  plot(scale01(unlist(matr[gene1,])), scale01(unlist(matr[gene2,])), col = farben, pch = 16, main = "Denoised", ylab = gene2, xlab = gene1)
  correl <- signif(cor(method = "spearman", scale01(unlist(matr[gene1,])), scale01(unlist(matr[gene2,]))), 2)
  legend("topright", paste("Spearman Rho", correl))
  
}

genCorPlot(gene1 = genes[2], gene2 = genes[1], matr = dca) # Panels E, F, G 
genCorPlot(gene1 = genes[2], gene2 = genes[1], matr = saver)
genCorPlot(gene1 = genes[2], gene2 = genes[1], matr = magic)
genCorPlot(gene1 = genes[2], gene2 = genes[1], matr = scimpute)
