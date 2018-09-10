# Imputation using SAVER ####
library(SAVER)
library(doParallel)
registerDoParallel(cores = 5)
sc <- read.csv("../data/francesconi/francesconi_withDropout.csv", row.names = 1)
sav <- saver(as.matrix(sc), parallel = TRUE)
sav <- sav$estimate
write.csv(sav, file = "../data/francesconi/francesconi_saver.csv", quote = F)

sc <- read.csv("../data/chu/chu_original.csv", row.names = 1)
sav <- saver(as.matrix(sc), parallel = TRUE)
sav <- sav$estimate
write.csv(sav, file = "../data/chu/chu_saver.csv", quote = F)

sc <- read.csv("../data/stoeckius/stoeckius_original.csv", row.names = 1)
sav <- saver(as.matrix(sc), parallel = TRUE)
sav <- sav$estimate
write.csv(sav, file = "../data/stoeckius/stoeckius_saver.csv", quote = F)
