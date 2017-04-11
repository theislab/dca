library(pscl)
library(readr)


# Load and save biochemists data ------------------------------------------
data("bioChemists", package = "pscl")
head(bioChemists)

#encode design matrix
design <- cbind.data.frame(art=bioChemists$art, model.matrix(art~., bioChemists)[,-1])
colnames(design) <- colnames(bioChemists)
head(design)
write_tsv(design, 'biochemists.tsv')


# NB fit ------------------------------------------------------------------
nb <- MASS::glm.nb(art ~ ., data = bioChemists)
coef.df <- rbind.data.frame(data.frame(coef(nb)), theta=nb$theta)
colnames(coef.df) <- 'val'
coef.df$coef <- rownames(coef.df)
coef.df
coef.df$coef <- c('intercept', colnames(bioChemists)[-1], 'theta')
coef.df
write_tsv(coef.df, 'biochemists-nb-coef.tsv')
pred.nb <- predict(nb, type='response')
write_tsv(data.frame(count=pred.nb), 'biochemists-nb-predictions.tsv')


# ZINB fit ----------------------------------------------------------------
zinb <- zeroinfl(art ~ . | ., data = bioChemists, dist = "negbin")
coef(zinb)
coef.df <- data.frame(count=zinb$coefficients$count,
                      zero=zinb$coefficients$zero)
coef.df <- rbind(coef.df, theta=zinb$theta)
coef.df$coef <- c('intercept', colnames(bioChemists)[-1], 'theta')
coef.df
write_tsv(coef.df, 'biochemists-zinb-coef.tsv')

pred.df <- cbind.data.frame(zero=predict(zinb, type='zero'),
                            count=predict(zinb, type='count'))

write_tsv(pred.df, 'biochemists-zinb-predictions.tsv')