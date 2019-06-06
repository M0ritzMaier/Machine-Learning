#----   Comprehension Check: Generative Models    ----
library(tidyverse)
library(caret)

#---    Q1      ----
library(dslabs)
library(caret)
data("tissue_gene_expression")

seed <- 1993
if(R.version$version.string  == "R version 3.6.0 (2019-04-26)"){
        set.seed(seed, sample.kind = "Rounding")
} else {
        set.seed(seed)
}

ind <- which(tissue_gene_expression$y %in% c("cerebellum", "hippocampus"))
y <- droplevels(tissue_gene_expression$y[ind])
x <- tissue_gene_expression$x[ind, ]
x <- x[, sample(ncol(x), 10)]

train(x = x, y = y, method = "lda")


#---    Q2      ----
train(x = x, y = y, method = "lda")$finalModel
str(train(x = x, y = y, method = "lda"))

M <- train(x = x, y = y, method = "lda")$finalModel$means
as.data.frame(t(M)) %>% ggplot(aes(hippocampus, cerebellum)) + geom_text(label = row.names(t(M)))


#---    Q3      ----
train(x = x, y = y, method = "qda")


#---    Q4      ----
M <- train(x = x, y = y, method = "qda")$finalModel$means
as.data.frame(t(M)) %>% ggplot(aes(hippocampus, cerebellum)) + geom_text(label = row.names(t(M)))


#---    Q5      ----
train(x = x, y = y, method = "lda", preProcess = "center")

M <- train(x = x, y = y, method = "lda", preProcess = "center")$finalModel$means
as.data.frame(t(M)) %>% ggplot(aes(hippocampus, cerebellum)) + geom_text(label = row.names(t(M)))


#---    Q6      ----
data("tissue_gene_expression")
seed <- 1993
if(R.version$version.string  == "R version 3.6.0 (2019-04-26)"){
        set.seed(seed, sample.kind = "Rounding")
} else {
        set.seed(seed)
}
y <- tissue_gene_expression$y
x <- tissue_gene_expression$x
x <- x[, sample(ncol(x), 10)]
train(x, y, method = "lda")
