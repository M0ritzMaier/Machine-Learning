#----   Comprehension Check: Cross-validation    ----
library(tidyverse)
library(caret)

#---    Q1      ----
if(R.version$version.string  == "R version 3.6.0 (2019-04-26)"){
        set.seed(1996, sample.kind = "Rounding")
} else {
        set.seed(1996)
}
n <- 1000
p <- 10000
x <- matrix(rnorm(n*p), n, p)
colnames(x) <- paste("x", 1:ncol(x), sep = "_")
y <- rbinom(n, 1, 0.5) %>% factor()

x_subset <- x[ ,sample(p, 100)]

fit <- train(x_subset, y, method = "glm")
fit$results


#---    Q2      ---
library(devtools)
devtools::install_bioc("genefilter")
library(genefilter)
tt <- colttests(x, y)

pvals <- tt$p.value

#---    Q3      ---
ind <- which(tt$p.value < 0.01)
length(ind)


#---    Q4      ---
x_subset <- x[,ind]
fit <- train(x_subset, y, method = "glm")
fit$results


#---    Q5      ---
k = seq(101, 301, 25)
fit <- train(x_subset, y, method = "knn", tuneGrid = data.frame(k = seq(101, 301, 25)))
ggplot(fit)


#---    Q6      ---
# We used the entire dataset to select the columns used in the model. 

#---    Q7      ---
k = seq(1,7,2)
library(dslabs)
data("tissue_gene_expression")
fit <- train(tissue_gene_expression$x, tissue_gene_expression$y, method = "knn", tuneGrid = data.frame(k = k))
ggplot(fit)
