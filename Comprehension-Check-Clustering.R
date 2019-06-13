#----   Comprehension Check: Clustering    ----
library(tidyverse)
library(caret)
library(rpart)
library(dslabs)


#---    Q1      ---
data("tissue_gene_expression")

d <- dist(tissue_gene_expression$x - rowMeans(tissue_gene_expression$x))


#---    Q2      ---
h <- hclust(d)

plot(h, cex = 0.65)


#---    Q3      ---
cl <- kmeans(tissue_gene_expression$x, centers = 7)
table(cl$cluster, tissue_gene_expression$y)


#---    Q4      ---
library(RColorBrewer)
sds <- matrixStats::colSds(tissue_gene_expression$x)
ind <- order(sds, decreasing = TRUE)[1:50]
colors <- brewer.pal(7, "Dark2")[as.numeric(tissue_gene_expression$y)]
heatmap(t(tissue_gene_expression$x[,ind]), col = brewer.pal(11, "RdBu"), scale = "row", ColSideColors = colors)
