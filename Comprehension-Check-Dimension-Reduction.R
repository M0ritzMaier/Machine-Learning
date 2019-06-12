#----   Comprehension Check: Dimension Reduction    ----
library(dslabs)
library(tidyverse)
library(caret)
library(rpart)
library(randomForest)
library(Rborist)

#---    Q1      ----
data("tissue_gene_expression")
dim(tissue_gene_expression$x)

# Principal components analysis
pca <- prcomp(tissue_gene_expression$x)


data.frame(pca$x[,1:2], tissue = tissue_gene_expression$y) %>% 
        ggplot(aes(PC1, PC2, fill = tissue))+
        geom_point(cex=3, pch=21) +
        coord_fixed(ratio = 1)


#---    Q2      ----
# Possible biases from exerimental design
# Average of predictors
avg_pred <- apply(tissue_gene_expression$x, 1, function(x) mean(x))

data.frame(pca$x[,1:2], avg_pred,tissue = tissue_gene_expression$y) %>% 
        ggplot(aes(PC1, avg_pred, fill = tissue))+
        geom_point(cex=3, pch=21)
# Correlation PC1 and average predictor
cor(pca$x[,1], avg_pred)


#---    Q3      ----
x <- with(tissue_gene_expression, sweep(x, 1, rowMeans(x)))
pc <- prcomp(x)
data.frame(pc_1 = pc$x[,1], pc_2 = pc$x[,2], 
           tissue = tissue_gene_expression$y) %>%
        ggplot(aes(pc_1, pc_2, color = tissue)) +
        geom_point()


#---    Q4      ----
# Boxplot for the first 10 PCA
# second largest median distance
data.frame(pca$x, tissue = tissue_gene_expression$y) %>% ggplot(aes(x = tissue, y = PC7)) + geom_boxplot()



#---    Q5      ----
var_explained <- cumsum(pca$sdev^2/sum(pca$sdev^2))
plot(var_explained)