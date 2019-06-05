#----   Comprehension Check: Nearest Neighbors    ----
library(tidyverse)

#---  Q1  ----
set.seed(1, sample.kind = "Rounding")
data("heights")
test_index <- caret::createDataPartition(y = heights$sex, times = 1, p = 0.5, list = F)

train_set <- heights %>% slice(-test_index)
test_set <- heights %>% slice(test_index)

ks <- seq(1, 101, 3)

F_1 <- sapply(ks, function(k){
 fit <- knn3(sex ~ height, data = train_set, k = k)
 y_hat <- predict(fit, test_set, type = "class")
 
 
 f_1 <- caret::F_meas(data = y_hat, reference = test_set$sex)
 f_1
})


max(F_1)
ks[F_1 == max(F_1)]


#---    Q2      ----
library(dslabs)
library(caret)
data("tissue_gene_expression")
ks <- c(1,3,5,6,7,9,11)
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(tissue_gene_expression$y, times = 1, p = 0.5, list = F)

train_set <- tissue_gene_expression$x[test_index,]
test_set <- tissue_gene_expression$x[-test_index,]
y_train <- tissue_gene_expression$y[test_index]
y_test <- tissue_gene_expression$y[-test_index]


sapply(ks, function(k) {
       
        fit <- knn3(train_set, y_train,  k = k)
        
        y_hat <- predict(fit, newdata = test_set, type = "class")
        
        oa <- confusionMatrix(data = y_hat, reference = y_test)$overall[1]
        names(oa) <- k
        oa
})
