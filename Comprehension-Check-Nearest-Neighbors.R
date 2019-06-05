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


library(dslabs)
data("tissue_gene_expression")