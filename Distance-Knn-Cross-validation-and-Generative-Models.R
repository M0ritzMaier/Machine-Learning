#############################################################################################################################
#                                                                                                                           #
#                                                                                                                           #
#                            DISTANCE, KNN, CROSS-VALIDATION, AND GENERATIVE MODELS                                         #
#                                                                                                                           #
#                                                                                                                           #
#############################################################################################################################
library(tidyverse)

#----  Distance  ----
set.seed(0, sample.kind = "Rounding")
if(!exists("mnist")) mnist <- read_mnist()
ind <- which(mnist$train$labels %in% c(2,7)) %>% sample(500)
x <- mnist$train$images[ind,]
y <- mnist$train$labels[ind]

# Distance between observation 1 and 2 => dist(1,2) = sqrt(sum_j=1^784 (x_1,j - x_2,j)^2)
# Example: first three observations
y[1:3]

x_1 <- x[1,]
x_2 <- x[2,]
x_3 <- x[3,]

# Distance between 7 and 7
sqrt(sum((x_1 - x_2)^2))
# Distance between 7 and 2
sqrt(sum((x_1 - x_3)^2))
# Distance between 7 and 2
sqrt(sum((x_2 - x_3)^2))

# Using matrix algebra => crossproduct
sqrt(crossprod(x_1 - x_2))
sqrt(crossprod(x_1 - x_3))
sqrt(crossprod(x_2 - x_3))

# All distances with dist-function
d <- dist(x)
as.matrix(d)[1:3, 1:3]

image(as.matrix(d))
# Order distances by label
image(as.matrix(d)[order(y), order(y)])

# Distance between predictors => dist(1,2) = sqrt(sum_i=1^N (x_i,1 - x_i,2)^2)
# Compute the distance between all pairs of 784 predictors
# transpose the data and then use dist-function
d <- dist(t(x))
dim(as.matrix(d))



#----  Knn  ----
# k-nearest neighbors algorithm
#  knn is related to smoothing (bin smoothing)
#  look at the k-nearest points and then calculate an average
library(caret)
# GLM predictions as benchmark
fit_glm <- glm(y ~ x_1 + x_2, data = mnist_27$train, family = "binomial")
p_hat_logistic <- predict(fit_glm, mnist_27$test)
y_hat_logistic <- factor(ifelse(p_hat_logistic > 0.5, 7, 2))
confusionMatrix(data = y_hat_logistic, reference = mnist_27$test$y)$overall[1]

# knn predictions
knn_fit <- knn3(y ~ ., data = mnist_27$train, k = 5)
# or
x <- as.matrix(mnist_27$train[,2:3])
y <- mnist_27$train$y
knn_fit <- knn3(x, y, k = 5)
# Predict function for knn produces either a probability for each class or it actually produces the outcome
y_hat_knn <- predict(knn_fit, mnist_27$test, type = "class")
confusionMatrix(data = y_hat_knn, reference = mnist_27$test$y)$overall[1]


#----  Overtraining and Oversmoothing  ----
# Understanding for overtraining
y_hat_knn <- predict(knn_fit, mnist_27$train, type = "class")
confusionMatrix(data = y_hat_knn, reference = mnist_27$train$y)$overall[1]

y_hat_knn <- predict(knn_fit, mnist_27$test, type = "class")
confusionMatrix(data = y_hat_knn, reference = mnist_27$test$y)$overall[1]
# The overall accuracy is higher in the train set than in the test set

# Almost perfect (train) fit with k = 1 
knn_fit_1 <- knn3(y ~ ., data = mnist_27$train, k = 1)
y_hat_knn_1 <- predict(knn_fit_1, mnist_27$train, type = "class")
confusionMatrix(data = y_hat_knn_1, reference = mnist_27$train$y)$overall[1]

# But worse accuracy in the test set
y_hat_knn_1 <- predict(knn_fit_1, mnist_27$test, type = "class")
confusionMatrix(data = y_hat_knn_1, reference = mnist_27$test$y)$overall[1]

# larger k => oversmoothing, to smooth to predict the data correct
knn_fit_401 <- knn3(y ~ ., data = mnist_27$train, k = 401)
y_hat_knn_401 <- predict(knn_fit_401, mnist_27$test, type = "class")
confusionMatrix(data = y_hat_knn_401, reference = mnist_27$test$y)$overall[1]

# Choose of k
ks <- seq(3, 251, 2)
library(purrr)
accuracy <- map_df(ks, function(k){
 fit <- knn3(y ~ ., data = mnist_27$train, k = k)
 
 # Just for illustration purpose
 y_hat <- predict(fit, mnist_27$train, type = "class")
 
 train_error <- confusionMatrix(data = y_hat, reference = mnist_27$train$y)$overall[1]
 
 # real prediction on the test set
 y_hat <- predict(fit, mnist_27$test, type = "class")
 
 test_error <- confusionMatrix(data = y_hat, reference = mnist_27$test$y)$overall[1]
 
 list(train = train_error, test = test_error)
})

# Plot of the accuracy on k
accuracy %>% mutate(k = ks) %>% gather(key = "set", value = "accuracy", -k) %>%  
 ggplot(aes(k, accuracy, col = set)) + geom_line() + geom_point()

Acc <-  accuracy %>% mutate(k = ks)
k_best <- Acc$k[which.max(Acc$test)]
# But we choose k based on the test set which is not a good choice => cross-validation