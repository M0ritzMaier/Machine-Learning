#############################################################################################################################
#                                                                                                                           #
#                                                                                                                           #
#                                   MODEL FITTING AND RECOMMENDATION SYSTEMS                                                #
#                                                                                                                           #
#                                                                                                                           #
#############################################################################################################################

library(dslabs)
library(matrixStats)
library(tidyverse)
library(caret)
library(randomForest)
library(Rborist)

#----   Case Study: MNIST    ----
mnist <- read_mnist()

# Two components: train and test set
names(mnist)

# Each includes a matrix with features and vector of labels (y)
dim(mnist$train$images)
class(mnist$train$labels)
table(mnist$train$labels)

# Subset of the data set, due to computational time 
seed <- 123
if(R.Version()$version.string == "R version 3.6.0 (2019-04-26)"){
  set.seed(seed, sample.kind = "Rounding")
} else {
  set.seed((seed))
}

index <- sample(nrow(mnist$train$images), 10000)
x <- mnist$train$images[index,]
y <- factor(mnist$train$labels[index])

index <- sample(nrow(mnist$test$images), 1000)
x_test <- mnist$test$images[index,]
y_test <- mnist$test$labels[index]


#----   Preprocessing MNIST Data    ----
# Transform predictors before applying machine learning algorithm
#   e.g. standardizing or taking the log, ...
# Remove predictors that are clearly not useful
#   e.g. predictors that are highly correlated with others, few non-unique values or close to zero variation

# Features with zero or almost zero variability
sds <- colSds(x)
qplot(sds, bins = 256, color = I("black"))

# Near zero variability
nzv <- nearZeroVar(x)

# Features that remain
col_index <- setdiff(1:ncol(x), nzv)
length(col_index)


#----   Model Fitting for MNIST Data    ----
# Implement k-nearest neighbors and random forests
# Set names for columns (requirement of caret-package)
colnames(x) <- 1:ncol(mnist$train$images)
colnames(x_test) <- colnames(x)

# knn
# 1. Optimize for the number of neighbors
# Possible checks before running code on full data set
# Computational time
n <- 1000
b <- 2
index <- sample(nrow(x), n)
control <- trainControl(method = "cv", number = b, p = .9)
train_knn <- train(x[index, col_index], y[index],
                   method = "knn",
                   tuneGrid = data.frame(k = c(3,5,7)),
                   trControl = control)

# Cross-validation
control <- trainControl(method = "cv", number = 10, p = .9)
train_knn <- train(x[, col_index], y, 
                   method = "knn",
                   tuneGrid = data.frame(k = c(1,3,5,7)),
                   trControl = control)
train_knn
plot(train_knn)

# Fit the entire data set with the optimized parameters
fit_knn <- knn3(x[,col_index], y, k = as.numeric(train_knn$bestTune))

# Prediction
y_hat_knn <- predict(fit_knn, 
                     x_test[, col_index],
                     type = "class")

# Confusion matrix
cm <- confusionMatrix(data = y_hat_knn, factor(y_test))
cm
cm$byClass[,1:2]

# Random forests
# Use of Rborist-package: less features but faster compared to random-forest-package
control <- trainControl(method = "cv", number = 5, p = 0.8)
grid <- expand.grid(minNode = c(1,5), predFixed = c(10, 15, 25, 35, 50))

# train_rf <- train(x[, col_index], y,
#                   method = "Rborist",
#                   nTree = 50,
#                   trControl = control,
#                   tuneGrid = grid,
#                   nSamp = 5000)

modelLookup("rf")
grid <- expand.grid(mtry = c(10, 15, 25, 35, 50))

train_rf <- train(x[, col_index], y,
                  method = "rf",
                  ntree = 50,
                  trControl = control,
                  tuneGrid = grid,
                  sampsize = 5000,
                  nodesize = c(1,5))

qqplot(train_rf)

train_rf$bestTune


# fit
# fit_rf <- Rborist(x[, col_index], y,
#                   nTree = 1000,
#                   minNode = train_rf$bestTune$minNode,
#                   predFixed = train_rf$bestTune$predFixed)

fit_rf <- randomForest(x[, col_index], y,
                      ntree = 1000,
                      nodesize = 1,
                      mtry = train_rf$bestTune$mtry)

# Prediction
y_hat_rf <- predict(fit_rf, x_test[,col_index])
# Confusion matrix
cm <- confusionMatrix(y_hat_rf, as.factor(y_test))
cm
cm$byClass[,1:2]


#----   Variable Importance    ----
