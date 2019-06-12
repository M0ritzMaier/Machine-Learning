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
# Limitation of Random Forests: not very interpretable
# Variable importance can help
rf <- randomForest(x, y, ntree = 50)

imp <- importance(rf)

# Visualization of importance of the features
image(matrix(imp, 28, 28))


#----   Ensembles    ----
# Idea of ensembling different machine learning algorithms into one
#  e.g. Compute new class probabilities by taking the average of the class probabilities provided by knn and random forests
p_rf <- predict(fit_rf, x_test[,col_index], type = "prob")
p_rf<- p_rf / rowSums(p_rf)
p_knn  <- predict(fit_knn, x_test[,col_index], type = "prob")
p <- (p_rf + p_knn)/2
y_pred <- factor(apply(p, 1, which.max)-1)
confusionMatrix(y_pred, factor(y_test))


#----   Recommendation Systems    ----
# Example Netflix challenge => Improve recommendation system by 10% and win 1 million dollar
data("movielens")

# Unqiue users that provide ratings and unique movies
movielens %>% summarize(n_users = n_distinct(userId), n_movies = n_distinct(movieId))

seed <- 755
if(R.version$version.string  == "R version 3.6.0 (2019-04-26)"){
 set.seed(seed, sample.kind = "Rounding")
} else {
 set.seed(seed)
}

test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.2, list = F)

train_set <- movielens[-test_index,]
test_set <- movielens[test_index,]

# make sure that we do not include users and movies in the test set that do not appear in the train set
test_set <- test_set %>% semi_join(train_set, by = "movieId") %>% semi_join(train_set, by = "userId")

# To compare different models => what means do well?
#  Netflix challange: Typical error => winner based on residual squared error
#   y_u,i: rating for movie i by user u
#   y^hat_u,i: prediction
#   RMSE = sqrt(1/N sum_u,i (y_u,i - y^hat_u,i)^2) with n numerb of user movie combination
RMSE <- function(true_ratings, predicted_ratings){
 sqrt(mean((true_ratings - predicted_ratings)^2))
}


#----   Building the Recommendation Systems    ----
# Matrix factorization

# Start: simplest recommendation model => same rating for all movies
# y_u,i = mu + e_u,i
mu_hat <- mean(train_set$rating)
mu_hat

naive_rmse <- RMSE(test_set$rating, mu_hat)
naive_rmse

# Result table
rmse_results <- data.frame(method = "Just the average", RMSE = naive_rmse)


# Average rating of each movie
#  y_u,i = mu + b_i + e_u,i with b_i average rating for movie i => effects or bias
fit <- lm(rating ~ as.factor(userId), data = movielens)

# Least squared estimate b^hat_i is in this situation just the average of y_u,i - mean(y)
mu <- mean(train_set$rating)
movie_avgs <- train_set %>% group_by(movieId) %>% summarize(b_i = mean(rating - mu))

ggplot(movie_avgs, aes(b_i)) + geom_histogram(bins = 20)

predicted_ratings <- mu + test_set %>% left_join(movie_avgs, by = "movieId") %>% .$b_i

model_1_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results, data_frame(method = "Movie Effect Model", RMSE = model_1_rmse))

rmse_results %>% knitr::kable()


# Average user effect (rated over 100 movies)
#  y_u,i = mu + b_i + b_u + e_u,i with b_u as user-specific effect
# fit either with lm or
# approximation 
user_avgs <- train_set %>% left_join(movie_avgs, by = "movieId") %>% group_by(userId) %>% 
 summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- test_set %>% left_join(movie_avgs, by = "movieId") %>% left_join(user_avgs, by = "userId") %>%
 mutate(pred = mu + b_i + b_u) %>% .$pred

model_2_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results, data_frame(method = "Movie + User Effects Model", RMSE = model_2_rmse))

rmse_results %>% knitr::kable()
