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


#----   Regularization    ----
# Despite large movie to movie variation, improvement only about 5%
# 10 of the largest mistakes when only using movie effects
test_set %>% left_join(movie_avgs, by = "movieId") %>% mutate(residual = rating - (mu+b_i)) %>%
  arrange(desc(abs(residual))) %>% select(title, residual) %>% slice(1:10) %>% knitr::kable()

# Top ten best and worst movies based on the estimate of movie effects b^hat_i
# Movie titles
movie_titles <- movielens %>% select(movieId, title) %>% distinct()

# Best 10
movie_avgs %>% left_join(movie_titles, by = "movieId") %>% arrange(desc(b_i)) %>% select(title, b_i) %>% slice(1:10) %>%
  knitr::kable()

# Worst 10
movie_avgs %>% left_join(movie_titles, by = "movieId") %>% arrange(b_i) %>% select(title, b_i) %>% slice(1:10) %>%
  knitr::kable()
# => all worse are obscure movies

# Times they are rated
train_set %>% count(movieId) %>% left_join(movie_avgs) %>% left_join(movie_titles, by = "movieId") %>%
  arrange(desc(b_i)) %>% select(title, b_i, n) %>% slice(1:10) %>% knitr::kable()

train_set %>% count(movieId) %>% left_join(movie_avgs) %>% left_join(movie_titles, by = "movieId") %>%
  arrange(b_i) %>% select(title, b_i, n) %>% slice(1:10) %>% knitr::kable()
# => Supposed best and worst movies are only rated a few times, mostly once
# => noisy estimates

# Regularization deals with this problem
#   penalize large estimates that came from small sample sizes => Penalized Least Squares
#   Idea: Add a large penalty for large valzes of b to the sum of squares equation that we minimize
#   => Minimize this equation: 1/N sum_u,i (y_u,i - mu - b_i)^2 + lambda * sum_i b_i^2
#       with penalty term lambda * sum_i b_i^2 => gets larger when many b's are large
#     Values of b that minimize the equation: b-hat_i(lambda) = 1/(lambda + n_i) sum_u=1^n_i (Y_u,i- mu-hat)
lambda <- 3
mu <- mean(train_set$rating)
movie_reg_avgs <- train_set %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu)/(n()+lambda) , n_i = n())

# Plot of regularized estimates vs. the least squared estimates, with circle size given by n_i
data_frame(original = movie_avgs$b_i, 
           regularlized = movie_reg_avgs$b_i, 
           n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)

# Top ten movies based on regularization estimates
train_set %>% count(movieId) %>% left_join(movie_reg_avgs) %>% left_join(movie_titles, by = "movieId") %>%
  arrange(desc(b_i)) %>% select(title, b_i, n) %>% slice(1:10) %>% knitr::kable()

# Worst ten movies based on regularization estimates
train_set %>% count(movieId) %>% left_join(movie_reg_avgs) %>% left_join(movie_titles, by = "movieId") %>%
  arrange(b_i) %>% select(title, b_i, n) %>% slice(1:10) %>% knitr::kable()

# Prediction with penalized least squares
predicted_ratings <- test_set %>% left_join(movie_reg_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>% .$pred

model_3_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results, 
                          data_frame(method = "Regularized Movie Effect", RMSE = model_3_rmse))

rmse_results %>% knitr::kable()

# lambda is a tuning parameter => we can use cross validation to choose it
lambdas <- seq(0, 10, 0.25)

mu <- mean(train_set$rating)
just_the_sum <- train_set %>% group_by(movieId) %>% summarize(s = sum(rating - mu), n_i = n())

rmses <- sapply(lambdas, function(l) {
  predicted_ratings <- test_set %>% left_join(just_the_sum, by = "movieId") %>%
    mutate(b_i = s/(n_i + l)) %>% mutate(pred = mu + b_i) %>% .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, rmses)
lambdas[which.min(rmses)]
# Just an example, we should use cross validation just on the training set without using the test set

# Regularization for movie and user effects
#   Minimize  1/N sum_u,i (y_u,i - mu - b_i - b_u)^2 + lambda * (sum_i b_i^2 + sum_u b_u^2)
#     with user effect sum_u b_u^2
#   => The estimate that minimize can be found similarly to the movie effect solution

# cross validation to find lambda
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, rmses)  
lambdas[which.min(rmses)]

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User Effect Model",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()


#----   Matrix Factorization    ----
# Related to factor analysis, singular value decomposition (SVD) and principaö component analysis (PCA)
# Groups of movies have similar rating patterns and groups of users have similar rating patterns
#   => Discover these pattern by studying the residuals
#       r_u,i = y_u,i - b-hat_i - b-hat_u

# Convert data into matrix, so that each user gets a row and each movie gets a column
# Subsample for illustration
train_small <- movielens %>% 
  group_by(movieId) %>%
  filter(n() >= 50 | movieId == 3252) %>% ungroup() %>% 
  group_by(userId) %>%
  filter(n() >= 50) %>% ungroup()

y <- train_small %>% 
  select(userId, movieId, rating) %>%
  spread(movieId, rating) %>%
  as.matrix()

# Row and column names
rownames(y)<- y[,1]
y <- y[,-1]

movie_titles <- movielens %>% 
  select(movieId, title) %>%
  distinct()

colnames(y) <- with(movie_titles, title[match(colnames(y), movieId)])

# Calculate residulas by substracting movie (colmeans) and user effects (rowmeans)
y <- sweep(y, 2, colMeans(y, na.rm=TRUE))
y <- sweep(y, 1, rowMeans(y, na.rm=TRUE))

# If the model (so far) includes all the signals then the residuals for different movies should be independent from each 
# other (noise) 
m_1 <- "Godfather, The"
m_2 <- "Godfather: Part II, The"
qplot(y[ ,m_1], y[,m_2], xlab = m_1, ylab = m_2)
# => Residuals ar correlated, if the like 'The Godfather' users are likely to like 'The Godfather II'
m_3 <- "Goodfellas"
qplot(y[ ,m_1], y[,m_3], xlab = m_1, ylab = m_3)
# => same applies for 'Goodfellas'
# or for those movies
m_4 <- "You've Got Mail" 
m_5 <- "Sleepless in Seattle" 
qplot(y[ ,m_4], y[,m_5], xlab = m_4, ylab = m_5)

# Pairwise correaltion for those movies
x <- y[, c(m_1, m_2, m_3, m_4, m_5)]
colnames(x)[1:2] <- c("Godfather", "Godfather 2")
cor(x, use="pairwise.complete")

#   => Stucture in the data, which the model (so far) do not account for
#   To model this we use matrix factorization
#     Define factors: We factorize the matrix r_u,i into two things a vector p and a vector q
#       r_u,i approx p_u * q_i (or p_u,1 * q_i,1 + p_u,2 * q_i,2 + ...+ p_u,N * q_i,N)
#       Reduce movies to few groups e.g. by genre
#       Reduce user to few groups e.g. those that like genre A and those like genre B and some how like both

# New model: Y_u,i = mu + b_i + b_u + p_u q_i + e_u,i

# How to estimate factors from data => one way fit models or SVD / PCA


#----   SVD and PCA    ----
# Find the vectors p and q that can decompose the residual matrix in the form above with the bonus that the  
# variability of these terms is decreasing with each factor and that the p's are uncorrelated to each other

# Example with movie data
# Convert NA to 0
y[is.na(y)] <- 0
y <- sweep(y, 1, rowMeans(y))

# PCA
pca <- prcomp(y)
# => Vector q are the principal components stored in pca$rotation
dim(pca$rotation)
# => p values = user effects are stored in pca$x
dim(pca$x)

plot(pca$sdev)

# Variance explained (of the residual variation)
var_explained <- cumsum(pca$sdev^2/sum(pca$sdev^2))
plot(var_explained)


# Plot of first two principal components with labels of the movies
library(ggrepel)
pcs <- data.frame(pca$rotation, name = str_trunc(colnames(y), 30))

highlight <- filter(pcs, PC1 < -0.1 | PC1 > 0.1 | PC2 < -0.075 | PC2 > 0.1)

pcs %>%  ggplot(aes(PC1, PC2)) + geom_point() + 
  geom_text_repel(aes(PC1, PC2, label=name),
                  data = highlight, size = 2)

pcs %>% select(name, PC1) %>% arrange(PC1) %>% slice(1:10) # critically acclaimed
pcs %>% select(name, PC1) %>% arrange(desc(PC1)) %>% slice(1:10) # blockbusters


pcs %>% select(name, PC2) %>% arrange(PC2) %>% slice(1:10) # independent films
pcs %>% select(name, PC2) %>% arrange(desc(PC2)) %>% slice(1:10) # nerd favorits


# Further info in regard to fitting a ,pdell that incorporates these estimates => recommenderlab package
