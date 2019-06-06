#############################################################################################################################
#                                                                                                                           #
#                                                                                                                           #
#                            DISTANCE, KNN, CROSS-VALIDATION, AND GENERATIVE MODELS                                         #
#                                                                                                                           #
#                                                                                                                           #
#############################################################################################################################
library(tidyverse)

#----           Distance                ----
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



#----           Knn             ----
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


#----           Overtraining and Oversmoothing          ----
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


#----           k-fold cross-validation         ----
# Often task in machine learning minimizing mean squared error
#       true error: MSE = E{1/N sum_i=1^N (Y_i^hat - Y_i)^2}
#       apparent error: MSE^hat = 1/N sum_i=1^N (y_i^hat - y_i)^2       => when you only have one data set
#               two characteristics to keep in mind:
#               1. apparent error is a random variable => algorithm with a lower apparent error may be due to luck
#               2. If the algorithm is trained on the same data, we be prone to overtraining

# The true error can be thought as average of many, many apparent errors obtained by applying the algorithm to B new random
# samples (none of which is used to the train the algorithm).
#       true error = 1/B sum_b=1^B 1/N sum_i=1^N (y_i^hat,b - y_i^b)^2
#       => Basic idea randomly generate smaller data sets that are not used for training and instead are used to estimate 
#          the true error

# typical choice of test sets are 10-20% of the original data set
# set of parameters: lambda
#       => optimize the algorithm paramters lambda without using the test set

# expected loss
#       MSE(lambda) = 1/B sum_b=^B 1/N sum_i=1^N (y_i^hat,b(lambda) - y_i^b)^2
#       => average of MSE over B samples

# Construct the first sample (obtained from the original train data, so the train set is devided into train and validate)
#       validation set: with observations M = N/K => rounded to the nearest integer
#       MSE_b^hat(lambda) = 1/M sum_i=1^M (y_i^hat,b(lambda) - y_i^b)^2
#       in k-fold cross-validation: randomly split the observations into k non-overlapping sets
#       => Use cross-validation to optimize model parameters

# Final step select lambda that minimizes the MSE
# however, optimization of lambda and therefore MSE is still on the train set
#       => Use test set to estimate accuracy and calculate final estimate of MSE

# Choice of k
#       the more k the better => better estimate
#       but the higher k the larger is the computation time
#       => k = 5 or k = 10 are quite popular

# One way to improve the variance of our final estimate
#       take more samples
#       to do this we can use overlapping training sets
#       => just pick k sets of some size at random
#               at each fold pick observations at random with replacement => bootstrap


#----           Bootstrap               ----
if(R.version$version.string  == "R version 3.6.0 (2019-04-26)"){
        set.seed(1, sample.kind = "Rounding")
} else {
        set.seed(1)
}
n <- 10^6
income <- 10^(rnorm(n, log10(45000), log10(3)))
qplot(log10(income), bins = 30, color = I("black"))

N <- 250
X <- sample(income, N)
M <- median(X)
M

B <- 10^5
Ms <- replicate(B, {
        X <- sample(income, N)
        M <- median(X)
})

library(gridExtra)
p1 <- qplot(Ms, bins = 30, color = I("black"))
p2 <- qplot(sample = scale(Ms)) + geom_abline()
grid.arrange(p1, p2, ncol = 2)

mean(Ms)
sd(Ms)

# bootstrap: Act as if the sample is the entire population and sample with replacement data sets of the same size
#       => Calculate summary statistics
#       => The distribution of the statistic obtained by the bootstrap samples apporximate the distribution of our 
#          actual statistic

B <- 10^5
M_stars <- replicate(B, {
        X_star <- sample(X, N, replace = T)
        M_star <- median(X_star)
})
qqplot(Ms, M_stars)

# Bootstrap confidence intervall
quantile(Ms, c(0.05, 0.95))
quantile(M_stars, c(0.05, 0.95))

# Confidence interval using the central limit theorem
median(X) + 1.96 * sd(X)/sqrt(N) * c(-1,1)


#----           Generative Models               ----
# In a binary case the best approach to developing a decision rule is to follow Bayes' rule
#       => Dicision rule based on the true conditional probability p(x) = Pr(Y = 1 | X = x)
#       dicrimitive approach: estimate the conditional probability without considering the distribution of predictors
#       generative methods: models the joint distribution of y and the predictors x

# naive Bayes
# Bayes Theorem: p(x) = Pr(Y = 1 | X = x) = f_X|Y=1 (x) Pr(Y = 1) / [f_X|Y=0 (x) Pr(Y = 0) + f_X|Y=1 (x) Pr(Y = 1)]
#       with f(.) representing the distribution functions of the predictors x


#----           Naive Bayes             ----
library(caret)
data("heights")
y <- heights$height
set.seed(2)
test_index <- createDataPartition(y, times = 1, p = 0.5, list = F)
train_set <- heights %>% slice(-test_index)
test_set <- heights %>% slice(test_index)

# conditional distribution
params <- train_set %>% group_by(sex) %>% summarize(avg = mean(height), sd = sd(height))
params

# prevelance: pi = Pr(Y = 1) => Proportion of females
pi <- train_set %>% summarize(pi = mean(sex == "Female")) %>% .$pi
pi

x <- test_set$height

f0 <- dnorm(x, params$avg[2], params$sd[2]) # male
f1 <- dnorm(x, params$avg[1], params$sd[1]) # female

p_hat_bayes <- f1 * pi / (f1*pi + f0 * (1-pi))

qplot(x, p_hat_bayes)


#----           Controlling prevalence          ----
# naive Bayes includes a parameter to account for differences in prevalance (pi)
y_hat_bayes <- ifelse(p_hat_bayes > 0.5, "Female", "Male")
sensitivity(data = factor(y_hat_bayes), reference = factor(test_set$sex))

# Algorithm gives more weight to specificity to account for low prevalence => pi_hat < 0.5
specificity(data = factor(y_hat_bayes), reference = factor(test_set$sex))

# Balance senstitivity and specificity with pi
pi <- .5
p_hat_bayes_unbiased <- f1 * pi / (f1*pi + f0 * (1-pi))
y_hat_bayes_unbiased <- ifelse(p_hat_bayes_unbiased > 0.5, "Female", "Male")

sensitivity(data = factor(y_hat_bayes_unbiased), reference = factor(test_set$sex))
specificity(data = factor(y_hat_bayes_unbiased), reference = factor(test_set$sex))

qplot(x, p_hat_bayes_unbiased)


#----           QDA and LDA             ----
# Quadratic discriminant analysis (QDA) is a version of Naive Bayes with (assumption of) multivariate normal distributions of
# the conditional probabilities of the predictors

data("mnist_27")
# two predictors => assume that conditional probability is bivariate normal
#       => estimate two averages, two sd and correlation for each case (7,2)
#       conditional distributions
#       => f_X_1,X_2|Y = 1 and f_X_1,X_2|Y = 0

params <- mnist_27$train %>% group_by(y) %>% summarize(avg_1 = mean(x_1), avg_2 = mean(x_2),
                                                       sd_1 = sd(x_1), sd_2 = sd(x_2),
                                                       r = cor(x_1,x_2))
params

mnist_27$train %>% mutate(y = factor(y)) %>% 
        ggplot(aes(x_1, x_2, fill = y, color=y)) + 
        geom_point(show.legend = FALSE) + 
        stat_ellipse(type="norm", lwd = 1.5)

library(caret)
train_qda <- train(y ~ ., data = mnist_27$train, method = "qda")
y_hat <- predict(train_qda, mnist_27$test)
confusionMatrix(data = y_hat, reference = mnist_27$test$y)

mnist_27$train %>% mutate(y = factor(y)) %>% 
        ggplot(aes(x_1, x_2, fill = y, color=y)) + 
        geom_point(show.legend = FALSE) + 
        stat_ellipse(type="norm") +
        facet_wrap(~y)

# QDA becomes harder if number of predictors increase
#       => Parameters to estimate: K * (2p + p * (p-1) / 2)

# Potential solution: Assume that the correlation structure is the same across all classes
params <- params %>% mutate(sd_1 = mean(sd_1), sd_2 = mean(sd_2), r = mean(r))
params

# Linear discriminant analysis (LDA)
# because of the assumption of the same sd and r, the boundary is a line (like in the logistic regression)
train_lda <- train(y ~ ., data = mnist_27$train, method = "lda")
y_hat <- predict(train_lda, mnist_27$test)
confusionMatrix(data = y_hat, reference = mnist_27$test$y)


#----           Case study: More than three classes             ----
mnist <- read_mnist()
set.seed(3456)
index_127 <- sample(which(mnist$train$labels %in% c(1,2,7)), 2000)

y <- mnist$train$labels[index_127]
x <- mnist$train$images[index_127,]

index_train <- createDataPartition(y, p = 0.8, times = 1, list = F)

# get quadrants
row_column <- expand.grid(row = 1:28, col = 1:28)
upper_left_ind <- which(row_column$col <= 14 & row_column$row <= 14)
lower_right_ind <- which(row_column$col > 14 & row_column$row > 14)

x <- x > 200
# binarize values
x <- cbind(rowSums(x[, upper_left_ind])/rowSums(x),
           # proportion of pixels in upper right quadrant)
           rowSums(x[, lower_right_ind])/rowSums(x))

train_set <- data.frame(y = factor(y[index_train]),
                        x_1 = x[index_train, 1],
                        x_2 = x[index_train, 2])

test_set <- data.frame(y = factor(y[-index_train]),
                        x_1 = x[-index_train, 1],
                        x_2 = x[-index_train, 2])

train_set %>% ggplot(aes(x_1, x_2, col = y)) + geom_point()

# QDA model
train_qda <- train(y ~ ., method = "qda", data = train_set)
predict(train_qda, test_set, type = "prob") %>% head()

predict(train_qda, test_set)

confusionMatrix(data = predict(train_qda, test_set), reference = test_set$y)

# LDA model
train_lda <- train(y ~ ., method = "lda", data = train_set)
confusionMatrix(data = predict(train_lda, test_set), reference = test_set$y)

test_set %>% mutate(y_hat = predict(train_lda, test_set)) %>% ggplot(aes(x_1, x_2, col = y_hat)) + geom_point()

# knn
train_knn <- train(y ~ ., method = "knn", tuneGrid = data.frame(k = seq(15,51,2)),
                   data = train_set)
confusionMatrix(data = predict(train_knn, test_set), reference = test_set$y)
test_set %>% mutate(y_hat = predict(train_knn, test_set)) %>% ggplot(aes(x_1, x_2, col = y_hat)) + geom_point()
# Generative models can be powerful but only when we are able to successfully qppoximate the joint distribution of the 
# predictor's condition on each class