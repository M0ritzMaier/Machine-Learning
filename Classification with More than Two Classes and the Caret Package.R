#############################################################################################################################
#                                                                                                                           #
#                                                                                                                           #
#                       CLASSIFICATION WITH MORE THAN TWO CLASSES AND THE CARET PACKAGE                                     #
#                                                                                                                           #
#                                                                                                                           #
#############################################################################################################################
library(tidyverse)
library(dslabs)
library(caret)
library(rpart)
library(randomForest)

#----           Trees Motivation                ----
# avoid the curse of dimensionality


#----           Classification and Regression Trees (CART)              ----
data("olive")
head(olive)

# predict the region using the fatty acid composition
table(olive$region)

olive <- select(olive, -area)

fit <- train(region ~ ., method = "knn", tuneGrid = data.frame(k = seq(1,15,2)), data = olive)
fit
ggplot(fit)

olive %>% gather(fatty_acid, percentage, -region) %>%
 ggplot(aes(region, percentage, fill = region)) +
 geom_boxplot() +
 facet_wrap(~fatty_acid, scales = "free") +
 theme(axis.text.x = element_blank())


olive %>% ggplot(aes(eicosenoic, linoleic, col = region)) + geom_point()

# Decision rule: if eicosenoic is larger than 0.065 predict Southern Italy else look at lineoleic and if this is larger than
# 10.535 predict Sardinia else Northern Italy

# Decision Tree: Define an algortihm that uses data to create trees
#  Outcome is continuous: Regression Tree

# Polls example
#  Partition feature space into J non-overlapping regions
#  Every observation that falls within a region x_i %in% R_j predict Y_hat with the average of the training observations Y_i
#  in region: x_i %in% R_j
#   Regression trees create partition recursively

# Define a predictor j and a value s to define two new partitions R_1 and R_2
#  R_1(j,s) = {X| X_j < s}
#  R_2(j,s) = {X| X_j => s}
# Define an average y_bar,R_1 and y_bar,R_2
# => Pick j and s so that the residual sum of squares is minimized
#    sum_i: x_i %in% R_1(j,s) (y_i - y^hat_R_1)^2 + sum_i: x_i %in% R_2(j,s) (y_i - y^hat_R_2)^2
#   => Applied recursively: find new region to split into two partitions

fit <- rpart(margin ~ ., data = polls_2008)
plot(fit, margin = 0.1)
text(fit, cex = 0.75)

polls_2008 %>% 
 mutate(y_hat = predict(fit)) %>% 
 ggplot() +
 geom_point(aes(day, margin)) +
 geom_step(aes(day, y_hat), col="red")

# compexity parameter (cp): algorithm requires a minimum of residual sum of squares improvement by cp
# sets a minimum number of observations to be pratitioned (minsplit)
# sets a minimum number of observations in each partition (minbucket)

# overfit example
fit <- rpart(margin ~ ., data = polls_2008, cp = 0, minsplit = 2)

polls_2008 %>% 
 mutate(y_hat = predict(fit)) %>% 
 ggplot() +
 geom_point(aes(day, margin)) +
 geom_step(aes(day, y_hat), col="red")

# prune can be used to cut off small trees

pruned_fit <- prune(fit, cp = 0.01)

polls_2008 %>% 
 mutate(y_hat = predict(pruned_fit)) %>% 
 ggplot() +
 geom_point(aes(day, margin)) +
 geom_step(aes(day, y_hat), col="red")

# Choice of cp => use cross validation
train_rpart <- train(margin ~., data = polls_2008, method = "rpart", tuneGrid = data.frame(cp = seq(0, 0.05, len = 25)))
ggplot(train_rpart)

plot(train_rpart$finalModel, margin = 0.1)
text(train_rpart$finalModel, cex = 0.75)


polls_2008 %>% 
 mutate(y_hat = predict(train_rpart$finalModel)) %>% 
 ggplot() +
 geom_point(aes(day, margin)) +
 geom_step(aes(day, y_hat), col="red")

#----           Classification (Decision) Trees              ----
# outcome is categorial
#  predict the class that appears the most in a node
#  we can no longer use residual sum of squares to decide on the partition
#  => instead we use Gini Index or Entropy

# p^hat_m,k: proportion of observations in partition m that are of class k
#  Gini = sum_k=1^K p^hat_m,k (1 - p^hat_m,k)
#  Entropy = - sum_k=1^K p^hat_m,k log(p^hat_m,k); with 0 * log(0) defined as 0
#  => Both metrics seek to partition observations into subsets that have the same class => purity

train_rpart <- train(y ~ ., method = "rpart", tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25)),
                     data = mnist_27$train)
plot(train_rpart)
confusionMatrix(data = predict(train_rpart, mnist_27$test), reference = mnist_27$test$y)

# Advantages of Decision Trees
#  1. Highly interoperable
#  2. Easy to visualize (if they are small enough)
#  3. Sometimes model human decision processes

# Disadvantages
#  1. Greedy approach via recursive paritioning is a bit harder to train
#  2. May not best performing due to a lack of flexibility
#  3. Instabil regarding new training data


#----           Random Forests              ----
# Basic idea: Improve prediction performance and reduce instability by averaging multiple decision trees, a forest of trees
#             constructed with randomness.
# Two features to accomplish this
#  bootstrap aggregation or bagging
#    1. Build many decision trees T_1, ..., T_B, using the training set
#    2. For every observation j in the test set, form a prediction y^hat_j using tree T_j
#  => Final predictions combines the predictions for each tree
#      for continuous outcomes y^hat = 1/B sum_j=1^B y^hat_j
#      for categorial data predict y^hat with majority vote (most frequent class among y^hat_1 ,... y^hat_T)

# To get many trees from the training set, we use the bootsrap
# Create tree T_j from a training set of size N, we sample N observations from the training set with replacement
# Build a decision tree for each bootstrap training set
fit <- randomForest(margin ~ ., data = polls_2008)
plot(fit)

polls_2008 %>% mutate(y_hat = predict(fit)) %>% ggplot() + geom_point(aes(day, margin)) + 
 geom_line(aes(day, y_hat), col = "red")


train_rf <- randomForest(y ~ ., data = mnist_27$train)
confusionMatrix(data = predict(train_rf, mnist_27$test), mnist_27$test$y)

# optimize parameters
fit <- train(y ~ ., data = mnist_27$train,
             method = "Rborist", tuneGrid = data.frame(predFixed = 2, minNode = seq(3,50)))
fit
confusionMatrix(data = predict(fit, mnist_27$test), mnist_27$test$y)

# Controlling the smoothness
#  1. size of each node => number of points per node larger
#  2. Random selesction of features to use for the splits, specifically when building each tree at each recursive partition,
#     we only consider a randomly selected subset of predictors to check for the best split. Every tree has different random
#     selection of features
#     => Reduces correlation between trees in the forest
#     => Improves prediction accuracy

# Disadvantage of random forests
#  - lose interpretability, by averaging many, many trees
# however variable importants helps us to interpret the results
#  => How much each predicor influences the final predictions