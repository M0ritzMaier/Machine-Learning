#############################################################################################################################
#                                                                                                                           #
#                                                                                                                           #
#                            BASICS OF EVALUATING MACHINE LEARNING ALGORITHMS                                               #
#                                                                                                                           #
#                                                                                                                           #
#############################################################################################################################

#----  Overall accuracy  ----
library(dslabs)
library(tidyverse)
library(caret)
data(heights)

# Outcome
y <- heights$sex

# Features
x <- heights$height

set.seed(2)
# Random index to divide train and test set
test_index <- createDataPartition(y, times = 1, p = 0.5, list = F)

train_set <- heights[-test_index,]
test_set <- heights[test_index,]

## just guessing
y_hat <- sample(c("Male", "Female"), length(test_index), replace = T) %>% factor(levels = levels(test_set$sex))

# overall accuracy
mean(y_hat == test_set$sex)


## Within two sd away from average male
y_hat <- ifelse(x > 62, "Male", "Female")

mean(y == y_hat)

## Verschiedene Cut-offs
cutoff <- seq(61,70)
accuracy <- map_dbl(cutoff, function(x){
 y_hat <- ifelse(train_set$height > x, "Male", "Female") %>% factor(levels = levels(test_set$sex))
 mean(y_hat == train_set$sex)
})
ggplot() + geom_line(aes(cutoff, accuracy))

# Wahl des Cutoffs mit der hoechsten Genauigkeit
best_cutoff <- cutoff[which.max(accuracy)]

y_hat <- ifelse(test_set$height > best_cutoff, "Male", "Female") %>% factor(levels = levels(test_set$sex))
y_hat <- factor(y_hat)
mean(y_hat == test_set$sex)


#----  Confusion matrix  ----  
table(predicted = y_hat, actual = test_set$sex)

# Accruacy for each sex separately
test_set %>% mutate(y_hat = y_hat) %>% group_by(sex) %>% summarize(accuracy = mean(y_hat == sex))

# Prevelance
#  reports a bias for males in the data set
prev <- mean(y == "Male")
prev

# Confusion matrix
#                       Actually positive      Actually negative
#  Predicted positive   True positives (TP)    False positives (FP)
#  Predivted negative   False negatives (FN)   True negatives (TN)

# Sensitivity: High sensitivity Y = 1 => Y_hat = 1
#  TP/(TP + FN) => True positive rate (TPR) - Recall

# Specificity:  High specificity Y = 0 => Y_hat = 0 or Y_hat = 1 => Y = 1
#  TN/(FP + TN) => True negative rate (TNR) or TP/(TP + FP) => precision - positive predictive value (PPV)

confusionMatrix(data = y_hat, reference = test_set$sex)


#----  Balanced accuracy and F1 score  ----
# average of sensitivity and specificity
# harmonic average F_1-score = 1/(0.5 * (1/recall + 1/precision)) or 2 * (precision * recall)/(precision + recall)

# If either sensitivity or specificity is more important
# weighted harmonic average 1/(beta^2/(1+beta^2) * 1/recall + 1/(1+beta^2) * 1/precision)

# maximizing the F-score
cutoff <- seq(61,70)
F_1 <- map_dbl(cutoff, function(x){
 y_hat <- ifelse(train_set$height > x, "Male", "Female") %>% factor(levels = levels(test_set$sex))
 F_meas(data = y_hat, reference = factor(train_set$sex))
})
ggplot() + geom_line(aes(cutoff, F_1)) + geom_point(aes(cutoff, F_1))
max(F_1)

best_cutoff <- cutoff[which.max(F_1)]
best_cutoff

y_hat <- ifelse(test_set$height > best_cutoff, "Male", "Female") %>% factor(levels = levels(test_set$sex))
confusionMatrix(data = y_hat, reference = test_set$sex)


#----  ROC and precision-recall curves  ----
# Male by guessing (with higher propability like in the given data set)
p <- 0.9
y_hat <- sample(c("Male", "Female"), length(test_index), replace = T, prob = c(p, 1-p)) %>% factor(levels = levels(test_set$sex))
mean(y_hat ==test_set$sex) # higher specificity compared to 0.5 but lower sensitivity

# Receiver operating characteristics (ROC) curve
#  plots sensitivity (TPR) versus 1 - specificity (or false positive rate - FPR)
cutoffs <- c(50, seq(60,75), 80)
height_cutoff <- map_df(cutoffs, function(x){
 y_hat <- ifelse(test_set$height > x, "Male", "Female") %>% factor(levels = levels(test_set$sex))
 list(method = "Height cutoff",
      FPR = 1-specificity(y_hat, test_set$sex),
      TPR = sensitivity(y_hat, test_set$sex))
})
height_cutoff %>% ggplot(aes(FPR, TPR)) + geom_line() + geom_point() + geom_text(label = cutoffs)

# Neither of the measures plotted depend on prevalence
# If prevalence matters => precision-recall plot
guessing <- map_df(seq(0, 1, 0.1), function(p){
 y_hat <- sample(c("Male", "Female"), length(test_index), replace = T, prob = c(p, 1-p)) %>% factor(levels = levels(test_set$sex))
 list(method = "Guessing",
      recall = sensitivity(y_hat, test_set$sex),
      precision = precision(y_hat, test_set$sex))
})

height_cutoff <- map_df(cutoffs, function(x){
 y_hat <- ifelse(test_set$height > x, "Male", "Female") %>% factor(levels = levels(test_set$sex))
 list(method = "Height cutoff",
      recall = sensitivity(y_hat, test_set$sex),
      precision = precision(y_hat, test_set$sex))
})

height_cutoff <- rbind(guessing, height_cutoff)

height_cutoff %>% ggplot(aes(recall, precision, col = method)) + geom_line() + geom_point()
