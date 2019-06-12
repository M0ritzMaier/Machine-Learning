#----   Comprehension Check: Ensembles    ----
library(dslabs)
library(tidyverse)
library(caret)
library(rpart)
library(randomForest)
library(Rborist)

#---    Q1      ----
# Train models
models <- c("glm", "lda",  "naive_bayes",  "svmLinear", 
            "gamboost",  "gamLoess", "qda", 
            "knn", "kknn", "loclda", "gam",
            "rf", "ranger",  "wsrf", "Rborist", 
            "avNNet", "mlp", "monmlp",
            "adaboost", "gbm",
            "svmRadial", "svmRadialCost", "svmRadialSigma")

seed <- 1
if(R.version$version.string  == "R version 3.6.0 (2019-04-26)"){
        set.seed(seed, sample.kind = "Rounding")
} else {
        set.seed(seed)
}
data("mnist_27")

fits <- lapply(models, function(model){ 
        print(model)
        train(y ~ ., method = model, data = mnist_27$train)
}) 

names(fits) <- models


#---    Q2      ----
# Predictions
y_hats <- sapply(fits, function(f){
        predict(f, mnist_27$test, class = "raw")
})


#---    Q3      ----
# Accuracy
acc <- apply(y_hats, 2, function(a){
        confusionMatrix(data = as.factor(a), reference = mnist_27$test$y)$overall["Accuracy"]
})
mean(acc)


#---    Q4      ----
# Ensemble by majority
y_pred <- apply(y_hats, 1, function(x){
        names(table(x))[table(x) == max(table(x))]
})

confusionMatrix(data = as.factor(y_pred), reference = mnist_27$test$y)$overall["Accuracy"]


#---    Q5      ----
# Single model that performs better than ensembled model
length(acc[acc > confusionMatrix(data = as.factor(y_pred), reference = mnist_27$test$y)$overall["Accuracy"]])
acc[acc > confusionMatrix(data = as.factor(y_pred), reference = mnist_27$test$y)$overall["Accuracy"]]
# Achtung Abweichung in den Antworten Models 1 und loclda


#---    Q6      ----
# With corss validation
fits <- lapply(models, function(model){ 
        print(model)
        train(y ~ ., method = model, data = mnist_27$train, trControl = trainControl("cv"))
}) 

names(fits) <- models

y_hats <- sapply(fits, function(f){
        predict(f, mnist_27$test, class = "raw")
})


acc <- apply(y_hats, 2, function(a){
        confusionMatrix(data = as.factor(a), reference = mnist_27$test$y)$overall["Accuracy"]
})
mean(acc)


#---    Q7      ----
# Ensembel with models with accuracy >= 0.8
cutoff <- 0.8

ind <- acc >= cutoff

y_pred <- apply(y_hats[,ind], 1, function(x){
        names(table(x))[table(x) == max(table(x))]
})

confusionMatrix(data = as.factor(y_pred), reference = mnist_27$test$y)$overall["Accuracy"]
