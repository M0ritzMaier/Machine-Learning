#----   Comprehension Check: Caret Package    ----
library(tidyverse)
library(caret)
library(rpart)
library(randomForest)

#---    Q1      ----
n <- 1000
sigma <- 0.25

seed <- 1
if(R.version$version.string  == "R version 3.6.0 (2019-04-26)"){
 set.seed(seed, sample.kind = "Rounding")
} else {
 set.seed(seed)
}

x <- rnorm(n, 0, 1)
y <- 0.75 * x + rnorm(n, 0, sigma)
dat <- data.frame(x = x, y = y)

modelLookup("Rborist")
grid <- expand.grid(minNode = seq(25,100,25), predFixed = 0)
fit <- train(y ~ x, data = dat, method ="Rborist", tuneGrid = grid)
fit$bestTune


#---    Q2      ----
dat %>% 
        mutate(y_hat = predict(fit)) %>% 
        ggplot() +
        geom_point(aes(x, y)) +
        geom_step(aes(x, y_hat), col = 2)


#---    Q3      ----
data("tissue_gene_expression")
seed <- 1991
if(R.version$version.string  == "R version 3.6.0 (2019-04-26)"){
        set.seed(seed, sample.kind = "Rounding")
} else {
        set.seed(seed)
}

modelLookup("rpart")
cp <- seq(0, 0.1, 0.01)

fit_q3 <- train(x = tissue_gene_expression$x , y = tissue_gene_expression$y, method = "rpart", tuneGrid = data.frame(cp =cp)) 
fit_q3
plot(fit_q3, highlight = T)
fit_q3$bestTune


#---    Q4      ----
confusionMatrix(fit_q3)


#---    Q5      ----
seed <- 1991
if(R.version$version.string  == "R version 3.6.0 (2019-04-26)"){
        set.seed(seed, sample.kind = "Rounding")
} else {
        set.seed(seed)
}



fit_q5 <- train(x = tissue_gene_expression$x , y = tissue_gene_expression$y, method = "rpart", tuneGrid = data.frame(cp =cp),
                control = rpart.control(minsplit = 0)) 
fit_q5


#---    Q6      ----
plot(fit_q5$finalModel, margin = 0.1)
text(fit_q5$finalModel, cex = 0.75)


#---    Q7      ----
seed <- 1991
if(R.version$version.string  == "R version 3.6.0 (2019-04-26)"){
        set.seed(seed, sample.kind = "Rounding")
} else {
        set.seed(seed)
}

modelLookup("rf")

fit_q7 <- train(x = tissue_gene_expression$x , y = tissue_gene_expression$y, method = "rf", 
                tuneGrid = data.frame(mtry = seq(50, 200, 25)),
                nodesize = 1) 
fit_q7
fit_q7$bestTune


#---    Q8      ----
imp <- varImp(fit_q7)


#---    Q9      ----
tree_terms <- as.character(unique(fit_q5$finalModel$frame$var[!(fit_q5$finalModel$frame$var == "<leaf>")]))
tree_terms

imp$importance[rownames(imp$importance) == "CFHR4",]

df <- data.frame(imp$importance) 

df2 <- df %>% mutate(rank = rank(-Overall))     
rownames(df2) <- rownames(df)
df2["CFHR4",]
