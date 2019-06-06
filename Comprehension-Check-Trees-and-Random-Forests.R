#----   Comprehension Check: Trees and Random Forests    ----
library(tidyverse)
library(caret)

#---    Q1      ----
library(rpart)
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

fit <- rpart(y ~ ., data = dat)

#---    Q2      ----
plot(fit, margin = 0.1)
text(fit)


#---    Q3      ----
dat %>% 
 mutate(y_hat = predict(fit)) %>% 
 ggplot() +
 geom_point(aes(x, y)) +
 geom_step(aes(x, y_hat), col = 2)


#---    Q4      ----
library(randomForest)
fit <- randomForest(y ~ x, data = dat)

dat %>% 
 mutate(y_hat = predict(fit)) %>% 
 ggplot() +
 geom_point(aes(x, y)) +
 geom_step(aes(x, y_hat), col = 2)


#---    Q5      ----
plot(fit)


#---    Q6      ----

fit <- randomForest(y ~ x, data = dat, nodesize = 50, maxnodes = 25)

dat %>% 
 mutate(y_hat = predict(fit)) %>% 
 ggplot() +
 geom_point(aes(x, y)) +
 geom_step(aes(x, y_hat), col = 2)
