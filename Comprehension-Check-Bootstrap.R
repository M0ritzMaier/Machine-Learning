#----   Comprehension Check: Bootstrap    ----
library(tidyverse)
library(caret)

#---    Q1      ----
seed <- 1995
if(R.version$version.string  == "R version 3.6.0 (2019-04-26)"){
        set.seed(seed, sample.kind = "Rounding")
} else {
        set.seed(seed)
}

indexes <- createResample(mnist_27$train$y, 10)
sapply(c(3,4,7), function(i){
        sum(indexes$Resample01 == i)
})


#---    Q2      ----
sum(sapply(indexes, function(x){
        sum(x == 3)
}))


#---    Q3      ----
y <- rnorm(100, 0, 1)
qnorm(0.75)
quantile(y, 0.75)

seed <- 1
if(R.version$version.string  == "R version 3.6.0 (2019-04-26)"){
        set.seed(seed, sample.kind = "Rounding")
} else {
        set.seed(seed)
}
B <- 10000
q75 <- replicate(B, {
        Y <- rnorm(100, 0, 1)
        quantile(Y, 0.75)
})
mean(q75)
sd(q75)


#---    Q4      ----
seed <- 1
if(R.version$version.string  == "R version 3.6.0 (2019-04-26)"){
        set.seed(seed, sample.kind = "Rounding")
} else {
        set.seed(seed)
}
y <- rnorm(100, 0, 1)

if(R.version$version.string  == "R version 3.6.0 (2019-04-26)"){
        set.seed(seed, sample.kind = "Rounding")
} else {
        set.seed(seed)
}
indexes <- createResample(y, 10)

b75 <- sapply(indexes, function(x){
        quantile(y[x], 0.75)
        })
mean(b75)
sd(b75)


#---    Q5      ----
if(R.version$version.string  == "R version 3.6.0 (2019-04-26)"){
        set.seed(seed, sample.kind = "Rounding")
} else {
        set.seed(seed)
}
indexes <- createResample(y, 10000)

b75 <- sapply(indexes, function(x){
        quantile(y[x], 0.75)
})
mean(b75)
sd(b75)
