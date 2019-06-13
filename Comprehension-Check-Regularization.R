#----   Comprehension Check: Regularization    ----
library(tidyverse)
library(caret)
library(rpart)


seed <- 1986
if(R.version$version.string  == "R version 3.6.0 (2019-04-26)"){
 set.seed(seed, sample.kind = "Rounding")
} else {
 set.seed(seed)
}

# Number of students
n <- round(2^rnorm(1000, 8, 1))

# True quality of each school that is completely independent from the size
seed <- 1
if(R.version$version.string  == "R version 3.6.0 (2019-04-26)"){
        set.seed(seed, sample.kind = "Rounding")
} else {
        set.seed(seed)
}

mu <- round(80 + 2 * rt(1000, 5))
range(mu)
schools <- data.frame(id = paste("PS", 1:100),
                      size = n,
                      quality = mu,
                      rank = rank(-mu))

# Top ten schools
schools %>% top_n(10, quality) %>% arrange(desc(quality))

# Simulated test scores (normally distributed N(avg_school, sd = 0.3)
seed <- 1
if(R.version$version.string  == "R version 3.6.0 (2019-04-26)"){
        set.seed(seed, sample.kind = "Rounding")
} else {
        set.seed(seed)
}
scores <- sapply(1:nrow(schools), function(i){
        scores <- rnorm(n = schools$size[i], mean = schools$quality[i], sd = 30)
        scores
})

# Combine the data sets, and set the scores for each school to the average score of the schools' students
schools <- schools %>% mutate(score = sapply(scores, mean))


#---    Q1      ----
# Top ten schools based on average score
schools %>% top_n(10, score) %>% arrange(desc(score)) %>% select(id, size, score)


#---    Q2      ----
# Median school size overall and for top 10 schools
median(schools$size)
median(schools %>% top_n(10, score) %>% .$size)

#---    Q3      ----
# Median school size for the bottom 10 schools
median(schools %>% top_n(10, -score) %>% .$size)


#---    Q4      ----
highlight <- schools %>% top_n(10, quality)
schools %>% ggplot(aes(size, score)) + geom_point() + geom_point(data = highlight, col = "red", size = 2)


#---    Q5      ----
overall <- mean(sapply(scores, mean))

alpha <- 25

# New calculation for each school using score obs from the students
schools <- schools %>% mutate(reg_score = overall + sapply(scores, function(s) {sum(s-overall)/(length(s) + alpha)}))
schools %>% top_n(10, reg_score) %>% arrange(desc(reg_score))

# or using size * mean
schools <- schools %>% mutate(reg_score = overall + size * (score - overall) / (size + alpha) )
schools %>% top_n(10, reg_score) %>% arrange(desc(reg_score))


#---    Q6      ----
alphas <- seq(1, 250, 1)

# RMSE for different levels of alpha 
rmses <- sapply(alphas, function(a){
        score <- schools %>% mutate(reg_score = overall + size * (score - overall) / (size + a)) %>% .$reg_score
        rmse <- sqrt(mean((score - schools$quality)^2))
})

qplot(alphas, rmses)
alphas[which.min(rmses)]

#---    Q7      ----
schools %>% mutate(reg_score = overall + size * (score - overall) / (size + alphas[which.min(rmses)]) ) %>%
        top_n(10, reg_score) %>% arrange(desc(reg_score))


#---    Q8      ----
alphas <- seq(10, 250, 1)
# RMSE for different levels of alpha without centering to 0
rmses <- sapply(alphas, function(a){
        score <- schools %>% mutate(reg_score = size * (score) / (size + a)) %>% .$reg_score
        rmse <- sqrt(mean((score - schools$quality)^2))
})

qplot(alphas, rmses)
alphas[which.min(rmses)]

