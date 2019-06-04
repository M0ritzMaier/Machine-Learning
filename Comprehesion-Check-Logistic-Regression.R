#----   Comprehension Check: Logistic Regression    ----
library(tidyverse)
library(caret)
library(e1071)
#---  Q1  ----
set.seed(2, sample.kind = "Rounding")
make_data <- function(n = 1000, p = 0.5, 
                      mu_0 = 0, mu_1 = 2, 
                      sigma_0 = 1,  sigma_1 = 1){
  
  y <- rbinom(n, 1, p)
  f_0 <- rnorm(n, mu_0, sigma_0)
  f_1 <- rnorm(n, mu_1, sigma_1)
  x <- ifelse(y == 1, f_1, f_0)
  
  test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)
  
  list(train = data.frame(x = x, y = as.factor(y)) %>% slice(-test_index),
       test = data.frame(x = x, y = as.factor(y)) %>% slice(test_index))
}
dat <- make_data()
dat$train %>% ggplot(aes(x, color = y)) + geom_density()

set.seed(1, sample.kind = "Rounding")
mu_1 <- seq(0, 3, len = 25)
res <- sapply(mu_1, function(m){
  dat <- make_data(mu_1 = m)
  
  fit_glm <- glm(y ~ x, data = dat$train, family = "binomial")
  p_hat <- predict(fit_glm, newdata = dat$test, type = "response")
  y_hat <- factor(ifelse(p_hat > 0.5, 1, 0))
  
  return(mean(y_hat == dat$test$y))
  
})

plot(mu_1, res)
