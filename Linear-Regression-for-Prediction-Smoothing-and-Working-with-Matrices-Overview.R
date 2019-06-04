#############################################################################################################################
#                                                                                                                           #
#                                                                                                                           #
#         LINEAR REGRESSION FOR PREDICTION, SMOOTHING, AND WORKING WITH MATRICES OVERVIEW                                   #
#                                                                                                                           #
#                                                                                                                           #
#############################################################################################################################

#----   Linear Regression for Prediction    ----#
library(HistData)
library(caret)
library(tidyverse)
library(e1071)
library(ggplot2)

galton_heights <- GaltonFamilies %>% 
  filter(childNum == 1 & gender == "male") %>%
  select(father, childHeight) %>%
  rename(son = childHeight)

# Train and test sets
y <- galton_heights$son
set.seed(1)
test_index <- createDataPartition(y, times = 1, p = 0.5, list =F)

train_set <- galton_heights %>% slice(-test_index)
test_set <- galton_heights %>% slice(test_index)

# Simple average (guessing)
avg <- mean(train_set$son)
avg

# Squared loss
mean((avg - test_set$son)^2)

# Conditional expectation => regression line
fit <- lm(son ~ father, data = train_set)
fit$coefficients

y_hat <- fit$coefficients[1] + fit$coefficients[2] * test_set$father
mean((y_hat - test_set$son)^2)


#----   Predict function    ----
y_hat <- predict(fit, test_set)
mean((y_hat - test_set$son)^2)


#----   Regression for Categorial Outcome   ----
library(dslabs)
data("heights")

y <- heights$height
set.seed(2, sample.kind = "Rounding")
test_index <- createDataPartition(y, times = 1, p = 0.5,
                                  list = F)
train_set <- heights %>% slice(-test_index)
test_set <- heights %>% slice(test_index)

# Conditional probability Pr(Y = 1 | X = x)
# rounding to the nearest inch and calculate cp that a 66 inch person is female
train_set %>% filter(round(height) == 66) %>%
  summarize(mean(sex == "Female"))

# p(x) = Pr(Y = 1 | X = x) = beta_0 + beta_1*x with Y = 1 equals "Female"
lm_fit <- mutate(train_set, y = as.numeric(sex == "Female")) %>% 
  lm(y ~ height, data = .)

# Decision rule: predict female if p^hat(x) > 0.5
p_hat <- predict(lm_fit, test_set)
y_hat <- ifelse(p_hat > 0.5, "Female", "Male") %>% factor()
confusionMatrix(y_hat, test_set$sex)


#----   Logistic Regression   ----
# assures that estimates between 0 and 1
# uses logistic transformation g(p) = log(p/(1-p)) => converts probabilities to log odds
# odds => how much more likely something happen, compared to not happen
# Maximum likelyhood estimate
glm_fit <- train_set %>% mutate(y = as.numeric(sex == "Female")) %>%
  glm(y ~ height, data = ., family = "binomial")


p_hat_logit <- predict(glm_fit, newdata = test_set, type = "response") # type = "response" gives back the conditional probabilities

y_hat_logit <- ifelse(p_hat_logit > 0.5, "Female", "Male") %>% factor()
confusionMatrix(y_hat_logit, test_set$sex)


#----   Case Study: 2 or 7    ----
# predictors: 
#   - proportion of dark pixel in the upper left quadrant => x_1
#   - proportion of dark pixels in the lower right quadrant => x-2
data(mnist_27)

mnist_27$train %>% ggplot(aes(x_1, x_2, col = y)) + geom_point()

# logistic regression
#   p(x_1, x_2) = Pr(Y = 1 | X_1 = x_1, X_2 = x_2) = g^-1(beta_0+beta_1*x_1+beta_2*x_2)
#   with g^-1 as the inverse of the logistic function
#     g^-1(x) = exp(x)/{1+exp(x)}

fit <- glm(y ~ x_1 + x_2, data = mnist_27$train, family = "binomial")

# decision rule p_hat > 0.5 : 7 else 2
p_hat <- predict(fit, newdata = mnist_27$test, type = "response")
y_hat <- factor(ifelse(p_hat > 0.5, 7, 2))
confusionMatrix(data = y_hat, reference = mnist_27$test$y)

mnist_27$true_p %>% ggplot(aes(x_1, x_2, z = p, fill = p)) +
  geom_raster() +
  scale_fill_gradientn(colors = c("#F8766D", "white", "#00BFC4")) + 
  stat_contour(breaks = c(0.5), color = "black")


#----   Introduction to Smoothing   ----
# also called  curve fiting and low band pass filtering
data("polls_2008")
qplot(day, margin, data = polls_2008)

# Mathematical model
#   Y_i = f(x_i) + e_i

# Predict Y for a given day x
#   f(x) = E(Y | X = x)

fit <- lm(margin~day, data = polls_2008)
polls_2008 %>% mutate(resid = ifelse(fit$residuals > 0, "+", "-")) %>%
  ggplot(aes(day, margin, col = resid)) + 
  geom_point() + 
  geom_abline(intercept = fit$coefficients[1], slope = fit$coefficients[2])


#----   Bin Smoothing amd Kernels   ----
# Idea: group data points into strata in which value of f(x) are constant
#   => Assumption if f(x) changes slowely
#       public opinion is constant within a week
#       x_0: day in the center of the week
#       other day |x - x_0| <= 3.5, we assume f(x) is constant f(x) = mu
#       E[Y_i | X_i = x_i] approx. mu if |x_i - x_0| <= 3.5
#       Size of the intervall satisfying the condition is called window size, bandwith or span.

#       A_0: set of indexes i such that |x_i - x_0| <= 3.5
#       N_0:  Number of indexes in A_0
#       f_hat(x_0) = 1/N sum_i element A_0 Y_i => Average in the window

span <- 7
fit <- with(polls_2008, ksmooth(day, margin, x.points = day, kernel = "box", bandwidth = span))

polls_2008 %>% mutate(smooth = fit$y) %>%
  ggplot(aes(day, margin)) + 
  geom_point(size = 3, alpha = 0.5, color = "grey") +
  geom_line(aes(day, smooth), color = "red")

# Smooth the lines if we take weighted averages that give the center of a point more weights
#   The Functions from which the weights are computed are called kernel

# With kernel = normal or Gaussion density
fit <- with(polls_2008, ksmooth(day, margin, x.points = day, kernel = "normal", bandwidth = span))

polls_2008 %>% mutate(smooth = fit$y) %>%
  ggplot(aes(day, margin)) + 
  geom_point(size = 3, alpha = 0.5, color = "grey") +
  geom_line(aes(day, smooth), color = "red")


#----   Local Weighted Regression (loess)   ----
# Make use of Taylor's theorem => if you look close enough at any smooth function f, it looks like a line
#   We assume the function is locally linear
#     enables us to look at larger windows
#     3 Week window
#     E[Y_i | X_i = x_i] = beta_0 + beta_1 * (x_i - x_0)
#     if |x_i - x_0| <= 10.5

total_days <- diff(range(polls_2008))
span <- 21/total_days

fit <- loess(margin ~ day, degree = 1, span = span, data = polls_2008) # degree = 1 tells loess to fit polynomials of degree 1 => lines

polls_2008 %>% mutate(smooth = fit$fitted) %>% 
  ggplot(aes(day, margin)) + 
  geom_point(size = 3, alpha = 0.5, color = "grey") + 
  geom_line(aes(day, smooth), color = "red")

# 3 differences between loess and (typical) bin smoother:
#   - rather than keeping the bin size the same, loess keeps the number of points used for the local fit the same. Is controlled with the span argument whichexpects a proportion
#   - when fitting a line locally, loess uses a weighted approach => minimizes a weighted version of OLS
#       sum_(i = 1)^N w_0 (x_i) [Y_i - {beta_0 + beta_1 * (x_i - x_0)}]^2
#       and instead of Gaussian kernel, loess uses a function called Tukey tri-weight
#       W(u) = (1- |u|^3)^3 if |u| <= 1
#       W(u) = 0 if |u| > 1
#       Weights w_0(x_i) = W((X_i - x_0) / h)
#   - loess has the option of fitting the local model robustly => outliers are detected and down-weighted for the next iteartion. Option family = "symmetric"

# Taylor's theorem also tells us that if you look at a function close enough, it looks like a parabola => do not need as close enough as a linear
#   => Windows can be even larger and fit parabolas instead of lines
#     E[Y_i | X_i = x_i] = beta_0 + beta_1 (x_i - x_0) + beta_2 (x_i - x_0)^2
#     if |x_i - x_0| <= h
#     => default behaviour for loess, degree = 2
fit <- loess(margin ~ day, degree = 1, span = span, data = polls_2008) # degree = 1 tells loess to fit polynomials of degree 1 => lines
fit_par <- loess(margin ~ day, span = span, data = polls_2008)

polls_2008 %>% mutate(smooth = fit$fitted, smooth_par = fit_par$fitted) %>% 
  ggplot(aes(day, margin)) + 
  geom_point(size = 3, alpha = 0.5, color = "grey") + 
  geom_line(aes(day, smooth), color = "red", lty = "dashed") + 
  geom_line(aes(day, smooth_par), color = "orange")


# ggplot uses loess in the smooth function
polls_2008 %>% ggplot(aes(day, margin)) +
  geom_point() + 
  geom_smooth()

polls_2008 %>% ggplot(aes(day, margin)) +
  geom_point() + 
  geom_smooth(color = "red", span = 0.15, method.args = list(degree = 1))


#----   Matrices   ----
mnist <- read_mnist()
class(mnist$train$images)

# first thousand predictors and outcomes
x <- mnist$train$images[1:1000,]
y <- mnist$train$labels[1:1000]

# Basic examples for matrices
my_vector <- 1:15
mat <- matrix(my_vector, 5, 3)
mat

mat_t <- matrix(my_vector, 3, 5, byrow = T)
mat_t

identical(t(mat), mat_t)

grid <- matrix(x[3,], 28, 28)

image(1:28, 1:28, grid)
image(1:28, 1:28, grid[,28:1])

# Total pixel darkness
sums <- rowSums(x)
avg <- rowMeans(x)



