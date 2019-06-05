#----   Comprehension Check: Working with Matrices    ----


#---  Q1  ----
x <- matrix(rnorm(100*10), 100, 10)


#---  Q2  ----
dim(x)
nrow(x)
ncol(x)


#---  Q3  ----
x <- x + seq(nrow(x))
x <- sweep(x, 1, 1:nrow(x),"+")


#---  Q4  ----
x <- sweep(x, 2, 1:ncol(x), FUN = "+")


#---  Q5  ----
rowMeans(x)
colMeans(x)


#---  Q6  ----
# Pixels in the grey area 50 to 205
mnist <- read_mnist()
dim(mnist$train$images)

sum((mnist$train$images >= 50 & mnist$train$images <= 205) * 1) / length(as.vector(mnist$train$images))