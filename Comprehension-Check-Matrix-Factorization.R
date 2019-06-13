#----   Comprehension Check: Matrix Factorization    ----
library(tidyverse)
library(caret)
library(rpart)


seed <- 1987
if(R.version$version.string  == "R version 3.6.0 (2019-04-26)"){
 set.seed(seed, sample.kind = "Rounding")
} else {
 set.seed(seed)
}

# Number of students
n <- 100
k <- 8

Sigma <- 64  * matrix(c(1, .75, .5, .75, 1, .5, .5, .5, 1), 3, 3) 
m <- MASS::mvrnorm(n, rep(0, 3), Sigma)
m <- m[order(rowMeans(m), decreasing = TRUE),]
y <- m %x% matrix(rep(1, k), nrow = 1) + matrix(rnorm(matrix(n*k*3)), n, k*3)
colnames(y) <- c(paste(rep("Math",k), 1:k, sep="_"),
                 paste(rep("Science",k), 1:k, sep="_"),
                 paste(rep("Arts",k), 1:k, sep="_"))

dim(y)

#---    Q1      ---
my_image <- function(x, zlim = range(x), ...){
        colors = rev(RColorBrewer::brewer.pal(9, "RdBu"))
        cols <- 1:ncol(x)
        rows <- 1:nrow(x)
        image(cols, rows, t(x[rev(rows),,drop=FALSE]), xaxt = "n", yaxt = "n",
              xlab="", ylab="",  col = colors, zlim = zlim, ...)
        abline(h=rows + 0.5, v = cols + 0.5)
        axis(side = 1, cols, colnames(x), las = 2)
}

my_image(y)
# The students that test well are at the top of the image and there seem to be three groupings by subject.


#---    Q2      ---
my_image(cor(y), zlim = c(-1,1))
range(cor(y))
axis(side = 2, 1:ncol(y), rev(colnames(y)), las = 2)
# There is correlation among all tests, but higher if the tests are in science and math and even higher within each subject.


#---    Q3      ---
# SVD of y
s <- svd(y)
names(s)

y_svd <- s$u %*% diag(s$d) %*% t(s$v)
max(abs(y-y_svd))

y_yv <- s$u %*% diag(s$d)

# Sum of squares
ss_y <- diag(t(y) %*% y)

ss_yv <- diag(t(y_yv) %*% y_yv)


#---    Q4      ---
plot(1:ncol(y), ss_y)
plot(1:ncol(y), ss_yv)

# ss_yv is decreasing and close to 0 for the 4th column and beyond.


#---    Q5      ---
plot(s$d, sqrt(ss_yv))
abline(a = 0, b = 1, col = "red")
# UD is equal to YV


#---    Q6      ---
sum(ss_yv[1:3])/sum(ss_yv)


#---    Q7      ---
identical(s$u %*% diag(s$d), sweep(s$u, 2, s$d, FUN = "*"))


#---    Q8      ---
ud <- s$u %*% diag(s$d)
average_score <- rowMeans(y)

plot(-ud[,1], average_score)
# There is a linearly increasing relationship between the average score for each student and U1d1,1.


#---    Q9      ---
my_image(s$v)
# The first column is very close to being a constant, which implies that the first column of YV is the sum of the rows of Y
# multiplied by some constant, and is thus proportional to an average.


#---    Q10     ---
# Y=U_1 d_1,1 V^T_1 + U_2 d_2,2 V^⊤_2 + ⋯+ U_p d_p,p V^⊤_p
plot(s$u[,1], ylim = c(-0.25, 0.25))
plot(s$v[,1], ylim = c(-0.25, 0.25))
with(s, my_image((u[, 1, drop=FALSE]*d[1]) %*% t(v[, 1, drop=FALSE])))
my_image(y)


#---    Q11     ---
resid <- y - with(s,(u[, 1, drop=FALSE]*d[1]) %*% t(v[, 1, drop=FALSE]))
my_image(cor(resid), zlim = c(-1,1))
axis(side = 2, 1:ncol(y), rev(colnames(y)), las = 2)

plot(s$u[,2], ylim = c(-0.25, 0.25))
plot(s$v[,2], ylim = c(-0.25, 0.25))
with(s, my_image((u[, 2, drop=FALSE]*d[2]) %*% t(v[, 2, drop=FALSE])))
my_image(resid)


#---    Q12     ---
resid <- y - with(s,sweep(u[, 1:2], 2, d[1:2], FUN="*") %*% t(v[, 1:2]))
my_image(cor(resid), zlim = c(-1,1))
axis(side = 2, 1:ncol(y), rev(colnames(y)), las = 2)

plot(s$u[,3], ylim = c(-0.25, 0.25))
plot(s$v[,3], ylim = c(-0.25, 0.25))
with(s, my_image((u[, 3, drop=FALSE]*d[3]) %*% t(v[, 3, drop=FALSE])))
my_image(resid)


#---    Q13     ---
resid <- y - with(s,sweep(u[, 1:3], 2, d[1:3], FUN="*") %*% t(v[, 1:3]))
my_image(cor(resid), zlim = c(-1,1))
axis(side = 2, 1:ncol(y), rev(colnames(y)), las = 2)

y_hat <- with(s,sweep(u[, 1:3], 2, d[1:3], FUN="*") %*% t(v[, 1:3]))
my_image(y, zlim = range(y))
my_image(y_hat, zlim = range(y))
my_image(y - y_hat, zlim = range(y))
