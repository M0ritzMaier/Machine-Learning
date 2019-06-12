#----   Comprehension Check: Recommendation Systems    ----
library(dslabs)
library(tidyverse)
library(caret)
library(rpart)
library(randomForest)
library(Rborist)

#---    Q1      ----
data("movielens")

# Train models
movielens %>% count(year) %>% ggplot(aes(year, sqrt(n))) + geom_point()

movielens %>% count(year) %>% filter(n == max(n))


#---    Q2      ----
# Movies in or after 1993, 25 movies with the most ratings per year and what is the average rating for each of those movies
top_25_movies <- movielens %>% filter(year >= 1993) %>% count(title) %>% arrange(desc(n))
top_25_movies <- top_25_movies[1:25,]

# Average rating
movielens %>% filter(title %in% top_25_movies$title) %>% group_by(title) %>% summarize(avg = mean(rating, na.rm = T))

                     
                     movielens %>% filter(year >= 1993) %>% rank(count(title))
