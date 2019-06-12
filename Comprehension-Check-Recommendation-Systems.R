#----   Comprehension Check: Recommendation Systems    ----
library(dslabs)
library(tidyverse)
library(caret)
library(rpart)
library(randomForest)
library(Rborist)
library(lubridate)

#---    Q1      ----
data("movielens")

# Train models
movielens %>% count(year) %>% ggplot(aes(year, sqrt(n))) + geom_point()

movielens %>% count(year) %>% filter(n == max(n))


#---    Q2      ----
# Movies in or after 1993, 25 movies with the most ratings per year and what is the average rating for each of those movies
top_25_movies <- movielens %>% filter(year >= 1993) %>% count(title) %>% arrange(desc(n)) %>% .$title
top_25_movies <- top_25_movies[1:25]

# Average rating
movielens %>% filter(title %in% top_25_movies) %>% group_by(title) %>% summarize(avg = mean(rating, na.rm = T)) %>% filter(title == "Shawshank Redemption, The")

movielens %>% filter(title == "Forrest Gump") %>% 
  mutate(rated_year = year(as.POSIXct(timestamp, origin = "1970-01-01", tz = "UTC"))) %>%
  count(rated_year) %>% summarize(mean(n))

movielens %>% filter(title == "Forrest Gump") %>% 
  mutate(rated_year = year(as.POSIXct(timestamp, origin = "1970-01-01", tz = "UTC"))) %>%
  count(rated_year) %>% summarize(sum(n))/(length(movielens %>% filter(title == "Forrest Gump") %>% 
                                                    mutate(rated_year = year(as.POSIXct(timestamp, origin = "1970-01-01", tz = "UTC"))) %>%
                                                    count(rated_year) %>% .$n)+3)                    


#---    Q3      ----
movielens %>% 
  filter(year >= 1993) %>%
  group_by(movieId) %>%
  summarize(n = n(), years = 2017 - first(year),
            title = title[1],
            rating = mean(rating)) %>%
  mutate(rate = n/years) %>%
  ggplot(aes(rate, rating)) +
  geom_point() +
  geom_smooth()


#---    Q5     ----
movielens <- mutate(movielens, date = as_datetime(timestamp))


#---    Q6     ----
movielens %>% mutate(date_week = round_date(date, unit = "week")) %>%
  group_by(date_week) %>% summarize(avg_rating = mean(rating)) %>%
  ggplot(aes(date_week, avg_rating)) + geom_point() + geom_smooth(method = "lm")


#---    Q8     ----
head(movielens)
movielens %>% group_by(genres) %>% summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n)) %>% filter(n > 1000) %>% 
  ggplot() + geom_bar(aes(genres)) + geom_errorbar(aes(genres, ymin = avg - 1.96 * se, ymax = avg + 1.96 *se))
