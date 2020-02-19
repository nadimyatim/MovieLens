if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1)
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#showing a sample of the edx dataset
head(edx) 

#showing summary statistics of the edx dataset and its features.
summary(edx) 

#number of unique users and unique movies in the edx dataset
edx %>% 
  summarize(NumberofUsers=n_distinct(userId),
            NumberofMovies=n_distinct(movieId)
  )

#histogram of the ratings
edx%>% ggplot(aes(rating))+
  geom_histogram(bins=10,fill="navy",col="red") +
  scale_x_continuous(breaks = seq(0.5,5,0.5)) 

# number of movie ratings done for the top 5 movies
edx %>% group_by(title) %>% 
        summarize(count=n()) %>% 
        arrange(desc(count)) %>% 
        top_n(5) 

# number of movie ratings per userId
edx %>% count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 20, fill="navy", col = "red")+
  scale_x_log10() +
  xlab("Ratings")+
  ylab("Users")

# number of movie ratings per movieId
edx %>% count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 20, fill="navy", col = "red")+
  scale_x_log10() +
  xlab("Ratings")+
  ylab("Movies") 

#Average rating per user
edx %>% group_by(userId) %>%
  summarize(rating=mean(rating))%>%
  ggplot(aes(rating))+
  geom_histogram(bins=40 ,fill="navy",col="red")+
  xlab("Average user rating") +
  ylab("Users") + 
  geom_vline(xintercept = mean(edx$rating),col="green") 

#Average rating per movie
edx %>% group_by(movieId) %>%
  summarize(rating=mean(rating))%>%
  ggplot(aes(rating))+
  geom_histogram(bins=40 ,fill="navy",col="red")+
  xlab("Average movie rating") +
  ylab("Movies") + 
  geom_vline(xintercept = mean(edx$rating),col="green") 

#Splitting the edx dataset into training and test sets
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train <- edx[-test_index,]
temp <- edx[test_index,]
# Making sure userId and movieId in test set are also in train set
test <- temp %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")
# Adding the rows that were removed from test set into the train set
removed <- anti_join(temp, test)
train <- rbind(train, removed)


#First Model(Just the Average model)


#average of all movies across all users
mu_hat <- mean(train$rating) 
mu_hat

#Calculating RMSE by predicting all unknown ratings with average mu_hat
firstRMSE <- RMSE(validation$rating,mu_hat) 
firstRMSE

RMSE_Results<- data_frame(method="Just the average",RMSE=firstRMSE)

#Displaying the RMSE results
RMSE_Results %>% knitr::kable() 





#Second Model

#Caclucating b_i
movie_averages <- train %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu_hat)) 

#Distribution of b_i
movie_averages %>% ggplot(aes(b_i)) +
               geom_histogram(bins=10,fill="navy",col="red") 

#Then in order to see if our predicitons improve or not, we execute the following piece of code.
predicted_ratings <- mu_hat + validation %>% 
  left_join(movie_averages, by='movieId') %>%
  .$b_i

secondRMSE <- RMSE(predicted_ratings, validation$rating)
RMSE_Results <- bind_rows(RMSE_Results,
                          data_frame(method="Movie Effect Model",
                                     RMSE = secondRMSE ))

#Displaying the RMSE results
RMSE_Results %>% knitr::kable() 




#Third Model


#Caclucating b_u
user_averages <- train %>% 
  left_join(movie_averages, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))

predicted_ratings <- validation %>% 
  left_join(movie_averages, by="movieId") %>%
  left_join(user_averages, by="userId") %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  .$pred

thirdRMSE <- RMSE(predicted_ratings, validation$rating)
thirdRMSE
RMSE_Results <- bind_rows(RMSE_Results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = thirdRMSE ))

#Displaying the RMSE results
RMSE_Results %>% knitr::kable()



#Fourth Model


#calculate the optmial lambda by using cross validation
lambdas <- seq(0, 10, 0.25)
total <- edx %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu_hat), n_i = n())
rmses <- sapply(lambdas, function(l){
  predicted_ratings <- validation %>% 
    left_join(total, by='movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu_hat + b_i) %>%
    .$pred
  return(RMSE(predicted_ratings, validation$rating))
})

#Plot of RMSES verses lambdas
qplot(lambdas, rmses, col="red")  

#Value of lambda that minimizes RMSE
lambdas[which.min(rmses)]

#Value of the minimum RMSE
min(rmses)

RMSE_Results <- bind_rows(RMSE_Results,
                          data_frame(method="Regularized Movie Effect Model",  
                                     RMSE = min(rmses)))

#Displaying the RMSE results
RMSE_Results %>% knitr::kable()



#Fifth Model
#calculate the optmial lambda by using cross validation
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu_hat <- mean(train$rating)
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu_hat)/(n()+l))
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu_hat)/(n()+l))
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu_hat + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, validation$rating))
})

#Plot of RMSES verses lambdas
qplot(lambdas,rmses,col="red")

#Value of lambda that minimizes RMSE
lambdas[which.min(rmses)]

#Value of the minimum RMSE
min(rmses)

RMSE_Results <- bind_rows(RMSE_Results,
                          data_frame(method="Regularized Movie + User Effect Model",  
                                     RMSE = min(rmses)))

#Displaying the RMSE results
RMSE_Results %>% knitr::kable()