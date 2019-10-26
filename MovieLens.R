################################
# Create edx set, validation set
################################

# Libraries to load
library(tidyverse)
library(lubridate)
library(Matrix)
library(recommenderlab)
library(Matrix.utils)
library(irlba)
library(recosystem)

# Note: this process could take a couple of minutes 


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

set.seed(1, sample.kind="Rounding")
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


# #####################################rm(dl, ratings, movies, test_index, temp, movielens, removed)


# looking at the data set edx provided

class(edx)
head(edx)
str(edx)
dim(edx)
length(unique(edx$movieId))  # number of different movies
length(unique(edx$userId))   # number of differen users
summary(edx)

# Exploring the genres

Genres <- edx %>% separate_rows(genres, sep = "\\|") %>% group_by(genres) %>% summarize(count = n()) %>% arrange(desc(count))
Genres %>% ggplot(aes(x=reorder(genres, count), y=count)) + 
  geom_bar(stat="identity", fill="blue") +
  coord_flip(y=c(0, 4000000)) +
  labs(x="Genres", y="Number of Movies per Genres") +
  scale_y_continuous(breaks = c(0,400000,1000000,2000000,4000000)) +
  geom_text(aes(label=count), hjust=-0.2, size=3) 

# Explore the Movies with highest number of ratings. 

Movie_rating_count <- edx %>% group_by(title) %>% summarize(count=n()) %>% top_n(30, count) %>% arrange(desc(count))
Movie_rating_count %>% ggplot(aes(x=reorder(title, count), y=count)) + 
  geom_bar(stat="identity", fill="blue") +
  coord_flip(y=c(0, 40000)) +
  labs(x="Top Movies", y="Number of Ratings") +
  scale_y_continuous(breaks = c(0,10000,20000,30000,40000)) +
  geom_text(aes(label=count), hjust=-0.2, size=3) 

# Explore the Genres by the average ratings. It seems the Genres has some effects on average rating

edx %>% group_by(genres) %>%
  summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
  filter(n >= 50000) %>% 
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Explore number of ratings by movieId and UserId. It seems there is a user and movie effect to consider.

edx %>%  count(movieId) %>% ggplot(aes(n)) +
  geom_histogram(bins=30, color="red") +
  scale_x_log10() +
  ggtitle("Movies") +
  labs(x="n in 30 bins", y="Count in bins") 
  
  edx %>% count(userId) %>% ggplot(aes(n)) +
  geom_histogram(bins=30, color="red") +
  scale_x_log10() +
  ggtitle("Users") +
  labs(x="n in 30 bins", y="Count in bins") 
  
# Exploring Time = There is some evidence of a time effect on average rating but it is mostly stable 
# also considering different time frame (weeks, months, etc) too. I decide to not work on timestamp.

Times <- mutate(edx, date = as_datetime(timestamp))
Times %>% mutate(date = round_date(date, unit = "week")) %>%
  group_by(date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth()

# Obtain a sparse Matrix in class realRatingMatrix 
edx.copy <- edx
edx.copy$userId <- as.factor(edx.copy$userId)
edx.copy$movieId <- as.factor(edx.copy$movieId)

edx.copy$userId <- as.numeric(edx.copy$userId)
edx.copy$movieId <- as.numeric(edx.copy$movieId)

sparse_ratings <- sparseMatrix(i = edx.copy$userId,
                               j = edx.copy$movieId,
                               x = edx.copy$rating,
                               dims = c(length(unique(edx.copy$userId)),
                                        length(unique(edx.copy$movieId))))

ratingMat <- new("realRatingMatrix", data = sparse_ratings)
image(ratingMat[1:50, 1:50])

# Matrix Reduction SVD 

set.seed(1)
Red_ratings <- irlba(sparse_ratings, tol=1e-4, verbose=TRUE, nv= 100, maxit=1000)
plot(cumsum(Red_ratings$d^2/sum(Red_ratings$d^2)), type="l", xlab="SIngular Vector", ylab="Variability Explained Cumulated")
lines(x=c(0,100), y= c(.90, .90))
k = max(which(cumsum(Red_ratings$d^2/sum(Red_ratings$d^2)) <= .90))
k # Number of Singular vectors with 90% of variability explained
U <- Red_ratings$u[, 1:k]
D <- Diagonal(x = Red_ratings$d[1:k])
V <- t(Red_ratings$v)[1:k,]

# U%*%D%*%V this provide us the original matrix at 90%
# U%*%V # this provide predicted ratings

# Different approach to reduce the original matrix (realRatingMatrix)

min_movies <- quantile(rowCounts(ratingMat),0.9)
min_movies
min_users <- quantile(colCounts(ratingMat), 0.9)
min_users
ratings_movies <- ratingMat[rowCounts(ratingMat) > min_movies, colCounts(ratingMat) > min_users]
ratings_movies

# Function Definition for RMSE calculation

RMSE <- function(true_rating, predicted_rating){
  sqrt(mean((true_rating - predicted_rating)^2))}

# Linear regression models using just the mean, movies, users and genres effect

mu <- mean(edx$rating)
naive_rmse <- RMSE(validation$rating, mu)
naive_rmse

movie_avg <- edx %>% group_by(movieId) %>% summarize(b_i = mean(rating - mu))
predicted_rating <- mu + validation %>% left_join(movie_avg, by = "movieId") %>% .$b_i
movie_rmse <- RMSE(validation$rating, predicted_rating)
movie_rmse

user_avg <- edx %>% left_join(movie_avg, by = "movieId") %>% group_by(userId) %>% 
  summarize(b_u = mean(rating - mu - b_i)) 
predicted_rating <- validation %>% left_join(movie_avg, by = "movieId") %>% 
  left_join(user_avg, by = "userId") %>% 
  mutate(pred = mu + b_i + b_u) %>% .$pred
movie_user_rmse <- RMSE(validation$rating, predicted_rating)
movie_user_rmse

genres_avg <- edx %>% left_join(movie_avg, by = "movieId") %>% left_join(user_avg, by = "userId") %>%
  group_by(genres) %>% summarize(b_g = mean(rating - mu - b_i - b_u))
predicted_rating <- validation %>% left_join(movie_avg, by = "movieId") %>% 
  left_join(user_avg, by = "userId") %>% left_join(genres_avg, by="genres") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>% .$pred
movie_genres_rmse <- RMSE(validation$rating, predicted_rating)
movie_genres_rmse

# Regularization approach

lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mu <- mean(edx$rating)
  b_i <- edx %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx %>% left_join(b_i, by='movieId') %>% group_by(userId) %>% summarize(b_u = sum(rating -b_i -mu)/(n()+l))
  predicted_rating <- validation %>% left_join(b_i, by = "movieId") %>% left_join(b_u, by = "userId") %>%
      mutate(pred = mu + b_i + b_u) %>% .$pred
  return(RMSE(validation$rating, predicted_rating))})

plot(lambdas, rmses)
lambda <- lambdas[which.min(rmses)]
lambda  
min(rmses)

# RECOMMENDER ENGINES

set.seed(1)
eval <- evaluationScheme(ratings_movies, method="split", train=0.9, given=-5)

model_svd <- Recommender(getData(eval,"train"), method="SVD")
pred_svd <- predict(model_svd, getData(eval,"known"), type="ratings")
rmse_svd <- calcPredictionAccuracy(pred_svd, getData(eval,"unknown"))[1]
rmse_svd

model_UBCF <- Recommender(getData(eval,"train"), method="UBCF", param=list(normalize="center", method="cosine", nn=50))
pred_UBCF <- predict(model_UBCF, getData(eval,"known"), type="ratings")
rmse_UBCF <- calcPredictionAccuracy(pred_UBCF, getData(eval,"unknown"))[1]
rmse_UBCF

model_IBCF <- Recommender(getData(eval,"train"), method="IBCF", param=list(normalize="center", method="cosine", k=350))
pred_IBCF <- predict(model_IBCF, getData(eval,"known"), type="ratings")
rmse_IBCF <- calcPredictionAccuracy(pred_IBCF, getData(eval,"unknown"))[1]
rmse_IBCF

# Matrix Factorization with parallel stochastic gradient descendent

# only three variables have been considered: "user";"item";"rating". 
# Both are converted in a Matrix format (this is required by the recosystem package);

edx.tiny <- edx %>% select(-c("genres","title","timestamp"))
names(edx.tiny) <- c("user","item","rating")
validation.tiny <- validation %>% select(-c("genres", "title", "timestamp"))
names(validation.tiny) <- c("user","item","rating")
edx.tiny <- as.matrix(edx.tiny)
validation.tiny <- as.matrix(validation.tiny)

# the model for prediction and validation output are written in two different flat files;

write.table(edx.tiny, file = "traindata.txt", sep=" ", row.names=FALSE, col.names=FALSE)
write.table(validation.tiny, file="validationdata.txt", sep=" ", row.names=FALSE, col.names=FALSE)

set.seed(55)
train_set <- data_file("traindata.txt")
validation_set <- data_file("validationdata.txt")

# create the model using Reco(), Tune() method to set parameters, 
# Implement train() and predict() to train and calculate predicted values.

r=Reco()
opts=r$tune(train_set, opt=list(dim=c(10,20,30), lrate=c(0.1,0.2), costp_l1=0, costq_l1=0, nthread=1, niter=10))
r$train(train_set , opts=c(opts$min, nthread=1, niter=20))
pred_file=tempfile()
r$predict(validation_set, out_file(pred_file))
scores_real <- read.table("validationdata.txt", header=FALSE, sep=" ")$V3
scores_pred <- scan(pred_file)
rmse_mf <- RMSE(scores_real, scores_pred)
rmse_mf

# Reduction based on Genres and PCA

Genres <- edx %>% filter(str_detect(genres, "Drama")) 

edx.copy <- Genres
edx.copy$userId <- as.factor(edx.copy$userId)
edx.copy$movieId <- as.factor(edx.copy$movieId)

edx.copy$userId <- as.numeric(edx.copy$userId)
edx.copy$movieId <- as.numeric(edx.copy$movieId)

sparse_ratings <- sparseMatrix(i = edx.copy$userId,
                               j = edx.copy$movieId,
                               x = edx.copy$rating,
                               dims = c(length(unique(edx.copy$userId)),
                                        length(unique(edx.copy$movieId))),
                               dimnames = list(paste("u", 1:length(unique(edx.copy$userId)), sep=""),
                                               paste("m",1:length(unique(edx.copy$movieId)), sep="")))

ratingMat <- new("realRatingMatrix", data = sparse_ratings)

min_n_movies <- quantile(rowCounts(ratingMat),0.95)
min_n_users <- quantile(colCounts(ratingMat), 0.95)
ratings_movies <- ratingMat[rowCounts(ratingMat) > min_n_movies, colCounts(ratingMat) > min_n_users]

image(ratings_movies[1:5,1:5])
getRatingMatrix(ratings_movies[1:5,1:5])
Genres_Matrix <- as(ratings_movies, "matrix")
Genres_Matrix[is.na(Genres_Matrix)] <- 0
mean_col <- colMeans(Genres_Matrix)

for (i in 1:nrow(Genres_Matrix)) {
  for (j in 1:ncol(Genres_Matrix)) if (Genres_Matrix[i,j]==0) Genres_Matrix[i,j]=mean_col[j]
}

cp <- prcomp(Genres_Matrix)

dim(cp$rotation)
dim(cp$x)

#summary(cp) - explained variability set as 60% with 76 principal components
x <- summary(cp)$importance[3,] 
plot(x, type="l")

var_explained <- cumsum(cp$sdev^2/sum(cp$sdev^2))
var_explained 

EV <- cp$x[,1:76]
r<-cor(Genres_Matrix,EV)

#The next line can be used to write a .csv file to assign an item to a principal component.
#write.table(r, file = "PCA_Corr.csv",row.names=TRUE, na="",col.names=TRUE, sep=",")

PCs <- c("619","3324","1822","425","2736","513","1498","664","270","623","1891","121",
         "1101","616","970","582","1382","1379","613","647","164","967","651","1141",
         "669","265","596","630","15","407","941","644","607","638","837","1274","2746",
         "143","274","244","519","466","181","242","3880","615","262","636","559","543",
         "149","184","286","968","276","4785","1476","36","1095","20","172","916","1328",
         "2483","1570","532","57","1968","10","631","524","145","1791","946","137","29")

edx.copy <- edx %>% filter(movieId %in% PCs)
edx.copy$userId <- as.factor(edx.copy$userId)
edx.copy$movieId <- as.factor(edx.copy$movieId)
edx.copy$userId <- as.numeric(edx.copy$userId)
edx.copy$movieId <- as.numeric(edx.copy$movieId)

sparse_ratings <- sparseMatrix(i = edx.copy$userId,
                               j = edx.copy$movieId,
                               x = edx.copy$rating,
                               dims = c(length(unique(edx.copy$userId)),
                                        length(unique(edx.copy$movieId))))

ratingMat <- new("realRatingMatrix", data = sparse_ratings)

min_movies <- quantile(rowCounts(ratingMat),0.9)
ratings_movies <- ratingMat[rowCounts(ratingMat) > min_movies]

  
set.seed(1)
eval <- evaluationScheme(ratings_movies, method="split", train=0.8, given=-1)

model_svd <- Recommender(getData(eval,"train"), method="SVD")
pred_svd <- predict(model_svd, getData(eval,"known"), type="ratings")
rmse_pca_svd <- calcPredictionAccuracy(pred_svd, getData(eval,"unknown"))[1]
rmse_pca_svd

model_svdf <- Recommender(getData(eval,"train"), method="SVDF")
pred_svdf <- predict(model_svdf, getData(eval,"known"), type="ratings")
rmse_pca_svdf <- calcPredictionAccuracy(pred_svdf, getData(eval,"unknown"))[1]
rmse_pca_svdf

model_UBCF <- Recommender(getData(eval,"train"), method="UBCF", param=list(normalize="center", method="cosine", nn=50))
pred_UBCF <- predict(model_UBCF, getData(eval,"known"), type="ratings")
rmse_pca_UBCF <- calcPredictionAccuracy(pred_UBCF, getData(eval,"unknown"))[1]
rmse_pca_UBCF

model_IBCF <- Recommender(getData(eval,"train"), method="IBCF", param=list(normalize="center", method="cosine", k=350))
pred_IBCF <- predict(model_IBCF, getData(eval,"known"), type="ratings")
rmse_pca_IBCF <- calcPredictionAccuracy(pred_IBCF, getData(eval,"unknown"))[1]
rmse_pca_IBCF

# Recap RMSEs

rmse_table <- data.frame(RMSE = c(naive_rmse,movie_rmse,movie_user_rmse,movie_genres_rmse,min(rmses),rmse_svd,rmse_UBCF,rmse_IBCF,rmse_mf,rmse_pca_svd,rmse_pca_svdf,rmse_pca_UBCF,rmse_pca_IBCF)) 
row.names(rmse_table) <- c("Naive","Movie","Movie & User","Movie & Users & Genres", "Movie & USers with Regularization", "SVD", "UBCF", "IBCF","Parallel Stochastic Gradient Descendent","PCA & SVD", "PCA & SVDF", "PCA & UBCF", "PCA & IBCF")
rmse_table


## --- End of file ---