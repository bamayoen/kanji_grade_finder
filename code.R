##########################################################
# Goal: Predict the JLPT grade of a kanji (from the Joyo kanji list)
##########################################################

# Install tidyverse, caret, and data.table packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# Install lubridate and readr packages
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")

library(lubridate)
library(readr)
library(gridExtra)

# Install rvest, rpart, and randomForest packages
if(!require(rvest)) install.packages("rvest", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")

library(rvest)
library(rpart)
library(randomForest)

# show 5 significant digits
options(digits = 5)

####################### Step 1: Pre-processing #######################
### Goal: Create training and test sets
## Training set = kanji_train
## Test set = kanji_test

# Joyo Kanji dataset:
# https://www.kaggle.com/anthaus/japanese-jy-kanji

# Kanji dataset: (data.csv)
# https://www.kanjidatabase.com/index.php

# Raw Data:
# https://en.wikipedia.org/wiki/List_of_kanji_radicals_by_stroke_count
# https://en.wikipedia.org/wiki/List_of_jōyō_kanji 
# https://www.kanjidatabase.com/kanji_show_selected.php

# download first dataset
h1 <- read_html("https://en.wikipedia.org/wiki/List_of_kanji_radicals_by_stroke_count")
tab <- h1 %>% html_nodes("table")
tab <- tab[[1]] %>% html_table
tab <- tab %>% setNames(c("radicalID", "radical", "radical_stroke_count", "meaning_reading", "freq",
                          "joyo_freq", "examples", "group"))

tab <- tab %>% mutate(freq = parse_number(freq)) %>%
  replace_na(list(freq = 0, joyo_freq = 0))

radical_df <- tab
rm(h1, tab)

# download second dataset
h2 <- read_html("https://en.wikipedia.org/wiki/List_of_jōyō_kanji")
tab <- h2 %>% html_nodes("table")
tab <- tab[[2]] %>% html_table
tab <- tab %>% setNames(c("kanjiID", "kanji", "old", "radical", "kanji_stroke_count", "grade", "year_added",
                          "meaning", "readings"))
tab <- tab %>% mutate(year_added = replace_na(year_added, 1946))

kanji_df <- tab
rm(h2, tab)

# download third dataset
h3 <- read_table2("data.csv")
tab <- as.data.frame(h3)
tab <- tab %>% setNames(c("ID", "kanji", "kanji_stroke_count", "grade", "classification", "JLPT_grade", "radical", 
                          "radical_freq", "n_on_readings", "n_on_meanings", "n_kun_meanings", "year_added2", 
                          "kanji_freq_proper", "kanji_freq", "symmetry"))
tab <- tab %>% mutate(symmetry = replace_na(symmetry, ""))

kanji_df2 <- tab
rm(h3, tab)

## tidy up data
# drop these columns from radical_df
radical_to_drop <- c("radicalID", "meaning_reading", "examples", "freq")

radical_df <- radical_df[, !(names(radical_df) %in% radical_to_drop)]
radical_df <- radical_df %>% mutate(group = as.factor(group)) # make group a factor

radical_df <- separate(radical_df, radical, into = c("radical", NA), sep = " ", extra = "drop", fill = "right") # select only the first form of the radical

# drop these columns from kanji_df
kanji_to_drop <- c("readings", "meaning", "kanjiID", "grade", "kanji_stroke_count")

kanji_df <- kanji_df[, !(names(kanji_df) %in% kanji_to_drop)]
kanji_df <- kanji_df %>% mutate(year_added = as.factor(year_added)) # make year_added a factor

# drop these columns from kanji_df2
kanji_to_drop2 <- c("year_added2")
kanji_df2 <- kanji_df2[, !(names(kanji_df2) %in% kanji_to_drop2)]
kanji_df2 <- kanji_df2 %>% mutate(grade = as.factor(grade), classification = as.factor(classification),
                                  JLPT_grade = as.factor(JLPT_grade), radical = as.factor(radical),
                                  symmetry = as.factor(symmetry)) # conver these columns to factors 

# merge kanji_df, kanji_df2 and radical_df
temp <- left_join(kanji_df2, kanji_df, by = "kanji")
temp <- rename(temp, radical = radical.y)
df <- left_join(temp, radical_df, by = "radical")

df <- df[, !(names(df) == "radical_stroke_count")] 
df <- df %>% replace_na(list(joyo_freq = 0, old = "", radical = "", year_added = "1981", group = "")) %>%
  mutate(radical = as.factor(radical), old_logical = ifelse(old == "", FALSE, TRUE))

levels(df$group) <- c("", "Top 25%", "Top 50%", "Top 50%", "Top 50%", "Top 50%", "Top 75%") # reduce "group" factors

# clear workspace
rm(kanji_to_drop, radical_to_drop, temp, kanji_to_drop2)

# create train and test sets (test = 20% of data)
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = df$JLPT_grade, times = 1, p = 0.2, list = FALSE)

kanji_train <- df[-test_index, ]
kanji_test <- df[test_index, ]

# clear workspace
rm(kanji_df, radical_df, test_index, kanji_df2)

####################### Step 2: Data Exploration #######################
# define most_common(x,y) function
most_common <- function(x, y){
  count <- vector(mode = "numeric", length = length(y))
  for(i in seq(1, length(y), 1)){
    count[i] <- sum(x == y[i])
  }
  
  return(y[which.max(count)])
}

# find most common element of each column
most_common_kanji_train <- kanji_train %>% group_by(JLPT_grade) %>%
  summarise(grade = most_common(grade, levels(grade)),
            classification = most_common(classification, levels(classification)),
            radical = most_common(radical, levels(radical)),
            avg_strokes = mean(kanji_stroke_count), percent_old = mean(old_logical),
            on_meanings = mean(n_on_meanings), kun_meanings = mean(n_kun_meanings),
            on_readings = mean(n_on_readings), radical_freq = mean(radical_freq),
            avg_freq = mean(kanji_freq), avg_freq_proper = mean(kanji_freq_proper),
            symmetry = most_common(symmetry, levels(symmetry)),
            year_added = most_common(year_added, levels(year_added)))
most_common_kanji_train

kanji_train %>% group_by(JLPT_grade) %>%
  summarise(group = most_common(group, levels(group)))

# Q: is there a relationship between a kanji having an alternative version and its grade?
kanji_old_stroke <- kanji_train %>% group_by(JLPT_grade, old_logical) %>%
  summarise(n = n(), avg_stroke_count = mean(kanji_stroke_count))

(kanji_train %>% filter(old_logical) %>% nrow()) / nrow(kanji_train) # 17.7% of kanji have an old vers.

####################### Step 3: Data Visualization #######################

# kanji stroke count distribution
kanji_train %>% ggplot(aes(kanji_stroke_count)) + 
  geom_histogram(bins = 20, col = "black") +
  geom_vline(xintercept = median(kanji_train$kanji_stroke_count), col = "blue", show.legend = TRUE) +
  theme(legend.position = "right") +
  labs(title = "Kanji Stroke Count Distribution", x = "Kanji Stroke Count", y = "Count")

# kanji frequency vs stroke count (most common year is 1946)
kanji_train %>% ggplot(aes(x=kanji_freq, y=kanji_stroke_count, col = JLPT_grade)) +
  geom_point() +
  facet_wrap(vars(year_added)) +
  labs(title = "Kanji Stroke Count vs Frequency (by Year Added)", y = "Kanji Stroke Count", x = "Kanji Frequency")

kanji_train %>% filter(year_added == "1946") %>%
  ggplot(aes(x=kanji_freq, y=kanji_stroke_count, col = JLPT_grade)) +
  geom_point() +
  labs(title = "Kanji Stroke Count vs Frequency (in 1946)", y = "Kanji Stroke Count", x = "Kanji Frequency")

# kanji frequency vs stroke count (by JLPT level)
kanji_train %>% ggplot(aes(y=kanji_freq, x=kanji_stroke_count, col = grade)) + 
  geom_point() +
  facet_wrap(vars(JLPT_grade)) +
  labs(title = "Kanji Frequency vs Stroke Count (by JLPT Grade)", x = "Kanji Stroke Count", y = "Kanji Frequency")

# kanji stroke count vs JLPT grade (with and without an old version)
kanji_old_stroke %>% 
  ggplot(aes(x = as.numeric(JLPT_grade)-1, y = avg_stroke_count, col = old_logical)) +
  geom_line() +
  scale_x_continuous(breaks = seq(0,5,1)) +
  labs(title = "Kanji Stroke Count vs JLPT Grade", x = "JLPT Grade", y = "Average Kanji Stroke Count")

# kanji frequency vs stroke count (by classification)
plot1 <- kanji_train %>% ggplot(aes(x=kanji_freq, y=kanji_stroke_count, col = classification)) +
  geom_point() +
  labs(title = "Kanji Stroke Count vs Frequency", x = "Kanji Frequency", y = "Kanji Stroke Count")

plot2 <- kanji_train %>% ggplot(aes(x=n_on_readings, y=kanji_stroke_count, col = grade)) +
  geom_point()
plot3 <- kanji_train %>% ggplot(aes(x=n_on_meanings, y=kanji_stroke_count, col = grade)) +
  geom_point()
plot4 <- kanji_train %>% ggplot(aes(x=n_kun_meanings, y=kanji_stroke_count, col = grade)) +
  geom_point()

# compile plot1-plot4
grid.arrange(plot1, plot2, plot3, plot4)

####################### Step 4: Prediction Models #######################

predictors <- kanji_train[, c(3:14, 16:20)] # ignore ID, kanji, and old elements

## decision tree (categorical)
rpart_fit <- train(JLPT_grade ~ ., predictors, method = "rpart", 
                   tuneGrid = data.frame(cp = seq(0, 0.1, 0.01)))
rpart_fit

# plot the tree
plot(rpart_fit$finalModel, margin = 0.1)
text(rpart_fit$finalModel, cex = 0.75)
varImp(rpart_fit) # top 10 most important variables

# model accuracy (confusion matrix)
rpart_cm <- confusionMatrix(predict(rpart_fit, kanji_test), kanji_test$JLPT_grade)
rpart_cm$overall["Accuracy"] # 0.587

## random forest (can't have predictors with more than 4 factor levels)
rforest_fit <- randomForest(JLPT_grade~., data = predictors[, c(1:4, 6:12, 14:17)])

rforest_fit # uses mtry = 3
varImp(rforest_fit) # most important variables

# model accuracy  (confusion matrix)
rforest_cm <- confusionMatrix(predict(rforest_fit, kanji_test), kanji_test$JLPT_grade)
rforest_cm$overall["Accuracy"] # 0.65

## k-nearest neighbours
ctrl <- trainControl(method = "cv", number = 10, p = 0.9)
knn_fit <- train(JLPT_grade ~ ., data = predictors, method = "knn", 
                 tuneGrid = data.frame(k = seq(20, 30, 2)), trControl = ctrl)

# analyze selected model
knn_fit$bestTune # k = 28
varImp(knn_fit)

ggplot(knn_fit, highlight = TRUE)

# model accuracy (confusion matrix)
knn_cm <- confusionMatrix(predict(knn_fit, kanji_test), kanji_test$JLPT_grade)
knn_cm$overall["Accuracy"] # 0.545

# loess - won't work for categorical data

# qda/lda - won't work for all because some predictors are factors 
# qda gives rank deficiency issues
lda_fit <- train(JLPT_grade ~ kanji_stroke_count + radical_freq + n_on_readings + n_on_meanings +
                   n_kun_meanings + kanji_freq_proper + kanji_freq + joyo_freq + old_logical + grade
                   , data = predictors, method = "lda")
lda_fit$finalModel

lda_cm <- confusionMatrix(predict(lda_fit, kanji_test), kanji_test$JLPT_grade)
lda_cm$overall["Accuracy"] # 0.566

# glm only works for two-class outcomes

## ensemble model
# combine predictions of all models
predictions <- data.frame(rpart = predict(rpart_fit, kanji_test),
                          rforest = predict(rforest_fit, kanji_test),
                          knn = predict(knn_fit, kanji_test),
                          lda = predict(lda_fit, kanji_test))
str(predictions)

rows <- seq(1:nrow(predictions))
outcomes <- levels(kanji_test$JLPT_grade)

# create ensemble prediction by selecting the most common prediction acrosss all models
ensemble_pred <- sapply(rows, function(r){
  most_common(predictions[r,], outcomes)
})

# compile accuracy of each model in the ensemble
ensemble_acc <- data.frame(rpart = rpart_cm$overall["Accuracy"],
                           rforest = rforest_cm$overall["Accuracy"],
                           knn = knn_cm$overall["Accuracy"],
                           lda = lda_cm$overall["Accuracy"])

# determine accuracy of the ensemble prediction
confusionMatrix(factor(ensemble_pred), kanji_test$JLPT_grade)$overall["Accuracy"] # 0.613

## summary
data.frame(model = c("decision tree", "random forest", "knn", "lda", "ensemble"), 
           accuracy = c(rpart_cm$overall[["Accuracy"]], rforest_cm$overall[["Accuracy"]], 
                        knn_cm$overall[["Accuracy"]], lda_cm$overall[["Accuracy"]],
                        confusionMatrix(factor(ensemble_pred), kanji_test$JLPT_grade)$overall["Accuracy"]))

####################### Step 5: Final Model #######################

## randomForest model
confusionMatrix(predict(rforest_fit, kanji_test), kanji_test$JLPT_grade)$overall["Accuracy"]
