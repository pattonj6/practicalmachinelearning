
library(knitr)
##output <- opts_knit$get("rmarkdown.pandoc.to")
##if (output=="html") opts_chunk$set(fig.width=10, fig.height=10)
##if (output=="docx") opts_chunk$set(fig.width=6,  fig.height=6)

## Data Processing

library(lubridate)
library(ggplot2)
library(dplyr)
library(data.table)
library(tidyr)

if (!file.exists("./data")) {
         dir.create("./data")
}

## Download, unzip, and read in the data.  This was done on Mac OS, 
## so downloaded in 'wb' mode.

fileurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(fileurl, destfile = "./data/pml-training.csv", mode = 'wb')

fileurl2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(fileurl2, destfile = "./data/pml-testing.csv", mode = 'wb')

pm1_train <- read.csv("./data/pml-training.csv", stringsAsFactors = TRUE)
pm1_test <- read.csv("./data/pml-testing.csv", stringsAsFactors = TRUE)
dt_pm1_train <- data.table(pm1_train)
dt_pm1_test <- data.table(pm1_test)

## Take a preliminary look at the data.  $classe is our response

str(dt_pm1_train)
str(dt_pm1_test)
unique(dt_pm1_train$classe)
tail(dt_pm1_train)

## looks like there are lots of columns that we should be able to get rid of :
## "max", "min", "amplitude", "var", "avg", "stddev", "skewness_yaw_belt", "kurtosis_yaw_belt"
## "X", "user_name", "raw_timestamp", "cvtd_timestamp", "new_window", "num_window"
#### lots of NA in columns, any we can throw out in $classe?
#### to save some time, I deleted these columns in excel, reimported

#### pm1_train_small <- read.csv("./data/pml-training_small.csv", stringsAsFactors = TRUE)

## lets get rid of the easy, unique columns first
drop_cols <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", 
               "cvtd_timestamp", "new_window", "num_window")

dt_pm1_train_small <- dt_pm1_train[, (drop_cols) := NULL]
dt_pm1_train_small <- data.table(dt_pm1_train_small)

dt_pm1_train_small <- dt_pm1_train_small %>% 
         select(-starts_with("max")) %>% 
         select(-starts_with("min")) %>% 
         select(-starts_with("amplitude")) %>% 
         select(-starts_with("var")) %>% 
         select(-starts_with("avg")) %>% 
         select(-starts_with("stddev")) %>% 
         select(-starts_with("skewness")) %>%
         select(-starts_with("kurtosis"))

#### dt_pm1_train_test <- dt_pm1_train_small[, .SD, .SDcols = !names(dt_pm1_train_small) %like% "max" | "min" | "amplitude" | "var"]
#### mydt[, .SD, .SDcols = ! names(mydt) %like% "foo"]
#### dfrm2 <- dfrm[ , -grep("\\.B$", names(dfrm)) ]
#### sapply(pm1_train$classe, function(x)all(is.na(x)))
#### no rows with NA
#### there are a lot of predictors (106), so let's try PCA

str(dt_pm1_train_small)
set.seed(42111)
library(caret)
library(lattice)
library(rpart)


#### pca_preprocess <- preProcess(pm1_train, method = "pca")
#### pca_train <- predict(pca_preprocess, pm1_train)
#### pca_model <- rpart(pm1_train$classe ~ ., method = "class", data = pca_train)

## referencing the github page: 
## https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md

install.packages("parallel")
install.packages("doParallel")
library(parallel)
library(doParallel)

cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE)

rf_train <- train(classe ~ ., method = "rf", 
                  data = pm1_train_small, verbose = FALSE, trControl = fitControl)

stopCluster(cluster)
registerDoSEQ()

rf_test <- predict(rf_train, dt_pm1_test)

rf_train$resample
confusionMatrix.train(rf_train)