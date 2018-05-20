---
title: "Exercise Data Prediction with Machine Learning"
author: "Jeff Patton"
date: "5/12/2018"
output:
  html_document:
    keep_md: yes
  pdf_document: null
---
## Synopsis

We built a random forest (rf) machine learning model to predict whether an arm weight lifting exercise was being performed correctly or incorrectly.  There were 5 different methods of performing the exercise, with Class A being correct - and Classes B-E being incorrect. ("classe" is the response in the dataset)

The rf training model showed an accuracy of greater than 99% for the cross-validation on a 5-fold number of tries.  When this random forest model was applied to the test (hold-out) set we measured 100% accuracy in classifying the 5 different exercise methods.  Additional, linear discriminant analysis and a gradient boosted model were also constructed and compared.  


```r
library(knitr)
output <- opts_knit$get("rmarkdown.pandoc.to")
if (output=="html") opts_chunk$set(fig.width=6, fig.height=6)
if (output=="docx") opts_chunk$set(fig.width=6,  fig.height=6)
```

## Data Processing


```r
library(lubridate)
library(ggplot2)
library(data.table)
library(tidyr)
```


```r
if (!file.exists("./data")) {
    dir.create("./data")
}
```

Download, unzip, and read in the data.  This was done on Mac OS, so downloaded in 'wb' mode.


```r
fileurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(fileurl, destfile = "./data/pml-training.csv", mode = 'wb')
fileurl2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(fileurl2, destfile = "./data/pml-testing.csv", mode = 'wb')

pm1_train <- read.csv("./data/pml-training.csv", stringsAsFactors = FALSE)
pm1_test <- read.csv("./data/pml-testing.csv", stringsAsFactors = FALSE)
```

Take a preliminary look at the data.


```r
pm1_train <- read.csv("./data/pml-training.csv", stringsAsFactors = TRUE)
pm1_test <- read.csv("./data/pml-testing.csv", stringsAsFactors = TRUE)
dt_pm1_train <- data.table(pm1_train)
dt_pm1_test <- data.table(pm1_test)
str(dt_pm1_train)
```

```
## Classes 'data.table' and 'data.frame':	19622 obs. of  160 variables:
##  $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
##  $ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
##  $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt        : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ kurtosis_roll_belt      : Factor w/ 397 levels "","-0.016850",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_picth_belt     : Factor w/ 317 levels "","-0.021887",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_yaw_belt       : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_roll_belt      : Factor w/ 395 levels "","-0.003095",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_roll_belt.1    : Factor w/ 338 levels "","-0.005928",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_yaw_belt       : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
##  $ max_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_belt            : Factor w/ 68 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ min_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_belt            : Factor w/ 68 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ amplitude_roll_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_belt    : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_belt      : Factor w/ 4 levels "","#DIV/0!","0.00",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ var_total_accel_belt    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_belt        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_belt       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_belt         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_belt_x            : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
##  $ gyros_belt_y            : num  0 0 0 0 0.02 0 0 0 0 0 ...
##  $ gyros_belt_z            : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
##  $ accel_belt_x            : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
##  $ accel_belt_y            : int  4 4 5 3 2 4 3 4 2 4 ...
##  $ accel_belt_z            : int  22 22 23 21 24 21 21 21 24 22 ...
##  $ magnet_belt_x           : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
##  $ magnet_belt_y           : int  599 608 600 604 600 603 599 603 602 609 ...
##  $ magnet_belt_z           : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
##  $ roll_arm                : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
##  $ pitch_arm               : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
##  $ yaw_arm                 : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm         : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ var_accel_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_arm         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_arm          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_arm_x             : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
##  $ gyros_arm_y             : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
##  $ gyros_arm_z             : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
##  $ accel_arm_x             : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
##  $ accel_arm_y             : int  109 110 110 111 111 111 111 111 109 110 ...
##  $ accel_arm_z             : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
##  $ magnet_arm_x            : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
##  $ magnet_arm_y            : int  337 337 344 344 337 342 336 338 341 334 ...
##  $ magnet_arm_z            : int  516 513 513 512 506 513 509 510 518 516 ...
##  $ kurtosis_roll_arm       : Factor w/ 330 levels "","-0.02438",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_picth_arm      : Factor w/ 328 levels "","-0.00484",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_yaw_arm        : Factor w/ 395 levels "","-0.01548",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_roll_arm       : Factor w/ 331 levels "","-0.00051",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_pitch_arm      : Factor w/ 328 levels "","-0.00184",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_yaw_arm        : Factor w/ 395 levels "","-0.00311",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ max_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_arm     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_arm       : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ roll_dumbbell           : num  13.1 13.1 12.9 13.4 13.4 ...
##  $ pitch_dumbbell          : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
##  $ yaw_dumbbell            : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
##  $ kurtosis_roll_dumbbell  : Factor w/ 398 levels "","-0.0035","-0.0073",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_picth_dumbbell : Factor w/ 401 levels "","-0.0163","-0.0233",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_yaw_dumbbell   : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_roll_dumbbell  : Factor w/ 401 levels "","-0.0082","-0.0096",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_pitch_dumbbell : Factor w/ 402 levels "","-0.0053","-0.0084",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_yaw_dumbbell   : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
##  $ max_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_dumbbell        : Factor w/ 73 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ min_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_dumbbell        : Factor w/ 73 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ amplitude_roll_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##   [list output truncated]
##  - attr(*, ".internal.selfref")=<externalptr>
```

```r
unique(dt_pm1_train$classe)
```

```
## [1] A B C D E
## Levels: A B C D E
```

There are 159 predictors, we probably need to look to remove a few.  It looks like there are lots of columns that have "na" values.  Additionally, when reviewing the background on this data, it sounds like we don't need skewness and kurtosis data. http://groupware.les.inf.puc-rio.br/har [ref 1]
We should be able to get rid of columns:
"max", "min", "amplitude", "var", "avg", "stddev", "skewness", "kurtosis", "X", "user_name", "raw_timestamp", "cvtd_timestamp", "new_window", "num_window"




```r
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

dim(dt_pm1_train_small)
```

```
## [1] 19622    53
```

Excellent, our training set is now a bit more manageable with just 52 predictors, from 159, before.  Now we will start trying to build a few different models on our training set.


```r
set.seed(42111)
library(caret)
library(lattice)
library(rpart)
library(gbm)
library(randomForest)
```


```r
## We are referencing the github page on how to run random forest with parallel processing:
## https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md [ref 2]
```


```r
##install.packages("parallel")
##install.packages("doParallel")
library(parallel)
library(doParallel)
```


```r
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE)

## In general, I would like to develop better ML skills for how I go from a 
## large training set > model selection.
## For this exercise, I picked 3 different model types and looked at their accuracy.

## here is our linear discriminant analysis model
lda_train <- train(classe ~., method = "lda", data = dt_pm1_train_small, 
                   verbose = FALSE, trControl = fitControl)
lda_train
```

```
## Linear Discriminant Analysis 
## 
## 19622 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 15696, 15699, 15697, 15698, 15698 
## Resampling results:
## 
##   Accuracy   Kappa   
##   0.7009993  0.621617
```

```r
## 70% accuracy on the training set using a 5-fold cross-validation method.

## here is our gradient boosted model
gbm_train <- train(classe ~., method = "gbm", data = dt_pm1_train_small, 
                   verbose = FALSE, trControl = fitControl)
gbm_train
```

```
## Stochastic Gradient Boosting 
## 
## 19622 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 15698, 15698, 15697, 15698, 15697 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.7526751  0.6863970
##   1                  100      0.8203544  0.7726509
##   1                  150      0.8570480  0.8190509
##   2                   50      0.8577105  0.8197238
##   2                  100      0.9067373  0.8819872
##   2                  150      0.9322187  0.9142355
##   3                   50      0.8961369  0.8684554
##   3                  100      0.9421568  0.9268071
##   3                  150      0.9618284  0.9517060
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```

```r
## 96% accuracy on the training set using a 5-fold cross-validation and n.trees = 150.

## here is our random forest model
rf_train <- train(classe ~ ., method = "rf", 
                  data = dt_pm1_train_small, verbose = FALSE, trControl = fitControl)
rf_train
```

```
## Random Forest 
## 
## 19622 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 15697, 15698, 15697, 15699, 15697 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9946999  0.9932955
##   27    0.9948527  0.9934889
##   52    0.9888894  0.9859451
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 27.
```

```r
## 99.5% accuracy on the training set using a 5-fold cross validation and mtry = 27.  mtry = 2 was nearly as 
## good as mtry = 27!

stopCluster(cluster)
registerDoSEQ()

## Since random forest had the highest accuracy on our training set, we will use it to predict on our
## test set (hold out) data.  
## This class project is a little different than normal since the test set we were given had missing predictors.
## This was done so that we could blind-test our prediction model and enter it in as a quiz.  
## I really wanted to show a confusion matrix of the test data set!
rf_test <- predict(rf_train, dt_pm1_test)
rf_test
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
## We used this output to enter into the quiz portion.  
## 100% on the quiz! That means our rf model is quite good at discerning correct exercise "A" versus 
## incorrect exercise "B-E" classes.  
#  So now I know that rf_test is basically the "classe" response column from the test set.

pm1_test_w_classe <- data.frame(dt_pm1_test, "classe"=rf_test) 

lda_test <- predict(lda_train, pm1_test_w_classe)
gbm_test <- predict(gbm_train, pm1_test_w_classe)

## rf_train$resample
confusionMatrix.train(rf_train)
```

```
## Cross-Validated (5 fold) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.4  0.1  0.0  0.0  0.0
##          B  0.0 19.2  0.1  0.0  0.0
##          C  0.0  0.0 17.3  0.1  0.0
##          D  0.0  0.0  0.1 16.2  0.0
##          E  0.0  0.0  0.0  0.0 18.3
##                             
##  Accuracy (average) : 0.9949
```

```r
confusionMatrix(rf_test, pm1_test_w_classe$classe)  
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction A B C D E
##          A 7 0 0 0 0
##          B 0 8 0 0 0
##          C 0 0 1 0 0
##          D 0 0 0 1 0
##          E 0 0 0 0 3
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.8316, 1)
##     No Information Rate : 0.4        
##     P-Value [Acc > NIR] : 1.1e-08    
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity              1.00      1.0     1.00     1.00     1.00
## Specificity              1.00      1.0     1.00     1.00     1.00
## Pos Pred Value           1.00      1.0     1.00     1.00     1.00
## Neg Pred Value           1.00      1.0     1.00     1.00     1.00
## Prevalence               0.35      0.4     0.05     0.05     0.15
## Detection Rate           0.35      0.4     0.05     0.05     0.15
## Detection Prevalence     0.35      0.4     0.05     0.05     0.15
## Balanced Accuracy        1.00      1.0     1.00     1.00     1.00
```

```r
## we expect this to be 100% accurate, and this is indeed true.

confusionMatrix(lda_test, pm1_test_w_classe$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction A B C D E
##          A 5 0 1 0 1
##          B 0 6 0 0 0
##          C 2 0 0 0 0
##          D 0 2 0 1 0
##          E 0 0 0 0 2
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7             
##                  95% CI : (0.4572, 0.8811)
##     No Information Rate : 0.4             
##     P-Value [Acc > NIR] : 0.006466        
##                                           
##                   Kappa : 0.589           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7143   0.7500   0.0000   1.0000   0.6667
## Specificity            0.8462   1.0000   0.8947   0.8947   1.0000
## Pos Pred Value         0.7143   1.0000   0.0000   0.3333   1.0000
## Neg Pred Value         0.8462   0.8571   0.9444   1.0000   0.9444
## Prevalence             0.3500   0.4000   0.0500   0.0500   0.1500
## Detection Rate         0.2500   0.3000   0.0000   0.0500   0.1000
## Detection Prevalence   0.3500   0.3000   0.1000   0.1500   0.1000
## Balanced Accuracy      0.7802   0.8750   0.4474   0.9474   0.8333
```

```r
## the lda model had lower accuracy, and this is proven out in the test data.

confusionMatrix(gbm_test, pm1_test_w_classe$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction A B C D E
##          A 7 0 0 0 0
##          B 0 8 0 0 0
##          C 0 0 1 0 0
##          D 0 0 0 1 0
##          E 0 0 0 0 3
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.8316, 1)
##     No Information Rate : 0.4        
##     P-Value [Acc > NIR] : 1.1e-08    
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity              1.00      1.0     1.00     1.00     1.00
## Specificity              1.00      1.0     1.00     1.00     1.00
## Pos Pred Value           1.00      1.0     1.00     1.00     1.00
## Neg Pred Value           1.00      1.0     1.00     1.00     1.00
## Prevalence               0.35      0.4     0.05     0.05     0.15
## Detection Rate           0.35      0.4     0.05     0.05     0.15
## Detection Prevalence     0.35      0.4     0.05     0.05     0.15
## Balanced Accuracy        1.00      1.0     1.00     1.00     1.00
```

```r
## interesting!  The gbm model had a slightly lower 96% accuracy for n.trees = 150 but 
## matched the performance of the rf model and predicted with 100% accuracy on the test data.
```


```r
plot(rf_train, main="Accuracy by Predictor Count")
```

![](index_files/figure-html/unnamed-chunk-11-1.png)<!-- -->

```r
varImpPlot(rf_train$finalModel, main="variable importance plot: Random Forest")
```

![](index_files/figure-html/unnamed-chunk-11-2.png)<!-- -->

```r
## The "belt" and "pitch forearm" predictors are most important to the rf model.
```

## Conclusions

We were able to build a random forest model with 100% accuracy to predict whether an arm weight lifting exercise was being performed correctly or incorrectly.  

## References

1.  Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. Accessed on 5/19/18.
http://groupware.les.inf.puc-rio.br/har

2.  Greski, Leonard; Improving Performance of Random Forest in caret::train(). Accessed on 5/19/2018. https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md
