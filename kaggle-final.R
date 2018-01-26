install.packages("mgcv")
# general data visualization
library(ggplot2)
library(corrplot)

# general data manipulation
library(tidyverse) 
library(readr) 
library(data.table) 
library(tidyr) 
library(stringr) 

# modeling
library(caret) 
library(car)
library(randomForest)
library(zoo)
library(tm)
library(e1071)
library(gmodels)
library(rpart)
library(ROCR)
library(lmtest)

# Read in the training and test sets.

train <- fread('train.csv', na.strings=c("-1","-1.0"))
test <- fread('test.csv', na.strings=c("-1","-1.0"))

## Turn categorical features in factors and binary features into logical values.

train <- train %>%
  mutate_at(vars(ends_with("cat")), funs(factor)) %>%
  mutate_at(vars(ends_with("bin")), funs(as.logical)) %>%
  mutate(target = as.factor(target))
test <- test %>%
  mutate_at(vars(ends_with("cat")), funs(factor)) %>%
  mutate_at(vars(ends_with("bin")), funs(as.logical))

# Combine train and test sets for similar treatment.

combine <- bind_rows(train %>% mutate(dset = "train"), 
                     test %>% mutate(dset = "test",
                                     target = NA))
combine <- combine %>% mutate(dset = factor(dset))

# Examine a correlation matrix to determine feature relationships.

train %>%
  mutate_at(vars(ends_with("cat")), funs(as.integer)) %>%
  mutate_at(vars(ends_with("bin")), funs(as.integer)) %>%
  mutate(target = as.integer(target)) %>%
  cor(use="complete.obs", method = "spearman") %>%
  corrplot(type="lower", tl.col = "black",  diag=FALSE)

# None of the features appear to be highly correlated with the target. Much of the correlation seems to be 
# between like variables, and mosty within the ind, reg, and car features, not so much with the calc features.

# Feature Engineering

# I do not want to leave NA values in the data, because that will cause problems with some of the modeling functions.
# However, we want to see if the number of NA values for each person has an effect on whether or not they will file a 
# claim, as there appear to be a large number of NA values, and those could provide some insight. So we will engineer
# a new feature that tells us the number of NAs in each row.

# Number of NAs
nano <- combine[3:59] %>%
  is.na() %>%
  rowSums() %>%
  as.integer()

combine$nano <- nano

hist(as.integer(combine$nano))

# There are no observations that have exactly 5 NA values. Also, there appears to be a difference in claim rates
# between those with 0 - 4 NA values and those with 6 - 8, based on the plots in the Kaggle Kernel. 
#Perhaps this could be an important new feature.

featNA <- ifelse(combine$nano <= 4, 1, 0)

# We also want to sum up the binary features (ind and calc) to see if the number of these features has an effect on claim rates.

# Sum up "ind" binary columns
bin_ind <- combine %>% dplyr::select(ends_with("bin")) %>%
  dplyr::select(starts_with("ps_ind")) %>%
  rowSums() %>%
  as.integer()

combine$bin_ind <- bin_ind

hist(as.integer(combine$bin_ind))

# There appears to be a difference in claim rates between those with 0-2 and those with 3 or more "ind" binary features.

featBININD <- ifelse(combine$bin_ind <= 2, 1, 0)

# Sum up "calc" binary columns
bin_calc <- combine %>% dplyr::select(ends_with("bin")) %>%
  dplyr::select(starts_with("ps_calc")) %>%
  rowSums() %>%
  as.integer()

combine$bin_calc <- bin_calc

# No observed differences between the claim rates.

# We also want to create a column that shows how different these binary features are from one another, to determine if
# a larger difference has an effect on claim rates.

# "ind" binary column differences per row
# (uses "rep" to create a combine-size data frame from the
# single reference row that we then subtract from all rows)
bins <- combine %>%
  dplyr::select(ends_with("bin")) %>%
  dplyr::select(starts_with("ps_ind")) %>%
  mutate_all(funs(as.integer))

ref_bin <- bins %>% head(1)
ref_bin <- ref_bin[rep(1,nrow(combine)),]

diff_ind <- rowSums(abs(bins - ref_bin))

combine$diff_ind <- diff_ind

# There appears to be a difference in claim rates between differences of 1 - 2 vs differences greater than 3.

featDIFFIND <- ifelse(combine$diff_ind >= 3, 0, 1)

# "calc" binary column differences per row
bins <- combine %>%
  dplyr::select(ends_with("bin")) %>%
  dplyr::select(starts_with("ps_calc")) %>%
  mutate_all(funs(as.integer))

ref_bin <- bins %>% head(1)
ref_bin <- ref_bin[rep(1,nrow(combine)),]

diff_calc <- rowSums(abs(bins - ref_bin))

combine$diff_calc <- diff_calc

# No observed difference between claim rates.

# What to do with missing values...

# Reread data so features are in numeric/integer form

trainx <- fread('train.csv', na.strings=c("-1","-1.0"))
testx <- fread('test.csv', na.strings=c("-1","-1.0"))

combinex <- bind_rows(trainx %>% mutate(dset = "train"), 
                      testx %>% mutate(dset = "test",
                                       target = NA))
combinex <- combinex %>% mutate(dset = factor(dset))

str(combinex)

# Replace NA values with column mean

combinex[,3:59] <- lapply(combinex[,3:59], na.aggregate)

str(combinex)

# Round the columns that should be whole numbers/factors

combinex[,c(3:21,24:35,43:59)] <- lapply(combinex[,c(3:21,24:35,43:59)], round)

# Change back to factor/logical

combinex <- combinex %>%
  mutate_at(vars(ends_with("cat")), funs(factor)) %>%
  mutate_at(vars(ends_with("bin")), funs(as.logical)) %>%
  mutate(target = as.factor(target))

# Add in engineered features

combinex$bin_calc <- bin_calc
combinex$bin_ind <- bin_ind
combinex$diff_calc <- diff_calc
combinex$diff_ind <- diff_ind
combinex$nano <- nano
combinex$featBININD <- featBININD
combinex$featDIFFIND <- featDIFFIND
combinex$featNA <- featNA

str(combinex)

combinex[,61:68] <- sapply(combinex[,61:68], as.numeric)

str(combinex)

combinex$bin_calc <- as.factor(combinex$bin_calc)
combinex$bin_ind <- as.factor(combinex$bin_ind)
combinex$diff_calc <- as.factor(combinex$diff_calc)
combinex$diff_ind <- as.factor(combinex$diff_ind)
combinex$nano <- as.factor(combinex$nano)
combinex$featBININD <- as.factor(combinex$featBININD)
combinex$featDIFFIND <- as.factor(combinex$featDIFFIND)
combinex$featNA <- as.factor(combinex$featNA)

str(combinex)

# Split into train vs test
trainz <- combinex %>%
  filter(dset == "train")
testz <- combinex %>%
  filter(dset == "test")

# Write important dfs to csv so you don't have to rerun all of this when R crashes

write_csv(trainz, "trainz.csv")
write_csv(testz, "testz.csv")
write_csv(combinex, "combinex.csv")

# Feature Selection

# Rename df to keep original intact
variables <- trainz

str(variables)

# Cannot have factor with 104 levels

variables$ps_car_11_cat <- as.numeric(variables$ps_car_11_cat)

# Set x and y dataframes for RandomForest
x <- variables
y <- x$target
x$target <- NULL
x$id <- NULL
x$dset <- NULL

# Set seed for ranomized trials
set.seed(7)

# Run RandomForest model
rftest <- randomForest(
  x = x,
  y = y,
  ntree = 30,
  data = variables,
  importance = TRUE
)


# Calculate variable importance

var.importance <- data.frame(rftest$importance)
var.importance$var <- row.names(var.importance)

# Reorder
var.importance <- var.importance %>% 
  arrange(-MeanDecreaseAccuracy)

write_csv(var.importance, "var.importance.csv")

# Select the top 60% of features.

num <- round(nrow(var.importance)*.6)

var.list <- var.importance[1:num,5]

# Partition training data.

train1 <- filter(trainz, target == 1)
train0 <- filter(trainz, target == 0)

set.seed(7)
shuff1 <- train1[sample(nrow(train1), nrow(train1)),]
set.seed(7)
shuff0 <- train0[sample(nrow(train0), nrow(train0)),]

trainx1 <- shuff1[c(1:15186),]
trainx0 <- shuff0[c(1:401463),]

testx1 <- shuff1[15187:21694,]
testx0 <- shuff0[401464:573518,]

trainx <- rbind(trainx1,trainx0)
testx <- rbind(testx1, testx0)

write_csv(trainx, "trainx.csv")
write_csv(testx, "testx.csv")

trainxx <- dplyr::select(trainx, "target", var.list)
testxx <- dplyr::select(testx, "target", var.list)
testsub <- dplyr::select(testz, var.list)


## From this point on, if R ever restarts, come back to here and run the code below (after loading libraries) 
## to reset global environment.

trainx <- fread('trainx.csv')
testx <- fread('testx.csv')
var.importance <- fread('var.importance.csv')
testz <- fread('testz.csv')

num <- round(nrow(var.importance)*.6)

var.list <- var.importance[1:num,5]
var.list <- var.list$var

trainxx <- dplyr::select(trainx, "target", var.list)
testxx <- dplyr::select(testx, "target", var.list)
testsub <- dplyr::select(testz, var.list)

str(trainxx)

trainxx$target <- as.factor(trainxx$target)
trainxx <- trainxx %>%
  mutate_at(vars(ends_with("cat")), funs(factor))
trainxx$nano <- as.factor(trainxx$nano)
trainxx$diff_ind <- as.factor(trainxx$diff_ind)
trainxx$featDIFFIND <- as.factor(trainxx$featDIFFIND)
trainxx$diff_calc <- as.factor(trainxx$diff_calc)
trainxx$bin_calc <- as.factor(trainxx$bin_calc)
trainxx$bin_ind <- as.factor(trainxx$bin_ind)


testxx$target <- as.factor(testxx$target)
testxx <- testxx %>%
  mutate_at(vars(ends_with("cat")), funs(factor))
testxx$nano <- as.factor(testxx$nano)
testxx$diff_ind <- as.factor(testxx$diff_ind)
testxx$featDIFFIND <- as.factor(testxx$featDIFFIND)
testxx$diff_calc <- as.factor(testxx$diff_calc)
testxx$bin_calc <- as.factor(testxx$bin_calc)
testxx$bin_ind <- as.factor(testxx$bin_ind)

testsub <- testsub %>%
  mutate_at(vars(ends_with("cat")), funs(factor))
testsub$nano <- as.factor(testsub$nano)
testsub$diff_ind <- as.factor(testsub$diff_ind)
testsub$featDIFFIND <- as.factor(testsub$featDIFFIND)
testsub$diff_calc <- as.factor(testsub$diff_calc)
testsub$bin_calc <- as.factor(testsub$bin_calc)
testsub$bin_ind <- as.factor(testsub$bin_ind)

trainxx$nano <- NULL
testxx$nano <- NULL
trainsub <- rbind(trainxx, testxx)

testsub$nano <- NULL


# Model 1: Logistic Regression 

logmod <- glm(target ~ ., family = "binomial", data = trainxx)

summary(logmod)

predictlog <- predict(logmod, testxx, type = "response")

# Full train set

logmodsub <- glm(target ~ ., family = "binomial", data = trainsub)

predictlogsub <- predict(logmodsub, testsub, type = "response")

id <- dplyr::select(testz, 1)

logsubmission <- data.frame(id, predictlogsub)

colnames(logsubmission)[2] <- "target"

write_csv(logsubmission, "logsubmission.csv")

## This model scored .252 and ranked 3840th place on the Kaggle leaderboard.


# Model 2: Naive Bayes

nbmod <- naiveBayes(trainxx[,2:39], trainxx$target)

prediction <- predict(nbmod, testxx[,2:39], type = "class")

CrossTable(prediction, testxx$target,
           prop.chisq = F,
           prop.t = F,
           dnn = c('predicted', 'actual'))

nbmodsub <- naiveBayes(trainsub[,2:38], trainsub$target)

predictnbsub <- predict(nbmodsub, testsub, type = "raw")

nbsubmission <- data.frame(id, predictnbsub)

nbsubmission <- dplyr::select(nbsubmission, 1, 3)

colnames(nbsubmission)[2] <- "target"

write_csv(nbsubmission, "nbsubmission.csv")

## This model scored .221 on the Kaggle leaderboard. The logistic regression model is still the best.

# Model 3: Decision Trees

treemod <- rpart(target ~ ., data = trainxx, method = "class")

summary(treemod)

# Does not split. I will try lessening the control constraints on the model.

treemod <- rpart(target ~ ., data = trainxx, method = "class", control=rpart.control(minsplit=1, minbucket=1, cp=0.001))

summary(treemod)

# Still does not split.

predicttree <- predict(treemod, testxx)

# This model does not do any better than chance.



# The model that I will try to improve is the logistic regression model, since that is the model with the best initial performance.


### Model improvement

# Model evaluation 

# Variable Importance

importance <- varImp(logmod)
importance$var <- rownames(importance)
importance <- arrange(importance, Overall)

# Area under the curve


# Compute AUC for predicting Class with the model

pred <- prediction(predictlog, testxx$target)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf)
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]
auc

# Area under the curve is .627, perhaps this can be improved by removing some variables that appear to be less important
# to the prediction.

summary(logmod)

logmod2 <- glm(target ~ . - diff_ind - featDIFFIND - ps_car_11 - diff_calc - bin_calc - bin_ind, 
               family = "binomial", data = trainxx)

predictlog2 <- predict(logmod2, testxx, type = "response")

pred <- prediction(predictlog2, testxx$target)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf)
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]
auc

# The Area under the Curve was improved by .001 and is now .628. Let's compare the two models.

# Model comparison

lrtest(logmod, logmod2)

# The two models do not appear to be significantly different. Perhaps removing more of the insignificant variable can provide more
# improvement.

summary(logmod2)

logmod3 <- glm(target ~ . - diff_ind - featDIFFIND - ps_car_11 - diff_calc - bin_calc - bin_ind - ps_ind_07_bin -
                 ps_car_08_cat - ps_car_02_cat - ps_car_05_cat - ps_car_03_cat - ps_ind_08_bin, 
               family = "binomial", data = trainxx)

predictlog3 <- predict(logmod3, testxx, type = "response")

pred <- prediction(predictlog3, testxx$target)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf)
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]
auc

lrtest(logmod3, logmod)
lrtest(logmod3, logmod2)

# The Area under the Curve is back down to .627 and there is still no significant difference between any of the models. 

# It appears that the second model might be the "best" 

# I need to create a threshold to determine the resulting prediction in binary form in order to further evaluate my model.

predictfin <- ifelse(predictlog2 > 0.036, 1, 0)

CrossTable(predictfin, testxx$target,
           prop.chisq = F,
           prop.t = F,
           dnn = c('predicted', 'actual'))

# It appears that only 60.8% of my predictions were accurate, while the remaining 39.2% were incorrect.

logmodsub <- glm(target ~ . - diff_ind - featDIFFIND - ps_car_11 - diff_calc - bin_calc - bin_ind, 
                 family = "binomial", data = trainsub)


predictlogsub <- predict(logmodsub, testsub, type = "response")

predictsubfin <- ifelse(predictlogsub > 0.036, 1, 0)

id <- dplyr::select(testz, 1)

logsubmission <- data.frame(id, predictlogsub)
logbinary <- data.frame(id, predictsubfin)

colnames(logsubmission)[2] <- "target"
colnames(logbinary)[2] <- "target"

write_csv(logsubmission, "logsubmission.csv")
write_csv(logbinary, "logbinary.csv")

## This submission still scored .252 on the Kaggle leaderboard. Not an improvement over the original model. However, since
## this model has a slightly higher AUC, I will stick with this as my final model.

odds <- as.data.frame(exp(cbind(OddsRatio=coef(logmodsub),confint(logmodsub))))
odds 
