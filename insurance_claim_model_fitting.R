
#setwd()
set.seed(500)
data_raw = read.csv("car_insurance_claim.csv")
#Dataset found at https://www.kaggle.com/datasets/xiaomengsun/car-insurance-claim-data

library(tidyr)
library(dplyr)
library(stringr)
library(mice)
library(rsample)
library(car)
library(splines)
library(caret)
library(MASS)
library(pROC)
library(car)
library(dplyr)
library(ggplot2)


### DATA CLEANING ###


#Renaming columns
colnames(data_raw) = c("id", "children_driving", "date_of_birth", "age", "children", "years_on_job",
                       "income", "single_parent", "home_value", "marrital_status", "gender",
                       "education", "occupation", "commute_time", "car_use", "car_value",
                       "policy_tenure", "car_type", "red_car", "claim_amount_last_5_years",
                       "number_claims_last_5_years", "revoked_license",
                       "license_points", "claim_amount", "car_age", "claim_flag", "urbanicity")


data_inter <- 0


data_inter <- data_raw %>%
  
  #removing duplicate rows
  distinct() %>%
  
  #Converting money columns to numeric data
  mutate(
    across(
      .cols = where(function(x) is.character(x) & any(str_detect(x, "\\$"))),
      .fns = ~.x %>% str_remove_all("[$,]") %>% as.numeric()
    )
  ) %>%
  
  #Removing "z_" from character data  
  mutate(
    across(
      .cols = where(\(x) is.character(x) & any(str_detect(x, "z_"))),
      .fns = ~.x %>% str_remove_all("z_")
    )
  ) %>%
  
  #Converting categorical data from chr to factor data
  mutate(
    across(
      .cols = where(function(x) is.character(x)),
      .fns = \(x) as.factor(x)
    )
  ) %>%
  
  #removing date of birth, not needed alongside age
  dplyr::select(-date_of_birth, -id) %>%
  
  #changing empty entries in occupation into NA values
  mutate(
    occupation = if_else(occupation == "", NA_character_, as.character(occupation)),
    occupation = factor(occupation)
  )


#Ordering factors that need it
data_inter$education <- ordered(data_inter$education,
                                levels = c("<High School", "High School", "Bachelors", "Masters", "PhD"))


#Functions to check for missing values

  #How many rows have a missing value
missing_val_prop <- function(x) {
  check <- apply(x, 1, \(x) any(is.na(x)))
  proportion <- sum(check) / nrow(x)
  return(proportion)
}

  #How many missing values does each variable have
missing_value_check <- function(x) {
  tab <- lapply(x, \(x) sum(is.na(x)))
  return(tab)
}

missing_val_prop(data_inter)
missing_value_check(data_inter)

#imputing full data set for EDA

eda_imputed_data <- mice(data_inter, m = 5, seed = 500)
data_clean_1 <- mice::complete(eda_imputed_data, 1)


### EXPLORATORY DATA ANALYSIS ###


#Viewing a snapshot of the imputed data and the proportion of customers that filed a claim
length(data_clean_1$claim_flag[data_clean_1$claim_flag == 1]) / nrow(data_clean_1)
head(data_clean_1)


#Box plots for numeric data vs Whether claim was filed

boxplot(data_clean_1$children_driving~data_clean_1$claim_flag,
        main = "Boxplot of Minors Driving Sorted by Claim Status",
        ylab = "Minors Drving",
        xlab = "(0 = No Claims Filed Within Policy Tenure)")

boxplot(data_clean_1$age~data_clean_1$claim_flag,
        main = "Boxplot of Age Sorted by Claim Status",
        ylab = "Age",
        xlab = "(0 = No Claims Filed Within Policy Tenure)")

boxplot(data_clean_1$children~data_clean_1$claim_flag,
        main = "Boxplot of Children Sorted by Claim Status",
        ylab = "Children",
        xlab = "(0 = No Claims Filed Within Policy Tenure)")

boxplot(data_clean_1$years_on_job~data_clean_1$claim_flag,
        main = "Boxplot of Years on Job Sorted by Claim Status",
        ylab = "Years on Job",
        xlab = "(0 = No Claims Filed Within Policy Tenure)")

boxplot(data_clean_1$income~data_clean_1$claim_flag,
        main = "Boxplot of Income Sorted by Claim Status",
        ylab = "Income",
        xlab = "(0 = No Claims Filed Within Policy Tenure)")

boxplot(data_clean_1$home_value~data_clean_1$claim_flag,
        main = "Boxplot of Home Value Sorted by Claim Status",
        ylab = "Home Value",
        xlab = "(0 = No Claims Filed Within Policy Tenure)")

boxplot(data_clean_1$commute_time~data_clean_1$claim_flag,
        main = "Boxplot of Daily Driving Time Sorted by Claim Status",
        ylab = "Driving Time in Minutes",
        xlab = "(0 = No Claims Filed Within Policy Tenure)")

boxplot(data_clean_1$car_value~data_clean_1$claim_flag,
        main = "Boxplot of Car Value Sorted by Claim Status",
        ylab = "Car Value",
        xlab = "(0 = No Claims Filed Within Policy Tenure)")

boxplot(data_clean_1$policy_tenure~data_clean_1$claim_flag,
        main = "Boxplot of Tenure Sorted by Claim Status",
        ylab = "# Months Policy Held",
        xlab = "(0 = No Claims Filed Within Policy Tenure)")

boxplot(data_clean_1$number_claims_last_5_years~data_clean_1$claim_flag,
        main = "Boxplot of Number of Claims Sorted by Claim Status",
        ylab = "Number of Claims",
        xlab = "(0 = No Claims Filed Within Policy Tenure)")

boxplot(data_clean_1$claim_amount_last_5_years~data_clean_1$claim_flag,
        main = "Boxplot of Claim Amount Sorted by Claim Status",
        ylab = "Claim Amount",
        xlab = "(0 = No Claims Filed Within Policy Tenure)")

boxplot(data_clean_1$license_points~data_clean_1$claim_flag,
        main = "Boxplot of License Points Sorted by Claim Status",
        ylab = "License Points",
        xlab = "(0 = No Claims Filed Within Policy Tenure)")

boxplot(data_clean_1$car_age~data_clean_1$claim_flag,
        main = "Boxplot of Car Age Sorted by Claim Status",
        ylab = "Car Age",
        xlab = "(0 = No Claims Filed Within Policy Tenure)")


#Tables for Categorical data vs whether a claim was Filed

ftable(data_clean_1$single_parent, data_clean_1$claim_flag)
ftable(data_clean_1$marrital_status, data_clean_1$claim_flag)
ftable(data_clean_1$gender, data_clean_1$claim_flag)
ftable(data_clean_1$education, data_clean_1$claim_flag)
ftable(data_clean_1$occupation, data_clean_1$claim_flag)
ftable(data_clean_1$car_use, data_clean_1$claim_flag)
ftable(data_clean_1$car_type, data_clean_1$claim_flag)
ftable(data_clean_1$red_car, data_clean_1$claim_flag)
ftable(data_clean_1$revoked_license, data_clean_1$claim_flag)
ftable(data_clean_1$urbanicity, data_clean_1$claim_flag)


###Checking linearity

#Age vs Logit Plot
age_binned <- data_clean_1 %>%
  mutate(bin = ntile(age, 20)) %>%       
  group_by(bin) %>%
  summarize(
    age_mid = mean(age),
    p = mean(claim_flag)
  ) %>%
  mutate(logit = log(p / (1 - p)))

ggplot(age_binned, aes(x = age_mid, y = logit)) +
  geom_point() +
  labs(title = "Age vs. Logit",
       x = "Age",
       y = "Log Odds")



#Years On Job vs Logit Plot
YoJ_binned <- data_clean_1 %>%
  mutate(bin = ntile(years_on_job, 10)) %>%       
  group_by(bin) %>%
  summarize(
    YoJ_mid = mean(years_on_job),
    p = mean(claim_flag)
  ) %>%
  mutate(logit = log(p / (1 - p)))

ggplot(YoJ_binned, aes(x = YoJ_mid, y = logit)) +
  geom_point() +
  labs(title = "Years on Job vs. Logit",
       x = "Years on Job",
       y = "Log Odds")


#Income vs Logit
income_binned <- data_clean_1 %>%
  mutate(bin = ntile(income, 20)) %>%       
  group_by(bin) %>%
  summarize(
    income_mid = mean(income),
    p = mean(claim_flag)
  ) %>%
  mutate(logit = log(p / (1 - p)))

ggplot(income_binned, aes(x = income_mid, y = logit)) +
  geom_point() +
  labs(title = "Income vs. Logit",
       x = "Income",
       y = "Log Odds")


#Home value vs Logit
home_value_binned <- data_clean_1 %>%
  mutate(bin = ntile(home_value, 20)) %>%       
  group_by(bin) %>%
  summarize(
    home_value_mid = mean(home_value),
    p = mean(claim_flag)
  ) %>%
  mutate(logit = log(p / (1 - p)))

ggplot(home_value_binned, aes(x = home_value_mid, y = logit)) +
  geom_point()+
  labs(title = "Home Value vs. Logit",
       x = "Home Value",
       y = "Log Odds")


#Commute Time vs Logit
commute_binned <- data_clean_1 %>%
  mutate(bin = ntile(commute_time, 20)) %>%       
  group_by(bin) %>%
  summarize(
    commute_mid = mean(commute_time),
    p = mean(claim_flag)
  ) %>%
  mutate(logit = log(p / (1 - p)))

ggplot(commute_binned, aes(x = commute_mid, y = logit)) +
  geom_point() +
  labs(title = "Commute Time vs. Logit",
       x = "Daily Driving Time in Minutes",
       y = "Log Odds")


#Car Value vs Logit
car_value_binned <- data_clean_1 %>%
  mutate(bin = ntile(car_value, 20)) %>%       
  group_by(bin) %>%
  summarize(
    car_value_mid = mean(car_value),
    p = mean(claim_flag)
  ) %>%
  mutate(logit = log(p / (1 - p)))

ggplot(car_value_binned, aes(x = car_value_mid, y = logit)) +
  geom_point() +
  labs(title = "Car Value vs. Logit",
       x = "Car Value",
       y = "Log Odds")



#Claim amount last 5 years vs Logit
amount_last_5_binned <- data_clean_1 %>%
  mutate(bin = ntile(claim_amount_last_5_years, 20)) %>%       
  group_by(bin) %>%
  summarize(
    amount_last_5_mid = mean(claim_amount_last_5_years),
    p = mean(claim_flag)
  ) %>%
  mutate(logit = log(p / (1 - p)))

ggplot(amount_last_5_binned, aes(x = amount_last_5_mid, y = logit)) +
  geom_point() +
  labs(title = "Claim Amount vs. Logit",
       x = "Total Claims in the Last 5 Years",
       y = "Log Odds")



#Claims last 5 years vs Logit
claim_last_5_binned <- data_clean_1 %>%
  mutate(bin = ntile(number_claims_last_5_years, 20)) %>%       
  group_by(bin) %>%
  summarize(
    claim_last_5_mid = mean(number_claims_last_5_years),
    p = mean(claim_flag)
  ) %>%
  mutate(logit = log(p / (1 - p)))

ggplot(claim_last_5_binned, aes(x = claim_last_5_mid, y = logit)) +
  geom_point() +
  labs(title = "Number of Claims vs. Logit",
       x = "Number of Claims in the Last 5 Years",
       y = "Log Odds")



#Car age vs Logit
car_age_binned <- data_clean_1 %>%
  mutate(bin = ntile(car_age, 20)) %>%       
  group_by(bin) %>%
  summarize(
    car_age_mid = mean(income),
    p = mean(claim_flag)
  ) %>%
  mutate(logit = log(p / (1 - p)))

ggplot(car_age_binned, aes(x = car_age_mid, y = logit)) +
  geom_point() +
  labs(title = "Car Age vs. Logit",
       x = "Car Age",
       y = "Log Odds")

#power transformations for claim number and claim amount
pt <- powerTransform(data_clean_1$number_claims_last_5_years, family = "yjPower")
number_claims_transformed <- yjPower(data_clean_1$number_claims_last_5_years, pt$lambda)
data_clean_1 <- data.frame(data_clean_1, number_claims_transformed)

pt2 <- powerTransform(data_clean_1$claim_amount_last_5_years, family = "yjPower")
claim_amount_transformed <- yjPower(data_clean_1$claim_amount_last_5_years, pt2$lambda)
data_clean_1 <- data.frame(data_clean_1, claim_amount_transformed)


#Looking at a model summary and checking multicollinearity
logit_1 <- glm(claim_flag ~ children_driving + bs(age, knots = c(49, 54), boundary.knots = c(15, 85), degree = 1) + children + years_on_job + 
                 income + single_parent + home_value + marrital_status + gender + education + occupation +
                 commute_time + car_use + car_value + policy_tenure + car_type +
                 red_car + number_claims_transformed + claim_amount_transformed +
                 revoked_license + license_points + car_age + urbanicity,
               family = binomial,
               data = data_clean_1)
vif(logit_1)
summary(logit_1)

#Removing occupation and transformed claim amount to eliminate multicollinearity
logit_2 <- glm(claim_flag ~ children_driving + bs(age, knots = c(49, 54), boundary.knots = c(15, 85), degree = 1) + children + years_on_job + 
                 income + single_parent + home_value + marrital_status + gender + education + 
                 commute_time + car_use + car_value + policy_tenure + car_type +
                 red_car + number_claims_transformed + 
                 revoked_license + license_points + car_age + urbanicity,
               family = binomial,
               data = data_clean_1)
vif(logit_2)
summary(logit_2)

#Using stepAIC to choose the final model
logit_3 <- stepAIC(logit_2, direction = "backward")
summary(logit_3)
vif(logit_3)

### CV MODELING ###

#power transformations for claim number and claim amount
pt <- powerTransform(data_inter$number_claims_last_5_years, family = "yjPower")
number_claims_transformed <- yjPower(data_inter$number_claims_last_5_years, pt$lambda)
data_inter <- data.frame(data_inter, number_claims_transformed)

pt2 <- powerTransform(data_inter$claim_amount_last_5_years, family = "yjPower")
claim_amount_transformed <- yjPower(data_inter$claim_amount_last_5_years, pt2$lambda)
data_inter <- data.frame(data_inter, claim_amount_transformed)

#creating folds for CV
K <- 5
folds <- createFolds(data_inter$claim_flag, k = K)

cv_results <- c()


for (k in 1:K) {
  test_idx <- folds[[k]]
  train_idx <- setdiff(seq_len(nrow(data_inter)), test_idx)
  
  train_data <- data_inter[train_idx, ]
  test_data  <- data_inter[test_idx, ]
  
  # 1. Impute ONLY the train data
  impute_train <- mice(train_data, m = 1, maxit = 5, printFlag = FALSE)
  
  # 2. Apply imputation *rules* to test data
  impute_test <- mice(test_data,
                       m = 5,
                       method = impute_train$method,
                       predictorMatrix = impute_train$predictorMatrix,
                       maxit = 0,        
                       visitSequence = impute_train$visitSequence,
                       printFlag = FALSE)
  imputed_test_data <- complete(impute_test, action = "all")
  
  
  # 3. Fit model on imputed training data
  fit <- with(impute_train,
              glm(claim_flag ~ children_driving + bs(age, knots = c(49, 54), boundary.knots = c(15, 85), degree = 1) + years_on_job + 
                    income + single_parent + home_value + marrital_status + education + 
                    commute_time + car_use + car_value + policy_tenure + car_type +
                    number_claims_transformed + revoked_license + license_points + urbanicity,
                  family = binomial,
                  data = train_data)
  )
  
  # 4. Predict on imputed test
  model_list <- fit$analyses
  
  pred_list <- Map(function(model, test_df) {
    predict(model, newdata = test_df, type = "response")
  }, model_list, imputed_test_data)
  
  final_fold_pred <- rowMeans(do.call(cbind, pred_list))
  
  
  
  cv_results[[k]] <- final_fold_pred
}

#combining each cv result into one vector
final_pred = 0
for(i in 1:K) {
  final_pred[folds[[i]]] <- cv_results[[i]]
}

#prediction threshold & Confusion Matrix
predicted_value <- factor(ifelse(final_pred > 0.35, 1, 0))
actual_value <- factor(data_inter$claim_flag)

confusion_matrix <- confusionMatrix(data = predicted_value,
                                    reference = actual_value,
                                    positive = "1")
print(confusion_matrix)

# Plot ROC curve
roc_obj <- roc(data_inter$claim_flag, final_pred)
plot(roc_obj, main = "ROC Curve", col = "blue", lwd = 2)

# AUC
auc(roc_obj)
