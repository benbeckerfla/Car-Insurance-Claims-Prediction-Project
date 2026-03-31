Car Insurance Claims Modeling
================
Benjamin Becker
2026-03-31

## Purpose

The goal of this project is to predict the probability that a customer
will file an insurance claim (claim_flag) using customer demographics
and vehicle history.

**Data Source:** This project utilizes the [Car Insurance Claim
Data](https://www.kaggle.com/datasets/xiaomengsun/car-insurance-claim-data)
provided by Xiaomengsun on Kaggle.

## Data Cleaning

First, all of the columns were renamed to be more clear and descriptive.
I cleaned the raw data by converting currency strings to numeric values
and converting data to the appropriate types. I also checked for missing
values, and used the MICE package to complete one imputed dataset for
EDA.

    ##    INCOME    OCCUPATION     EDUCATION
    ## 1 $67,349  Professional           PhD
    ## 2 $91,449 z_Blue Collar z_High School
    ## 3 $52,881       Manager     Bachelors
    ## 4 $16,039      Clerical z_High School
    ## 5         z_Blue Collar  <High School

    ##   income    education   occupation
    ## 1  67349          PhD Professional
    ## 2  91449  High School  Blue Collar
    ## 3  52881    Bachelors      Manager
    ## 4  16039  High School     Clerical
    ## 5     NA <High School  Blue Collar

I created two short functions to check for missing values. The first
determines how many values are null for each column; the second returns
the proportion of observations that contained a null value for at least
one variable.

``` r
missing_value_check(data_inter)
```

    ## $children_driving
    ## [1] 0
    ## 
    ## $age
    ## [1] 7
    ## 
    ## $children
    ## [1] 0
    ## 
    ## $years_on_job
    ## [1] 548
    ## 
    ## $income
    ## [1] 570
    ## 
    ## $single_parent
    ## [1] 0
    ## 
    ## $home_value
    ## [1] 575
    ## 
    ## $marrital_status
    ## [1] 0
    ## 
    ## $gender
    ## [1] 0
    ## 
    ## $education
    ## [1] 0
    ## 
    ## $occupation
    ## [1] 665
    ## 
    ## $commute_time
    ## [1] 0
    ## 
    ## $car_use
    ## [1] 0
    ## 
    ## $car_value
    ## [1] 0
    ## 
    ## $policy_tenure
    ## [1] 0
    ## 
    ## $car_type
    ## [1] 0
    ## 
    ## $red_car
    ## [1] 0
    ## 
    ## $claim_amount_last_5_years
    ## [1] 0
    ## 
    ## $number_claims_last_5_years
    ## [1] 0
    ## 
    ## $revoked_license
    ## [1] 0
    ## 
    ## $license_points
    ## [1] 0
    ## 
    ## $claim_amount
    ## [1] 0
    ## 
    ## $car_age
    ## [1] 639
    ## 
    ## $claim_flag
    ## [1] 0
    ## 
    ## $urbanicity
    ## [1] 0

``` r
missing_val_prop(data_inter)
```

    ## [1] 0.2567712

Because more than 25% of the observations contained a null value for at
least one variable, simply deleting these rows would lose a large amount
of information. Therefore, I decided to use the MICE package to create
imputed datasets. I completed one dataset to use for exploratory data
analysis.

## Exploratory Data Analysis and Model Selection

Box plots and frequency tables were created to view the distribution of
the values of each variable. I used logit plots to check the assumption
of linearity. The plots for age, claim_amount_last_5_years, and
number_claims_last_5_years all showed non-linear relationships:

![](Car-Insurance-Claims-Markdown_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->![](Car-Insurance-Claims-Markdown_files/figure-gfm/unnamed-chunk-8-2.png)<!-- -->![](Car-Insurance-Claims-Markdown_files/figure-gfm/unnamed-chunk-8-3.png)<!-- -->

I applied Yeo-Johnson transformations to the claim count and amount
variables to linearize their relationship with the log-odds of a claim.
These were added to the dataset as number_claims_transformed and
claim_amount_transformed:

![](Car-Insurance-Claims-Markdown_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->![](Car-Insurance-Claims-Markdown_files/figure-gfm/unnamed-chunk-9-2.png)<!-- -->

The relationship between age and the log-odds of a claim was linear
until around 50 years; I used a b spline within the model’s formula to
account for this higher risk associated with older drivers.

``` r
#First model
logit_1 <- glm(claim_flag ~ children_driving +
          bs(age, knots = c(49, 54), Boundary.knots = c(15, 85), degree = 1) +
          children + years_on_job + income + single_parent + home_value +
          marrital_status + gender + education + occupation +
          commute_time + car_use + car_value + policy_tenure + car_type +
          red_car + number_claims_transformed + claim_amount_transformed +
          revoked_license + license_points + car_age + urbanicity,
               family = binomial,
               data = data_clean_1)
```

I used the vif function to check for multicollinearity, and the Akaike
Information Criterion to select the model. Number_claims_transformed and
claim_amount_transformed both had high adjusted GVIFs, so the claim
amount was removed from the model.

|                           | GVIF^(1/(2\*Df)) |
|:--------------------------|-----------------:|
| children_driving          |             1.18 |
| age_spline                |             1.10 |
| children                  |             1.50 |
| years_on_job              |             1.23 |
| income                    |             1.69 |
| single_parent             |             1.39 |
| home_value                |             1.44 |
| marrital_status           |             1.47 |
| gender                    |             1.92 |
| education                 |             1.35 |
| occupation                |             1.23 |
| commute_time              |             1.02 |
| car_use                   |             1.52 |
| car_value                 |             1.47 |
| policy_tenure             |             1.01 |
| car_type                  |             1.20 |
| red_car                   |             1.35 |
| number_claims_transformed |             5.26 |
| claim_amount_transformed  |             5.28 |
| revoked_license           |             1.02 |
| license_points            |             1.11 |
| car_age                   |             1.47 |
| urbanicity                |             1.07 |

Multicollinearity Check (VIF)

|                           | GVIF^(1/(2\*Df)) |
|:--------------------------|-----------------:|
| children_driving          |             1.18 |
| age_spline                |             1.10 |
| children                  |             1.50 |
| years_on_job              |             1.23 |
| income                    |             1.69 |
| single_parent             |             1.39 |
| home_value                |             1.44 |
| marrital_status           |             1.47 |
| gender                    |             1.92 |
| education                 |             1.35 |
| occupation                |             1.23 |
| commute_time              |             1.02 |
| car_use                   |             1.52 |
| car_value                 |             1.47 |
| policy_tenure             |             1.01 |
| car_type                  |             1.20 |
| red_car                   |             1.35 |
| number_claims_transformed |             1.12 |
| revoked_license           |             1.00 |
| license_points            |             1.11 |
| car_age                   |             1.47 |
| urbanicity                |             1.07 |

Multicollinearity Check (VIF)

## Cross Validation and Model Evaluation

5 Fold Cross Validation was used to ensure the model generalizes well.
Imputation was performed within each fold to prevent data leakage.

``` r
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
              glm(claim_flag ~ children_driving +
          bs(age, knots = c(49, 54), Boundary.knots = c(15, 85), degree = 1) +
          years_on_job + income + single_parent + home_value +
          marrital_status + education + occupation + commute_time +
          car_use + car_value + policy_tenure + car_type +
          number_claims_transformed + revoked_license + license_points +
          urbanicity,
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
```

I used the confusion matrix and ROC curve to evaluate the model.
Originally, a prediction threshold of 0.5 was used. However, this
resulted in the model having a very low sensitivity, with a high number
of false negative predictions.

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 6927 1533
    ##          1  628 1213
    ##                                          
    ##                Accuracy : 0.7902         
    ##                  95% CI : (0.7822, 0.798)
    ##     No Information Rate : 0.7334         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.4006         
    ##                                          
    ##  Mcnemar's Test P-Value : < 2.2e-16      
    ##                                          
    ##             Sensitivity : 0.4417         
    ##             Specificity : 0.9169         
    ##          Pos Pred Value : 0.6589         
    ##          Neg Pred Value : 0.8188         
    ##              Prevalence : 0.2666         
    ##          Detection Rate : 0.1178         
    ##    Detection Prevalence : 0.1787         
    ##       Balanced Accuracy : 0.6793         
    ##                                          
    ##        'Positive' Class : 1              
    ## 

This is a poor initial result, as failing to predict this many claims
could lead an insurance company to lose a lot of money. I changed the
prediction threshold to 0.35, which greatly improved the sensitivity of
the model at the cost of only a small amount of accuracy.

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 6098  940
    ##          1 1457 1806
    ##                                          
    ##                Accuracy : 0.7673         
    ##                  95% CI : (0.759, 0.7754)
    ##     No Information Rate : 0.7334         
    ##     P-Value [Acc > NIR] : 1.688e-15      
    ##                                          
    ##                   Kappa : 0.4386         
    ##                                          
    ##  Mcnemar's Test P-Value : < 2.2e-16      
    ##                                          
    ##             Sensitivity : 0.6577         
    ##             Specificity : 0.8071         
    ##          Pos Pred Value : 0.5535         
    ##          Neg Pred Value : 0.8664         
    ##              Prevalence : 0.2666         
    ##          Detection Rate : 0.1753         
    ##    Detection Prevalence : 0.3168         
    ##       Balanced Accuracy : 0.7324         
    ##                                          
    ##        'Positive' Class : 1              
    ## 

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

![](Car-Insurance-Claims-Markdown_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

    ## Area under the curve: 0.8152
