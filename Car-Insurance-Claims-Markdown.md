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
           children + years_on_job + 
           income + single_parent + home_value +
           marrital_status + gender + education + occupation +
           commute_time + car_use + car_value + policy_tenure + car_type +
           red_car + number_claims_transformed + claim_amount_transformed +
           revoked_license + license_points + car_age + urbanicity,
          family = binomial,
          data = data_clean_1)
```

I used the vif function to check for multicollinearity, and the Akaike
Information Criterion to select the model. Education, occupation,
number_claims_transformed, and claim_amount_transformed all had high
variance inflation factors (Greater than 10).

    ##                                                                         GVIF Df
    ## children_driving                                                    1.395542  1
    ## bs(age, knots = c(49, 54), Boundary.knots = c(15, 85), degree = 1)  1.806750  3
    ## children                                                            2.247500  1
    ## years_on_job                                                        1.501948  1
    ## income                                                              2.859038  1
    ## single_parent                                                       1.936250  1
    ## home_value                                                          2.066210  1
    ## marrital_status                                                     2.171742  1
    ## gender                                                              3.690382  1
    ## education                                                          10.811983  4
    ## occupation                                                         18.594840  7
    ## commute_time                                                        1.037108  1
    ## car_use                                                             2.308267  1
    ## car_value                                                           2.169075  1
    ## policy_tenure                                                       1.011778  1
    ## car_type                                                            6.433528  5
    ## red_car                                                             1.820918  1
    ## number_claims_transformed                                          27.624467  1
    ## claim_amount_transformed                                           27.906608  1
    ## revoked_license                                                     1.036966  1
    ## license_points                                                      1.235498  1
    ## car_age                                                             2.166280  1
    ## urbanicity                                                          1.145601  1
    ##                                                                    GVIF^(1/(2*Df))
    ## children_driving                                                          1.181330
    ## bs(age, knots = c(49, 54), Boundary.knots = c(15, 85), degree = 1)        1.103612
    ## children                                                                  1.499166
    ## years_on_job                                                              1.225540
    ## income                                                                    1.690869
    ## single_parent                                                             1.391492
    ## home_value                                                                1.437432
    ## marrital_status                                                           1.473683
    ## gender                                                                    1.921037
    ## education                                                                 1.346599
    ## occupation                                                                1.232171
    ## commute_time                                                              1.018385
    ## car_use                                                                   1.519298
    ## car_value                                                                 1.472778
    ## policy_tenure                                                             1.005872
    ## car_type                                                                  1.204606
    ## red_car                                                                   1.349414
    ## number_claims_transformed                                                 5.255898
    ## claim_amount_transformed                                                  5.282671
    ## revoked_license                                                           1.018315
    ## license_points                                                            1.111529
    ## car_age                                                                   1.471829
    ## urbanicity                                                                1.070328

Education and Occupation are likely highly correlated, as are the number
of claims and the amount of claims, so occupation and claim amount were
removed to decrease multicollinearity.

    ##                                                                        GVIF Df
    ## children_driving                                                   1.393066  1
    ## bs(age, knots = c(49, 54), Boundary.knots = c(15, 85), degree = 1) 1.769264  3
    ## children                                                           2.211319  1
    ## years_on_job                                                       1.231204  1
    ## income                                                             2.459435  1
    ## single_parent                                                      1.937068  1
    ## home_value                                                         1.975479  1
    ## marrital_status                                                    2.142924  1
    ## gender                                                             3.647515  1
    ## education                                                          3.206399  4
    ## commute_time                                                       1.036311  1
    ## car_use                                                            1.590823  1
    ## car_value                                                          2.167936  1
    ## policy_tenure                                                      1.010398  1
    ## car_type                                                           5.853559  5
    ## red_car                                                            1.815503  1
    ## number_claims_transformed                                          1.242685  1
    ## revoked_license                                                    1.005446  1
    ## license_points                                                     1.220993  1
    ## car_age                                                            2.152538  1
    ## urbanicity                                                         1.138058  1
    ##                                                                    GVIF^(1/(2*Df))
    ## children_driving                                                          1.180282
    ## bs(age, knots = c(49, 54), Boundary.knots = c(15, 85), degree = 1)        1.099762
    ## children                                                                  1.487051
    ## years_on_job                                                              1.109596
    ## income                                                                    1.568259
    ## single_parent                                                             1.391786
    ## home_value                                                                1.405517
    ## marrital_status                                                           1.463873
    ## gender                                                                    1.909847
    ## education                                                                 1.156784
    ## commute_time                                                              1.017994
    ## car_use                                                                   1.261278
    ## car_value                                                                 1.472391
    ## policy_tenure                                                             1.005186
    ## car_type                                                                  1.193279
    ## red_car                                                                   1.347406
    ## number_claims_transformed                                                 1.114758
    ## revoked_license                                                           1.002719
    ## license_points                                                            1.104986
    ## car_age                                                                   1.467153
    ## urbanicity                                                                1.066798

Using the stepAIC function gave the following as the final model:

    ## 
    ## Call:
    ## glm(formula = claim_flag ~ children_driving + bs(age, knots = c(49, 
    ##     54), Boundary.knots = c(15, 85), degree = 1) + years_on_job + 
    ##     income + single_parent + home_value + marrital_status + education + 
    ##     commute_time + car_use + car_value + policy_tenure + car_type + 
    ##     number_claims_transformed + revoked_license + license_points + 
    ##     urbanicity, family = binomial, data = data_clean_1)
    ## 
    ## Coefficients:
    ##                                                                       Estimate
    ## (Intercept)                                                         -1.868e+00
    ## children_driving                                                     5.166e-01
    ## bs(age, knots = c(49, 54), Boundary.knots = c(15, 85), degree = 1)1 -1.383e+00
    ## bs(age, knots = c(49, 54), Boundary.knots = c(15, 85), degree = 1)2 -8.917e-01
    ## bs(age, knots = c(49, 54), Boundary.knots = c(15, 85), degree = 1)3  8.735e-01
    ## years_on_job                                                        -1.170e-02
    ## income                                                              -3.389e-06
    ## single_parentYes                                                     3.198e-01
    ## home_value                                                          -1.373e-06
    ## marrital_statusYes                                                  -4.817e-01
    ## education.L                                                         -6.327e-01
    ## education.Q                                                          9.867e-02
    ## education.C                                                          1.322e-01
    ## education^4                                                         -1.667e-01
    ## commute_time                                                         1.619e-02
    ## car_usePrivate                                                      -8.715e-01
    ## car_value                                                           -2.349e-05
    ## policy_tenure                                                       -5.659e-02
    ## car_typePanel Truck                                                  5.041e-01
    ## car_typePickup                                                       4.252e-01
    ## car_typeSports Car                                                   8.328e-01
    ## car_typeSUV                                                          5.968e-01
    ## car_typeVan                                                          5.822e-01
    ## number_claims_transformed                                            8.018e-01
    ## revoked_licenseYes                                                   7.634e-01
    ## license_points                                                       9.237e-02
    ## urbanicityHighly Urban/ Urban                                        2.429e+00
    ##                                                                     Std. Error
    ## (Intercept)                                                          2.151e-01
    ## children_driving                                                     5.016e-02
    ## bs(age, knots = c(49, 54), Boundary.knots = c(15, 85), degree = 1)1  1.707e-01
    ## bs(age, knots = c(49, 54), Boundary.knots = c(15, 85), degree = 1)2  1.525e-01
    ## bs(age, knots = c(49, 54), Boundary.knots = c(15, 85), degree = 1)3  4.680e-01
    ## years_on_job                                                         6.742e-03
    ## income                                                               9.316e-07
    ## single_parentYes                                                     8.974e-02
    ## home_value                                                           3.022e-07
    ## marrital_statusYes                                                   7.334e-02
    ## education.L                                                          9.648e-02
    ## education.Q                                                          7.214e-02
    ## education.C                                                          6.192e-02
    ## education^4                                                          5.330e-02
    ## commute_time                                                         1.674e-03
    ## car_usePrivate                                                       6.600e-02
    ## car_value                                                            4.200e-06
    ## policy_tenure                                                        6.583e-03
    ## car_typePanel Truck                                                  1.272e-01
    ## car_typePickup                                                       8.749e-02
    ## car_typeSports Car                                                   9.493e-02
    ## car_typeSUV                                                          7.579e-02
    ## car_typeVan                                                          1.079e-01
    ## number_claims_transformed                                            1.075e-01
    ## revoked_licenseYes                                                   7.134e-02
    ## license_points                                                       1.240e-02
    ## urbanicityHighly Urban/ Urban                                        1.044e-01
    ##                                                                     z value
    ## (Intercept)                                                          -8.681
    ## children_driving                                                     10.299
    ## bs(age, knots = c(49, 54), Boundary.knots = c(15, 85), degree = 1)1  -8.103
    ## bs(age, knots = c(49, 54), Boundary.knots = c(15, 85), degree = 1)2  -5.848
    ## bs(age, knots = c(49, 54), Boundary.knots = c(15, 85), degree = 1)3   1.867
    ## years_on_job                                                         -1.735
    ## income                                                               -3.638
    ## single_parentYes                                                      3.564
    ## home_value                                                           -4.543
    ## marrital_statusYes                                                   -6.568
    ## education.L                                                          -6.558
    ## education.Q                                                           1.368
    ## education.C                                                           2.136
    ## education^4                                                          -3.127
    ## commute_time                                                          9.669
    ## car_usePrivate                                                      -13.205
    ## car_value                                                            -5.593
    ## policy_tenure                                                        -8.596
    ## car_typePanel Truck                                                   3.964
    ## car_typePickup                                                        4.860
    ## car_typeSports Car                                                    8.773
    ## car_typeSUV                                                           7.874
    ## car_typeVan                                                           5.395
    ## number_claims_transformed                                             7.459
    ## revoked_licenseYes                                                   10.701
    ## license_points                                                        7.446
    ## urbanicityHighly Urban/ Urban                                        23.268
    ##                                                                     Pr(>|z|)
    ## (Intercept)                                                          < 2e-16
    ## children_driving                                                     < 2e-16
    ## bs(age, knots = c(49, 54), Boundary.knots = c(15, 85), degree = 1)1 5.38e-16
    ## bs(age, knots = c(49, 54), Boundary.knots = c(15, 85), degree = 1)2 4.97e-09
    ## bs(age, knots = c(49, 54), Boundary.knots = c(15, 85), degree = 1)3 0.061951
    ## years_on_job                                                        0.082687
    ## income                                                              0.000275
    ## single_parentYes                                                    0.000366
    ## home_value                                                          5.55e-06
    ## marrital_statusYes                                                  5.08e-11
    ## education.L                                                         5.46e-11
    ## education.Q                                                         0.171379
    ## education.C                                                         0.032687
    ## education^4                                                         0.001764
    ## commute_time                                                         < 2e-16
    ## car_usePrivate                                                       < 2e-16
    ## car_value                                                           2.24e-08
    ## policy_tenure                                                        < 2e-16
    ## car_typePanel Truck                                                 7.36e-05
    ## car_typePickup                                                      1.17e-06
    ## car_typeSports Car                                                   < 2e-16
    ## car_typeSUV                                                         3.43e-15
    ## car_typeVan                                                         6.84e-08
    ## number_claims_transformed                                           8.70e-14
    ## revoked_licenseYes                                                   < 2e-16
    ## license_points                                                      9.60e-14
    ## urbanicityHighly Urban/ Urban                                        < 2e-16
    ##                                                                        
    ## (Intercept)                                                         ***
    ## children_driving                                                    ***
    ## bs(age, knots = c(49, 54), Boundary.knots = c(15, 85), degree = 1)1 ***
    ## bs(age, knots = c(49, 54), Boundary.knots = c(15, 85), degree = 1)2 ***
    ## bs(age, knots = c(49, 54), Boundary.knots = c(15, 85), degree = 1)3 .  
    ## years_on_job                                                        .  
    ## income                                                              ***
    ## single_parentYes                                                    ***
    ## home_value                                                          ***
    ## marrital_statusYes                                                  ***
    ## education.L                                                         ***
    ## education.Q                                                            
    ## education.C                                                         *  
    ## education^4                                                         ** 
    ## commute_time                                                        ***
    ## car_usePrivate                                                      ***
    ## car_value                                                           ***
    ## policy_tenure                                                       ***
    ## car_typePanel Truck                                                 ***
    ## car_typePickup                                                      ***
    ## car_typeSports Car                                                  ***
    ## car_typeSUV                                                         ***
    ## car_typeVan                                                         ***
    ## number_claims_transformed                                           ***
    ## revoked_licenseYes                                                  ***
    ## license_points                                                      ***
    ## urbanicityHighly Urban/ Urban                                       ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 11945.5  on 10300  degrees of freedom
    ## Residual deviance:  9188.3  on 10274  degrees of freedom
    ## AIC: 9242.3
    ## 
    ## Number of Fisher Scoring iterations: 5

## Cross Validation and Model Evaluation

5 Fold Cross Validation was used to ensure the model generalizes well.
Imputation was performed within each fold to prevent data leakage.

``` r
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
            glm(claim_flag ~ children_driving + 
            bs(age, knots = c(49, 54), Boundary.knots = c(15, 85), degree = 1) +
            years_on_job + income + single_parent + home_value +
            marrital_status + education + commute_time + car_use +
            car_value + policy_tenure + car_type + number_claims_transformed +
            revoked_license + license_points + urbanicity,
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
    ##          0 6943 1590
    ##          1  612 1156
    ##                                           
    ##                Accuracy : 0.7862          
    ##                  95% CI : (0.7782, 0.7941)
    ##     No Information Rate : 0.7334          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.3834          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ##                                           
    ##             Sensitivity : 0.4210          
    ##             Specificity : 0.9190          
    ##          Pos Pred Value : 0.6538          
    ##          Neg Pred Value : 0.8137          
    ##              Prevalence : 0.2666          
    ##          Detection Rate : 0.1122          
    ##    Detection Prevalence : 0.1716          
    ##       Balanced Accuracy : 0.6700          
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
    ##          0 6141  977
    ##          1 1414 1769
    ##                                          
    ##                Accuracy : 0.7679         
    ##                  95% CI : (0.7596, 0.776)
    ##     No Information Rate : 0.7334         
    ##     P-Value [Acc > NIR] : 5.556e-16      
    ##                                          
    ##                   Kappa : 0.435          
    ##                                          
    ##  Mcnemar's Test P-Value : < 2.2e-16      
    ##                                          
    ##             Sensitivity : 0.6442         
    ##             Specificity : 0.8128         
    ##          Pos Pred Value : 0.5558         
    ##          Neg Pred Value : 0.8627         
    ##              Prevalence : 0.2666         
    ##          Detection Rate : 0.1717         
    ##    Detection Prevalence : 0.3090         
    ##       Balanced Accuracy : 0.7285         
    ##                                          
    ##        'Positive' Class : 1              
    ## 

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

![](Car-Insurance-Claims-Markdown_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

    ## Area under the curve: 0.8124
