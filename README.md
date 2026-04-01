# Car-Insurance-Claims-Prediction-Project
This repository contains an R-based machine learning project that predicts the likelihood of an auto insurance customer filing a claim (claim_flag). The project utilizes logistic regression and demonstrates an end-to-end data science workflow, including data cleaning, exploratory data analysis, feature engineering, and cross-validated model evaluation. The goal is to build a model to predict whether or not a customer will file a claim during the tenure of their policy.

## Data Source
The dataset used in this project is the Car Insurance Claim Data provided by xiaomengsun on Kaggle. You can find the original data and documentation here: https://www.kaggle.com/datasets/xiaomengsun/car-insurance-claim-data

## Methodology
- Data Cleaning & Formatting: Cleaned raw text fields (e.g., removing $ and z_ prefixes), formatted currency values to numeric, and appropriately typed categorical features as factors (including ordered factors for education levels).
- Missing Data Imputation: Handled missing values using Multiple Imputation by Chained Equations (mice).
- Exploratory Data Analysis (EDA): Generated boxplots, frequency tables, and binned logit plots to visualize relationships between predictors and the target variable, as well as to check for linearity assumptions.
- Feature Engineering: Applied Yeo-Johnson power transformations to heavily skewed variables (e.g., historical claim counts and amounts). Used B-splines to model non-linear relationships (e.g., Age).
- Model Selection: Built a baseline binomial logistic regression model.
- Checked for multicollinearity using Variance Inflation Factors (VIF) and iteratively removed problematic predictors.
- Performed backward selection using stepAIC to isolate the most parsimonious, high-performing model.
- Cross-Validation: Implemented 5-fold cross-validation. To prevent data leakage, imputation rules were fit strictly on the training folds and subsequently applied to the test folds.

## Model Evaluation
The model's predictive capability is assessed using:
- A confusion matrix with a 0.35 probability threshold
- ROC Curve & AUC to evaluate the model's ability to distinguish between classes across all classification thresholds.

## Dependencies
This project requires R and the following packages:
- dplyr, tidyr, stringr (Data manipulation)
- mice (Missing data imputation)
- rsample, caret (Data splitting and cross-validation)
- car, MASS, splines (Transformations, VIF, and Stepwise selection)
- ggplot2 (Data visualization)
- pROC (Model evaluation)

