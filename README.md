# IPL 1st Inning Score Prediction using Machine Learning

An end-to-end Machine Learning pipeline implemented in Python to predict the final 1st-inning score of an IPL match based on live match features (runs, wickets, overs, and recent run-rates). 

## Project Overview
Predicting a cricket match score dynamically requires handling sequence-like changes in run rates and wicket falls. This project walks through the entire data science workflow—from data cleaning and handling categorical layout mismatches using Scikit-Learn's `ColumnTransformer`, to optimizing model training and evaluating multiple regression algorithms.

## Dataset & Feature Engineering
The initial dataset consists of **76,014 rows** with 15 columns representing match specifics. 

### Data Cleaning Steps:
1. **Irrelevant Feature Removal:** Dropped identifiers and high-cardinality features (`mid`, `date`, `venue`, `batsman`, `bowler`, `striker`, `non-striker`).
2. **Team Filtering:** Narrowed down the dataset to the 8 most consistent/active IPL franchises.
3. **Thresholding Constraints:** Removed the first 5 overs of every match (`overs >= 5.0`) since score trends stabilize better after the powerplay.
4. **Categorical Encoding:** Applied `OneHotEncoder` via a `ColumnTransformer` to safely map `batting_team` and `bowling_team` without creating structural layout disparities.

## Models Trained & Evaluated
The performance of five different regression algorithms was evaluated using $R^2$ Score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE):

| Model | Train Score ($R^2$) | Test Score ($R^2$) | MAE | RMSE |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **99.06%** | **93.25%** | **4.54** | **7.77** |
| **Decision Tree Regressor** | 99.98% | 86.68% | 3.87 | 10.92 |
| **Neural Network (MLPRegressor)** | 86.01% | 84.17% | 8.44 | 11.91 |
| **Linear Regression** | 66.18% | 64.84% | 13.28 | 17.75 |
| **LinearSVR (Support Vector Machine)** | 66.05% | 64.70% | 13.27 | 17.78 |

*Note: Random Forest emerged as the optimal model, offering a balanced fit and the lowest prediction variance (RMSE: 7.77).*



### Prerequisites
Make sure you have the following packages installed:
```bash
pip install numpy pandas seaborn scikit-learn joblib

```

## This project is only made for educational purpose. It is not intended for any commercial use.
