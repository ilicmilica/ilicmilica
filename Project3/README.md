# Project 3

## Target Variable Prediction: A Time-Series Modeling Project
### Overview
This project focuses on predictive modeling for time-series data, analyzing relationships between input, output, and feature variables to optimize delay compensation and forecast future outcomes. Using a combination of statistical analysis and machine learning, the project identifies influential features and achieves a high-performing predictive model.
### Dataset Description
The dataset consists of seven time-series variables:

- Five feature variables (sampled every 5 minutes)
- One input variable (sampled every 2 hours)
- One output/target variable (sampled every 2 hours)

The objective is to:

1. Determine the delay (lag) between the input and output variables.
2. Identify the most influential features affecting the output.
3. Build a predictive model to forecast the output variable based on the input and selected features.
### Key Insights:
#### 1. Delay Analysis
The project calculated the delay (lag) between the input and output variables to align their time-series.
Results:
- Optimal Lag: -2 samples
- Delay: -4.0 hours
- The time-series data was shifted accordingly by -4.0 hours to account for this lag.
#### 2. Feature Selection
The correlation between the features and the target variable was analyzed using Pearson and Spearman correlation coefficients.
The top two most influential features were identified as:
- Feature4
- Feature5

Correlation analysis results are illustrated in the graph below:

![feature correlations with target variable](https://github.com/user-attachments/assets/6586e62c-3737-40ca-9170-70519a92ed0b)


#### 3. Predictive Modeling:
A predictive model was developed to estimate the output variable based on the input variable and the most influential features.

#### Model Results:
- R² Score: 0.98 (indicating strong predictive accuracy)
- Mean Absolute Error (MAE): 0.26
- Root Mean Squared Error (RMSE): 0.11

#### Visual Results
The model’s performance is visualized below, showing the close alignment between true and predicted values:

1. Scatterplot of True vs Predicted Values:

![True vs Predicted](https://github.com/user-attachments/assets/24e2c2f3-03b8-4e32-bbda-350c233fbfd3)

2. Time-Series Comparison of True vs Predicted Values:

![True vs Predicted time series](https://github.com/user-attachments/assets/23811115-5fdb-4c78-a096-f5557fbaeba0)

### Skills Demonstrated

#### 1. Time-Series Analysis

Handling time-series data with varying sampling intervals (e.g., 5 minutes vs. 2 hours).
Identifying and compensating for delays (lags) in time-series.
#### 2. Feature Engineering

Using correlation analysis (Pearson and Spearman) for feature selection.
Selecting influential predictors for improved model performance.
#### 3. Predictive Modeling

Building regression models for high-dimensional time-series data.
Evaluating models using metrics like R², MAE, and RMSE.
Visualizing model performance through scatterplots and time-series comparisons.
#### 4. Data Manipulation and Visualization

Shifting and aligning datasets based on lag analysis.
Plotting insights for feature relationships and model performance.
### Conclusion
This project successfully predicts a target variable in a time-series context, achieving high accuracy with an R² of 0.98. The approach of combining delay analysis, feature selection, and robust modeling highlights its utility in scenarios where time-dependent relationships are critical.
