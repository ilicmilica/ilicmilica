# Project 3

## Target Variable Prediction

### Key Insights:
The code calculates the delay between the input and target variable and shifts it:

Optimal lag: -2 samples
Delay: -4.0h
Shifted by -4.0h...

The code then calculates the top N features that have the most influence on the target variable:

Top 2 influential features based on Pearson and Spearman Correlation:
['feature4', 'feature5']
![feature correlations with target variable](https://github.com/user-attachments/assets/6586e62c-3737-40ca-9170-70519a92ed0b)


### Predictive Modeling:
- The model achieved an R² score of 0.98, indicating strong predictive performance.
- Other results include:
  - Mean Absolute Error: 0.26
  - Root Mean Squared Error: 0.11
- The following graphs show true vs predicted target values:
![True vs Predicted](https://github.com/user-attachments/assets/24e2c2f3-03b8-4e32-bbda-350c233fbfd3)
![True vs Predicted time series](https://github.com/user-attachments/assets/23811115-5fdb-4c78-a096-f5557fbaeba0)

