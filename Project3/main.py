'''
This script performs target variable delay calculation, feature selection, 
and model training to predict target variable value based on input features.

All variables are time series data with a timestamp column.
Variables include:
- input1: features
- input2: input variable
- target: target variable
'''
#========= Import the libraries ============

import pandas as pd
from functions import get_files, calculate_delay, get_top_influential_features, adjust_features_freq, train_model_lin_reg, plot_target_vs_feature

#========= Get the files ============

path = 'Project3/csv_files' # folder with csv files

dataframes= get_files(path) # files are stored in the dataframe

features = dataframes['input1'] # input features
input = dataframes['input2'] # input variable
target = dataframes['target'] # target variable

print("Files uploaded...")

#========= Calculate the delay ============
# Delay for the target variable is calculated and then it is shifted accordingly

print("Calculating the delay...")
optimal_lag, delay_in_time_units, target_shifted = calculate_delay(input, target)

print(f"Optimal lag: {optimal_lag} samples")
print(f"Delay: {delay_in_time_units}h")
print(f"Shifted by {delay_in_time_units}h...\n")

#========= Adjusting the features frequency (every 5 minutes) to target frequency (every 2 hours) ============

features_adjusted = adjust_features_freq(features)

#========= Feature Analysis: Identification of input features that are most crucial in predicting the target variable =========
# I decided to do Pearson and Spearman correlation to see which features have the most correlation with the target variable
# The results showed that feature4 and feature5 have the most influence on target variable
# I also visually inspected the signals just to confirm it

                                        # ====== Pearson and Spearman =======

# Call the function with the specified top_n value
top_n=2
top_n_features = get_top_influential_features(target_shifted.iloc[:,1], features_adjusted, top_n)

feature_names = top_n_features['Feature Name'].tolist()

# Print the top influential features
print(f"Top {top_n} influential features based on Pearson and Spearman Correlation:")
print(feature_names)
print("\n")


                                        # ========= Plotting target vs features ==========

#plot_target_vs_feature(target_shifted, features_adjusted)                                        


#========= Model training =========

# After determining the influential features, model is trained
# X - feature 4, feature 5 and input variable
# y - target variable

                                        # ========= Linear Regression ==========

# combining input variable and selected features into one dataframe
selected_features = features_adjusted[feature_names]
combined_df = pd.concat([input.iloc[:,1], selected_features], axis=1)
combined_df = combined_df.dropna()

X = combined_df # no timestamp column
y = target_shifted.iloc[:,1] # removing the timestamp column

trained_model = train_model_lin_reg(X, y, model_path = "Project3/models/linear_regression_model.joblib")

# When evaluated the model shows really good values: 
# R-squared (R² -> 1)~ 0.982, Mean Absolute Error (MAE -> 0)~ 0.256 and Root Mean Squared Error (RMSE -> 0)~ 0.109

