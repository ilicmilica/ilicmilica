#========= Import the libraries ============

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import joblib 
import seaborn as sns

#========= Functions ============

def get_files(csv_folder):

    dataframes = {}

    for file_name in os.listdir(csv_folder):
        
        if file_name.endswith('.csv'): # check if the file is a CSV file
            
            file_path = os.path.join(csv_folder, file_name)
            df = pd.read_csv(file_path) # read the CSV file into a DataFrame
            
            df.columns.values[0] = 'timestamp' # rename the first column to 'timestamp'
            df['timestamp'] = pd.to_datetime(df['timestamp'], format="%d.%m.%Y %H:%M") # change the 'timestamp' column to datetime format
            
            dataframes[file_name[:-4]] = df # store the DataFrame in the dictionary with the filename as the key

    return dataframes


                                        # ========= Delay calculation ==========
def calculate_delay(input, target):
    
    # extract the input and output columns
    input_variable = input.iloc[:, 1].values
    target_variable = target.iloc[:, 1].values

    lags = np.arange(-len(target_variable) + 1, len(target_variable)) # generate the lags

    # compute cross-correlation
    cross_corr = np.correlate(input_variable - input_variable.mean(), 
                            target_variable- target_variable.mean(), 
                            mode='full')

    optimal_lag = lags[np.argmax(cross_corr)] # find the lag with the highest correlation

    # calculate the delay in time units
    time_difference = pd.Timedelta(input['timestamp'].iloc[1] - input['timestamp'].iloc[0])
    total_hours = time_difference.total_seconds() / 3600
    delay_in_time_units = optimal_lag * total_hours

    # shift the target column by x rows back
    target_shifted =  target.shift(optimal_lag)

    return optimal_lag, delay_in_time_units, target_shifted


                                        # ========= Pearson and Spearman Correlation ==========

def get_top_influential_features(target, features_adjusted, top_n=2):

    features_columns = features_adjusted.columns  # extract all columns except the first one (timestamp)

    correlation_results = []

    # loop through each feature
    for column in features_columns:
        
        pearson_corr = target.corr(features_adjusted[column], method = 'pearson') # calculate Pearson Correlation 
        spearman_corr = target.corr(features_adjusted[column], method='spearman') # calculate Spearman Correlation
        
        correlation_results.append({ 
            'Feature Name': column,
            'Pearson Correlation': pearson_corr,
            'Spearman Correlation': spearman_corr
        }) # save the results

    correlation_df = pd.DataFrame(correlation_results)

    # sort by absolute value of Pearson Correlation, then Spearman Correlation
    correlation_df = correlation_df.reindex(correlation_df[['Pearson Correlation', 'Spearman Correlation']].abs()
        .sort_values(by=['Pearson Correlation', 'Spearman Correlation'], ascending=False).index)

    # vizualize the correlations
    x = np.arange(len(correlation_df))
    width = 0.2
    plt.figure(figsize=(10, 6))

    # plot side-by-side bars 
    plt.bar(x - width/2, correlation_df['Pearson Correlation'], width = 0.2, label='Pearson', color='cornflowerblue')
    plt.bar(x + width/2, correlation_df['Spearman Correlation'], width = 0.2, label='Spearman', color='indianred')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title("Feature Correlations with Target Variable", fontsize=14)
    plt.xlabel("Feature", fontsize=12)
    plt.ylabel("Correlation Coefficient", fontsize=12)
    plt.xticks(ticks=x, labels=correlation_df['Feature Name'], rotation=0, ha='center', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()

    # Select the top n features
    top_features_df = correlation_df.head(top_n)

    return top_features_df


                                        # ========= Adjust the features' frequency ==========

def adjust_features_freq(features):
   
    mean_values = []

    intervals_per_2h = 24 # number of 5-minute intervals in 2 hours (24 intervals)

    # loop through the input data manually in chunks of 24
    for i in range(0, len(features), intervals_per_2h):

        chunk = features.iloc[i:i + intervals_per_2h]
        chunk_mean = chunk.mean(numeric_only=True) # calculate the mean
        mean_values.append(chunk_mean) # append the result to the list

    features_adjusted = pd.DataFrame(mean_values) # create a new DataFrame from the list of means
    features_adjusted.reset_index(drop=True, inplace=True) # reset the index to have sequential values starting from 0
    
    return features_adjusted


                                        # ========= Model training - Linear Regression ==========

def train_model_lin_reg(X, y, model_path=None):
    """
    Trains a linear regression model on the provided features and target,
    evaluates it using MAE, RMSE, and R-squared, and visualizes the results.
    
    Parameters:
    X (pd.DataFrame)
    y (pd.Series): Target vector
    
    Returns:
    Trained linear regression model
    """
    
    # find the minimum length of the two
    min_length = min(len(X), len(y))

    # truncate both DataFrames to the minimum length
    X = X.iloc[:min_length]
    y = y.iloc[:min_length]

    # train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) # no shuffle because its timeseries

    # initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # predict the values for test set
    y_pred = model.predict(X_test)

    # evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # print the results
    print("Model evaluation results:")
    print("R-squared:", r2)
    print("Mean Absolute Error:", mae)
    print("Root Mean Squared Error:", rmse)
    print("\n")

    # plot true vs. predicted values (time series)
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label="True Values", linestyle='-', marker='o')
    plt.plot(y_pred, label="Predicted Values", linestyle='--', marker='x')
    plt.title("True vs. Predicted Values for Target Variable")
    plt.xlabel("Sample Index")
    plt.ylabel("Target Value")
    plt.legend()
    

    # scatter plot that shows true vs. predicted values
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 45-degree line
    plt.title("True vs. Predicted Values for Target Variable")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.show()

    # save the model if a path is provided
    if model_path:
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

    return model


                                        # ========= Plotting target vs features ==========


def plot_target_vs_feature(target, feature_columns):
    
    target = target.dropna()
    timestamp = target['timestamp']
    target = target.iloc[:,1].values

    # loop through each feature
    for column in feature_columns:

        plt.figure()

        # scaling the data so we can see their relationship better
        # set up MinMaxScaler with range [-1, 1]
        scaler_input = MinMaxScaler(feature_range=(-1, 1))
        scaler_target = MinMaxScaler(feature_range=(-1, 1))

        # apply the scaling
        input_scaled = scaler_input.fit_transform(feature_columns[column].values.reshape(-1, 1))
        target_scaled = scaler_target.fit_transform(target.reshape(-1, 1))

        # find the minimum length of the two
        min_length = min(len(target), len(feature_columns))

        # cut to the minimum length
        target_scaled = target_scaled[:min_length]
        input_scaled = input_scaled[:min_length]
        timestamp = timestamp.iloc[:min_length]

        # plot (time, target)
        plt.plot(timestamp, target_scaled, label='Target')

        # plot (time, feature)
        plt.plot(timestamp, input_scaled, label=f'{column}')

        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.title(f'Target Variable and {column} over time')
        plt.legend()
        plt.show()
