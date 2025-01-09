import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Path to the folder containing CSV files
data_folder = "Project1/data"

# Get a list of all CSV files in the folder
csv_files = [file for file in os.listdir(data_folder) if file.endswith('.csv')]

# Dictionary to store DataFrames for better management (optional)
dict = {}

# Loop through each CSV file
for file in csv_files:
    # Extract the name without extension
    name = os.path.splitext(file)[0]
    
    # Load the CSV file into a DataFrame
    df_ = pd.read_csv(os.path.join(data_folder, file))
    
    # Assign the DataFrame to a dynamically generated variable
    globals()[f"df_{name}"] = df_
    
    # Optionally, store it in the dictionary for easy access
    dict[name] = df_


for key in dict:
    print(key)
    #print(dict[key].head())

# plot the data
#sns.scatterplot(x='Health (Life Expectancy)', y='Happiness Score', data=dict['2015'])
#plt.show()

happiest = dict['2015'].nlargest(10, 'Happiness Score')
least_happy = dict['2015'].nsmallest(10, 'Happiness Score')

# Combine the two DataFrames
combined = pd.concat([happiest, least_happy])
df_sorted = combined.sort_values(by="Happiness Score", ascending=True)

# Create a color gradient from red to green based on scores
colors = plt.cm.RdYlGn((np.array(df_sorted['Happiness Score']) - min(df_sorted['Happiness Score'])) / (max(df_sorted['Happiness Score']) - min(df_sorted['Happiness Score'])))

# Create a horizontal bar chart
plt.barh(df_sorted['Country'], df_sorted['Happiness Score'], color = colors)

# Add labels and title
plt.xlabel('Happiness Score')
plt.ylabel('Country')
plt.title('Top 10 most and least happy countries')

# Add numerical values to the end of each bar
for index, value in enumerate(df_sorted["Happiness Score"]):
    plt.text(value - 0.2, index, f"{value:.2f}", va='center', ha='right', color='white', fontsize=9)

# Display the chart
#plt.show()

# Assuming dict['2015'] is a pandas DataFrame
df_2015 = dict['2015']  # Extract the DataFrame for the year 2015

# Extract the Happiness Score as a Series
happiness_score = df_2015['Happiness Score']

# Extract columns starting from the 5th column
x = df_2015.iloc[:, 5:]  # Adjusted to 0-based indexing

# Calculate Pearson correlation of each column with Happiness Score
correlation = x.corrwith(happiness_score)

# Print the correlation values
print(correlation)

# Plot the correlation values
plt.figure(figsize=(10, 5))
sns.barplot(x=correlation.values, y=correlation.index, palette='coolwarm')
plt.xlabel('Correlation')
plt.ylabel('Factors')
plt.title('Correlation of factors with Happiness Score')

# Number of columns to plot
num_columns = len(x.columns)
import math
# Calculate the number of rows and columns for subplots
cols = 3  # Number of columns in the grid
rows = math.ceil(num_columns / cols)

# Create subplots
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))

# Flatten the axes array for easy iteration
axes = axes.flatten()
from sklearn.linear_model import LinearRegression

# Linear regression model
model = LinearRegression()

# Iterate through each column and plot
for i, column in enumerate(x.columns):
    # Scatter plot
    axes[i].scatter(x[column], happiness_score, alpha=0.7, label='Data Points')
    
    # Fit linear regression
    x_values = x[column].values.reshape(-1, 1)  # Reshape for sklearn
    y_values = happiness_score.values
    model.fit(x_values, y_values)
    y_pred = model.predict(x_values)
    
    # Plot best-fit line
    axes[i].plot(x[column], y_pred, color='red', linestyle='-', linewidth=2, label='Best-Fit Line')
    
    # Titles and labels
    axes[i].set_title(f"{column} vs Happiness Score")
    axes[i].set_xlabel(column)
    axes[i].set_ylabel("Happiness Score")
    axes[i].legend()

# Turn off any unused axes
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing between rows and columns

# Display the plots
plt.show()