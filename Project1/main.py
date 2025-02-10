import os
import pandas as pd
import matplotlib.pyplot as plt
from functions import topN, top3_corr, plot_world_map

# path to the folder that contains the CSV files
data_folder = "Project1/data"

df_2015 = pd.read_csv(os.path.join(data_folder, "2015.csv"))
df_2016 = pd.read_csv(os.path.join(data_folder, "2016.csv"))
df_2017 = pd.read_csv(os.path.join(data_folder, "2017.csv"))
df_2018 = pd.read_csv(os.path.join(data_folder, "2018.csv"))
df_2019 = pd.read_csv(os.path.join(data_folder, "2019.csv"))

# plot top 10 happiest and least happy countries
topN(df_2015, N=10)

# calculate top 3 features that contribute the most to the happiness score based on Pearson correlation
top_features_2015 = top3_corr(df_2015, year=2015)
#top_features_2016 = top3_corr(df_2016, year=2016)

# plot the world map of happiness scores
plot_world_map(df_2015, year=2015)

# which region dominates the top 10 happiest countries, take first 10 values of region column
x = df_2015['Region'].head(10)
region = x.value_counts().idxmax()
print(f"The region that dominates the top 10 happiest countries in 2015 is {region}.")

# which region dominated the top 10 lowest happiness scores, take last 10 values of region column
x = df_2015['Region'].tail(10)
region = x.value_counts().idxmax()
print(f"The region that dominates the top 10 least happy countries in 2015 is {region}.")

##############################################

# predict the happiness score of a country based on the top 3 features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# prepare the data
X = pd.concat([df_2015[top_features_2015], df_2016[top_features_2015], df_2017[top_features_2015]])
y = pd.concat([df_2015['Happiness Score'], df_2016['Happiness Score'], df_2017['Happiness Score']])

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)   

# create a linear regression model
model = LinearRegression()

# train the model
model.fit(X_train, y_train)

# make predictions
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

# plot the predicted vs actual happiness score
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Actual Happiness Score')
plt.ylabel('Predicted Happiness Score')
plt.title('Predicted vs Actual Happiness Score')
plt.show()

#When evaluated the model shows really good values: 
''' R-squared (R² -> 1), Mean Absolute Error (MAE -> 0) and Root Mean Squared Error (RMSE -> 0)

Model evaluation results:
R-squared: 0.714747953263662
Mean Absolute Error: 0.47400573165033194
Root Mean Squared Error: 0.3642449104976103
'''