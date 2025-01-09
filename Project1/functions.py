import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px



def topN(df, N):

    # extract the top 10 happiest countries in 2015
    happiest_2015 = df.nlargest(N, 'Happiness Score')

    # extract the top 10 least happy countries in 2015
    least_happy_2015 = df.nsmallest(N, 'Happiness Score')

    # combine the two DataFrames
    combined = pd.concat([happiest_2015, least_happy_2015])
    df_sorted = combined.sort_values(by="Happiness Score", ascending=True)

    # create a color gradient from red to green based on scores
    colors = plt.cm.RdYlGn((np.array(df_sorted['Happiness Score']) - min(df_sorted['Happiness Score'])) / (max(df_sorted['Happiness Score']) - min(df_sorted['Happiness Score'])))

    # Dynamically adjust the figure size
    width = 10  # Fixed width
    height_per_row = 0.5  # Adjust the scaling factor as needed
    height = max(2, len(df_sorted) * height_per_row)  # Ensure a minimum height

    # create a horizontal bar chart
    plt.figure(figsize=(width, height))
    plt.barh(df_sorted['Country'], df_sorted['Happiness Score'], color=colors)

    # add labels and title
    plt.xlabel('Happiness Score')
    plt.ylabel('Country')
    plt.title(f'Top {N} most and least happy countries')

    # add numerical values to the end of each bar
    for index, value in enumerate(df_sorted["Happiness Score"]):
        plt.text(value - 0.2, index, f"{value:.2f}", va='center', ha='right', color='white', fontsize=9)

    plt.tight_layout()  # Ensure the layout fits well

    # Ensure the directory exists
    os.makedirs('Project1/images', exist_ok=True)
    # Save the figure
    file_path = f'Project1/images/top_{N}.png'
    plt.savefig(file_path)
    print(f"Figure saved to {file_path}")

    plt.show()

################################################################################################

def top3_corr(df, year):

    happiness_score = df['Happiness Score'] # extract the Happiness Score as a Series

    x = df.iloc[:, 5:]  # extract columns starting from the 5th column

    # calculate Pearson correlation of each column with Happiness Score
    correlation = x.corrwith(happiness_score)

    # sort the correlation values in descending order
    correlation = correlation.sort_values(ascending=False)

    # extract the top 3 features
    top_features = correlation.nlargest(3).index.tolist()
    print(f"Top 3 features that contribute the most to the happiness score in {year} are {top_features[0]}, {top_features[1]}, and {top_features[2]}.")

    return top_features

################################################################################################

def plot_world_map(df, year):

    fig = px.choropleth(
        df,
        locations="Country",
        locationmode="country names",
        color="Happiness Score",
        hover_name="Country",
        title="World Happiness Map in " + str(year),
        color_continuous_scale=["black","red", "orange", "green", "lime"]  # Custom color scheme
    )

    # Save the plot as an HTML file
    #output_folder = "Project1/images"
   # fig.write_html(os.path.join(output_folder, "graph.html"))

    fig.show()