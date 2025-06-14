import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend suitable for servers

import seaborn as sns
import os

def plot_delay_distribution(df):
    plt.figure(figsize=(6, 4))
    df['Delay_Minutes'].hist(bins=30, color='skyblue', edgecolor='black')
    plt.title('Flight Delay Distribution')
    plt.xlabel('Delay (minutes)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    save_plot('delay_distribution.png')


def plot_airline_delays(df):
    plt.figure(figsize=(10, 5))
    airline_delay = df.groupby('Airline')['Delay_Minutes'].mean().sort_values(ascending=False)
    airline_delay.plot(kind='bar', color='orange')
    plt.title('Average Delay by Airline')
    plt.ylabel('Average Delay (minutes)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot('airline_delays.png')


def plot_weather_vs_delay(df):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x='Dep_Visibility', y='Delay_Minutes', hue='Dep_Weather')
    plt.title('Departure Weather Visibility vs. Delay')
    plt.xlabel('Visibility (km)')
    plt.ylabel('Delay (minutes)')
    plt.tight_layout()
    save_plot('weather_vs_delay.png')


def plot_airport_delay_trends(df):
    plt.figure(figsize=(10, 5))
    airport_delay = df.groupby('Dep_Airport')['Delay_Minutes'].mean().sort_values(ascending=False).head(10)
    airport_delay.plot(kind='bar', color='teal')
    plt.title('Top 10 Departure Airports by Average Delay')
    plt.ylabel('Average Delay (minutes)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot('airport_delays.png')


def plot_feature_importance(model, feature_names):
    import matplotlib.pyplot as plt
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[sorted_idx])
    plt.xticks(range(len(importances)), [feature_names[i] for i in sorted_idx], rotation=90)
    plt.title("Feature Importance (XGBoost)")
    plt.tight_layout()
    plt.savefig("static/plots/feature_importance.png")
    plt.close()



def save_plot(filename):
    output_dir = 'app/static/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
