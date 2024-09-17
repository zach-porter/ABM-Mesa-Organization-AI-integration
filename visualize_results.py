# visualize_results.py

import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metrics(data_path='data/results.csv'):
    """
    Plots various metrics from the simulation data.
    """
    if not os.path.exists(data_path):
        print(f"Data file '{data_path}' not found.")
        return

    data = pd.read_csv(data_path)

    # Plot Average Knowledge Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(data["Average Knowledge"], label="Average Knowledge", color='blue')
    plt.xlabel("Time Steps")
    plt.ylabel("Average Knowledge")
    plt.title("Average Knowledge Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/average_knowledge_over_time.png')
    plt.show()

    # Plot AI Utilization Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(data["Positive Attitudes"], label="Positive Attitudes", color='green')
    plt.plot(data["Neutral Attitudes"], label="Neutral Attitudes", color='orange')
    plt.plot(data["Negative Attitudes"], label="Negative Attitudes", color='red')
    plt.xlabel("Time Steps")
    plt.ylabel("Number of Agents")
    plt.title("AI Utilization Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/ai_utilization_over_time.png')
    plt.show()

    # Plot AI Knowledge Contribution Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(data["AI Knowledge Contribution"], label="AI Knowledge Contribution", color='purple')
    plt.xlabel("Time Steps")
    plt.ylabel("AI Knowledge Contribution")
    plt.title("AI Knowledge Contribution Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/ai_contribution_over_time.png')
    plt.show()

    # Additional Plots can be added as needed

if __name__ == "__main__":
    plot_metrics()
