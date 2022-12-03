import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def graphLabels(label: pd.DataFrame, title: str) -> None:
    """Make a bar chart of the labels to see if it's balanced or not

    Args:
        label (pd.DataFrame): the dataframe of labels that you want to graph
        title (str): The title of the graph
    """

    labels, counts = np.unique(label, return_counts=True)
    data = pd.DataFrame({'labels': labels, 'count': counts})

    plt.bar(x=data['labels'], height=data['count'])
    plt.legend(title=title)
    plt.xlabel("Label")
    plt.ylabel("Frequency of Label")
    plt.tight_layout()
    plt.show()

def main():
    popularity = pd.read_csv("pruned datasets/popularity.csv")
    rank = pd.read_csv("pruned datasets/ranks.csv")
    popularityReduced = pd.read_csv("pruned datasets/popularity-reduced.csv")
    rankReduced = pd.read_csv("pruned datasets/ranks-reduced.csv")

    graphLabels(popularity, "Popularity")
    graphLabels(rank, "Peak Rank")
    graphLabels(popularityReduced, "Popularity Reduced")
    graphLabels(rankReduced, "Peak Rank Reduced")

if __name__ == "__main__":
    main()