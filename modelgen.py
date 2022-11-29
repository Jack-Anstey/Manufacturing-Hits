import pandas as pd
import sklearn as sk
from sklearn.linear_model import SGDRegressor
from subframes import getSubFrames

def linReg(frames: dict(dict())) -> None:
    """Given a dictionary of dictionaries of dataframes,
    this method creates a linear regression model for all of them

    Args:
        frames (dict(dict())): A dictionary of dictionaries of dataframes
    """

    for key in frames.keys():
        frames[key]['linear-regression'] = {'model-pop': SGDRegressor(max_iter=1000, tol=1e-3).fit(frames[key]['data-train'], frames[key]['popularity-train']),
        'model-rank': SGDRegressor(max_iter=1000, tol=1e-3).fit(frames[key]['data-train'], frames[key]['peak-rank-train'])}

    # no need to return frames since adding keys does that implicitly

def randomForest(frames: dict(dict())) -> None:
    """Given a dictionary of dictionaries of dataframes,
    this method creates a random forest model for all of them

    Args:
        frames (dict(dict())): A dictionary of dictionaries of dataframes
    """
    
    # TODO randomforest. Make sure to use: 
    # popularity-reduced-train and peak-rank-reduced-train for the labels!
    for key in frames.keys():
        frames[key]['random-forest'] = {'model-pop': "random forest goes here", "model-rank": "random forest goes here"} 

def main():
    data = pd.read_csv("pruned datasets/data.csv")
    popularity = pd.read_csv("pruned datasets/popularity.csv")
    rank = pd.read_csv("pruned datasets/ranks.csv")

    subFrames = getSubFrames(data, popularity, rank, 1950, 1950)
    linReg(subFrames)
    print(subFrames)

if __name__ == "__main__":
    main()