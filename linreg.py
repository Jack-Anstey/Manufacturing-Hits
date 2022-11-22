import pandas as pd
import sklearn as sk
from subframes import getSubFrames

def linReg(frames: dict(dict())) -> None:
    """Given a dictionary of dictionaries of dataframes,
    this method creates a linear regression model for all of them

    Args:
        frames (dict(dict())): A dictionary of dictionaries of dataframes
    """
    # TODO do cvgridsearch on every single subframe, then take the resulting hyperparameters
    # and put them into the dictionary along with the model

    # add a model key to each sub dictionary whose value is a linear regression model
    # fitted to each dataset
    for key in frames.keys():
        frames[key]['hyperparameters'] = 'for our paper later'
        frames[key]['model'] = 'insert linreg model here'

    # no need to return frames since adding keys affects it for us


def main():
    data = pd.read_csv("pruned datasets/data.csv")
    popularity = pd.read_csv("pruned datasets/popularity.csv")
    rank = pd.read_csv("pruned datasets/ranks.csv")

    subFrames = getSubFrames(data, popularity, rank, 1950, 1950)
    linReg(subFrames)
    print(subFrames)

if __name__ == "__main__":
    main()