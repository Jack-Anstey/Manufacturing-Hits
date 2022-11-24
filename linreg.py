import pandas as pd
import sklearn as sk
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from subframes import getSubFrames

def linReg(frames: dict(dict())) -> None:
    """Given a dictionary of dictionaries of dataframes,
    this method creates a linear regression model for all of them

    Args:
        frames (dict(dict())): A dictionary of dictionaries of dataframes
    """

    # add some model keys to each sub dictionary whose value is a linear regression model fitted to each dataset
    # one model using the popularity labels and another using the peak-rank labels
    # also take the elastic cv hyperparameters for analysis later
    for key in frames.keys():
        # regr = ElasticNetCV(cv=5)
        # frames[key]['hyperparameters'] = regr.get_params()
        # frames[key]['model-pop'] = regr.fit(frames[key]['data'], frames[key]['popularity'])
        # frames[key]['model-rank'] = regr.fit(frames[key]['data'], frames[key]['peak-rank'])

        # TODO gridsearchcv using the SGDRegressor

        frames[key]['model-pop'] = SGDRegressor(max_iter=1000, tol=1e-3).fit(frames[key]['data'], frames[key]['popularity'])
        frames[key]['model-rank'] = SGDRegressor(max_iter=1000, tol=1e-3).fit(frames[key]['data'], frames[key]['peak-rank'])

    # no need to return frames since adding keys does that implicitly


def main():
    data = pd.read_csv("pruned datasets/data.csv")
    popularity = pd.read_csv("pruned datasets/popularity.csv")
    rank = pd.read_csv("pruned datasets/ranks.csv")

    subFrames = getSubFrames(data, popularity, rank, 1950, 1950)
    linReg(subFrames)
    print(subFrames)

if __name__ == "__main__":
    main()