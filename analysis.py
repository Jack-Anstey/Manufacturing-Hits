import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDRegressor
from subframes import getSubFrames
from modelgen import randomForest
from modelgen import linReg
from normalization import normalize
import matplotlib.pyplot as plt

def getR2(subFrame: dict()) -> tuple():
    """Get the r^2 value of a given linear regression model

    Args:
        subFrame (dict): the dictionary that holds the model and the data

    Returns:
        (float, float): the r^2 using the popularity labels, and the r^2 using the rank labels
    """

    return (r2_score(y_true=subFrame['popularity-test'], y_pred=subFrame['linear-regression']['model-pop'].predict(subFrame['data-test'])), 
    r2_score(y_true=subFrame['peak-rank-test'], y_pred=subFrame['linear-regression']['model-rank'].predict(subFrame['data-test'])))

def getAcc(subFrame: dict()) -> tuple():
    """Get the accuracy score of a given random forest model

    Args:
        subFrame (dict): the dictionary that holds the model and the data

    Returns:
        (float, float): the accuracy score using the popularity labels, and the accuracy score using the rank labels
    """

    return (accuracy_score(y_true=subFrame['popularity-reduced-test'], y_pred=subFrame['random-forest']['model-pop'].predict(subFrame['data-test'])), 
    accuracy_score(y_true=subFrame['peak-rank-reduced-test'], y_pred=subFrame['random-forest']['model-rank'].predict(subFrame['data-test'])))

def main():
    # load the data
    data = pd.read_csv("pruned datasets/data.csv")
    popularity = pd.read_csv("pruned datasets/popularity.csv")
    rank = pd.read_csv("pruned datasets/ranks.csv")
    popularityReduced = pd.read_csv("pruned datasets/popularity-reduced.csv")
    rankReduced = pd.read_csv("pruned datasets/ranks-reduced.csv")

    # get subframes from data
    subFrames = getSubFrames(data, popularity, rank, popularityReduced, rankReduced, 1950, 2020)
    
    # create models using subframes
    linReg(subFrames)
    randomForest(subFrames)
    
    # display r^2
    for key in subFrames:
        print("Year range:", str(key), "to", str(key+9))
        pop, rank = getR2(subFrames[key])
        print("SGD Linear Regression model r^2 value (popularity):", pop)
        print("SGD Linear Regression model r^2 value (peak-rank):", rank)
        print()

    # TODO uncomment this when random forest is implemented in modelgen.py
    # display accuracy
    # for key in subFrames:
    #     print("Year range:", str(key), "to", str(key+9))
    #     pop, rank = getAcc(subFrames[key])
    #     print("Random Forest model accuracy (popularity):", pop)
    #     print("Random Forest model accuracy (peak-rank):", rank)
    #     print()

    # TODO
    # Combine training data by decade together and combine test data by decade
    # Then, model and test the r^2 and accuracy of this combined dataset using the SGDLinReg and random forest respectively

if __name__ == "__main__":
    main()