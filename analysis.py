import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import SGDRegressor
from subframes import getSubFrames
from modelgen import randomForest
from modelgen import linReg
from modelgen import knn
from normalization import normalize
import matplotlib.pyplot as plt

def getR2(subFrame: dict(), modelName: str) -> tuple():
    """Get the r^2 value of a given linear regression model

    Args:
        subFrame (dict): the dictionary that holds the model and the data
        modelName (str): the name of the model that you are going to use

    Returns:
        (float, float): the r^2 using the popularity labels, and the r^2 using the rank labels
    """

    return (r2_score(y_true=subFrame['popularity-test'], y_pred=subFrame[modelName]['model-pop'].predict(subFrame['data-test'])), 
    r2_score(y_true=subFrame['peak-rank-test'], y_pred=subFrame[modelName]['model-rank'].predict(subFrame['data-test'])))

def getAcc(subFrame: dict(), modelName: str) -> tuple():
    """Get the accuracy score of a given model

    Args:
        subFrame (dict): the dictionary that holds the model and the data
        modelName (str): the name of the model that you are going to use

    Returns:
        (float, float): the accuracy score using the popularity labels, and the accuracy score using the rank labels
    """

    return (accuracy_score(y_true=subFrame['popularity-reduced-test'], y_pred=subFrame[modelName]['model-pop'].predict(subFrame['data-test'])), 
    accuracy_score(y_true=subFrame['peak-rank-reduced-test'], y_pred=subFrame[modelName]['model-rank'].predict(subFrame['data-test'])))

def getF1(subFrame: dict, modelName: str, average: str) -> tuple():
    """Get the F1 score of a given model

    Args:
        subFrame (dict): the dictionary that holds the model and the data
        modelName (str): the name of the model that you are going to use
        average (str): the average value that you want to have applied to the F1 score method

    Returns:
        (float, float): the F1 score using the popularity labels, and the F1 score using the rank labels
    """

    return (f1_score(y_true=subFrame['popularity-reduced-test'], y_pred=subFrame[modelName]['model-pop'].predict(subFrame['data-test']), average=average), 
    f1_score(y_true=subFrame['peak-rank-reduced-test'], y_pred=subFrame[modelName]['model-rank'].predict(subFrame['data-test']), average=average))

def analyzeLinReg(subFrames: dict) -> None:
    """Analyze how effective the linear regression model is

    Args:
        subFrames (dict): the dictionary that holds the model and the data
    """

    # create models using subframes
    linReg(subFrames)

    # display r^2
    for key in subFrames:
        print("Year range:", str(key), "to", str(key+9))
        pop, rank = getR2(subFrames[key], 'linear-regression')
        print("SGD Linear Regression model r^2 value (popularity):", pop)
        print("SGD Linear Regression model r^2 value (peak-rank):", rank)
        print()

def analyzeRF(subFrames: dict) -> None:
    """Analyze how effective the random forest model is

    Args:
        subFrames (dict): the dictionary that holds the model and the data
    """

    # create models using subframes
    randomForest(subFrames)

    # display accuracy and F1 for rf
    for key in subFrames:
        print("Year range:", str(key), "to", str(key+9))
        pop1, rank1 = getAcc(subFrames[key], 'random-forest')
        pop2, rank2 = getF1(subFrames[key], 'rf', 'weighted')
        print("Random Forest model accuracy (popularity):", pop1)
        print("Random Forest F1 (popularity):", pop2)
        print("Random Forest model accuracy (peak-rank):", rank1)
        print("Random Forest F1 (peak-rank):", rank2)
        print()

def analyzeKNN(subFrames: dict) -> None:
    """Analyze how effective the KNN model is

    Args:
        subFrames (dict): the dictionary that holds the model and the data
    """

    # create models using subframes
    knn(subFrames)

    # display accuracy and F1 for knn
    for key in subFrames:
        print("Year range:", str(key), "to", str(key+9))
        pop1, rank1 = getAcc(subFrames[key], 'knn')
        pop2, rank2 = getF1(subFrames[key], 'knn', 'weighted')
        print("KNN model accuracy (popularity):", pop1)
        print("KNN F1 (popularity):", pop2)
        print("KNN model accuracy (peak-rank):", rank1)
        print("KNN F1 (peak-rank):", rank2)
        print()

def main():
    # load the data
    data = pd.read_csv("pruned datasets/data.csv")
    popularity = pd.read_csv("pruned datasets/popularity.csv")
    rank = pd.read_csv("pruned datasets/ranks.csv")
    popularityReduced = pd.read_csv("pruned datasets/popularity-reduced.csv")
    rankReduced = pd.read_csv("pruned datasets/ranks-reduced.csv")

    # get subframes from data
    subFrames = getSubFrames(data, popularity, rank, popularityReduced, rankReduced, 1950, 2020)
    
    # perform r^2, acc, and F1 analysis (as applicable) and print the results
    # analyzeLinReg(subFrames)
    # analyzeRF(subFrames)
    analyzeKNN(subFrames)

    # TODO
    # Combine training data by decade together and combine test data by decade
    # Then, model and test the r^2 and accuracy of this combined dataset using the SGDLinReg and random forest respectively

    # combined training and test datasets
    # build all the neccessary dataframes
    combDataTrain = pd.DataFrame(columns=subFrames[list(subFrames.keys())[0]]['data-train'].columns)
    combDataTest = pd.DataFrame(columns=subFrames[list(subFrames.keys())[0]]['data-train'].columns)
    combPopTrain = pd.DataFrame(columns=subFrames[list(subFrames.keys())[0]]['popularity-train'].columns)
    combPopTest = pd.DataFrame(columns=subFrames[list(subFrames.keys())[0]]['popularity-test'].columns)
    combPopRTrain = pd.DataFrame(columns=subFrames[list(subFrames.keys())[0]]['popularity-reduced-train'].columns)
    combPopRTest = pd.DataFrame(columns=subFrames[list(subFrames.keys())[0]]['popularity-reduced-test'].columns)
    combRankTrain = pd.DataFrame(columns=subFrames[list(subFrames.keys())[0]]['peak-rank-train'].columns)
    combRankTest = pd.DataFrame(columns=subFrames[list(subFrames.keys())[0]]['peak-rank-test'].columns)
    combRankRTrain = pd.DataFrame(columns=subFrames[list(subFrames.keys())[0]]['peak-rank-reduced-train'].columns)
    combRankRTest = pd.DataFrame(columns=subFrames[list(subFrames.keys())[0]]['peak-rank-reduced-test'].columns)

    # combine!
    for key in subFrames:
        combDataTrain = pd.concat([combDataTrain, subFrames[key]['data-train']])
        combDataTrain.reset_index(drop=True, inplace=True)

        combDataTest = pd.concat([combDataTest, subFrames[key]['data-test']])
        combDataTest.reset_index(drop=True, inplace=True)

        combPopTrain = pd.concat([combPopTrain, subFrames[key]['popularity-train']])
        combPopTrain.reset_index(drop=True, inplace=True)

        combPopTest = pd.concat([combPopTest, subFrames[key]['popularity-test']])
        combPopTest.reset_index(drop=True, inplace=True)
        
        combRankTrain = pd.concat([combRankTrain, subFrames[key]['peak-rank-train']])
        combRankTrain.reset_index(drop=True, inplace=True)

        combRankTest = pd.concat([combRankTest, subFrames[key]['peak-rank-test']])
        combRankTest.reset_index(drop=True, inplace=True)

        combPopRTrain = pd.concat([combPopRTrain, subFrames[key]['popularity-reduced-train']])
        combPopRTrain.reset_index(drop=True, inplace=True)

        combPopRTest = pd.concat([combPopRTest, subFrames[key]['popularity-reduced-test']])
        combPopRTest.reset_index(drop=True, inplace=True)
        
        combRankRTrain = pd.concat([combRankRTrain, subFrames[key]['peak-rank-reduced-train']])
        combRankRTrain.reset_index(drop=True, inplace=True)

        combRankRTest = pd.concat([combRankRTest, subFrames[key]['peak-rank-reduced-test']])
        combRankRTest.reset_index(drop=True, inplace=True)

if __name__ == "__main__":
    main()