import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import SGDRegressor
from subframes import getSubFrames
from modelgen import randomForest
from modelgen import linReg
from modelgen import knn
from modelgen import xgboost
from normalization import normalize
from sklearn import metrics
import matplotlib.pyplot as plt

def combine(subFrames: dict) -> dict:
    """Given a dictionary of data and labels, combine them appropriately

    Args:
        subFrames (dict): the original data and labels

    Returns:
        dict: a dictionary of the combined data and labels
    """
    combDataTrain = pd.DataFrame(columns=subFrames[list(subFrames.keys())[0]]['data-train'].columns)
    combDataTest = pd.DataFrame(columns=subFrames[list(subFrames.keys())[0]]['data-train'].columns)
    combPopTrain = pd.DataFrame(columns=subFrames[list(subFrames.keys())[0]]['popularity-train'].columns, dtype=int)
    combPopTest = pd.DataFrame(columns=subFrames[list(subFrames.keys())[0]]['popularity-test'].columns, dtype=int)
    combPopRTrain = pd.DataFrame(columns=subFrames[list(subFrames.keys())[0]]['popularity-reduced-train'].columns, dtype=int)
    combPopRTest = pd.DataFrame(columns=subFrames[list(subFrames.keys())[0]]['popularity-reduced-test'].columns, dtype=int)
    combRankTrain = pd.DataFrame(columns=subFrames[list(subFrames.keys())[0]]['peak-rank-train'].columns, dtype=int)
    combRankTest = pd.DataFrame(columns=subFrames[list(subFrames.keys())[0]]['peak-rank-test'].columns, dtype=int)
    combRankRTrain = pd.DataFrame(columns=subFrames[list(subFrames.keys())[0]]['peak-rank-reduced-train'].columns, dtype=int)
    combRankRTest = pd.DataFrame(columns=subFrames[list(subFrames.keys())[0]]['peak-rank-reduced-test'].columns, dtype=int)

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

    return {'everything': {'data-train': combDataTrain, 'data-test': combDataTest, 'popularity-train': combPopTrain, 
    'popularity-test': combPopTest, 'popularity-reduced-train': combPopRTrain, 
    'popularity-reduced-test': combPopRTest, 'peak-rank-train': combRankTrain, 
    'peak-rank-test': combRankTest, 'peak-rank-reduced-train': combRankRTrain, 
    'peak-rank-reduced-test': combRankRTest}}

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

def analyzeLinReg(subFrames: dict, fullData: bool) -> None:
    """Analyze how effective the linear regression model is

    Args:
        subFrames (dict): the dictionary that holds the model and the data
        fullData (bool): true if you are using the entire dataset, false if not
    """

    # create models using subframes
    linReg(subFrames)

    # display r^2
    for key in subFrames:
        if fullData: print("The Entire Dataset:")
        else: print("Year range:", str(key), "to", str(key+9))
        pop, rank = getR2(subFrames[key], 'linear-regression')
        print("SGD Linear Regression model r^2 value (popularity):", pop)
        print("SGD Linear Regression model r^2 value (peak-rank):", rank)
        print()

def analyzeRF(subFrames: dict, fullData: bool) -> None:
    """Analyze how effective the random forest model is

    Args:
        subFrames (dict): the dictionary that holds the model and the data
        fullData (bool): true if you are using the entire dataset, false if not
    """

    # create models using subframes
    randomForest(subFrames)

    # display accuracy and F1 for rf
    for key in subFrames:
        if fullData: print("The Entire Dataset:") 
        else: print("Year range:", str(key), "to", str(key+9))
        pop1, rank1 = getAcc(subFrames[key], 'random-forest')
        pop2, rank2 = getF1(subFrames[key], 'random-forest', 'weighted')
        print("Random Forest model accuracy (popularity):", pop1)
        print("Random Forest F1 (popularity):", pop2)
        print("Random Forest model accuracy (peak-rank):", rank1)
        print("Random Forest F1 (peak-rank):", rank2)
        print()

def analyzeKNN(subFrames: dict, fullData: bool) -> None:
    """Analyze how effective the KNN model is

    Args:
        subFrames (dict): the dictionary that holds the model and the data
        fullData (bool): true if you are using the entire dataset, false if not
    """

    # create models using subframes
    knn(subFrames)

    # display accuracy and F1 for knn
    for key in subFrames:
        if fullData: print("The Entire Dataset:") 
        else: print("Year range:", str(key), "to", str(key+9))
        pop1, rank1 = getAcc(subFrames[key], 'knn')
        pop2, rank2 = getF1(subFrames[key], 'knn', 'weighted')
        print("KNN model accuracy (popularity):", pop1)
        print("KNN F1 (popularity):", pop2)
        print("KNN model accuracy (peak-rank):", rank1)
        print("KNN F1 (peak-rank):", rank2)
        print()

def analyzeXGB(subFrames: dict, fullData: bool) -> None:
    """Analyze how effective the XGBoost model is

    Args:
        subFrames (dict): the dictionary that holds the model and the data
        fullData (bool): true if you are using the entire dataset, false if not
    """

    # create models using subframes
    xgboost(subFrames)

    # display accuracy and F1 for knn
    for key in subFrames:
        if fullData: print("The Entire Dataset:") 
        else: print("Year range:", str(key), "to", str(key+9))
        pop1, rank1 = getAcc(subFrames[key], 'xgb')
        pop2, rank2 = getF1(subFrames[key], 'xgb', 'weighted')
        print("XGB model accuracy (popularity):", pop1)
        print("XGB F1 (popularity):", pop2)
        print("XGB model accuracy (peak-rank):", rank1)
        print("XGB F1 (peak-rank):", rank2)
        print()

def confusionMatrix(subFrames: dict, fullData: bool, model: str) -> None:
    """
    Create confusion matrices for the subFrames input as parameters
    Args:
        subFrames (dict): the dictionary that holds the model and the data
        fullData (bool): true if entire dataset, else false
        model (string): which model to use for error analysis
    """

    if model == "knn":
        knn(subFrames)
    elif model == "linear-regression":
        linReg(subFrames)
    elif model == "random-forest":
        randomForest(subFrames)
    else:
        print("Please enter\n1. knn\n2. linear-regression\n3. random-forest\nas the \"model\" parameter")

    for key in subFrames:
        if fullData:
            print("The Entire Dataset:")
        else:
            print("Year range: {} to {}".format(str(key), str(key+9)))
        subFrame = subFrames[key]

        # get actual and predicted labels for popularity and rank
        y_true_pop = subFrame['popularity-reduced-test']
        y_pred_pop = subFrame[model]['model-pop'].predict(subFrame['data-test'])
        y_true_rank = subFrame['peak-rank-reduced-test']
        y_pred_rank = subFrame[model]['model-rank'].predict(subFrame['data-test'])

        # get confusion matrices
        matrix_pop = metrics.confusion_matrix(y_true_pop, y_pred_pop)
        matrix_rank = metrics.confusion_matrix(y_true_rank, y_pred_rank)

        # display matrices
        matrix_pop_display = metrics.ConfusionMatrixDisplay(confusion_matrix=matrix_pop)
        matrix_rank_display = metrics.ConfusionMatrixDisplay(confusion_matrix=matrix_rank)

        matrix_pop_display.plot()
        plt.title("Confusion matrix for popularity using model {}".format(model))
        plt.show()

        matrix_rank_display.plot()
        plt.title("Confusion matrix for peak-rank using model {}".format(model))
        plt.show()

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
    analyzeLinReg(subFrames, False)
    analyzeRF(subFrames, False)
    analyzeKNN(subFrames, False)
    analyzeXGB(subFrames, False)

    # see confusion matrices for each model
    confusionMatrix(subFrames, False, "knn")
    confusionMatrix(subFrames, False, "linear-regression")
    confusionMatrix(subFrames, False, "random-forest")
    confusionMatrix(subFrames, False, "xgb")

    # Combining training and test datasets
    combinedEverything = combine(subFrames)

    # Check the results using the entire dataset!
    analyzeLinReg(combinedEverything, True)
    analyzeRF(combinedEverything, True)
    analyzeKNN(combinedEverything, True)
    analyzeXGB(combinedEverything, True)

    # Confusion matrices using entire dataset
    confusionMatrix(combinedEverything, True, "knn")
    confusionMatrix(combinedEverything, True, "linear-regression")
    confusionMatrix(combinedEverything, True, "random-forest")
    confusionMatrix(combinedEverything, True, "xgb")

if __name__ == "__main__":
    main()