import pandas as pd
import sklearn as sk
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import RandomizedSearchCV
from subframes import getSubFrames
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def linReg(frames: dict(dict())) -> None:
    """Given a dictionary of dictionaries of dataframes,
    this method creates a linear regression model for all of them

    Args:
        frames (dict(dict())): A dictionary of dictionaries of dataframes
    """

    for key in frames.keys():
        frames[key]['linear-regression'] = {'model-pop': SGDRegressor(max_iter=1000, tol=1e-3).fit(frames[key]['data-train'], frames[key]['popularity-train'].values.ravel()),
        'model-rank': SGDRegressor(max_iter=1000, tol=1e-3).fit(frames[key]['data-train'], frames[key]['peak-rank-train'].values.ravel())}

    # no need to return frames since adding keys does that implicitly
def getAcc(subFrame: dict()) -> tuple():
    """Get the accuracy score of a given random forest model

    Args:
        subFrame (dict): the dictionary that holds the model and the data

    Returns:
        (float, float): the accuracy score using the popularity labels, and the accuracy score using the rank labels
    """

    return (accuracy_score(y_true=subFrame['popularity-reduced-test'], y_pred=subFrame['random-forest']['model-pop'].predict(subFrame['data-test'])),
    accuracy_score(y_true=subFrame['peak-rank-reduced-test'], y_pred=subFrame['random-forest']['model-rank'].predict(subFrame['data-test'])))

def randomForest(frames: dict(dict())) -> None:
    """Given a dictionary of dictionaries of dataframes,
    this method creates a random forest model for all of them

    Args:
        frames (dict(dict())): A dictionary of dictionaries of dataframes
    """
    # hyperparameters for tuning
    n_estimators = [10, 100, 300, 500, 800]
    max_depth = [5, 8, 15]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2]
    hyperF = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    rf_RandomGridPop = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=hyperF, cv=10, verbose=2, n_jobs=4)
    rf_RandomGridRank = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=hyperF, cv=10, verbose=2, n_jobs=4)

    # popularity-reduced-train and peak-rank-reduced-train for the labels!
    for key in frames.keys():
        # no tuning
        # frames[key]['random-forest'] = {'model-pop': RandomForestClassifier(n_estimators=750, max_depth=15, min_samples_leaf=5).fit(frames[key]['data-train'], frames[key]['popularity-reduced-train']),
        #                                 "model-rank": RandomForestClassifier(n_estimators=750, max_depth=15, min_samples_leaf=5).fit(frames[key]['data-train'], frames[key]['peak-rank-reduced-train'])}

        # with tuning
        rf_RandomGridPop.fit(frames[key]['data-train'], frames[key]['popularity-reduced-train'].values.ravel())
        rf_RandomGridRank.fit(frames[key]['data-train'], frames[key]['peak-rank-reduced-train'].values.ravel())
        frames[key]['random-forest'] = {'model-pop': rf_RandomGridPop, 'model-rank' : rf_RandomGridRank}
        print("pop rf grid best params: {}\n".format(rf_RandomGridPop.best_params_))
        print("rank rf grid best params: {}\n".format(rf_RandomGridRank.best_params_))

 
def knn(frames: dict(dict())) -> None:
    """Given a dictionary of dictionaries of dataframes,
    this method creates a KNN model for all of them
    Args:
        frames (dict(dict())): A dictionary of dictionaries of dataframes
    """

    # hyperparameters for tuning
    n_neighbors = [1,3,5,10,20,30,50,75,100,150,200]
    metric = ['euclidean','manhattan','minkowski']
    hyperF = dict(n_neighbors = n_neighbors,metric = metric)
    knn_RandomGridPop = RandomizedSearchCV(estimator= KNeighborsClassifier(), param_distributions=hyperF, cv=10, verbose=2, n_jobs=4)
    knn_RandomGridRank = RandomizedSearchCV(estimator= KNeighborsClassifier(), param_distributions=hyperF, cv=10, verbose=2, n_jobs=4)

    for key in frames.keys():
        knn_RandomGridPop.fit(frames[key]['data-train'], frames[key]['popularity-reduced-train'].values.ravel())
        knn_RandomGridRank.fit(frames[key]['data-train'], frames[key]['peak-rank-reduced-train'].values.ravel())
        frames[key]['knn'] = {'model-pop': knn_RandomGridPop, 'model-rank' : knn_RandomGridRank}
        print("pop knn grid best params: {}\n".format(knn_RandomGridPop.best_params_))
        print("rank knn grid best params: {}\n".format(knn_RandomGridRank.best_params_))
        
       
        
        
def main():
    data = pd.read_csv("pruned datasets/data.csv")
    popularity = pd.read_csv("pruned datasets/popularity.csv")
    rank = pd.read_csv("pruned datasets/ranks.csv")

    subFrames = getSubFrames(data, popularity, rank, 1950, 1950)
    linReg(subFrames)
    print(subFrames)

if __name__ == "__main__":
    main()
