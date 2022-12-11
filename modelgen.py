import pandas as pd
import sklearn as sk
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import RandomizedSearchCV
from subframes import getSubFrames
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

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
        # frames[key]['random-forest'] = {'model-pop': RandomForestClassifier(n_estimators=750, max_depth=15, min_samples_leaf=5).fit(frames[key]['data-train'], frames[key]['popularity-reduced-train'].values.ravel()),
        #                                 "model-rank": RandomForestClassifier(n_estimators=750, max_depth=15, min_samples_leaf=5).fit(frames[key]['data-train'], frames[key]['peak-rank-reduced-train'].values.ravel())}

        # with tuning
        # rf_RandomGridPop.fit(frames[key]['data-train'], frames[key]['popularity-reduced-train'].values.ravel())
        # rf_RandomGridRank.fit(frames[key]['data-train'], frames[key]['peak-rank-reduced-train'].values.ravel())
        # frames[key]['random-forest'] = {'model-pop': rf_RandomGridPop, 'model-rank' : rf_RandomGridRank}
        # print("pop rf grid best params: {}\n".format(rf_RandomGridPop.best_params_))
        # print("rank rf grid best params: {}\n".format(rf_RandomGridRank.best_params_))

        # hard coded tuning
        estimatorsP, max_depthP, mlsP, mssP, estimatorsR, mssR, mlsR, max_depthR = getBestParams(key)

        frames[key]['random-forest'] = {'model-pop': RandomForestClassifier(n_estimators=estimatorsP, max_depth=max_depthP, min_samples_split=mssP, min_samples_leaf=mlsP).fit(frames[key]['data-train'], frames[key]['popularity-reduced-train'].values.ravel()),
                                        "model-rank": RandomForestClassifier(n_estimators=estimatorsR, max_depth=max_depthR, min_samples_split=mssR, min_samples_leaf=mlsR).fit(frames[key]['data-train'], frames[key]['peak-rank-reduced-train'].values.ravel())}


def getBestParams(key: str) -> tuple():
    """Get the best parameters for random forest given a key

    Args:
        key (str): the key that defines what best parameters we chose

    Returns:
        tuple: a tuple of the best parameters
    """

    if key == 1950:
        estimatorsP = 10
        mssP = 10
        mlsP = 1
        max_depthP = 5

        estimatorsR = 100
        mssR = 5
        mlsR = 2
        max_depthR = 5
    elif key == 1960:
        estimatorsP = 10
        mssP = 2
        mlsP = 2
        max_depthP = 5

        estimatorsR = 300
        mssR = 10
        mlsR = 2
        max_depthR = 5
    elif key == 1970:
        estimatorsP = 300
        mssP = 10
        mlsP = 2
        max_depthP = 15

        estimatorsR = 300
        mssR = 2
        mlsR = 2
        max_depthR = 5
    elif key == 1980:
        estimatorsP = 300
        mssP = 5
        mlsP = 1
        max_depthP = 5

        estimatorsR = 300
        mssR = 5
        mlsR = 2
        max_depthR = 5
    elif key == 1990:
        estimatorsP = 300
        mssP = 2
        mlsP = 2
        max_depthP = 5

        estimatorsR = 100
        mssR = 10
        mlsR = 2
        max_depthR = 8
    elif key == 2000:
        estimatorsP = 100
        mssP = 2
        mlsP = 1
        max_depthP = 8

        estimatorsR = 500
        mssR = 5
        mlsR = 2
        max_depthR = 5
    elif key == 2010:
        estimatorsP = 300
        mssP = 10
        mlsP = 1
        max_depthP = 15

        estimatorsR = 500
        mssR = 10
        mlsR = 2
        max_depthR = 5
    elif key == 2020:
        estimatorsP = 800
        mssP = 5
        mlsP = 1
        max_depthP = 15

        estimatorsR = 300
        mssR = 2
        mlsR = 2
        max_depthR = 8
    elif key == "everything":
        estimatorsP = 800
        mssP = 2
        mlsP = 1
        max_depthP = 8

        estimatorsR = 800
        mssR = 10
        mlsR = 1
        max_depthR = 8
    else:  # defaults
        estimatorsP = 750
        mssP = 5
        mlsP = 5
        max_depthP = 15

        estimatorsR = 750
        mssR = 5
        mlsR = 5
        max_depthR = 15

    return estimatorsP, max_depthP, mlsP, mssP, estimatorsR, mssR, mlsR, max_depthR
 
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

        
        
def xgboost(frames: dict(dict())) -> None:
    """Given a dictionary of dictionaries of dataframes,
    this method creates a XGBoost model for all of them
    Args:
        frames (dict(dict())): A dictionary of dictionaries of dataframes
    """

    # hyperparameters for tuning
    n_estimators = [25, 50, 75, 100, 150, 200, 250]
    max_depth = [2, 4, 6, 8]
    eta = [0.2, 0.3, 0.4, 0.5]

    hyperF = dict(n_estimators = n_estimators,max_depth = max_depth, eta=eta)
    xgb_RandomGridPop = RandomizedSearchCV(estimator= xgb.XGBClassifier(), param_distributions=hyperF, cv=10, verbose=2, n_jobs=4)
    xgb_RandomGridRank = RandomizedSearchCV(estimator= xgb.XGBClassifier(), param_distributions=hyperF, cv=10, verbose=2, n_jobs=4)

    for key in frames.keys():
        xgb_RandomGridPop.fit(frames[key]['data-train'], frames[key]['popularity-reduced-train'].values.ravel())
        xgb_RandomGridRank.fit(frames[key]['data-train'], frames[key]['peak-rank-reduced-train'].values.ravel())
        frames[key]['xgb'] = {'model-pop': xgb_RandomGridPop, 'model-rank' : xgb_RandomGridRank}
        print("pop xgb grid best params: {}\n".format(xgb_RandomGridPop.best_params_))
        print("rank xgb grid best params: {}\n".format(xgb_RandomGridRank.best_params_))
    
        
        
        
        
def main():
    data = pd.read_csv("pruned datasets/data.csv")
    popularity = pd.read_csv("pruned datasets/popularity.csv")
    rank = pd.read_csv("pruned datasets/ranks.csv")

    subFrames = getSubFrames(data, popularity, rank, 1950, 1950)
    linReg(subFrames)
    print(subFrames)

if __name__ == "__main__":
    main()
