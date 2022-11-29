import pandas as pd
import sklearn as sk
from normalization import normalize

def getDataRange(minYear: int, maxYear: int, data: pd.DataFrame, popularity: pd.DataFrame, rank: pd.DataFrame, popReduced: pd.DataFrame, rankReduced: pd.DataFrame) -> list():
    """Given a minimum year and a maximum year, get the rows that are within that range (inclusive)

    Args:
        minYear (int): the minimum value of the range of years 
        maxYear (int): the maximum value of the range of years
        data (pd.DataFrame): the data to comb through
        popularity (pd.DataFrame): the popularity labels of the data
        rank (pd.DataFrame): the rank labels of the data
        popReduced (pd.DataFrame): the popularity labels reduced to only have 10 classes
        rankReduced (pd.DataFrame): the rank labels reduced to only have 10 classes

    Returns:
        list(): a list of dataframes with only range of years provided in the args
    """

    if minYear > maxYear:
        print("The range values given are invalid. Cannot have a minYear be greater than a maxYear")
        return None, None, None

    # Make a copy of the data and then add the labels to the copy dataframe
    # We do this to make sure the labels continue to be associated with the correct rows
    dataC = data.copy()
    dataC['popularity'] = popularity
    dataC['peak-rank'] = rank
    dataC['popR'] = popReduced
    dataC['rankR'] = rankReduced

    # get a subdata frame with only the range specified
    dataC = dataC.loc[(dataC['release_year'] >= minYear) & (dataC['release_year'] <= maxYear)]
    dataC.reset_index(drop=True, inplace=True)  # resets the garbage indexes

    #shuffle the combined dataset
    shuffled = dataC.sample(frac=1).reset_index(drop=True)

    #do 70/30 train test splits
    train = shuffled[:int(len(shuffled)*0.7)].copy()
    test = shuffled[int(len(shuffled)*0.7):].copy()

    # prep data for returning
    popTrain = train['popularity'].copy()
    rankTrain = train['peak-rank'].copy()
    popRTr = train['popR'].copy()
    rankRTr = train['rankR'].copy()
    train.drop(columns=['popularity', 'peak-rank', 'popR', 'rankR'], inplace=True)

    popTest = test['popularity'].copy()
    rankTest = test['peak-rank'].copy()
    popRTe = test['popR'].copy()
    rankRTe = test['rankR'].copy()
    test.drop(columns=['popularity', 'peak-rank', 'popR', 'rankR'], inplace=True)

    shuffled.drop(columns=['popularity', 'peak-rank', 'popR', 'rankR'], inplace=True)

    # build and return a dictionary!
    return {'data': shuffled, 'data-train': train, 'popularity-train': popTrain, 'popularity-reduced-train': popRTr,  
            'peak-rank-train': rankTrain, 'peak-rank-reduced-train': rankRTr, 'data-test': test, 'popularity-test': popTest,
            'popularity-reduced-test': popRTe, 'peak-rank-test': rankTest, 'peak-rank-reduced-test': rankRTe}


def getSubFrames(data: pd.DataFrame, popularity: pd.DataFrame, rank: pd.DataFrame, popReduced: pd.DataFrame, rankReduced: pd.DataFrame, minDec, maxDec) -> dict(dict()):
    """Generate sub dataframes from our initial dataset. The number of subframes that will be generated
    is defined by the input of minDec and maxDec inclusive. I.e. if you set maxDec to 
    2020, then 2020 will be considered. Each subframe will be the size of a decade.
    It's encouraged for the minDec and maxDec values to end in a 0, but it isn't neccessary.
    Each subframe is normalized before being returned

    Args:
        data (pd.DataFrame): the data to comb through
        popularity (pd.DataFrame): the popularity labels of the data
        rank (pd.DataFrame): the rank labels of the data
        popReduced (pd.DataFrame): the popularity labels reduced to only have 10 classes
        rankReduced (pd.DataFrame): the rank labels reduced to only have 10 classes
        minDec (int): minimum year to consider
        maxDec (int): maximum year to consider

    Returns:
        dict(list()): A dictionary of dictionaries. These subdictionaries have 3 dataframes: the sub dataframe, 
        sub popularity labels, then the sub peak-rank labels. The key to each sublist is the associated decade
        i.e. the 1950's, 1960's, etc. The key within each sublist is either 'data', 'popularity', or 'peak-rank'
    """

    batches = {}  # where we will store all the sub dataframes

    # add subframes to the dictionary (+10 is to make call inclusive)
    for minYear in range(minDec, maxDec+10, 10):  # iterate through each decade from 1950 to 2020
        batches[minYear] = getDataRange(minYear, minYear+9, data, popularity, rank, popReduced, rankReduced)
        # print("Min:", minYear, "Max:", minYear+9)

    # normalize the training and test sub dataframes utilizing the entire dataset before the train/test split
    for key in batches.keys():
        batches[key]['data-train'] = normalize(batches[key]['data'], batches[key]['data-train'])
        batches[key]['data-test'] = normalize(batches[key]['data'], batches[key]['data-test'])

    return batches

def main():
    """
    Take our dataset, normalize, then break into decades for analysis
    """

    # we can only normalize once we get all our final datasets
    data = pd.read_csv("pruned datasets/data.csv")
    popularity = pd.read_csv("pruned datasets/popularity.csv")
    rank = pd.read_csv("pruned datasets/ranks.csv")

    subFrames = getSubFrames(data, popularity, rank, 1950, 1960)
    print(subFrames)  # testing to see if the output is what we expect

if __name__ == "__main__":
    main()