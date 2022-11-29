import pandas as pd

def pruneData(data: pd.DataFrame) -> None:
    """Takes the data and reduces the granularity of time to include only a year.
    Outputs results to a .csv

    Args:
        data (pd.DataFrame): the dataframe that you want to reduce
    """

    newColumns = ['release_year']
    newColumns.extend(data.columns[2:]) # create a new set of columns
    pruned = pd.DataFrame(columns = newColumns)  # make a dataframe using the new columns

    for row in data.iterrows():  # go through each row of the dataframe
        year = -1  # reset the year to be -1
        if row[1]['release_date_precision'] == 'day':
            year = row[1]['release_date'][-4:]
            if "-" in year:
                year = row[1]['release_date'][:4]
        elif row[1]['release_date_precision'] == 'month':
            year = row[1]['release_date'][:4]
        else:
            year = row[1]['release_date']
        newRow = {'release_year': year}  # start making a new row using the appropriate year value
        for columnName, value in zip(data.columns[2:], row[1][2:]):  # build a dictionary using the remaining values
            newRow[columnName] = value
        pruned = pruned.append(pd.Series(newRow), ignore_index=True)  # add the dictionary to make a new row for our pruned dataset

    pruned.to_csv('pruned datasets/data.csv', index=False)  # output the pruned data to a new .csv

def pruneNulls(data: pd.DataFrame) -> pd.DataFrame:
    """Removes every null row and seperates the labels

    Args:
        data (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: a dataframe of just the data, no labels
    """

    data.dropna(inplace=True)  # drop all empty rows
    data.reset_index(drop=True, inplace=True)  # reset index values

    newPop = data['popularity'].copy()
    newRank = data['peak-rank'].copy()

    # force the popularity and peak-rank to be a particular datatype
    newPop = newPop.astype(int)
    newRank = newRank.astype(int)

    data.drop(columns=['popularity', 'peak-rank'], inplace=True)

    newPop.to_csv('pruned datasets/popularity.csv', index=False)
    newRank.to_csv('pruned datasets/ranks.csv', index=False) 

    return data

def reduceLabels() -> None:
    """Take the newly generated popularity.csv and ranks.csv,
    and then create popularity-reduced.csv and ranks-reduced.csv
    where we reduce the number of classes from 100 to 10
    """
    
    # TODO reductions
    print("Do the reductions here!")

def main():
    """
    Take the original output from Spotify-Scraper and prune null rows 
    and set full dates to just be the year. Also makes sure that our labels are pruned in the same way
    """

    data = pruneNulls(pd.read_csv("original datasets/data.csv", skip_blank_lines=False))
    pruneData(data)
    reduceLabels()


if __name__ == "__main__":
    main()