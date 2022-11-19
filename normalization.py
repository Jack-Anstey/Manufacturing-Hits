import pandas as pd
import sklearn as sk
import sklearn.preprocessing as pre

def normalize(data: pd.DataFrame) -> pd.DataFrame:
    """Take a dataframe and normalize it using sklearn's standard scaler

    Args:
        data (pd.DataFrame): the data that you want to normalize

    Returns:
        pd.DataFrame: the normalized data
    """

    # make our scaler using the training data
    scaler = pre.StandardScaler().fit(data)

    # return the resulting dataframes through transformation
    return pd.DataFrame(scaler.transform(data), columns=data.columns)

def main():
    """
    Take our dataset and normalize it
    """

    data = normalize(pd.read_csv("pruned datasets/data.csv"))
    print(data)

if __name__ == "__main__":
    main()