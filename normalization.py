import pandas as pd
import sklearn as sk
import sklearn.preprocessing as pre

def normalize(fullData: pd.DataFrame, portion: pd.DataFrame) -> pd.DataFrame:
    """Take a dataframe and normalize it using sklearn's standard scaler

    Args:
        fullData (pd.DataFrame): the entire dataset that defines the scalar
        portion (pd.DataFrame): the data that you want to normalize

    Returns:
        pd.DataFrame: the normalized data
    """

    # make our scaler using the full dataset
    scaler = pre.StandardScaler().fit(fullData)

    # return the normalized portion through transformation
    return pd.DataFrame(scaler.transform(portion), columns=portion.columns)

def main():
    """
    Take our dataset and normalize it
    """

    data = normalize(pd.read_csv("pruned datasets/data.csv"))
    print(data)

if __name__ == "__main__":
    main()