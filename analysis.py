import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import SGDRegressor
from subframes import getSubFrames
from linreg import linReg
from normalization import normalize
import matplotlib.pyplot as plt

def getAccuracy(subFrame, year):
    print("Year range:", str(year), "to", str(year+9))
    # print(subFrame['model-pop'].predict(subFrame['data']))
    print("Model accuracy (popularity):", r2_score(y_true=subFrame['popularity'], y_pred=subFrame['model-pop'].predict(subFrame['data']).round()))
    print("Model accuracy (peak-rank):", r2_score(y_true=subFrame['peak-rank'], y_pred=subFrame['model-rank'].predict(subFrame['data']).round()))
    print()

def main():
    # load the data
    data = pd.read_csv("pruned datasets/data.csv")
    popularity = pd.read_csv("pruned datasets/popularity.csv")
    rank = pd.read_csv("pruned datasets/ranks.csv")

    subFrames = getSubFrames(data, popularity, rank, 1950, 2020)
    linReg(subFrames)
    
    for key in subFrames:
        getAccuracy(subFrames[key], key)

    data = normalize(data)  # normalize the entire dataset
    
    print("Entire dataset")
    print("Model accuracy (popularity):", r2_score(y_true=popularity, 
        y_pred=SGDRegressor(max_iter=1000, tol=1e-3).fit(data, popularity['popularity']).predict(data).round()))
    print("Model accuracy (peak-rank):", r2_score(y_true=rank, 
        y_pred=SGDRegressor(max_iter=1000, tol=1e-3).fit(data, rank['peak-rank']).predict(data).round()))

if __name__ == "__main__":
    main()