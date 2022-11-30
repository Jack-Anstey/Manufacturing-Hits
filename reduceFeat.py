import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from normalization import normalize



def reduceFeatures(data: pd.DataFrame, threshold:float) -> None:
    """Take a dataframe and remove highly correlated features based on Pearson Correlation, plots heatmap as well
    Args:
        data (pd.DataFrame): the dataset 
        portion (pd.DataFrame): the threshold used to evaluate whether a feature should be eliminated
    Returns:
        None
    """
    
    feats = data.columns.to_numpy()
    size = len(feats)
    mydrops = []

    p_matrix = data.corr(method='pearson')
    sns.heatmap(p_matrix,vmin=-1.0,vmax =1.0, annot = True,fmt = '.2f',annot_kws={"fontsize":7})
    plt.show()
    p_matrix = p_matrix.to_numpy()
    
    for i in range(0,size):
        if not i in mydrops:
            for j in range(i,size):
                val = p_matrix[i,j]
                if abs(val) >= threshold and (not val in mydrops) and i!=j:
                    mydrops.append(j)
                

    dropfeats = feats[mydrops]
    data.drop(columns = dropfeats,axis=1,inplace = True)
    
    
    
def main():
    """
    Take our dataset and reduce unecessary features
    """

    data = normalize(pd.read_csv("pruned datasets/data.csv"),pd.read_csv("pruned datasets/data.csv"))

    reduceFeatures(data,0.6)
    
    print(data)


if __name__ == "__main__":
    main()
    
    
    
    
    
    
