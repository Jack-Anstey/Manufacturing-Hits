import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split



def pca(xTrain, xTest):
    """Conducts PCA on training and test set of features until 95% of variance is captured (must be used AFTER normalization)
    Args:
        xTrain (pd.DataFrame): the training set of features which the PCA will be fit on
        xTest (pd.DataFrame) : the test set of features
    Returns:
        new_xTrain (pd.DataFrame): the transformed training data, columns relabled as the principal components they represent
        new_xTest (pd.DataFrame): the transformed training data, columns relabled as the principal components they represent
    """

    new_xTrain = None
    new_xTest = None
    comp = []
    
    for i in range(1,xTrain.shape[1]+1):
        comp.append(f'pc{i}')
        p = PCA(n_components = i)
        p.fit(xTrain)
        variance_captured = np.sum(p.explained_variance_ratio_)
        
        if variance_captured>0.95:
            new_xTrain = p.transform(xTrain)
            new_xTest = p.transform(xTest)
            break

    return pd.DataFrame(new_xTrain,columns = comp), pd.DataFrame(new_xTest,columns = comp)



def main():
    data = pd.read_csv("pruned datasets/data.csv")
    xTrain, xTest = train_test_split(data, test_size=0.3)
    new_xTrain,new_xTest = pca(xTrain,xTest)
    


if __name__ == "__main__":
    main()
