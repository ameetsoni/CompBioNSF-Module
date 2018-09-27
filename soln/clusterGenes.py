#Libraries to include; you can add more libraries to extend beyond the
# functionality in the tutorial
import numpy as np
import geneMLLib as ml #our custom library
from sklearn import metrics
from sklearn.cluster import KMeans

def main():
    #load data, X is gene values, genes is names of genes, y are the labels
    dir = "../data/yeast/"
    X = ml.loadGeneExpression(dir+'expression.csv')
    geneNames = ml.loadGeneNames(dir+'names.txt')

    model = KMeans(n_clusters=11, init="random")
    clusterIndex = model.fit_predict(X)
    print(metrics.silhouette_score(X,clusterIndex))
    for i in range(11):
        print(geneNames[clusterIndex == i])

main()
