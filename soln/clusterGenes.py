#Libraries to include; you can add more libraries to extend beyond the
# functionality in the tutorial
import numpy as np
import geneMLLib as ml #our custom library
from sklearn import metrics
from sklearn.cluster import KMeans

def main():
    #load data, X is gene values, genes is names of genes, y are the labels
    dir = "../data/sample-yeast/"
    X = ml.loadGeneExpression(dir+'expression.csv')
    geneNames = ml.loadGeneNames(dir+'names.txt')

    k = 5

    model = KMeans(n_clusters=k)
    model.fit(X)
    clusterIndex = model.predict(X)
    print(metrics.silhouette_score(X,clusterIndex))
    for i in range(k):
        print("Cluster %d" % i)
        for name in geneNames[clusterIndex == i]:
            print("\t"+name)
        print()

main()
