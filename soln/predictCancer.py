#Libraries to include; you can add more libraries to extend beyond the
# functionality in the tutorial
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import geneMLLib as ml #our custom library


def main():
    #load data, X is gene values, genes is names of genes, y are the labels
    dir = "../data/colonCancer/"
    X = ml.loadGeneExpression(dir+'expression.csv')
    geneNames = ml.loadGeneNames(dir+'names.txt')
    y = ml.loadLabels(dir+'ids.txt')

    print("Running a train-test split...")
    #set up cross-validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    #Test one run using default settings
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    trainScore = clf.score(X_train, y_train)
    testScore = clf.score(X_test, y_test)
    print("Accuracy on training data: %.3f" % trainScore)
    print("Accuracy on held-aside test data: %.3f" % testScore)

    print("\nRunning tune-test cross-validation...")
    #Using full tune-test 5-fold cross-validation
    parameters = {"C": [.1, 1, 10, 100]}
    scores, weights = ml.runTuneTest(clf, parameters, X, y)

    print("%5s %10s" % ("Fold","AUC Score"))
    print("%5s %10s" % ("----","---------"))

    for fold in range(len(scores)):
        print("%5d %10.3f" % (fold+1, scores[fold]))
    print("-"*16)
    print("%5s %10.3f" % ("Mean", np.mean(scores)))
    print()

    rankedGenes = ml.getGeneRanks(weights)
    print("Top 10 genes for predicting cancer tissue:")
    print("------------------------------------------")
    for index in rankedGenes[0:10]:
        print(geneNames[index])

main()
