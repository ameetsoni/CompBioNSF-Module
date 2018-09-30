import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

def runTuneTest(learner, parameters, X, y):
    """
    This method takes in a learning algorithm, the possible settings you would use for the algorithm and the full data set
    It performs cross-validation to tune the algorithm; that is, it creates held-aside data to determine which settings are best

    Returns the accuracy and model weights for each fold
    """

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True)
    outer_cv = outer_cv.split(X,y)

    results = []
    fold = 0
    features = []
    #For each chunk (fold) created above, treat it as the held-aside test set and group the other four to train the model
    for train, test in outer_cv:
        fold += 1
        X_train, X_test =  X[train], X[test]
        y_train, y_test = y[train], y[test]
        clf = GridSearchCV(learner, parameters, cv=3)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        results.append(score)
        #Use the code below instead if you want roc values
        #preds = clf.predict_proba(X_test)
        #results.append(roc_auc_score(y_test,preds[:,1]))
        try: # works for random forests or logistic regression
            features.append(clf.best_estimator_.coef_[0,:])
        except:
            features.append(clf.best_estimator_.feature_importances_)

    return results, features

def getGeneRanks(weights):
    """
    Takes the weights learned from an algorithm and sorts them by the most significant weights for predicting cancer tissue.  Returns an array of length numGenes, with the ith entry being the index of the ith most significant gene
    """
    rank_weights = []
    for w in weights:
        rank_weights.append(np.argsort(np.argsort(w)))
    feat_weights = np.average(rank_weights,axis=0)
    indices = np.argsort(feat_weights)
    return indices


def loadGeneExpression(filename, delim=","):
    """
    Parses a csv file containing gene expression data.  The format of the file
    should be one expression profile per line.

    Returns: 2D numpy array of size (numProfiles, numGenes)
    """

    rawdata = np.genfromtxt(filename,delimiter=delim)
    scaled = preprocessing.normalize(rawdata, norm='l2', axis=0)
    return scaled


def loadGeneNames(filename):
    """
    Reads the names of the genes in the expression profile.  There should be
    the same number of gene names as columns from loadGeneExpression

    Returns a 1D numpy array of length numGenes
    """
    with open(filename,'r') as f:
        genes = np.array([line.rstrip() for line in f])
    return genes

def loadLabels(filename):
    """
    Reads the labels (classifications) of each expression profile loaded from
    loadGeneExpression.  There should be on label per profile.  Negative values
    indicate cancer while positive indicate normal tissue

    Returns a 1D numpy array of length numProfiles
    """
    y = np.genfromtxt(filename)
    y = np.where(y>0,1,-1) #positive is normal
    return y
