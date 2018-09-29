# CompBioNSF-Module
Code for module presented at Carleton College, September 29, 2018

## Setup

Download this repository by either cloning the repo using git or by downloading a zip version and unpackaging the contents.  You may need to install the following libraries:
 * Numpy
 * Scikit-Learn (sklearn)
 * SciPy

This can be done using `pip`:

```bash
pip install numpy
pip install sklearn
pip install scipy
```

## Files

There are two Python files of interest in the [`src`](src/) folder:

  * [`geneMLLib.py`](src/geneMLLib.py) - this contains provided libraries.  If you are a novice, you can safely ignore this file and follow the documentation to use the provided functionality.  If you are familiar with Python and various libraries (numpy, scikit-learn, etc.) feel free to peruse the code to see how the data is processed.
  * [`predictCancer.py`](src/predictCancer.py) - the main program for building a supervised prediction algorithm for predicting colon cancer. You will implement this following the instructions below
  * [`clusterGenes.py`](src/clusterGenes.py) - the main program for building an unsupervised clustering algorithm for grouping genes. You will implement this following the instructions below


## Data

You can find the relevant files in your [`data`](data).  
For prediction , we will use the [colon cancer](data/colonCancer) data set, detailed in the seminal paper [Broad patterns of gene expression revealed by clustering analysis of tumor and normal colon tissues probed by oligonucleotide arrays](https://www.ncbi.nlm.nih.gov/pubmed/10359783) by Alon et al.  For gene clustering, we will use a subset of the [yeast](data/sampleYeast) expression data set from the paper [Cluster analysis and display of genome-wide expression patterns](http://www.pnas.org/content/95/25/14863.full) by Eisen et al.

In each data directory, there are three files made available:
 * `expression.csv` - the gene expression data, where rows are profiles and columns are measurements.  For the cancer data set, each column represents a gene and each row represents a patient.  For the yeast data set, each column is an experiment and each row is a gene profile.  These files have already been processed, as detailed in the original paper. Rows correspond to patients and columns correspond to genes.
 * `names.txt` - the names of the genes in the expression data set.  The names file is in order according to the columns of `expression.csv` (e.g., column 1 in the expression data is the gene in row 1 of the names file) for colon cancer, but corresponds to the rows in the yeast data set.
 * `DATA.md `- information on how the data set was generated.

 In addition, for the colon cancer data set, we have:
 * `ids.txt` - identifies normal vs cancerous tissues (i.e., the labels) from the samples provided in `expressions.txt`.  The rows correspond, with a negative value indicating cancerous tissue and positive values indicating normal tissue.  The magnitude of the value is not relevant for our application.


## Part 2: Clustering Gene Expression

We will use the data provided in [data/sample-yeast](data/sample-yeast).  Each gene in our data set has over 70 measurements from different samples forming a gene profile.  The experimental conditions (e.g., point-in-time, individual, etc. ) are the same within one column by vary between columns.  

#### Getting started

Move into the `src` directory and open the file `clusterGenes.py`:

```bash
$ cd src
$ vim clusterGenes.py
```

#### Load the Data

We need to load all of our data into the program so that we can use it to train and test our models.  I have written functions in the `geneMLLib.py` that will handle the parsing of files for you.  If you are curious, feel free to inspect that given code.

```python
dir = "../data/sample-yeast/"
X = ml.loadGeneExpression(dir+'expression.csv')
geneNames = ml.loadGeneNames(dir+'names.txt')
```

The first line defines the directory for our data, the second line loads in our expression data into a two-dimensional array of dimension numGenes x numMeasurements, and the last is an array of gene names, with entry `i` corresponding to the gene expression in the `i`th row of `X`.

#### Train a cluster model

Now that we have our gene expression data, we will cluster the genes based on their expression profiles (the names will not be used in the clustering, but rather only for evaluation).  We will use the [K-means clustering algorithm](https://en.wikipedia.org/wiki/K-means_clustering), which is available in [scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans).  You can explore [other clustering](http://scikit-learn.org/stable/modules/clustering.html) algorithms implemented in scikit-learn.

```python
k=5 #number of clusters
model = KMeans(n_clusters=k)
model.fit(X)
```

The above specifies that we will construct a model using k-means.  We initialize the model with 5 cluster centers, but this choice can be changed.  See the scikit-learn [documentation]((http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)) for other settings we can modify in the k-means algorithm.  The model is then fit to the data, giving us `k` cluster centers.  While we could display the cluster centers, it is probably more useful to see what genes belong to each cluster.

#### Inspect the gene clusters

We can use the predict function to identify the closest cluster center for each gene in our data set:

```python
clusterIndex = model.predict(X)
```

`clusterIndex` has one entry per row of `X`, with a value from `0` to `k-1`.  To print the genes in each cluster, you can run the following code:

```python
for i in range(k):
   print("Cluster %d" % i)
   for name in geneNames[clusterIndex == i]:
       print("\t"+name)
   print()
```

Note that the cluster ordering is random and does not have significance.  Also, if you rerun the algorithm you may get different results depending on where the cluster centers began.  View your results.  How did the clustering algorithm do?

#### Inferring Gene Function

In `names.txt` I have removed the gene annotations of four genes (labeled GENE 10, 18, 23, and 34).  Based on the clustering results, what would you infer about the likely function of these genes with unknown function?  In a real-world scenario, we would use clustering in a similar manner.  In this simulated exercise, you can verify your answers by looking at the information in the [DATA.md](data/sample-yeast/DATA.md) file.

### Algorithm Understanding

K-means is the most popular algorithm for clustering, but no means always the best option.  To get an understanding of why, experiment with the data sets in this [interactive tutorial](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/
).  What properties of the data set determine if k-means gives a plausible answer or not?

How does [DBSCAN (Density-based Spatial Clustering of Applications of Noise)](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/) compare?

## Part 2: Cancer Prediction Implementation

In the provided data, the gene expressions was measured for tissue samples from 62 patients.  40 of the samples were identified as positive for colon cancer and 22 were negative (normal tissue).  One common method for analyzing gene expression data is to perform classification to learn differences in the expression patterns of samples from different categories (cancerous vs normal).  We will utilize [Scikit-Learn](http://scikit-learn.org/), a Python library for accessible, yet efficient data mining.

### Getting started


Move into the `src` directory and open the main program with your favorite editor e.g.,

```bash
$ cd src
$ vim predictCancer.py
```

We will start our implementation of the `main()` function.

### Load the data

We need to load all of our data into the program so that we can use it to train and test our models.  I have written functions in the `geneMLLib.py` that will handle the parsing of files for you.  If you are curious, feel free to inspect that given code.  Place these four lines of code at the top of `main`:

```python
dir = "../data/colonCancer/"
X = ml.loadGeneExpression(dir+'expression.csv')
y = ml.loadLabels(dir+'ids.txt')
geneNames = ml.loadGeneNames(dir+'names.txt')
```

The first line defines the directory for the data.  The next three lines load, respectively, the gene expression data (a 2D array of size numPatients by numGenes), the correct tissue classifications (+1 for normal, -1 for cancer), and the names of the genes corresponding to the columns of `X`. This last one is optional, and is only used for trying to interpret our models.

Second, we will carve out a portion of our data to be a held-aside test set.  The purpose of this is to evaluate how good our model is; we need to evaluate how accurately it can identify cancerous tissues on samples it has not seen before.  

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
```

This creates a training set that will be used to build the model, and a test set which will be used for evaluating the quality of the model.  We will default to using a third of or the data for testing - more test data comes at the cost of data for training our model, while less test data makes our estimates less robust.

### Train a model

Now that we have data, we will train a classifier.  We will use [Logistic Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), a simple yet effective model.  You can free free to experiment with different [supervised learning models](http://scikit-learn.org/stable/supervised_learning.html).

```python
clf = LogisticRegression()
clf.fit(X_train, y_train)
```

The first line defines our classifier model.  You can customize logistic regression by setting parameters.  For example,

```python
clf = LogisticRegression(penalty='l1')
```

uses a L1 penalty instead of L2 to keep weights lower.  Read the [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) if you would like to change the default settings.

The `fit()` function trains the model based on your settings for the classifier.  It builds a model of the patterns in `X_train` in order to induce the correct labels in `y_train`.

### Evaluate the model

To see how well the model does, we could see well it does on the training data:

```python
trainScore = clf.score(X_train, y_train)
print("Accuracy on training data: %.3f" % trainScore)
```

This will provide an over-confident estimate; of course our model should do well on the data set it was fitting to!  We would like to see if the model *generalizes* to unseen data.  Holding aside data simulates a real-world setting, where a diagnostic tool needs to be accurate on new patients.

```python
testScore = clf.score(X_test, y_test)
print("Accuracy on held-aside test data: %.3f" % testScore)
```

### Extensions

The above demonstration is only a portion of the full machine learning pipeline.  If you have time and want to explore machine learning in depth, consider these extensions.

#### Parameter Tuning

Almost every machine learning model has some set of parameters that affect the learned model.  For example, if you choose to use a neural network model, how many layers do you want?  How big is a layer?  How fast should the model learn?  How long it should it be trained?

It is usually not self-evident what the best settings are, though there exists plenty of literature on best practices.  Usually, the parameter presents a trade-off between competing needs - to a) learn as much as possible from the training data while b) not *overlearning* to those specific examples (aka overfitting).  Recall, our goal is *generalization error* - we want to minimize future error, which is not the same as current error.

For logistic regression, there are a few parameters that you can find in the [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression).  One important parameter is how to penalize the weights (if it all) and how to balance the weight penalty vs the penalty for being inaccurate on the training data .  A weight penalty discourages high weights (usually a sign that the model is trying to hard to learn specific examples) but also increases the error rate.  The default penalty in sci-kit learn is the [L2](https://en.wikipedia.org/wiki/L2_norm) norm and the weight to this penalty is the *complexity* parameter (**C**).  Higher values of C favor reducing the error rate and allowing larger weights, while lower weights *regularize* the model to avoid too much variance.

Instead of arbitrarily picking C, we can try out different values of C.  To do so, we can establish the set of parameter settings to consider:

```python
parameters = {"C": [.1, 1, 10, 100]}
```

Then, we use the grid-search feature to have scikit-learn try out each setting and pick the best one:

```python
tune_clf = GridSearchCV(clf, parameters, cv=3)
//This is code you had from before to train/test
trainScore = tune_clf.score(X_train, y_train)
print("Accuracy on training data: %.3f" % trainScore)
testScore = tune_clf.score(X_test, y_test)
print("Accuracy on held-aside test data: %.3f" % testScore)
```
Only the first line differs; the other lines were from before, but rather than fit the logistic regression classifier (`clf`), we fit and score a version of logistic regression with the best `C` parameter picked by `GridSearchCV`.  To understand the `cv` setting, read the next section.  This repeats the train/test process internally 3 times and picks the C with the best average error.

To see the value of C the model picked, you can access the `best_params_` setting:

```python
print("The chosen C value: %d" % tune_clf.best_params_)
```

Experiment with multiple parameters and different algorithms.  One tip: if your model consistently chooses the lowest or highest parameter option, adjust your range.  Usually, we space in order of magnitude (e.g., 1, 10, 100, etc.) for numerical parameter settings.

#### N-fold cross validation

If you re-run your program multiple times, you'll notice that there is a lot of variance in your error rate.  This is due to the randomness involved in partitioning training/testing examples on small data sets.  By chance, you may get an extra easy example or two and that bumps your accuracy significantly.  This, however, is not a robust estimate of future error.  We can simulate having more data by dividing up the data set into train/test splits multiple times and then taking the average error across the different runs.  The most popular approach for this is known as [N-fold cross-validation](http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation) where `N` is the number of repetitions.  

Take a look in `geneMLLib.py` at the function `runTuneTest`.  This function does two of the extensions: it divides up the data set into `N=5` partitions.  For each, it treats 4 of the partitions as the training set and 1 as the test set:

```python
outer_cv = StratifiedKFold(n_splits=5, shuffle=True)
outer_cv = outer_cv.split(X,y)
for train, test in outer_cv:
      X_train, X_test =  X[train], X[test]
      y_train, y_test = y[train], y[test]
      ...
```

The first two lines specify that we should divide the data up 5 times, and the for loop picks up the individual train/test splits that were created. The rest of the for loop which look familiar - it is also doing the tuning process from the previous section.  To use this function in your main code, modify your classifier call to be:

```python
parameters = {"C": [.1, 1, 10, 100]}
scores, weights = ml.runTuneTest(clf, parameters, X, y)
```

and then print out the results from the different folds:

```python
print("%5s %10s" % ("Fold","AUC Score"))
print("%5s %10s" % ("----","---------"))

for fold in range(len(scores)):
    print("%5d %10.3f" % (fold+1, scores[fold]))
print("-"*16)
print("%5s %10.3f" % ("Mean", np.mean(scores)))
print()
```

#### Interpreting Models

One important aspect of machine learning in computational biology is that nearly as important as accuracy is the *interpretation* of the models; biologists want to gain insight into the underlying biological phenomena that the algorithm picked up on.  The ability to interpret models varies; decision trees are a favorite in the medical community as they mimic human decision processes, while neural networks are non-linear with many parameters and thus very opaque.

Sci-kit learns support for interpreting models is not uniform, so you will need to search for strategies related to your preferred model.  For algorithms that learn weights like logistic regression, we can extract the learned coefficients.  These are an array of values, which each value corresponding to the importance of that column in our data set.  So, if the gene in column `i` is very strongly correlated with colon cancer, the `i`th coefficient should be large.  We can use this to find which genes were most predictive of colon cancer.

Note this is given to you as one of the results from `runTuneTest`; essentially, we just need to retrieve the classifiers coefficients which is a one-dimensional array of length numGenes:

```python
features.append(clf.best_estimator_.coef_)
```

I've written a function `getGeneRanks()` that takes these coefficients and sorts the gene names you loaded in earlier by importance:

```python
rankedGenes = ml.getGeneRanks(weights)
print("Top 10 genes for predicting cancer tissue:")
print("------------------------------------------")
for index in rankedGenes[0:10]:
    print(geneNames[index])
```

Read the original paper and see if your results match theirs.  You should see ribosomal proteins in your list, as supported in the biological literature.  
