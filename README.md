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
  * [`predictCancer.py`](src/predictCancer.py) - the main program you will implement following the instructions below


## Data

We will develop using the [colon cancer](data/colonCancer) data set, detailed in the seminal paper [Broad patterns of gene expression revealed by clustering analysis of tumor and normal colon tissues probed by oligonucleotide arrays](https://www.ncbi.nlm.nih.gov/pubmed/10359783) from Alon et al.  You can find the relevant files in your [`data`](data).  The original files are also [publicy available](http://genomics-pubs.princeton.edu/oncology/affydata/index.html).

There are three files made available:
 * `expression.csv` - the gene expression data from 2000 genes for 62 patients.  This file has already been processed, as detailed in the original paper. Rows correspond to patients and columns correspond to genes.
 * `names.txt` - the names of the genes in the expression data set.  The names file is in order according to the columns of `expression.csv` (e.g., column 1 in the expression data is the gene in row 1 of the names file).
 * `ids.txt` - identifies normal vs cancerous tissues (i.e., the labels) from the samples provided in `expressions.txt`.  The rows correspond, with a negative value indicating cancerous tissue and positive values indicating normal tissue.  The magnitude of the value is not relevant for our application.

## Implementation

In the provided data, the gene expressions was measured for tissue samples from 62 patients.  40 of the samples were identified as positive for colon cancer and 22 were negative (normal tissue).  One common method for analyzing gene expression data is to perform classification to learn differences in the expression patterns of samples from different categories (cancerous vs normal).  We will utilize [Scikit-Learn](http://scikit-learn.org/), a Python library for accessible, yet efficient data mining.

### Getting started

Clone or download the repo and start in the main directory.

```bash
$ git clone https://github.com/ameetsoni/CompBioNSF-Module.git
$ cd CompBioNSF-Module/
```

Move into the `src` directory and open the main program with your favorite editor e.g.,

```bash
$ cd src
$ atom predictCancer.py
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

```
testScore = clf.score(X_test, y_test)
print("Accuracy on held-aside test data: %.3f" % testScore)
```

## Extensions

* tuning parameters
* using n-fold cross validation
* interpreting models
