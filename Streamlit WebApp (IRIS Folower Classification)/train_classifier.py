#Import required packages
import joblib
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

#Load the dataset
iris = datasets.load_iris()

#Fetch features and labels
X = iris.data
Y = iris.target

#Create the classifier
clf = RandomForestClassifier()

#Fit the classifier to the data
clf.fit(X, Y)

#Save the classifier
joblib.dump(clf, "./random_forest.joblib")