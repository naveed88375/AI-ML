#Import required packages
import pandas as pd
import numpy as np
import load_data
import data_prep
import argparse
from sklearn import tree, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dataplot

#Set input arguments
parser = argparse.ArgumentParser()
parser.add_argument("classifier", nargs='?', type=str)
parser.add_argument("data_type", nargs='?', type=str)
parser.add_argument("data_dir", nargs='?', type=str)
args = parser.parse_args()

#Get entered arguments
clf = args.classifier
d_type = args.data_type
d_dir = args.data_dir

#Check for classifier type
if clf == 'RF':
    model = RandomForestClassifier()
elif clf == 'SVM':
    model = svm.SVC()
elif clf == 'TREE':
    model = tree.DecisionTreeClassifier()
#Load the dataset from the data directory
data = load_data.data_load(d_dir)
#Check if any transformation is required
if d_type == 'Original':
    df = data.copy()
else:
    df = data_prep.preprocess(data, d_type)

#Plot 3D scatter plot
dataplot.plot_data(d_type, df)
#Get features and labels
X = df.drop('label',axis=1)
Y = df['label'].to_numpy()
#Perform data normalization
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
#Setup kfold validation
kfold = model_selection.KFold(n_splits=10, random_state=42, shuffle=True)
#Lists to store each metric scores
P_score, R_score, A_score, C_Mat = [], [], [], []

print("10 fold validation started.")
#Counter
i = 1
#Loop over each data fold
for train_index, test_index in kfold.split(X_norm):
    #Get train and test splits
    X_train, X_test = X_norm[train_index], X_norm[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    #Model training
    model.fit(X_train, y_train)
    #Model predictions
    pred = model.predict(X_test)
    #Calculate metric scores and store
    P_score.append(precision_score(y_test, pred, average='macro'))
    R_score.append(recall_score(y_test, pred, average='macro'))
    A_score.append(accuracy_score(y_test, pred))
    C_Mat.append(confusion_matrix(y_test, pred))
    #%completetion message
    print(str(i)+" fold completed.")
    i+=1
print("Training and validation completed.")    

#Emotion classes
labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]
#Take average of the scores
scores = {'Precision':np.mean(P_score),
         'Recall':np.mean(R_score),
         'Accuracy':np.mean(A_score)}
conf_mat = sum(C_Mat)/len(C_Mat)

#Save the results
print("Saving obtained results.")
# open file for writing
f = open(clf+"_"+d_type+"_scores.txt","w")
# write file
f.write( str(scores) )
# close file
f.close()

#Plot and save confusion matrix
fig = plt.figure()
sns.heatmap(conf_mat,
            cmap='Blues',
            linecolor='white',
            linewidths=1,
            xticklabels=labels,
            yticklabels=labels,
            annot=True,
           fmt='.2f')
plt.title('Confusion Matrix')
plt.ylabel('Ground Truth')
plt.xlabel('Predictions')
plt.savefig(clf+"_"+d_type+"_ConfusionMatrix.png")
print("Results saved successfully.")