#Import required packages
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from feature_extraction import feature_extraction
from plot_data import plot_data
import argparse

#Read input arguments
parser = argparse.ArgumentParser()
parser.add_argument("data_type", nargs='?', type=str)
parser.add_argument("data_file", nargs='?', type=str)
args = parser.parse_args()
dtype = args.data_type
data_path = args.data_file

#Read data file
data = pd.read_csv(data_path, header=None)
data.rename(columns = {0:'ID', 1:'DataType', 2:'Class'}, inplace = True)

#Check data type, extract features and explore data
if dtype=='dia':
    df = data[data['DataType'] == 'BP Dia_mmHg']
    df_feat = feature_extraction(df, dtype)
    plot_data(df, df_feat)
elif dtype=='sys':
    df = data[data['DataType'] == 'LA Systolic BP_mmHg']
    df_feat = feature_extraction(df, dtype)
    plot_data(df, df_feat)
elif dtype=='eda':
    df = data[data['DataType'] == 'EDA_microsiemens']
    df_feat = feature_extraction(df, dtype)
    plot_data(df, df_feat)
elif dtype=='res':
    df = data[data['DataType'] == 'Respiration Rate_BPM']
    df_feat = feature_extraction(df, dtype)
    plot_data(df, df_feat)
else:
    df_dia = feature_extraction(data[data['DataType'] == 'BP Dia_mmHg'], 'dia')
    df_sys = feature_extraction(data[data['DataType'] == 'LA Systolic BP_mmHg'], 'sys')
    df_eda = feature_extraction(data[data['DataType'] == 'LA Systolic BP_mmHg'], 'eda')
    df_res = feature_extraction(data[data['DataType'] == 'Respiration Rate_BPM'], 'res')
    df_feat = pd.concat([df_dia,df_sys.iloc[:,3:],df_eda.iloc[:,3:],df_res.iloc[:,3:]],axis=1)

#Separate features and labels
X = df_feat.drop(['Class','ID', 'DataType'],axis=1).to_numpy()
Y = df_feat['Class'].to_numpy()

#Create random forest model
model = RandomForestClassifier()

#Empty lists to store results
P_score, R_score, A_score, C_Mat = [], [], [], []

#Set Kfold strategy with 10 folds
kfold = model_selection.GroupKFold(n_splits=10)

#Create groups of users
groups = np.repeat(np.arange(0, 10),12)

#Perform 10 fold training and validation
for train_index, test_index in kfold.split(X, groups=groups):
    #Get train and test splits
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    #Model training
    model.fit(X_train, y_train)
    #Model predictions
    pred = model.predict(X_test)
    #Calculate metric scores and store
    P_score.append(precision_score(y_test, pred, average='micro'))
    R_score.append(recall_score(y_test, pred, average='macro'))
    A_score.append(accuracy_score(y_test, pred))
    C_Mat.append(confusion_matrix(y_test, pred))

#Aggregate results from each fold
scores = {'Precision':np.mean(P_score),
         'Recall':np.mean(R_score),
         'Accuracy':np.mean(A_score)}
conf_mat = sum(C_Mat)/len(C_Mat)

#Print results to screen
print("Precision score: ", scores['Precision'])
print("Recall score: ", scores['Recall'])
print("Accuracy score: ", scores['Accuracy'])

#Plot confusion matrix
fig = plt.figure()
sns.heatmap(conf_mat,
            cmap='Blues',
            linecolor='white',
            linewidths=1,
            xticklabels=['No Pain', 'Pain'],
            yticklabels=['No Pain', 'Pain'],
            annot=True,
           fmt='.2f')
#Label the plot
plt.title('Confusion Matrix')
plt.ylabel('Ground Truth')
plt.xlabel('Predictions')
plt.savefig("ConfusionMatrix.png")
plt.show()