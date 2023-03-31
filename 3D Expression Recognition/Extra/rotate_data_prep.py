#Packages used
import pandas as pd
import numpy as np

#Read data file and remove unwanted columns
df = pd.read_csv('final_data.csv').dropna()
if sum(df.columns == 'Unnamed: 0'):
    df = df.drop('Unnamed: 0',axis=1)

#Create copies to store each data type
df_x = df.copy()
df_y = df.copy()
df_z = df.copy()

#Matrix to rotate data around x-axis
rot_mat_x = np.array([
    [1,  0,             0            ],
    [0,  np.cos(np.pi), np.sin(np.pi)],
    [0, -np.sin(np.pi), np.cos(np.pi)]
])
#Matrix to rotate data around y-axis
rot_mat_y = np.array([
    [np.cos(np.pi), 0, -np.sin(np.pi)],
    [0,             1,  0            ],
    [np.sin(np.pi), 0,  np.cos(np.pi)]
])
#Matrix to rotate data around z-axis
rot_mat_z = np.array([
    [np.cos(np.pi) , np.sin(np.pi), 0],
    [-np.sin(np.pi), np.cos(np.pi), 0],
    [0             , 0            , 1]
])

#Rotate each data point
for i in range(len(df)):
    df_x.iloc[i,:-1] = np.dot(rot_mat_x, df.iloc[:,:-1].to_numpy().reshape(len(df),3, 83)[i]).reshape(-1,1).squeeze()
    df_y.iloc[i,:-1] = np.dot(rot_mat_y, df.iloc[:,:-1].to_numpy().reshape(len(df),3, 83)[i]).reshape(-1,1).squeeze()
    df_z.iloc[i,:-1] = np.dot(rot_mat_z, df.iloc[:,:-1].to_numpy().reshape(len(df),3, 83)[i]).reshape(-1,1).squeeze()
    if i%1000 == 0:
        print(str(i)+" iteration completed.")
        print(str(60000-i)+" remaining.")
    
#Store final result to csv files
df_x.to_csv("data_rotated_x.csv")
df_y.to_csv("data_rotated_y.csv")
df_z.to_csv("data_rotated_z.csv")

print("Data conversion completed.")