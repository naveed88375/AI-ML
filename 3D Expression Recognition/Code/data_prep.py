#Packages used
import pandas as pd
import numpy as np

def preprocess(data, type):
    if type=='Translated':
        print("Translating data...")
        #Calculate mean for x, y and z axis
        x_mean = data.iloc[:,:83].mean(axis=1)
        y_mean = data.iloc[:,83:166].mean(axis=1)
        z_mean = data.iloc[:,166:249].mean(axis=1)
        #Translate data for x, y and z axis
        x_translated = pd.DataFrame(data.iloc[:,:83].to_numpy() - x_mean.to_numpy().reshape(-1,1))
        y_translated = pd.DataFrame(data.iloc[:,83:166].to_numpy() - y_mean.to_numpy().reshape(-1,1))
        z_translated = pd.DataFrame(data.iloc[:,166:249].to_numpy() - z_mean.to_numpy().reshape(-1,1))
        #Merge columns and assign labels
        df_translated = pd.concat([x_translated,y_translated,z_translated],axis=1)
        df_translated['label'] = data['label'].to_numpy()
        print("Translation completed.")
        return df_translated
    
    #Rotate data around X-axis
    elif type == 'RotatedX':
        df_x = data.copy()
        df_len = len(df_x)
        #Matrix to rotate data around x-axis
        rot_mat= np.array([
            [1,  0,             0            ],
            [0,  np.cos(np.pi), np.sin(np.pi)],
            [0, -np.sin(np.pi), np.cos(np.pi)]])
        print("Data rotation started...")
        #Rotate each data point
        for i in range(df_len):
            df_x.iloc[i,:-1] = np.dot(rot_mat, data.iloc[:,:-1].to_numpy().reshape(len(data),3, 83)[i]).reshape(-1,1).squeeze()
            if i%1000 == 0:
                print(str(round((i/df_len)*100))+"% completed")
        print("Data rotation completed.")
        return df_x
    
    #Rotate data around Y-axis
    elif type == 'RotatedY':
        df_y = data.copy()
        df_len = len(df_y)
        #Matrix to rotate data around y-axis
        rot_mat = np.array([
            [np.cos(np.pi), 0, -np.sin(np.pi)],
            [0,             1,  0            ],
            [np.sin(np.pi), 0,  np.cos(np.pi)]])
        print("Data rotation started...") 
        #Rotate each data point
        for i in range(df_len):
            df_y.iloc[i,:-1] = np.dot(rot_mat, data.iloc[:,:-1].to_numpy().reshape(len(data),3, 83)[i]).reshape(-1,1).squeeze()
            if i%1000 == 0:
                print(str(round((i/df_len)*100))+"% completed")
        print("Data rotation completed.")
        return df_y
    
    #Rotate data round Z-axis
    elif type == 'RotatedZ':
        df_z = data.copy()
        df_len = len(df_z)
        #Matrix to rotate data around z-axis
        rot_mat = np.array([
            [np.cos(np.pi) , np.sin(np.pi), 0],
            [-np.sin(np.pi), np.cos(np.pi), 0],
            [0             , 0            , 1]])
        print("Data rotation started...")        
        #Rotate each data point
        for i in range(df_len):
            df_z.iloc[i,:-1] = np.dot(rot_mat, data.iloc[:,:-1].to_numpy().reshape(len(data),3, 83)[i]).reshape(-1,1).squeeze()
            if i%1000 == 0:
                print(str(round((i/df_len)*100))+"% completed")
        print("Data rotation completed.")
        return df_z