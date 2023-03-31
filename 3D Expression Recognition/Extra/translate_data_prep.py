#Required packages
import pandas as pd

#Read data file and remove unwanted columns
df = pd.read_csv('final_data.csv')
if sum(df.columns == 'Unnamed: 0'):
    df = df.drop('Unnamed: 0',axis=1)

print("Translating data...")

#Calculate mean for x, y and z axis
x_mean = df.iloc[:,:83].mean(axis=1)
y_mean = df.iloc[:,83:166].mean(axis=1)
z_mean = df.iloc[:,166:249].mean(axis=1)

#Translate data for x, y and z axis
x_translated = pd.DataFrame(df.iloc[:,:83].to_numpy() - x_mean.to_numpy().reshape(-1,1))
y_translated = pd.DataFrame(df.iloc[:,83:166].to_numpy() - y_mean.to_numpy().reshape(-1,1))
z_translated = pd.DataFrame(df.iloc[:,166:249].to_numpy() - z_mean.to_numpy().reshape(-1,1))

#Merge columns and assign labels
df_translated = pd.concat([x_translated,y_translated,z_translated],axis=1)
df_translated['label'] = df['label']

#Save data to csv file
df_translated.to_csv("data_translated.csv")

print("Conversion completed.")