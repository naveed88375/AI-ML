#Required Packages
import pandas as pd
import glob
from os import listdir

def data_load(base_dir):
    #Set path
    base_dir = base_dir+"/"
    #Emotion labels
    labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]
    #Number of folders in data direcotry
    folders = listdir(base_dir)
    #Remove unnecessary files
    if '4DFE_Featurepoints83.JPG' in folders:
        folders.remove('4DFE_Featurepoints83.JPG')

    #To store data fetched from directories
    data = []
    f_len = len(folders)

    print("Data loading started...")

    #Loop over folders in data
    for i in range(f_len):
        person_id = folders[i]
        #Set path for current folder
        person_path = base_dir + person_id
        #Loop over each emotion class
        for p in labels:
            #Set path for current emotion
            emotion_path = person_path + "/" + p + "/"
            #Files in the emotion folder
            sub_files = listdir(emotion_path)
            d_temp = []
            #Loop over each file
            for m in sub_files:
                file_path = emotion_path+m
                #Read data from the file
                d_temp.append(pd.read_csv(file_path, delimiter=" ", header=None,encoding='latin-1',on_bad_lines='skip',lineterminator='\n').dropna().iloc[:,1:].to_numpy().reshape(-1,1).squeeze())
            #Store data from all files in a dataframe
            df = pd.DataFrame(d_temp)
            #Create corresponding label
            df['label']=p
            #Append dataframe to a list
            data.append(df)
        print(str(round((i/f_len)*100))+"% completed")
    #Merge all dataframes
    df_final = pd.concat(data).dropna()
    print("Data loaded successfully.")

    return df_final