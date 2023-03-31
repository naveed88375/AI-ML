#Import required packages
import pandas as pd
import matplotlib.pyplot as plt

#Function to plot 3d scatter plot
def plot_data(d_type, df):
    #Get sample from the dataframe and reshape it to x, y and z axis
    df_plot = df.iloc[:,:-1].to_numpy().reshape(len(df),83, 3)[400]
    #Create empty figure
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    #Plot data on the 3-axis
    surf = ax.scatter(df_plot[:,0],
                      df_plot[:,1],
                      df_plot[:,2],
                      c='cyan',
                      edgecolor='b')
    #Set figure dimensions
    ax.view_init(elev=60.)
    plt.title(d_type+" Happy")
    #Save plot to image file
    plt.savefig(d_type+"_3Dscatter.png")
    print("Data 3D scatter plot saved to file.")