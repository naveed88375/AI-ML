#Required packages
import matplotlib.pyplot as plt
import seaborn as sns

#Function to plot data
def plot_data(signal, features):
    #Plot boxplot for variability
    plt.figure(dpi=100)
    sns.set_theme(style='whitegrid')
    sns.boxplot(features.iloc[:,3:7])
    #Label the plot
    plt.xlabel('features')
    plt.ylabel('variability')
    plt.title('Features Variability')
    plt.savefig("boxplot.png")
    
    #Plot raw signal using line chart
    plt.figure(dpi=100)
    signal[signal['Class']=='No Pain'].iloc[0,3:].plot()
    signal[signal['Class']=='Pain'].iloc[0,3:].plot()
    #Label the plot
    plt.legend(['No Pain', 'Pain'])
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title('Signal Variability')
    plt.savefig("linechart.png")