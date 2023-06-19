from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def detection(errs, test_labels):
    
    #Target labels
    target_labels = ["Normal", "Anomaly"]
    
    #Set threshold to be 10% more than the mean on normal sound signals
    thresh = np.mean(errs[:120]) + 0.1*np.mean(errs[:120])
    
    #Print the error threshold
    print("The error threshold is set to be: ",thresh)
    
    #Get prediction labels using error threshold
    pred_labels = np.array(errs)>thresh
    
    #Print the classification report
    print(classification_report(test_labels, pred_labels,target_names=target_labels))
    
    #Plot confusion matrix
    fig = plt.figure(dpi=100)
    sns.heatmap(confusion_matrix(test_labels,pred_labels),
                cmap='Blues',
                linecolor='white',
                linewidths=1,
                xticklabels=target_labels,
                yticklabels=target_labels,
                annot=True)
    #Label the plot
    plt.title('Confusion Matrix')
    plt.ylabel('Ground Truth')
    plt.xlabel('Predictions')
    plt.show()
    
    