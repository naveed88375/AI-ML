#Import required packages
import numpy as np
from utils import extract_signal_features
from tqdm import tqdm
from utils import load_sound_file
import matplotlib.pyplot as plt

def reconstruction(model, test_files, test_labels, n_mels, frames, n_fft):
    
    #list to store reconstruction errors
    reconstruction_errors = []
    
    #Iterate over each test file
    for index, eval_filename in tqdm(enumerate(test_files), total=len(test_files)):
        # Load signal
        signal, sr = load_sound_file(eval_filename)
        
        # Extract features from this signal:
        eval_features = extract_signal_features(
            signal, 
            sr, 
            n_mels=n_mels, 
            frames=frames, 
            n_fft=n_fft)
        
        # Get predictions from our autoencoder:
        prediction = model.predict(eval_features)

        # Estimate the reconstruction error:
        mse = np.mean(np.mean(np.square(eval_features - prediction), axis=1))
        reconstruction_errors.append(mse)
        
    #Plot reconstruction errors for normal and anomaly signals
    data = np.column_stack((range(len(reconstruction_errors)), reconstruction_errors))
    #Set bins
    bin_width = 0.25
    bins = np.arange(min(reconstruction_errors), max(reconstruction_errors) + bin_width, bin_width)

    fig = plt.figure(figsize=(10,6), dpi=100)
    plt.hist(data[test_labels==0][:,1], bins=bins, color='b', label='Normal signals', edgecolor='#FFFFFF')
    plt.hist(data[test_labels==1][:,1], bins=bins, color='r', label='Anomaly signals', edgecolor='#FFFFFF')
    #Label the plots
    plt.xlabel("MSE")
    plt.ylabel("# Samples")
    plt.title('Reconstruction error distribution on the testing set', fontsize=16)
    plt.legend();
    plt.show()
        
    return reconstruction_errors