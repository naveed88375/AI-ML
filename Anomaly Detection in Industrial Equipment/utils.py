from tqdm import tqdm
import librosa
import numpy as np

#Function to generate training dataset
def generate_dataset(files_list, n_mels=64, frames=5, n_fft=1024, hop_length=512):
    # Number of dimensions for each frame:
    dims = n_mels * frames
    
    for index in tqdm(range(len(files_list)), desc='Extracting features'):
        # Load signal
        signal, sr = load_sound_file(files_list[index])
        
        # Extract features from this signal:
        features = extract_signal_features(
            signal, 
            sr, 
            n_mels=n_mels, 
            frames=frames, 
            n_fft=n_fft)
        
        if index == 0:
            dataset = np.zeros((features.shape[0] * len(files_list), dims), np.float32)
            
        dataset[features.shape[0] * index : features.shape[0] * (index + 1), :] = features

    return dataset

def extract_signal_features(signal, sr, n_mels=64, frames=5, n_fft=1024):
    # Compute a mel-scaled spectrogram:
    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels
    )
    
    # Convert to decibel (log scale for amplitude):
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Generate an array of vectors as features for the current signal:
    features_vector_size = log_mel_spectrogram.shape[1] - frames + 1
    
    # Skips short signals:
    dims = frames * n_mels
    if features_vector_size < 1:
        return np.empty((0, dims), np.float32)
    
    # Build N sliding windows (=frames) and concatenate them to build a feature vector:
    features = np.zeros((features_vector_size, dims), np.float32)
    for t in range(frames):
        features[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t:t + features_vector_size].T
        
    return features

#Load sound file
def load_sound_file(wav_name, mono=False, channel=0):
    multi_channel_data, sampling_rate = librosa.load(wav_name, sr=None, mono=mono)
    signal = np.array(multi_channel_data)[channel, :]
    
    return signal, sampling_rate