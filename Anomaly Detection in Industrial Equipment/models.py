#Required packages
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten

def autoencoder_baseline(input_dims):
    #input layer
    inputLayer = Input(shape=(input_dims,))
    #Encoder block
    x = Dense(128, activation="relu")(inputLayer)
    x = Dense(64, activation="relu")(x)
    #Latent space
    x = Dense(32, activation="relu")(x)
    #Decoder block
    x = Dense(64, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    #Output layer
    x = Dense(input_dims, activation=None)(x)
    #Create and return the model
    return Model(inputs=inputLayer, outputs=x)
