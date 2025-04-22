import librosa
import pandas as pd
import numpy as np
import skimage
import os
from os.path import join

train_rec = pd.read_csv('./Bird-audio/csv/chiffchaff-withinyear-fg-trn.csv')
test_rec = pd.read_csv('./Bird-audio/csv/chiffchaff-withinyear-fg-tst.csv')

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X-X.min())/(X.max()-X.min())
    X_scaled = X_std * (max-min) + min
    return X_scaled

for i in range(len(train_rec)):
    for n in train_rec.columns[1:]:
        if train_rec[n].iloc[i] == 1:
            filename = str(train_rec.iloc[i]['wavfilename'])
            audio_file = join('./Bird-audio/chiffchaff-fg/', filename)

            x, sr = librosa.load(audio_file, sr=None)

            # Converts to spectrogram
            # Parameters from Stowell 2019 AAII paper
            S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=1024, hop_length=512, window='hamming', n_mels=40, fmax=sr/2)

            # Small value added to prevent log of 0
            S = np.log(S+1e-9)

            # Scales image to fit in 8 bits
            img = scale_minmax(S, 0, 255).astype(np.uint8)
            img = np.flip(img, axis=0) # Low freq on bottom
            img = 255-img # Strips colour saving space

            # Splits the name of the file to find what folder it should save it in
            destination = join('./cnn-data/train', str(n))

            # Saves image
            skimage.io.imsave(destination + '/'+ filename[:-4]+".png", img)
for i in range(len(test_rec)):
    for n in test_rec.columns[1:]:
        if test_rec[n].iloc[i] == 1:
            filename = str(test_rec.iloc[i]['wavfilename'])
            audio_file = join('./Bird-audio/chiffchaff-fg/', filename)

            x, sr = librosa.load(audio_file, sr=None)

            # Converts to spectrogram
            # Parameters from Stowell 2019 AAII paper
            S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=1024, hop_length=512, window='hamming', n_mels=40, fmax=sr/2)

            # Small value added to prevent log of 0
            S = np.log(S+1e-9)

            # Scales image to fit in 8 bits
            img = scale_minmax(S, 0, 255).astype(np.uint8)
            img = np.flip(img, axis=0) # Low freq on bottom
            img = 255-img # Strips colour saving space

            # Splits the name of the file to find what folder it should save it in
            destination = join('./cnn-data/test', str(n))

            # Saves image
            skimage.io.imsave(destination +'/'  + filename[:-4]+".png", img)