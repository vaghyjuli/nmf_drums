import glob
import os
import librosa
import numpy as np

"""
Calculate and save magnitude STFTs of all audio files as numpy matrices,
in order to save computation time during optimization.
"""

EPS = 2.0 ** -52

def stft_magnitude(wav_file, window):
    hop = int(window / 2)
    x, Fs = librosa.load(wav_file)
    X = librosa.stft(x, n_fft=window, hop_length=hop, win_length=window, window='hann', center=True, pad_mode='constant')
    Y = np.abs(X) + EPS
    return Y

data_folder = "/Users/juliavaghy/Desktop/data"
windows = [256, 1024, 2048, 4096]
for window in windows:
    os.chdir(os.path.join(data_folder, "drum-loops"))
    sample_directories = [item for item in os.listdir() if os.path.isdir(item)]
    for sample_directory in sample_directories:
        os.chdir(os.path.join(data_folder, "drum-loops", sample_directory))
        make_path = lambda f : os.path.join(data_folder, "drum-loops", sample_directory, f)
        wav_files = glob.glob("*.wav")
        if len(wav_files) != 1:
            print(f"There should be a single WAV file in {sample_directory}")
            continue
        f_name = wav_files[0]
        wav_file = os.path.join(data_folder, "drum-loops", sample_directory, f_name)
        stft = stft_magnitude(wav_file, window)
        np.save(f"{f_name[:-4]}-{window}.npy", stft)

    os.chdir(os.path.join(data_folder, "kits"))
    kits = [item for item in os.listdir() if os.path.isdir(item)]
    for kit in kits:
        os.chdir(os.path.join(data_folder, "kits", kit, "instruments"))
        instruments = glob.glob("*.wav")
        for instrument in instruments:
            wav_file = os.path.join(data_folder, "kits", kit, "instruments", instrument)
            stft = stft_magnitude(wav_file, window)
            np.save(f"{instrument[:-4]}-{window}.npy", stft)

    os.chdir(os.path.join(data_folder, "background"))
    noises = glob.glob("*.wav")
    for noise in noises:
        wav_file = os.path.join(data_folder, "background", noise)
        stft = stft_magnitude(wav_file, window)
        print(stft.shape)
        np.save(f"{noise[:-4]}-{window}.npy", stft)

    os.chdir(os.path.join(data_folder, "background-loud"))
    noises = glob.glob("*.wav")
    for noise in noises:
        wav_file = os.path.join(data_folder, "background-loud", noise)
        stft = stft_magnitude(wav_file, window)
        print(stft.shape)
        np.save(f"{noise[:-4]}-{window}.npy", stft)