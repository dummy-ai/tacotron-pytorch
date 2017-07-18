import sys
import copy
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from modules.hyperparams import Hyperparams as hp
from utils import ProgressBar


# credits: https://github.com/Kyubyong/tacotron/blob/master/utils.py
def compute_spectrograms(audio_file):
    '''Extracts melspectrogram and log magnitude from given `audio_file`.
    Args:
        audio_file: A string, full path of an audio file

    Returns:
      S: A 2D numpy array, a melscale spectrogram
        with shape of (n_mels, T)
      magnitude: A 2D numpy array, with shape (1+hp.n_fft//2, T)
    '''

    # loading audio file
    y, sr = librosa.load(audio_file, sr=hp.sr)  # or set sr to hp.sr.

    # pre-emphasis
    emphasized_y = emphasize(y)

    # stft. D: (1+n_fft//2, T)
    D = librosa.stft(y=emphasized_y,
                     n_fft=hp.n_fft,
                     hop_length=hp.hop_length,
                     win_length=hp.win_length)

    # magnitude spectrogram
    magnitude = np.abs(D)  # (1+n_fft/2, T)

    # power spectrogram
    power = magnitude**2  # (1+n_fft/2, T)

    # mel spectrogram
    # (n_mels, T)
    S = librosa.feature.melspectrogram(S=power, n_mels=hp.n_mels)

    # (n_mels, T), (1+n_fft/2, T)
    return S.astype(np.float32), magnitude.astype(np.float32)


def emphasize(signal):
    return np.append(signal[0], signal[1:] - hp.pre_emphasis * signal[:-1])


def spectrogram2wav(spectrogram):
    '''
    spectrogram: [t, f], i.e. [t, nfft // 2 + 1]
    '''
    spectrogram = spectrogram.T  # [f, t]
    X_best = copy.deepcopy(spectrogram)  # [f, t]
    bar = ProgressBar(hp.n_iter, unit='')
    for i in range(hp.n_iter):
        bar.update(i)
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(
            X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)  # [f, t]
        phase = est / np.maximum(1e-8, np.abs(est))  # [f, t]
        X_best = spectrogram * phase  # [f, t]
    X_t = invert_spectrogram(X_best)

    return np.real(X_t)


def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length,
                         win_length=hp.win_length, window="hann")


def griffinlim(spectrogram, n_iter=hp.n_iter, window='hann',
               n_fft=2048, hop_length=hp.hop_length):
    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

    for i in range(hp.n_iter):
        full = np.abs(spectrogram).astype(np.complex) * angles
        inverse = librosa.istft(full, hop_length=hop_length, window=window)
        rebuilt = librosa.stft(inverse, n_fft=n_fft, hop_length=hop_length, window=window)
        angles = np.exp(1j * np.angle(rebuilt))

    full = np.abs(spectrogram).astype(np.complex) * angles
    inverse = librosa.istft(full, hop_length=hop_length, window=window)

    return np.real(inverse)


def usage(cmd):
    print("Usage: {0} [audio_file]".format(cmd))
    print("\tVisualize mel scale spectrogam")
    print("\tInvoke this script with pythonw on Mac OS X")


def main():
    if len(sys.argv) == 2:
        audio_file = sys.argv[1]
        S = melscale(audio_file)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            librosa.power_to_db(S, ref=np.max),
            y_axis='mel',
            x_axis='time',
            fmax=8000
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()
        plt.show()
    else:
        usage(sys.argv[0])

if __name__ == "__main__":
    main()
