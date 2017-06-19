import sys
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def melscale(audio_file, sr=None, n_fft=2048, 
             hop_length=512, n_mels=80):
    y, sr = librosa.load(audio_file, sr=sr)
    return librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=80)

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

