import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np

def spectrogram( sig, Fs ):
    f, t, Sxx = signal.spectrogram(np.real(sig), Fs)
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()