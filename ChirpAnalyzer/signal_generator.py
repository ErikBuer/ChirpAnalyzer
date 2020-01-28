from scipy.signal import chirp, sweep_poly, spectrogram, welch
from scipy.special import factorial
import scipy.signal as signal
import analysis as analyze
import matplotlib.pyplot as plt
import numpy as np
import rftool.radar as radar
from rftool.utility import *
import rftool.utility as util

Fs=np.intc(100e3)           # receiver sample rate
frameSize=np.float(128e-3)  # seconds
"""
a = 10e3
b = 1e3
d = 1e3
c = np.array([10e3,10,1e6,0,a,0,b,0,d,0])
sig = radar.cascadedIntegralChirpGenerator( frameSize, c, Fs )
"""


"""
f, t, Sxx = signal.spectrogram(sig, Fs)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


f, Pxx_den = welch(sig, Fs, 'flattop', 1024, scaling='density')
Pxx_den_dB = pow2db(np.abs(Pxx_den))
plt.plot(f, Pxx_den_dB)
plt.grid()
plt.ylim([-120,0])
plt.title("Welch PSD Estimate")
plt.xlabel('frequency [Hz]')
plt.ylabel('dBW/Hz')
plt.show()
"""

# Time domain window for the function to match
window_f = signal.bartlett(256)
window_f = np.multiply(np.subtract(window_f, window_f.max()), 100)
#window_f = np.fft.fft(window_t,2048)/2048
# Shift and normalize

# Remove infinitesimally small components
#window_f = util.mag2db(np.maximum(window_f, 1e-10))
f = np.linspace(-Fs/4, Fs/4, len(window_f))

"""
plt.plot(f, window_f)
plt.title("Frequency Response")
plt.ylabel("Normalized magnitude [dB]")
plt.xlabel("Frequency [Hz]")
plt.show()
"""

chirp = radar.chirp(frameSize, Fs)

coeff = chirp.getCoefficients( window_f, f, order=10, symm=True, fftLen = 8192)
print( coeff )

sig = chirp.generate()
radar.ACF(sig, singleSided=False)
#radar.hilbert_spectrum(np.real(sig), Fs)
plt.show()