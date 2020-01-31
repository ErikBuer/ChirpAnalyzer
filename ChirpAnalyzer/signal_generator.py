from scipy.signal import chirp, sweep_poly, spectrogram, welch
from scipy.special import factorial
import scipy.signal as signal
import analysis as analyze
import matplotlib.pyplot as plt
import numpy as np
import rftool.radar as radar
import rftool.utility as util

Fs=np.intc(500e3)           # receiver sample rate
frameSize=np.float(10e-3)  # seconds


# Time domain window for the function to match
chirp = radar.chirp(frameSize, Fs)
chirp.fftLen = 1024

"""
# Generate target autocorrelation window
#window_t = signal.chebwin(512, 60)
#r_xx = chirp.PSD(window_t)
#r_xx = np.fft.fftshift(r_xx / abs(r_xx).max())
#r_xx_dB = util.mag2db( r_xx )
"""

# Synthesize the target autocorrelation function
window_t = signal.chebwin(4096, 45)
r_xx = np.fft.fft(window_t, chirp.fftLen)
r_xx = np.abs(np.fft.fftshift(r_xx / abs(r_xx).max()))
r_xx_dB = util.mag2db(np.maximum(r_xx, 1e-10))
chirp = radar.chirp(frameSize, Fs)

coeff = chirp.getCoefficients( r_xx_dB, order=8, symm=True)
print( coeff )

sig = chirp.generate()
#radar.ACF(sig, singleSided=False)
radar.hilbert_spectrum(np.real(sig), Fs)
plt.show()