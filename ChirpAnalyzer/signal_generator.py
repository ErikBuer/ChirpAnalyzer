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
chirp = radar.chirp(Fs)
chirp.fftLen = 1024

# Synthesize the target autocorrelation function
window_t = signal.chebwin(np.intc(512), 60)
plt.plot(window_t)
plt.show()


coeff = chirp.getCoefficients( window_t, order=8, symm=True)
print( coeff )

sig = chirp.generate()
#radar.ACF(sig, singleSided=False)
radar.hilbert_spectrum(np.real(sig), Fs)
plt.show()