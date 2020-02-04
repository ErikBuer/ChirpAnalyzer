from scipy.signal import chirp, sweep_poly, spectrogram, welch
from scipy.special import factorial
import scipy.signal as signal
import analysis as analyze
import matplotlib.pyplot as plt
import numpy as np
import rftool.radar as radar
import rftool.utility as util

Fs=np.intc(1.6e6) # receiver sample rate
T=np.float(10e-3)  # Pulse duration


# Time domain window for the function to match
NLFM = radar.chirp(Fs)
NLFM.fftLen = 2048

# Synthesize the target autocorrelation function
window_t = signal.chebwin(np.intc(2048), 60)
#window_t = signal.gaussian(np.intc(2048), 360)
#window_t = signal.gaussian(np.intc(2048), 400)
NLFM.getCoefficients( window_t, targetBw=160e3, centerFreq=100e3, symm=True, T=T)



# Time domain window for the function to match
LFM = radar.chirp(Fs)
LFM.fftLen = 2048

# Synthesize the target autocorrelation function
window_t = np.array([1,1,1])
LFM.getCoefficients( window_t, targetBw=160e3, centerFreq=100e3, symm=True, T=T)

signals = np.stack((LFM.sig, NLFM.sig), axis=-1)

radar.ACF(signals)
radar.hilbert_spectrum(np.real(LFM.sig), Fs)
radar.hilbert_spectrum(np.real(NLFM.sig), Fs)
plt.show()