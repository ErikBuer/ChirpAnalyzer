from scipy.signal import chirp, sweep_poly, spectrogram, welch
from scipy.special import factorial
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
import rftool.radar as radar
import rftool.utility as util

Fs=np.intc(802e3) # receiver sample rate
T=np.float(6e-3)  # Pulse duration


# Time domain window for NLFM generation
NLFM = radar.chirp(Fs)
NLFM.fftLen = 2048

# Synthesize the target autocorrelation function
window_t = signal.chebwin(np.intc(2048), 60)
#window_t = signal.gaussian(np.intc(2048), 360)
#window_t = signal.gaussian(np.intc(2048), 400)
NLFM.getCoefficients( window_t, targetBw=200e3, centerFreq=100e3, T=T)



# Time domain window for LFM generation
LFM = radar.chirp(Fs)

# Generate linear chirp
window_t = np.array([1,1,1])
LFM.getCoefficients( window_t, targetBw=200e3, centerFreq=100e3, T=T)

""""
LFMsig = LFM.genFromPoly()
NLFMsig = NLFM.genFromPoly()
signals = np.stack((LFMsig, NLFMsig), axis=-1)

radar.ACF(signals, label=['LFM', 'NLFM'])
radar.hilbert_spectrum(np.real(LFMsig), Fs, label='LFM')
radar.hilbert_spectrum(np.real(NLFMsig), Fs, label='NLFM')

radar.hilbert_spectrum(np.real(NLFM.modulate([1,1,1,1,0,1,0,1])), Fs, label='NLFM')
"""
NLFM.demodulate(NLFM.modulate())

plt.show()


"""
TODO Generation
- Modulate signal

TODO Analyze
- List of parameters to estimate on single chirp
- Skriv i rapport hvordan HH-transform foregår.
- Lag cyclostationary transform.
- Vurder Om kurvene skal karakteriseres fra sentrum, eller frs siden (i tid-frekvens).
"""