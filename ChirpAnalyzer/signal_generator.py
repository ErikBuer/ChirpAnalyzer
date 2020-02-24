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
#window_t = signal.chebwin(np.intc(2048), 60)
window_t = signal.hamming(np.intc(2048))
#window_t = signal.gaussian(np.intc(2048), 360)
#window_t = signal.gaussian(np.intc(2048), 400)

NLFM.getCoefficients( window_t, targetBw=100e3, centerFreq=100e3, T=T)


"""
LFM = radar.chirp(Fs)

# Generate linear chirp
window_t = np.array([1,1,1])
LFM.getCoefficients( window_t, targetBw=200e3, centerFreq=100e3, T=T)

LFMsig = LFM.genNumerical()
NLFMsig = NLFM.genFromPoly()
signals = np.stack((LFMsig, NLFMsig), axis=-1)

radar.ACF(signals, label=['LFM', 'NLFM'])
#radar.hilbert_spectrum(np.real(LFMsig), Fs, label='LFM')
radar.hilbert_spectrum(np.real(NLFMsig), Fs, label='NLFM')
"""

package = np.random.randint(0, 1, 32)
modSig = NLFM.modulate( package )
modSig = util.wgnSnr( modSig, -59) #-53
util.welch(modSig, Fs=Fs)

#radar.hilbert_spectrum(np.real(modSig), Fs, label='NLFM')


SCD, f, alpha = radar.FAM(modSig, Fs = Fs, plot=False, method='conj', scale='linear')
radar.cyclicEstimator( SCD, f, alpha )

plt.show()


"""
TODO Generation
- Generate binary files

TODO Analyze
- Time domain multiplication of frequency fomain triangle window convolution. For IF and symbol rate estimation.
- Bandwidth estimation

- List of parameters to estimate on single chirp
- Skriv i rapport hvordan HH-transform foregår.
- Vurder Om kurvene skal karakteriseres fra sentrum, eller frs siden (i tid-frekvens).

TODO Report
- How chich parameters in FAM decides the estimation resolution?
"""