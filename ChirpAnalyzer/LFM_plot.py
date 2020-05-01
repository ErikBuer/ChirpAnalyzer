from scipy.signal import chirp, sweep_poly, spectrogram, welch
from scipy.special import factorial
import scipy.signal as signal
import matplotlib.pyplot as plt
plt.style.use('masterThesis')
import matplotlib as mpl

mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

imagePath = '../figures/LFM_example/'

import numpy as np
import rftool.radar as radar
import rftool.utility as util

Fs=np.intc(100e3)   # receiver sample rate
T=np.float(10e-3)    # Pulse duration
points = np.intc(Fs*T)      
t = np.linspace(0, T, points)

targetBw=6e3    # Pulse BW
centerFreq=4e3   # Pulse center frequency

# Generate linear chirp
LFM = radar.chirp(Fs)
window_t = np.array([1,1,1])
LFM.getCoefficients( window_t, targetBw=targetBw, centerFreq=centerFreq, T=T)
LFMsig = LFM.genNumerical()

fig, ax = plt.subplots()
fig.set_size_inches(7,1.75)
ax.plot(t, np.real(LFMsig))
#plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude [$\Re\{\cdot\}$]")
plt.savefig(imagePath + 'LFM_time-domain.png', bbox_inches='tight')
plt.savefig(imagePath + 'LFM_time-domain.pgf', bbox_inches='tight')

fig, ax = plt.subplots()
fig.set_size_inches(7,1.75)
ax.plot(t, LFM.targetOmega_t)
#plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Instantaneous Frequency [Hz]")
plt.savefig(imagePath + 'LFM_omega_time-domain.png', bbox_inches='tight')
plt.savefig(imagePath + 'LFM_omega_time-domain.pgf', bbox_inches='tight')

util.periodogram(LFMsig,Fs)
plt.savefig(imagePath + 'LFM_welch.png', bbox_inches='tight')
plt.savefig(imagePath + 'LFM_welch.pgf', bbox_inches='tight')

radar.ACF(LFMsig, label='LFM')
plt.savefig(imagePath + 'LFM_ACF.png', bbox_inches='tight')
plt.savefig(imagePath + 'LFM_ACF.pgf', bbox_inches='tight')

"""radar.hilbert_spectrum(np.real(LFMsig), Fs, label='NLFM')
plt.savefig('LFM_Hilbert.png', bbox_inches='tight')
plt.savefig('LFM_Hilbert.pgf', bbox_inches='tight')"""

plt.show()