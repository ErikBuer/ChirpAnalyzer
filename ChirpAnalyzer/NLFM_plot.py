from scipy.signal import chirp, sweep_poly, spectrogram, welch
from scipy.special import factorial
import scipy.signal as signal
import matplotlib.pyplot as plt
plt.style.use('masterThesis')
import matplotlib as mpl


debug = True

if debug == False:
    mpl.use("pgf")
    mpl.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

imagePath = '../figures/NLFM_example/'

import numpy as np
import rftool.radar as radar
import rftool.utility as util

Fs=np.intc(100e3)   # receiver sample rate
T=np.float(10e-3)    # Pulse duration
points = np.intc(Fs*T)      
t = np.linspace(0, T, points)

targetBw=6e3    # Pulse BW
centerFreq=4e3   # Pulse center frequency


# Generate chirp
NLFM = radar.chirp(Fs)
window_t = signal.hamming(np.intc(2048))
NLFM.getCoefficients( window_t, targetBw=targetBw, centerFreq=centerFreq, T=T)
NLFMsig = NLFM.genNumerical()


fig, ax = plt.subplots()
fig.set_size_inches(7,1.75)
ax.plot(window_t)
ax.set_xticklabels([])
plt.xlabel("Instantaneous Freuqncy")
plt.ylabel("Weighting")
plt.tight_layout()
if debug == False:
    plt.savefig(imagePath + 'NLFM_window.png', bbox_inches='tight')
    plt.savefig(imagePath + 'NLFM_window.pgf', bbox_inches='tight')

fig, ax = plt.subplots()
fig.set_size_inches(7,1.75)
ax.plot(t, np.real(NLFMsig))
plt.xlabel("Time [s]")
plt.ylabel("Amplitude [$\Re\{\cdot\}$]")
plt.tight_layout()
if debug == False:
    plt.savefig(imagePath + 'NLFM_time-domain.png', bbox_inches='tight')
    plt.savefig(imagePath + 'NLFM_time-domain.pgf', bbox_inches='tight')

fig, ax = plt.subplots()
fig.set_size_inches(7,1.75)
ax.plot(t, NLFM.targetOmega_t)
plt.xlabel("Time [s]")
plt.ylabel("Instantaneous Frequency [Hz]")
plt.tight_layout()
if debug == False:
    plt.savefig(imagePath + 'NLFM_omega_time-domain.png', bbox_inches='tight')
    plt.savefig(imagePath + 'NLFM_omega_time-domain.pgf', bbox_inches='tight')

util.periodogram(NLFMsig, Fs)
if debug == False:
    plt.savefig(imagePath + 'NLFM_welch.png', bbox_inches='tight')
    plt.savefig(imagePath + 'NLFM_welch.pgf', bbox_inches='tight')

radar.ACF(NLFMsig, label='NLFM')
if debug == False:
    plt.savefig(imagePath + 'NLFM_ACF.png', bbox_inches='tight')
    plt.savefig(imagePath + 'NLFM_ACF.pgf', bbox_inches='tight')

"""radar.hilbert_spectrum(np.real(NLFMsig), Fs, label='NLFM')
plt.savefig('NLFM_Hilbert.png', bbox_inches='tight')
plt.savefig('NLFM_Hilbert.pgf', bbox_inches='tight')"""

plt.show()