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

imagePath = '../figures/ModChirpPsd/'

import numpy as np
import rftool.radar as radar
import rftool.LFM as LFM
import rftool.utility as util

Fs=np.intc(802e3)   # receiver sample rate
T=np.float(6e-3)    # Pulse duration
points = np.intc(Fs*T)      
t = np.linspace(0, T, points)

targetBw=6e3    # Pulse BW
centerFreq=4e3   # Pulse center frequency

# Generate linear chirp
FM = LFM.chirp(Fs,6e-3,50e3,100e3,4, direction='up')

#! Study a single symbol
"""sig_t = FM.getSymbolSig(1)
util.welch(sig_t, Fs, nfft=np.intc(2**15))"""

# Periodogram of packet
bitstream = np.random.randint(0, 4, 32)
sig_t = FM.modulate( bitstream )

#util.periodogram(sig_t, Fs)
# Calculates Power Spectral Density in dBW/Hz.
fftLen = np.intc(2**15)
f, psd = signal.welch(sig_t, fs=Fs, nfft=fftLen, nperseg=fftLen,
noverlap = fftLen/4, return_onesided=False) # window = signal.blackmanharris(fftLen)

# Remove infinitesimally small components
# psd_dB = util.pow2db(np.maximum(psd, 1e-14))
psd_dB = util.pow2db(psd)
fig, ax = plt.subplots()
ax.plot(f, psd_dB)
ax.set_xlim(0,150e3)
ax.set_ylim(-85,-35)
ax.set_title("Welch's PSD Estimate")
ax.set_ylabel("dBW/Hz")
ax.set_xlabel("$f$ [Hz]")
fig.set_size_inches(7,2)
plt.tight_layout()

if debug == False:
    plt.savefig(imagePath + 'LFM_modulated_PSD.png', bbox_inches='tight')
    plt.savefig(imagePath + 'LFM_modulated_PSD.pgf', bbox_inches='tight')

plt.show()