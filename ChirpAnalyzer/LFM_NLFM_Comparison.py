from scipy.signal import chirp, sweep_poly, spectrogram, welch
from scipy.special import factorial
import scipy.signal as signal
import matplotlib.pyplot as plt
plt.style.use('masterThesis')
import matplotlib as mpl

debug=True
imagePath = '../figures/LfmNlfmComparison/'

if debug==False:
    mpl.use("pgf")
    mpl.rcParams.update({
        "pgf.texsystem": "lualatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    
import numpy as np
import rftool.radar as radar
import rftool.utility as util
import rftool.estimation as estimate

Fs=np.intc(802e3)   # receiver sample rate
T=np.float(6e-3)    # Pulse duration
points = np.intc(Fs*T)
t = np.linspace(0, T, points)

targetBw=50e3      # Pulse BW
centerFreq=75e3    # Pulse center frequency

# Time domain window for NLFM generation
NLFM = radar.chirp(Fs)
NLFM.fftLen = 2048


# Synthesize the target autocorrelation function
#indow_t = signal.chebwin(np.intc(2048), 60)
window_t = signal.hamming(np.intc(2048))
#window_t = signal.gaussian(np.intc(2048), 360)
#window_t = signal.gaussian(np.intc(2048), 400)
NLFM.getCoefficients( window_t, targetBw=targetBw, centerFreq=centerFreq, T=T)

# Time domain window for LFM generation
LFMsig = signal.chirp(t, centerFreq-(targetBw/2), t[-1], centerFreq+(targetBw/2), method='linear')
"""plt.figure()
plt.plot(t, LFMsig)"""

NLFMsig = NLFM.genFromPoly()
signals = np.stack((LFMsig, NLFMsig), axis=-1)

radar.ACF(signals, Fs=Fs, label=['LFM', 'NLFM'])
if debug==False:
    plt.savefig(imagePath+'ACF_compare'+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'ACF_compare'+'.pgf', bbox_inches='tight')


# Cross correlation of symbols
sig_t = NLFM.modulate( np.array([0]) )
sig_t2 = NLFM.modulate( np.array([1]) )

Rxy = signal.correlate(sig_t, sig_t2, method='fft')
Rxx = signal.correlate(sig_t, sig_t, method='fft')
tau = np.linspace(-np.floor(len(Rxy)/2)/Fs, np.floor(len(Rxy)/2)/Fs, len(Rxy))
RxyNorm_dB = util.pow2db(np.divide(np.abs(Rxy), np.max(np.abs(Rxx))))

fig, ax = plt.subplots()
ax.plot(tau, RxyNorm_dB)
ax.grid()
ax.set_xlabel('$t$ [Hz]')
ax.set_ylabel('Normalized Correlation [dB]')
plt.tight_layout()

if debug == False:
    plt.savefig(imagePath + 'NLFM_Symbol_Xcorr.png', bbox_inches='tight')
    plt.savefig(imagePath + 'NLFM_Symbol_Xcorr.pgf', bbox_inches='tight')



plt.show()