from scipy.signal import chirp, sweep_poly, spectrogram, welch
from scipy.special import factorial
import scipy.signal as signal
import matplotlib.pyplot as plt
plt.style.use('masterThesis')
import matplotlib as mpl
import numpy as np
import rftool.radar as radar
import rftool.LFM as LFM
import rftool.utility as util
import rftool.estimation as estimate
#import tftb as tftb                        # WVD
from pyhht.visualization import plot_imfs   # Hilbert-Huang TF analysis
from pyhht import EMD                       # Hilbert-Huang TF analysis
imagePath = '../figures/EMD/'

debug = False
if debug== False:
    mpl.use("pgf")
    mpl.rcParams.update({
        "pgf.texsystem": "lualatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })


Fs=np.intc(802e3) # receiver sample rate
dt = 1/Fs
T=np.float(6e-3)/4  # Pulse duration


# Time domain window for NLFM generation
NLFM = radar.chirp(Fs)
NLFM.fftLen = 2048

# Synthesize the target autocorrelation function
#window_t = signal.chebwin(np.intc(2048), 60)
window_t = signal.hamming(np.intc(2048))
#window_t = signal.gaussian(np.intc(2048), 360)
#window_t = signal.gaussian(np.intc(2048), 400)

NLFM.getCoefficients( window_t, targetBw=100e3/4, centerFreq=100e3/4, T=T)
sig_t = NLFM.genNumerical()
t = np.linspace(-T/2, (T/2)-dt, len(sig_t))
SNR = 10 # dB
SnrString = 'SNR_'+str(SNR)
sig_t = util.wgnSnr(sig_t, SNR)

########################################################################

# Generate linear chirp (simple)
T=1e-6
T=T
"""FM = LFM.chirp(Fs=Fs,T=T, fStart=40e3, fStop=60e3, nChirps=4, direction='up')
sig_t1 = np.real(FM.getSymbolSig(0))

FM = LFM.chirp(Fs=Fs,T=T, fStart=10e3, fStop=40e3, nChirps=4, direction='up')
sig_t2 = np.real(FM.getSymbolSig(1))

sig_t = sig_t1+sig_t2
t = np.linspace(-T/2, (T/2)-dt, len(sig_t))"""

decomposer = EMD(np.real(sig_t))
imfs = decomposer.decompose()
print('len(imfs)', len(imfs))


fig = plt.figure(figsize=(7, 5))
axs = fig.subplots(len(imfs)+1,1)
axs[0].plot(sig_t)
axs[0].set_yticklabels([])
axs[0].set_xticklabels([])
axs[0].set_ylabel('$s(t)$')

for i in range(1,len(imfs)+1):
    axs[i].plot(t, imfs[i-1])
    axs[i].set_ylabel('I '+str(i))
    axs[i].set_yticklabels([])
    if i<len(imfs):
        axs[i].set_xticklabels([])

axs[-1].set_ylabel('Res.')
axs[-1].set_xlabel('$t$ [s]')
axs[-1].ticklabel_format(useMathText=True, scilimits=(0,3), axis='x')

plt.tight_layout()

if debug == False:
    plt.savefig(imagePath+'EMD_'+SnrString+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'EMD_'+SnrString+'.pgf', bbox_inches='tight')
plt.show()