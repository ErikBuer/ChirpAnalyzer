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
import tftb as tftb
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
sig_t = NLFM.genNumerical()
SNR = 10 # dB
SnrString = 'SNR_'+str(SNR)

sig_t = util.wgnSnr(sig_t, SNR)

########################################################################
""" # TODO
IFmaxDHHT = estimate.instFreq(sig_t, Fs, method='maxDHHT')
IFmaxDHHT_AE = np.abs(np.subtract(NLFM.targetOmega_t, IFmaxDHHT))

plt.figure()
plt.subplot(211)
plt.plot(IFmaxDHHT, label='')
plt.ylim(0,300000)
plt.ylabel("$f$ [Hz]")
plt.title('Estimated IF')
plt.tight_layout()
#plt.legend()

plt.subplot(212)
plt.plot(IFmaxDHHT_AE, label='Absolute Error')
plt.ylabel("Error [Hz]")
plt.title('Absolute Error')
plt.tight_layout()
#plt.legend()

if debug== False:
    plt.savefig(imagePath+'IFmaxDHHT_'+SnrString+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'IFmaxDHHT_'+SnrString+'.pgf', bbox_inches='tight')
    plt.savefig(imagePath+'Hilbert_'+SnrString+'.pgf', bbox_inches='tight')
"""

"""# Generate linear chirp
FM = LFM.chirp(Fs=Fs,T=T, fStart=50e3, fStop=150e3, nChirps=4, direction='up')
symbol=1
sig_t = FM.getSymbolSig(symbol)"""

# Generate linear chirp (simple)
FM = LFM.chirp(Fs=Fs,T=T/4, fStart=20e3, fStop=80e3, nChirps=4, direction='up')
sig_t = FM.getSymbolSig(1)

HH = estimate.HilberHuang(np.real(sig_t), Fs)
fig, ax = HH.discreteSpectrum( frequencyBins=200, decimateTime=4, filterSigma=1 )
ax.set_ylim(0, 100e3)
#fig, ax = HH.spectrum()
plt.title=('')

if debug == False:
    plt.savefig(imagePath+'Hilbert_LFM_MOD_SIG'+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'Hilbert_LFM_MOD_SIG'+'.pgf', bbox_inches='tight')
plt.show()