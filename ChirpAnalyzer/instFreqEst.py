from scipy.signal import chirp, sweep_poly, spectrogram, welch
from scipy.special import factorial
import scipy.signal as signal
import matplotlib.pyplot as plt
plt.style.use('masterThesis')
import matplotlib as mpl
import numpy as np
import rftool.radar as radar
import rftool.utility as util
import rftool.estimation as estimate
import tftb as tftb
imagePath = '../figures/IFestimation'

pgf= True
if pgf== True:
    mpl.use("pgf")
    mpl.rcParams.update({
        "pgf.texsystem": "lualatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })


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
sig_t = NLFM.genNumerical()
SNR = 10 # dB
SnrString = 'SNR_'+str(SNR)

sig_t = util.wgnSnr(sig_t, SNR)

"""
package = np.random.randint(0, 2, 2)
modSig = NLFM.modulate( package )
modSig = util.wgnSnr(modSig, 20)
"""


########################################################################

DerivIF = estimate.instFreq(sig_t, Fs)
Deriv_AE = np.abs(np.subtract(NLFM.targetOmega_t, DerivIF))

plt.figure()
plt.subplot(211)
plt.plot(DerivIF, label='Derivative Method')
plt.ylim(0,300000)
plt.ylabel("f [Hz]")
plt.title('Estimated IF')
plt.tight_layout()
#plt.legend()

ax = plt.subplot(212)
ax.plot(Deriv_AE, label='Absolute Error')
ax.set_ylabel("f [Hz]")
ax.set_title('Absolute Error')
plt.tight_layout()
#plt.legend()
if pgf== True:
    plt.savefig(imagePath+'DerivIf_'+SnrString+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'DerivIf_'+SnrString+'.pgf', bbox_inches='tight')


########################################################################

IFBarnesTwo = estimate.instFreq(sig_t, Fs, method='BarnesTwo')
IFBarnesTwo_AE = np.abs(np.subtract(NLFM.targetOmega_t, IFBarnesTwo))

plt.figure()
plt.subplot(211)
plt.plot(IFBarnesTwo, label='Barnes Two-Point FIR')
plt.ylim(0,300000)
plt.ylabel("f [Hz]")
plt.title('Estimated IF')
plt.tight_layout()
#plt.legend()

plt.subplot(212)
plt.plot(IFBarnesTwo_AE, label='Absolute Error')
plt.ylabel("f [Hz]")
plt.title('Absolute Error')
plt.tight_layout()
#plt.legend()

plt.savefig(imagePath+'BarnesTwoIF_'+SnrString+'.png', bbox_inches='tight')
plt.savefig(imagePath+'BarnesTwoIF_'+SnrString+'.pgf', bbox_inches='tight')
"""

########################################################################
"""
IFmaxDHHT = estimate.instFreq(sig_t, Fs, method='maxDHHT')
IFmaxDHHT_AE = np.abs(np.subtract(NLFM.targetOmega_t, IFmaxDHHT))

plt.figure()
plt.subplot(211)
plt.plot(IFmaxDHHT, label='')
plt.ylim(0,300000)
plt.ylabel("f [Hz]")
plt.title('Estimated IF')
plt.tight_layout()
#plt.legend()

plt.subplot(212)
plt.plot(IFmaxDHHT_AE, label='Absolute Error')
plt.ylabel("f [Hz]")
plt.title('Absolute Error')
plt.tight_layout()
#plt.legend()

if pgf== True:
    plt.savefig(imagePath+'IFmaxDHHT_'+SnrString+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'IFmaxDHHT_'+SnrString+'.pgf', bbox_inches='tight')
    plt.savefig(imagePath+'Hilbert_'+SnrString+'.pgf', bbox_inches='tight')


"""HH = estimate.HilberHuang(np.real(sig_t), Fs)
fig = HH.discreteSpectrum( frequencyBins=128, decimateTime=13, filterSigma=1 )
plt.title=('')
plt.savefig(imagePath+'Hilbert_'+SnrString+'.png', bbox_inches='tight')
"""
########################################################################

windowsize=50
instFreqPolyMle = estimate.instFreq(sig_t, Fs, method='polyMle', windowSize=windowsize, order=2)
polyML_AE = np.abs(np.subtract(NLFM.targetOmega_t, instFreqPolyMle))

plt.figure()
plt.subplot(211)
plt.plot(instFreqPolyMle, label='Piecewise Poly MLE')
plt.ylim(0,200000)
plt.ylabel('f [Hz]')
plt.title('Estimated IF')
plt.tight_layout()
#plt.legend()

plt.subplot(212)
plt.plot(polyML_AE)
plt.ylabel('f [Hz]')
plt.title('Absolute Error')
plt.tight_layout()
#plt.legend()

if pgf== True:
    plt.savefig(imagePath+'polyMleIF_'+SnrString+'_windowsize_'+str(windowsize)+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'polyMleIF_'+SnrString+'_windowsize_'+str(windowsize)+'.pgf', bbox_inches='tight')

########################################################################
"""
instFreqPolyLS = estimate.instFreq(sig_t, Fs, method='polyLeastSquares', order=24)
polyLS_AE = np.abs(np.subtract(NLFM.targetOmega_t, instFreqPolyLS))

plt.figure()
plt.subplot(211)
plt.plot(instFreqPolyLS, label='Poly Least Squares')
plt.ylim(0,200000)
plt.ylabel('f [Hz]')
plt.title('Estimated IF')
plt.tight_layout()
#plt.legend()

plt.subplot(212)
plt.plot(polyLS_AE)
plt.ylabel('f [Hz]')
plt.title('Absolute Error')
plt.tight_layout()
#plt.legend()

if pgf== True:
    plt.savefig(imagePath+'polyLSIF_'+SnrString+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'polyLSIF_'+SnrString+'.pgf', bbox_inches='tight')
"""
########################################################################

IFmaxWVT = estimate.instFreq(sig_t, Fs, method='maxWVD')
IFmaxWVT_AE = np.abs(np.subtract(NLFM.targetOmega_t, IFmaxWVT))

plt.figure()
plt.subplot(211)
plt.plot(IFmaxWVT, label='WVD MLE')
plt.ylim(0,200000)
plt.ylabel('f [Hz]')
plt.title('Estimated IF')
plt.tight_layout()
#plt.legend()

plt.subplot(212)
plt.plot(IFmaxWVT_AE)
plt.ylabel('f [Hz]')
plt.title('Absolute Error')
plt.ylim(0,10000)
plt.tight_layout()
#plt.legend()

if pgf== True:
    plt.savefig(imagePath+'IFmaxWVT_'+SnrString+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'IFmaxWVT_'+SnrString+'.pgf', bbox_inches='tight')

########################################################################
"""
tfr = tftb.processing.WignerVilleDistribution(sig_t)
timeFreqMat, t, f = tfr.run()
#tfr.plot(kind='contour')
f_t = Fs*f[np.argmax(timeFreqMat,0)]
f = f*Fs

timeFreqMat = signal.decimate(timeFreqMat, q=13, axis=0)
f = np.linspace(0, Fs/2, timeFreqMat.shape[0])
timeFreqMat = signal.decimate(timeFreqMat, q=12, axis=1)
t = np.linspace(0, len(sig_t)/Fs, timeFreqMat.shape[1])

fig = plt.figure()
ax = fig.add_subplot(111)
cset = ax.pcolormesh(t, f, timeFreqMat)
plt.colorbar(cset, ax=ax)

#ax.set_title("Wigner-Ville Distribution")
ax.set_xlabel('t [s]')
ax.set_ylabel('f [Hz]')
ax.set_ylim(0,np.divide(Fs,2))
ax.set_xlim(t[1], t[-1])
plt.tight_layout()

if pgf== True:
    plt.savefig(imagePath+'WVD_'+SnrString+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'WVD_'+SnrString+'.pgf', bbox_inches='tight')
"""
########################################################################

#plt.show()