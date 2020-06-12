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
imagePath = '../figures/IFestimation/'

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

"""#! Excluded
package = np.random.randint(0, 2, 2)
modSig = NLFM.modulate( package )
modSig = util.wgnSnr(modSig, 20)
"""

########################################################################
"""#! In Report
DerivIF = estimate.instFreq(sig_t, Fs)
Deriv_AE = np.abs(np.subtract(NLFM.targetOmega_t, DerivIF))

plt.figure()
plt.subplot(211)
plt.plot(DerivIF, label='Derivative Method')
plt.ylim(0,300000)
plt.ylabel("$f$ [Hz]")
plt.title('Estimated IF')
plt.tight_layout()
#plt.legend()

ax = plt.subplot(212)
ax.plot(Deriv_AE, label='Absolute Error')
ax.set_ylabel("Error [Hz]")
ax.set_title('Absolute Error')
plt.tight_layout()
#plt.legend()
if debug == False:
    plt.savefig(imagePath+'DerivIf_'+SnrString+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'DerivIf_'+SnrString+'.pgf', bbox_inches='tight')
"""
########################################################################
"""#! In Report
IFBarnesTwo = estimate.instFreq(sig_t, Fs, method='BarnesTwo')
IFBarnesTwo_AE = np.abs(np.subtract(NLFM.targetOmega_t, IFBarnesTwo))

plt.figure()
plt.subplot(211)
plt.plot(IFBarnesTwo, label='Barnes Two-Point FIR')
plt.ylim(0,300000)
plt.ylabel("$f$ [Hz]")
plt.title('Estimated IF')
plt.tight_layout()
#plt.legend()

plt.subplot(212)
plt.plot(IFBarnesTwo_AE, label='Absolute Error')
plt.ylabel("Error [Hz]")
plt.title('Absolute Error')
plt.tight_layout()
#plt.legend()

plt.savefig(imagePath+'BarnesTwoIF_'+SnrString+'.png', bbox_inches='tight')
plt.savefig(imagePath+'BarnesTwoIF_'+SnrString+'.pgf', bbox_inches='tight')
"""

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

if debug == False:
    plt.savefig(imagePath+'IFmaxDHHT_'+SnrString+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'IFmaxDHHT_'+SnrString+'.pgf', bbox_inches='tight')
    plt.savefig(imagePath+'Hilbert_'+SnrString+'.pgf', bbox_inches='tight')
"""

"""
HH = estimate.HilberHuang(np.real(sig_t), Fs)
fig = HH.discreteSpectrum( frequencyBins=128, decimateTime=13, filterSigma=1 )
plt.title=('')
plt.savefig(imagePath+'Hilbert_'+SnrString+'.png', bbox_inches='tight')
"""
########################################################################
#! In report
"""windowsize=50
IFpolyMLE = estimate.instFreq(sig_t, Fs, method='polyMle', windowSize=windowsize, order=2)
IFpolyMLE_AE = np.abs(np.subtract(NLFM.targetOmega_t, IFpolyMLE))

time = np.linspace(-T/2,(T/2)-dt, len(IFpolyMLE))

fig = plt.figure(figsize=(7, 2.5))
ax = fig.add_subplot(211)
ax.plot(time, IFpolyMLE, label='Piece-wise Polynomial MLE')
ax.plot(time, NLFM.targetOmega_t, label='True IF')
ax.set_ylim(0,200000)
ax.set_ylabel('$f$ [Hz]')
ax.ticklabel_format(useMathText=True, scilimits=(0,3))
ax.set_title('Estimated IF')
plt.legend(loc='upper right')

ax = fig.add_subplot(212)
ax.plot(time, IFpolyMLE_AE)
ax.set_ylabel('Error [Hz]')
ax.set_xlabel('$t$ [s]')
ax.set_title('Absolute Error')
#ax.set_ylim(10,10010)
ax.ticklabel_format(useMathText=True, scilimits=(0,3))
plt.tight_layout()

if debug == False:
    plt.savefig(imagePath+'polyMleIF_'+SnrString+'_windowsize_'+str(windowsize)+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'polyMleIF_'+SnrString+'_windowsize_'+str(windowsize)+'.pgf', bbox_inches='tight')"""

########################################################################
""" #! Excluded
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
plt.ylabel('$f$ [Hz]')
plt.title('Absolute Error')
plt.tight_layout()
#plt.legend()

if pgf == True:
    plt.savefig(imagePath+'polyLSIF_'+SnrString+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'polyLSIF_'+SnrString+'.pgf', bbox_inches='tight')
"""#! Excluded
########################################################################
# WVD MLE for NLFM
""" #! In Report
IFmaxWVT = estimate.instFreq(sig_t, Fs, method='maxWVD')
IFmaxWVT_AE = np.abs(np.subtract(NLFM.targetOmega_t, IFmaxWVT))

plt.figure()
plt.subplot(211)
plt.plot(IFmaxWVT, label='WVD MLE')
plt.ylim(0,200000)
plt.ylabel('$f$ [Hz]')
plt.title('Estimated IF')
plt.tight_layout()
#plt.legend()

plt.subplot(212)
plt.plot(IFmaxWVT_AE)
plt.ylabel('Error [Hz]')
plt.title('Absolute Error')
plt.ylim(0,10000)
plt.tight_layout()
#plt.legend()

if debug == False:
    plt.savefig(imagePath+'IFmaxWVT_'+SnrString+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'IFmaxWVT_'+SnrString+'.pgf', bbox_inches='tight')
"""
########################################################################
# WVD NLFM Chirp
"""#! In report
tfr = tftb.processing.WignerVilleDistribution(sig_t)
timeFreqMat, t, f = tfr.run()
#tfr.plot(kind='contour')
f_t = Fs*f[np.argmax(timeFreqMat,0)]
f = f*Fs

timeFreqMat = signal.decimate(timeFreqMat, q=13, axis=0)
f = np.linspace(0, Fs/2, timeFreqMat.shape[0])
timeFreqMat = signal.decimate(timeFreqMat, q=13, axis=1)
t = np.linspace(0, len(sig_t)/Fs, timeFreqMat.shape[1])

fig = plt.figure()
ax = fig.add_subplot(111)
cset = ax.pcolormesh(t, f, timeFreqMat)
plt.colorbar(cset, ax=ax)

#ax.set_title("Wigner-Ville Distribution")
ax.set_xlabel('$t$ [s]')
ax.set_ylabel('$f$ [Hz]')
ax.set_ylim(0,200e3)#np.divide(Fs,2))
ax.set_xlim(t[1], t[-1])
ax.ticklabel_format(useMathText=True, scilimits=(0,3))
plt.tight_layout()

if debug == False:
    plt.savefig(imagePath+'WVD_'+SnrString+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'WVD_'+SnrString+'.pgf', bbox_inches='tight')
"""
########################################################################
# WVD Non-Continious LFM Chirp (X-products)
#! In report
"""
# Generate linear chirp
FM = LFM.chirp(Fs=Fs,T=T/4, fStart=20e3, fStop=80e3, nChirps=4, direction='up')
sig_t = FM.getSymbolSig(1)

tfr = tftb.processing.WignerVilleDistribution(sig_t)
timeFreqMat, t, f = tfr.run()
#tfr.plot(kind='contour')
f_t = Fs*f[np.argmax(timeFreqMat,0)]
f = f*Fs

timeFreqMat = signal.decimate(timeFreqMat, q=2, axis=0)
f = np.linspace(0, Fs/2, timeFreqMat.shape[0])
timeFreqMat = signal.decimate(timeFreqMat, q=4, axis=1)
t = np.linspace(0, len(sig_t)/Fs, timeFreqMat.shape[1])

fig, ax = plt.subplots()
cset = ax.pcolormesh(t, f, timeFreqMat)
#cset = ax.pcolormesh(t, f, util.pow2db(np.abs(timeFreqMat)))
plt.colorbar(cset, ax=ax)

#ax.set_title("Wigner-Ville Distribution")
ax.set_xlabel('$t$ [s]')
ax.set_ylabel('$f$ [Hz]')
ax.set_ylim(0,100e3)#np.divide(Fs,2))
ax.set_xlim(t[1], t[-1])
ax.ticklabel_format(useMathText=True, scilimits=(0,3))
plt.tight_layout()

if debug == False:
    plt.savefig(imagePath+'WVD_LFM_MOD'+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'WVD_LFM_MOD'+'.pgf', bbox_inches='tight')"""

########################################################################
# WVD MLE for troublesome LFM Symbol
 #! Don't comment out this!
# Generate linear chirp
FM = LFM.chirp(Fs=Fs,T=T, fStart=50e3, fStop=150e3, nChirps=4, direction='up')
symbol=1
sig_t = FM.getSymbolSig(symbol)
sig_t = util.wgnSnr(sig_t, SNR)

"""#! In Report
IFmaxWVT = estimate.instFreq(sig_t, Fs, method='maxWVD')
IFmaxWVT_AE = np.abs(np.subtract(FM.getSymbolIF(symbol), IFmaxWVT))

time = np.linspace(-T/2,(T/2)-dt, len(IFmaxWVT))

fig = plt.figure(figsize=(7, 2.5))
ax = fig.add_subplot(211)
ax.plot(time, IFmaxWVT, label='WVD MLE')
ax.plot(time, FM.getSymbolIF(symbol), label='True IF')
ax.set_ylim(0,200000)
ax.set_ylabel('$f$ [Hz]')
ax.ticklabel_format(useMathText=True, scilimits=(0,3))
ax.set_title('Estimated IF')
plt.legend()

ax = fig.add_subplot(212)
ax.plot(time, IFmaxWVT_AE)
ax.set_ylabel('Error [Hz]')
ax.set_xlabel('$t$ [s]')
ax.set_title('Absolute Error')
#ax.set_ylim(10,10010)
ax.ticklabel_format(useMathText=True, scilimits=(0,3))
plt.tight_layout()
#plt.legend()

if debug == False:
    plt.savefig(imagePath+'IFmaxWVT_MOD_LFM_'+SnrString+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'IFmaxWVT_MOD_LFM_'+SnrString+'.pgf', bbox_inches='tight')
"""
########################################################################
# HHT MLE for troublesome LFM Symbol
#! In Report
"""IFmaxHHT = estimate.instFreq(sig_t, Fs, method='maxDHHT')
IFmaxHHT_AE = np.abs(np.subtract(FM.getSymbolIF(symbol), IFmaxHHT))

time = np.linspace(-T/2,(T/2)-dt, len(IFmaxHHT))

fig = plt.figure(figsize=(7, 2.5))
ax = fig.add_subplot(211)
ax.plot(time, IFmaxHHT, label='HHT MLE')
ax.plot(time, FM.getSymbolIF(symbol), label='True IF')
ax.set_ylim(0,200000)
ax.set_ylabel('$f$ [Hz]')
ax.ticklabel_format(useMathText=True, scilimits=(0,3))
ax.set_title('Estimated IF')
plt.legend(loc='upper right')

ax = fig.add_subplot(212)
ax.plot(time, IFmaxHHT_AE)
ax.set_ylabel('Error [Hz]')
ax.set_xlabel('$t$ [s]')
ax.set_title('Absolute Error')
#ax.set_ylim(10,10010)
ax.ticklabel_format(useMathText=True, scilimits=(0,3))
plt.tight_layout()
#plt.legend()

if debug == False:
    plt.savefig(imagePath+'IFmaxHHT_MOD_LFM_'+SnrString+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'IFmaxHHT_MOD_LFM_'+SnrString+'.pgf', bbox_inches='tight')"""
########################################################################
# Poly MLE for troublesome LFM Symbol
#! In Report
"""windowsize=50
IFpolyMle = estimate.instFreq(sig_t, Fs, method='polyMle', windowSize=windowsize, order=2)
IFpolyMle_AE = np.abs(np.subtract(FM.getSymbolIF(symbol), IFpolyMle))

time = np.linspace(-T/2,(T/2)-dt, len(IFpolyMle))

fig = plt.figure(figsize=(7, 2.5))
ax = fig.add_subplot(211)
ax.plot(time, IFpolyMle, label='Piece-wise Polynomial MLE')
ax.plot(time, FM.getSymbolIF(symbol), label='True IF')
ax.set_ylim(0,200000)
ax.set_ylabel('$f$ [Hz]')
ax.ticklabel_format(useMathText=True, scilimits=(0,3))
ax.set_title('Estimated IF')
plt.legend(loc='upper right')

ax = fig.add_subplot(212)
ax.plot(time, IFpolyMle_AE)
ax.set_ylabel('Error [Hz]')
ax.set_xlabel('$t$ [s]')
ax.set_title('Absolute Error')
#ax.set_ylim(10,10010)
ax.ticklabel_format(useMathText=True, scilimits=(0,3))
plt.tight_layout()
#plt.legend()

if debug == False:
    plt.savefig(imagePath+'IFpolyMle_MOD_LFM_'+SnrString+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'IFpolyMle_MOD_LFM_'+SnrString+'.pgf', bbox_inches='tight')"""
########################################################################
# Derivative method for troublesome LFM Symbol
#! In Report
"""IF = estimate.instFreq(sig_t, Fs)# , method='maxDHHT')
IF_AE = np.abs(np.subtract(FM.getSymbolIF(symbol), IF))

time = np.linspace(-T/2,(T/2)-dt, len(IF))

fig = plt.figure(figsize=(7, 2.5))
ax = fig.add_subplot(211)
ax.plot(time, IF, label='Derivative Method')
ax.plot(time, FM.getSymbolIF(symbol), label='True IF')
ax.set_ylim(0,200000)
ax.set_ylabel('$f$ [Hz]')
ax.ticklabel_format(useMathText=True, scilimits=(0,3))
ax.set_title('Estimated IF')
plt.legend(loc='upper right')

ax = fig.add_subplot(212)
ax.plot(time, IF_AE)
ax.set_ylabel('Error [Hz]')
ax.set_xlabel('$t$ [s]')
ax.set_title('Absolute Error')
#ax.set_ylim(10,10010)
ax.ticklabel_format(useMathText=True, scilimits=(0,3))
plt.tight_layout()
#plt.legend()

if debug == False:
    plt.savefig(imagePath+'IFDerivative_MOD_LFM_'+SnrString+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'IFDerivative_MOD_LFM_'+SnrString+'.pgf', bbox_inches='tight')"""
########################################################################
# Derivative method for troublesome LFM Symbol
#! In Report
"""IF = estimate.instFreq(sig_t, Fs, method='BarnesTwo')
IF_AE = np.abs(np.subtract(FM.getSymbolIF(symbol), IF))

time = np.linspace(-T/2,(T/2)-dt, len(IF))

fig = plt.figure(figsize=(7, 2.5))
ax = fig.add_subplot(211)
ax.plot(time, IF, label='Barnes Method')
ax.plot(time, FM.getSymbolIF(symbol), label='True IF')
ax.set_ylim(0,200000)
ax.set_ylabel('$f$ [Hz]')
ax.ticklabel_format(useMathText=True, scilimits=(0,3))
ax.set_title('Estimated IF')
plt.legend(loc='upper right')

ax = fig.add_subplot(212)
ax.plot(time, IF_AE)
ax.set_ylabel('Error [Hz]')
ax.set_xlabel('$t$ [s]')
ax.set_title('Absolute Error')
#ax.set_ylim(10,10010)
ax.ticklabel_format(useMathText=True, scilimits=(0,3))
plt.tight_layout()
#plt.legend()

if debug == False:
    plt.savefig(imagePath+'IFBarness_MOD_LFM_'+SnrString+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'IFBarness_MOD_LFM_'+SnrString+'.pgf', bbox_inches='tight')"""
########################################################################
#TODO
"""IFmaxDHHT = estimate.instFreq(sig_t, Fs, method='maxDHHT')
IFmaxDHHT_AE = np.abs(np.subtract(NLFM.targetOmega_t, IFmaxDHHT))

fig = plt.figure(figsize=(7, 2.5))
ax = fig.add_subplot(211)
ax.plot(time, IFmaxDHHT, label='')
ax.plot(time, FM.getSymbolIF(symbol), label='True IF')
plt.ylim(0,300000)
plt.ylabel("$f$ [Hz]")
plt.title('Estimated IF')
plt.tight_layout()
#plt.legend()

ax = fig.add_subplot(212)
ax.plot(IFmaxDHHT_AE, label='Absolute Error')
ax.ylabel("Error [Hz]")
ax.title('Absolute Error')
ax.tight_layout()
#plt.legend()

if debug== False:
    plt.savefig(imagePath+'IFmaxDHHT_'+SnrString+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'IFmaxDHHT_'+SnrString+'.pgf', bbox_inches='tight')
"""
########################################################################
"""#! In report
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
    plt.savefig(imagePath+'Hilbert_LFM_MOD_SIG'+'.pgf', bbox_inches='tight')"""

########################################################################
# Unwrap phase
#! In report
# Generate linear chirp (simple)
FM = LFM.chirp(Fs=Fs,T=T/4, fStart=20e3, fStop=80e3, nChirps=4, direction='up')
sig_t = FM.getSymbolSig(1)
t = np.linspace(0, len(sig_t)/Fs, len(sig_t))

phi_t = np.angle(sig_t)
phi_tU = np.unwrap(np.angle(sig_t))

fig = plt.figure(figsize=(7, 2.5))
ax = fig.add_subplot(211)
ax.plot(t, phi_t, label='$\Phi(t)$, Wrapped Phase')
ax.set_ylabel('Angle [rad]')
ax.set_xticklabels([])
ax.legend(loc='lower right')

ax = fig.add_subplot(212)
ax.plot(t, phi_tU, label='$\Phi(t)$, Unwrapped Phase')
ax.set_ylabel('Angle [rad]')
ax.set_xlabel('$t$ [s]')
ax.ticklabel_format(useMathText=True, scilimits=(0,3))
ax.legend(loc='lower right')
plt.tight_layout()

if debug== False:
    plt.savefig(imagePath+'unWrapPhase_LFM_'+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'unWrapPhase_LFM_'+'.pgf', bbox_inches='tight')

plt.show()