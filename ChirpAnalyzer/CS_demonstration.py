from scipy.signal import chirp, sweep_poly, spectrogram, welch
from scipy.special import factorial
import scipy.signal as signal
import matplotlib.pyplot as plt
plt.style.use('masterThesis')
import matplotlib as mpl

import scipy.optimize as optimize
import pickle

debug=True
imagePath = '../figures/cycloDemo/'

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
T=np.float(6e-3)  # Pulse duration
points = np.intc(Fs*T)
t = np.linspace(0, T, points)

targetBw=50e3      # Pulse BW
centerFreq=100e3    # Pulse center frequency

# Time domain window for NLFM generation
NLFM = radar.chirp(Fs)
NLFM.fftLen = 2048


# Synthesize the target autocorrelation function
#indow_t = signal.chebwin(np.intc(2048), 60)
window_t = signal.hamming(np.intc(2048))
#window_t = signal.gaussian(np.intc(2048), 360)
#window_t = signal.gaussian(np.intc(2048), 400)
NLFM.getCoefficients( window_t, targetBw=targetBw, centerFreq=centerFreq, T=T)
NLFMsig = NLFM.genFromPoly()

bitStream = np.random.randint(0, 2, 32)
modSig = NLFM.modulate(bitStream)
#modSig = util.wgnSnr( modSig, -40 )


# Plot autocorrelation of signal for report
Rxx = np.abs(signal.correlate(modSig, modSig, mode='full', method='fft'))
Rxx = Rxx[np.intc(len(Rxx)/2):]
f0 = estimate.f0MleTime(Rxx=Rxx, f=Fs, peaks=5)

dt = (1/Fs)
t = np.linspace(0, len(Rxx)*dt,len(Rxx))
fig, ax = plt.subplots()
ax.set_xmargin(0.01)
ax.plot(t, Rxx)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Correlation')
ax.ticklabel_format(useMathText=True, scilimits=(0,3))
plt.tight_layout()

if debug == False:
    fileName = 'Rxx_32_bit_sig'
    plt.savefig(imagePath + fileName + '.png', bbox_inches='tight')
    plt.savefig(imagePath + fileName + '.pgf', bbox_inches='tight')


# Spectral Correlation Density
SCD, f, alpha = estimate.FAM(modSig, Fs = Fs, plot=False, method='non-conj', scale='linear')
estimate.cyclicEstimator( SCD, f, alpha, bandLimited=False )

SCDplt = np.abs(SCD)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# Plot for positive frequencies
im = ax.pcolormesh(alpha, f[np.intc(len(f)/2):len(f)-np.intc(len(f)/4)], SCDplt[np.intc(len(f)/2):len(f)-np.intc(len(f)/4),:], edgecolors='none')
ax.ticklabel_format(useMathText=True, scilimits=(0,3))
#plt.title("Spectral Correlation Density")
ax.set_xlabel("alpha [Hz]")
ax.set_ylabel("$f$ [Hz]")
fig.colorbar(im)
plt.tight_layout()
 #! in Report
"""if debug==False:
    plt.savefig(imagePath+'SCD_FM_32'+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'SCD_FM_32'+'.pgf', bbox_inches='tight')"""

# plot the alpha distribution for f=f_c
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(alpha, np.abs(SCD[144,:]))
ax.ticklabel_format(useMathText=True, scilimits=(0,3))
ax.set_xlabel("alpha [Hz]")
ax.set_ylabel("Correlation")
plt.tight_layout()
if debug==False:
    plt.savefig(imagePath+'S_fc'+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'S_fc'+'.pgf', bbox_inches='tight')


# Fit a GMM
cyclicVec = np.abs(SCD[144,:])
cyclicVec[np.intc(len(cyclicVec)/2)] = 0

def func(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
    return y

# gaussian[\mu, \phi, \sigma]
Fsymb = 1/T
width = 3
numbOfGausses = 6 
#gaussian1 = np.array([1*Fsymb,1,1*width])
Gaussian = np.array([])
for i in range(-numbOfGausses,numbOfGausses+1):
    if i!=0:
        GmVecIt = np.array([i*Fsymb,np.abs(1/i),i*width])
        Gaussian = np.append(Gaussian, GmVecIt)
    """else:
        GmVecIt = np.array([i*Fsymb,np.abs(1/2),2*width])
        Gaussian = np.append(Gaussian, GmVecIt)"""

fit = func(alpha, *Gaussian)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
#ax.plot(alpha, cyclicVec, label='Cyclic Autorcorrelation')
ax.plot(alpha, fit, label='GMM PDF')
ax.set_xlabel('alpha [Hz]')
ax.set_ylabel('Weighting')
ax.ticklabel_format(useMathText=True, scilimits=(0,3))
ax.legend()
plt.tight_layout()

if debug==False:
    plt.savefig(imagePath+'Spectral_Autocorr_Density_GMM'+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'Spectral_Autocorr_Density_GMM'+'.pgf', bbox_inches='tight')

    # Write GMM to binary file
    path = '../jobs/'
    filename = 'SCD_GMM'
    destination = path + filename + '.pkl'
    # Save job to binary file
    with open(destination,'wb') as F:
        pickle.dump(fit, F)


# Plot cyclic correlation at alpha = 0 and multiples of f_symb
# Only plot for positive frequencies
zeroAplhaIndex = np.intc(len(alpha)/2)
len200kHz = np.intc(len(f)/4)
index200kHz = len(f)-len200kHz
zeroF = np.intc(len(f)/2)
fPos = f[zeroF:index200kHz]

# Single-sided cyclic plot
SCDsingle = np.add(np.abs(SCD[:,zeroAplhaIndex:]),np.abs(np.flip(SCD[:,:zeroAplhaIndex], axis=1)))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# alpha = 0
alpha0Index = 0#np.intc(len(alpha)/2)
ax.plot(fPos, np.abs(SCDsingle[zeroF:index200kHz:, alpha0Index]), label='alpha=0')
# alpha = Fsymbs
deltaAlpha = (alpha[-1]-alpha[0])/(len(alpha)-1)
FsymbAlpha = alpha0Index+np.intc(Fsymb/deltaAlpha)
ax.plot(fPos, np.abs(SCDsingle[zeroF:index200kHz:, FsymbAlpha]), label='alpha=|1/T|')
# alpha = 2Fsymb
twoFsymbAlpha = alpha0Index+2*np.intc(Fsymb/deltaAlpha)
ax.plot(fPos, np.abs(SCDsingle[zeroF:index200kHz:, twoFsymbAlpha]), label='alpha=|2/T|')
"""# alpha = 3Fsymb
threeFsymbAlpha = alpha0Index+3*np.intc(Fsymb/deltaAlpha)
ax.plot(fPos, np.abs(SCDsingle[zeroF:index200kHz:, threeFsymbAlpha]), label='alpha=|3/T|')"""
"""# alpha = 4Fsymb
fourFsymbAlpha = alpha0Index+4*np.intc(Fsymb/deltaAlpha)
ax.plot(fPos, np.abs(SCDsingle[zeroF:index200kHz:, fourFsymbAlpha]), label='alpha=|4/T|')"""

ax.ticklabel_format(useMathText=True, scilimits=(0,3))
ax.set_xlabel("$f$ [Hz]")
ax.set_ylabel("Correlation")
plt.tight_layout()
ax.legend()
if debug==False:
    plt.savefig(imagePath+'S_alpha'+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'S_alpha'+'.pgf', bbox_inches='tight')

#####################################################################################################
# Conjugate SCD
SCD, f, alpha = estimate.FAM(modSig, Fs = Fs, plot=False, method='conj', scale='linear')
estimate.cyclicEstimator( SCD, f, alpha, bandLimited=False )

SCDplt = np.abs(SCD)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# Plot for positive frequencies
im = ax.pcolormesh(alpha, f[np.intc(len(f)/2):len(f)-np.intc(len(f)/4)], SCDplt[np.intc(len(f)/2):len(f)-np.intc(len(f)/4),:], edgecolors='none')
ax.ticklabel_format(useMathText=True, scilimits=(0,3))
#plt.title("Spectral Correlation Density")
ax.set_xlabel("alpha [Hz]")
ax.set_ylabel("$f$ [Hz]")
fig.colorbar(im)
plt.tight_layout()

#! in Report
if debug==False:
    plt.savefig(imagePath+'SCD_CONJ_FM_32'+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'SCD_CONJ_FM_32'+'.pgf', bbox_inches='tight')

plt.show()