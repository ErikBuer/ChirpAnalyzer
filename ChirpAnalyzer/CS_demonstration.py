from scipy.signal import chirp, sweep_poly, spectrogram, welch
from scipy.special import factorial
import scipy.signal as signal
import matplotlib.pyplot as plt
plt.style.use('masterThesis')
import matplotlib as mpl

import scipy.optimize as optimize

import pickle

pgf=True
imagePath = '../figures/cycloDemo/'

if pgf==True:
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

Fs=np.intc(800e3)   # receiver sample rate
T=np.float(6e-3)  # Pulse duration
points = np.intc(Fs*T)
t = np.linspace(0, T, points)

targetBw=100e3      # Pulse BW
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

SCD, f, alpha = estimate.FAM(modSig, Fs = Fs, plot=False, method='conj', scale='linear')
estimate.cyclicEstimator( SCD, f, alpha, bandLimited=True )

SCDplt = np.abs(SCD)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# Plot for positive frequencies
im = ax.pcolormesh(alpha, f[np.intc(len(f)/2):-1], SCDplt[np.intc(len(f)/2):-1,:], edgecolors='none')
ax.ticklabel_format(useMathText=True, scilimits=(0,3))
#plt.title("Spectral Correlation Density")
ax.set_xlabel("alpha [Hz]")
ax.set_ylabel("f [Hz]")
fig.colorbar(im)
plt.tight_layout()
""" #! in Report
if pgf==True:
    plt.savefig(imagePath+'SCD_NLFM_32'+'_pgf_'+str(pgf)+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'SCD_NLFM_32'+'_pgf_'+str(pgf)+'.pgf', bbox_inches='tight')
"""
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(alpha, np.abs(SCD[144,:]))
ax.ticklabel_format(useMathText=True, scilimits=(0,3))
ax.set_xlabel("alpha [Hz]")
ax.set_ylabel("Correlation")
if pgf==True:
    plt.savefig(imagePath+'S_fc'+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'S_fc'+'.pgf', bbox_inches='tight')

"""
mean, var, skew, kurt = norm.stats(moments='mvsk')
x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, norm.pdf(x), label='norm pdf')
"""


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
width = 5
#gaussian1 = np.array([1*Fsymb,1,1*width])
Gaussian = np.array([])
for i in range(-6,6+1):
    if i!=0:
        GmVecIt = np.array([i*Fsymb,np.abs(1/i),i*width])
        Gaussian = np.append(Gaussian, GmVecIt)

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

if pgf==True:
    plt.savefig(imagePath+'Spectral_Autocorr_Density_GMM'+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'Spectral_Autocorr_Density_GMM'+'.pgf', bbox_inches='tight')

# Write GMM to binary file
path = '../jobs/'
filename = 'SCD_GMM'
destination = path + filename + '.pkl'
# Save job to binary file
with open(destination,'wb') as f:
    pickle.dump(fit, f)

plt.show()