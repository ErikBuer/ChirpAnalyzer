from scipy.signal import chirp, sweep_poly, spectrogram, welch
from scipy.special import factorial
import scipy.signal as signal
import numpy as np
import random

import rftool.radar as radar
import rftool.utility as util
import rftool.estimation as estimate
import rftool.communications as comm
from utility import *

import matplotlib.pyplot as plt
import matplotlib as mpl

import pickle
#import os

Debug = False

Fs = np.intc(802e3) # Receiver sample rate. #! Must be the same as the signals
T = np.float(6e-3)  # Pulse duration.       #! Must be the same as the signals
nIterations = 500
packetSize = 32

# Load alpha window function a-priori
path = '../jobs/'
filename = 'SCD_GMM'
destination = path + filename + '.pkl'
with open(destination,'rb') as f:
    alphaWindow = pickle.load(f)

# Plot results
import matplotlib.pyplot as plt
plt.style.use('masterThesis')
import matplotlib
imagePath = '../figures/symRateEst/'

if Debug==False:
    mpl.use("pgf")
    mpl.rcParams.update({
        "pgf.texsystem": "lualatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })


# Compare the method of 
def symbolrateAutocorr(sig, Fs, **kwargs):
    Rxx = np.abs(signal.correlate(sig, sig, mode='full', method='fft'))
    f0 = estimate.f0MleTime(Rxx=Rxx, f=Fs, peaks=5)
    return f0

# Wrapper for estimation function 
def symbolRateEstimator(sig, Fs, aPrioriFCenter=False, **kwargs):
    # Ensure that the true center frequency is only used in the intended case
    if aPrioriFCenter==False:
        kwargs.pop('fCenterPriori')   # removes fCenterPriori from kwargs library

    SCD, f, alpha = estimate.FAM(sig, Fs = Fs, plot=False, method='conj', scale='linear', **kwargs)
    fCenter, R_symb = estimate.cyclicEstimator( SCD, f, alpha, **kwargs)
    return R_symb

# Configure estimators
estimators = []
estimators.append(estimator('Autocorrelation MLE', symbolrateAutocorr, Fs=Fs))
estimators.append(estimator('Cyclic MLE Method', symbolRateEstimator, Fs=Fs))
estimators.append(estimator('Cyclic MLE Method, Full BW', symbolRateEstimator, Fs=Fs, bandLimited=False))
estimators.append(estimator('Cyclic MLE A-Priori $f_c$', symbolRateEstimator, aPrioriFCenter=True, Fs=Fs))
estimators.append(estimator('Cyclic MLE A-Priori $f_c$, $\Omega$', symbolRateEstimator, aPrioriFCenter=True, Fs=Fs, alphaWindow=alphaWindow, fWindow='triangle', fWindowWidthHertz=50e3))

# Create analysis object
m_analysis = analysis('Symbol_Rate_Estimation', estimators=estimators, lossFcn='MAE')

# Generate Eb/N0 range for statistics gathering.
EbN0Start = 40
EbN0End = 10

m_analysis.axis.displayName = '$E_b/N_0$ [dB]'
m_analysis.axis.displayVector = np.linspace(EbN0End, EbN0Start, EbN0Start-EbN0End+1)
m_analysis.axis.name = 'S/N [dB]'
m_analysis.axis.vector = comm.EbN0toSNRdB(m_analysis.axis.displayVector, 2, Fs, 1/T)
m_analysis.analyze(iterations=nIterations, parameter='symbolRate', packetSize=packetSize, debug=Debug)

# Write to binary file
path = '../jobs/'
jobname = 'SRateJob'
destination = path + jobname + str(m_analysis.iterations) + '.pkl'
# Save job to binary file
with open(destination,'wb') as f:
    pickle.dump(m_analysis, f)

iterations =  nIterations #! Must be same as job file
"""
# Read from binary file
path = '../jobs/'
jobname = 'SRateJob'
destination = path + jobname + str(iterations) + '.pkl'
with open(destination,'rb') as f:
    m_analysis = pickle.load(f)"""

fig, ax = m_analysis.plotResults(pgf=not Debug, scale='semilogy', plotYlabel='MAE [Hz]')
ax.legend(loc='upper right')
#fig.set_figheight(2.5)
plt.tight_layout()


if Debug == False:
    fileName = m_analysis.name +'_'+ str(iterations) + '_iterations' # str(m_analysis.iterations)
    plt.savefig(imagePath + fileName + '.png', bbox_inches='tight')
    plt.savefig(imagePath + fileName + '.pgf', bbox_inches='tight')

plt.show()