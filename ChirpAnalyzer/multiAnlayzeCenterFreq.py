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
nIterations = 1
packetSize = 32

# Load alpha window function a-priori
path = '../jobs/'
filename = 'SCD_GMM'
destination = path + filename + '.pkl'
with open(destination,'rb') as f:
    alphaWindow = pickle.load(f)


# Wrapper for estimation function 
def cyclicFreqEstimator(sig, Fs, **kwargs):
    SCD, f, alpha = estimate.FAM(sig, Fs = Fs, plot=False, method='conj', scale='linear', **kwargs)
    fCenter, R_symb = estimate.cyclicEstimator( SCD, f, alpha, bandLimited=True , **kwargs)
    return fCenter

# Wrapper for CRLB calculator
def CRLB(sig, Fs, packetSize, SNR, cleanSig, **kwargs):
    symblen = np.intc(len(sig)/packetSize)
    p_n =  cleanSig[0:symblen] # the energy of one pulse is the same as sum(abs(symbol)). The symbols has a magnitude of one
    # Number of pulses
    K = packetSize
    # Calculate discrete pulse times l_k
    l_k = np.linspace(0, K-1, K)*len(p_n)
    # Calculate Noise power N
    N =  util.db2pow(util.powerdB(p_n) - SNR)
    CRLBOmega = estimate.pulseCarrierCRLB(p_n, K, l_k, N)
    AbsoluteErrorHertz = np.sqrt(CRLBOmega)*Fs/(2*np.pi)
    #CRLBHertz = np.power(np.sqrt(CRLBOmega)*Fs/(2*np.pi), 2)
    return AbsoluteErrorHertz

# Configure estimators
estimators = []
estimators.append(estimator('DFT MLE Method', estimate.carierFrequencyEstimator, Fs=Fs, method='mle' , nfft=459))
#estimators.append(estimator('Cyclic MLE Method', cyclicFreqEstimator, Fs=Fs))
estimators.append(estimator('Cyclic MLE A-Priori $T_s$', cyclicFreqEstimator, Fs=Fs, alphaWindow=alphaWindow))
estimators.append(estimator('Cyclic MLE A-Priori $T_s$, $\Omega$', cyclicFreqEstimator, Fs=Fs, alphaWindow=alphaWindow, fWindow='rectangle', fWindowWidthHertz=50e3))
estimators.append(estimator('$\sqrt{CRLB}$ [Hz]', CRLB, packetSize=packetSize, Fs=Fs))

# Create analysis object
m_analysis = analysis('Center_Frequency_Estimation', estimators=estimators, lossFcn='MAE')

# Generate Eb/N0 range for statistics gathering.
EbN0Start = 40
EbN0End = 10

m_analysis.axis.displayName = '$E_b/N_0$ [dB]'
m_analysis.axis.displayVector = np.linspace(EbN0End, EbN0Start, EbN0Start-EbN0End)
m_analysis.axis.name = 'S/N [dB]'
m_analysis.axis.vector = comm.EbN0toSNRdB(m_analysis.axis.displayVector, 2, Fs, 1/T)
m_analysis.analyze(iterations=nIterations, parameter='fCenter', packetSize=packetSize, debug=Debug)

"""# Write to binary file
path = '../jobs/'
filename = 'centerFrequencyJob'
destination = path + filename + str(m_analysis.iterations) + '.pkl'
# Save job to binary file
with open(destination,'wb') as f:
    pickle.dump(m_analysis, f)


# Read from binary file
path = '../jobs/'
filename = 'centerFrequencyJob'
destination = path + filename + str(iterations) + '.pkl'
with open(destination,'rb') as f:
    m_analysis = pickle.load(f)"""

# Plot results
import matplotlib.pyplot as plt
plt.style.use('masterThesis')
import matplotlib

iterations = nIterations
fig, ax = m_analysis.plotResults(pgf=not Debug, scale='semilogy', plotYlabel='MAE [Hz]')
ax.legend(loc='lower right')
#ax.set_ymargin = 0.1
fig.set_figheight(2.5)
plt.tight_layout()
imagePath = '../figures/centerFreqEst/'
if Debug == False:
    fileName = m_analysis.name +'_'+ str(iterations) + '_iterations' # str(m_analysis.iterations)
    plt.savefig(imagePath + fileName + '.png', bbox_inches='tight')
    plt.savefig(imagePath + fileName + '.pgf', bbox_inches='tight')

ax.lines.pop(-1)
#ax.set_ymargin = 0.1
ax.legend(loc='upper right')
ax.set_ylim(100,100000)
plt.tight_layout()
if Debug == False:
    fileName = m_analysis.name +'_noCRLB_'+ str(iterations) + '_iterations' # str(m_analysis.iterations)
    plt.savefig(imagePath + fileName + '.png', bbox_inches='tight')
    plt.savefig(imagePath + fileName + '.pgf', bbox_inches='tight')

plt.show()