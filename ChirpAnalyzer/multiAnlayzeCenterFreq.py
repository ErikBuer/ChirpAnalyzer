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

"""
import matplotlib.pyplot as plt
import matplotlib
"""

import pickle
#import os

Debug = False

Fs = np.intc(802e3) # Receiver sample rate. #! Must be the same as the signals
T = np.float(6e-3)  # Pulse duration.       #! Must be the same as the signals
nIterations = 99
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
def CRLB(sig, Fs, packetSize, SNR):
    symblen = np.intc(len(sig)/packetSize)
    p_n = sig
    # Number of pulses
    K = packetSize
    # Calculate discrete pulse times l_k
    l_k = np.linspace(0, K-1, K)*len(p_n)
    # Calculate Noise power N
    N =  util.db2pow(util.powerdB(p_n) - SNR)
    CRLBOmega = estimate.pulseCarrierCRLB(p_n, K, l_k, N)
    CRLB = np.sqrt(CRLBOmega)*Fs/(2*np.pi)
    return CRLB

# Configure estimators
estimators = []
#estimators.append(estimator('DFT MLE Method', estimate.carierFrequencyEstimator, Fs=Fs, method='mle' , nfft=459))
estimators.append(estimator('Cyclic MLE Method', cyclicFreqEstimator, Fs=Fs))
estimators.append(estimator('Cyclic MLE A-Priori Symbol-Rate', cyclicFreqEstimator, Fs=Fs, alphaWindow=alphaWindow))
#! estimators.append(estimator('CRLB', CRLB, packetSize=packetSize, Fs=Fs))

# Create analysis object
m_analysis = analysis('Center_Frequency_Estimation', estimators=estimators, lossFcn='MAE')

# Generate Eb/N0 range for statistics gathering.
EbN0Start = 40
EbN0End = 0

m_analysis.axis.displayName = '$E_b/N_0$ [dB]'
m_analysis.axis.displayVector = np.linspace(EbN0End, EbN0Start, 41)
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
m_analysis.plotResults(pgf=not Debug, scale='semilogy')

imagePath = '../figures/'
fileName = m_analysis.name +'_'+ str(iterations) + '_iterations' # str(m_analysis.iterations)
plt.savefig(imagePath + fileName + '.png', bbox_inches='tight')
plt.savefig(imagePath + fileName + '.pgf', bbox_inches='tight')