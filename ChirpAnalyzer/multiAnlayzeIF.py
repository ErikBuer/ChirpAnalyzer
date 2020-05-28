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
from matplotlib import cm
colorMap = cm.coolwarm
import matplotlib
"""

import pickle
#import os
path = '../jobs/'
debug = False

Fs = np.intc(802e3) # Receiver sample rate. #! Must be the same as the signals
T = np.float(6e-3)  # Pulse duration.       #! Must be the same as the signals
nIterations = 5 # 12

# Generate Eb/N0 range for statistics gathering.
EbN0Start = 100
EbN0End = 20

# Configure estimators
estimators = []
estimators.append(estimator('Barnes Two-Point FIR', estimate.instFreq, Fs=Fs, method='BarnesTwo'))
estimators.append(estimator('Derivative', estimate.instFreq, Fs=Fs, method='derivative'))
estimators.append(estimator('Hilbert-Huang MLE', estimate.instFreq, Fs=Fs, method='maxDHHT'))
estimators.append(estimator('Piecewise Polynomial MLE', estimate.instFreq, Fs=Fs, method='polyMle', windowSize=50, order=2)) #! In report
estimators.append(estimator('WVD MLE', estimate.instFreq, Fs=Fs, method='maxWVD'))

######################################################################################################
# LFM Case
"""
# Create analysis object
m_analysis = analysis('IF_Estimation_NLFM', estimators=estimators, lossFcn='MAE')

m_analysis.axis.displayName = '$E_b/N_0$ [dB]'
m_analysis.axis.displayVector = np.linspace(EbN0End, EbN0Start, np.intc(abs(EbN0End-EbN0Start)/4))
m_analysis.axis.name = '$S/N$ [dB]'
m_analysis.axis.vector = comm.EbN0toSNRdB(m_analysis.axis.displayVector, 2, Fs, 1/T)
m_analysis.analyze(iterations=nIterations, parameter='omega_t')

# Write to binary file
filename = 'jobNLFM'
destination = path + filename + str(m_analysis.iterations) + '.pkl'
# Save job to binary file
with open(destination,'wb') as f:
    pickle.dump(m_analysis, f)
"""
######################################################################################################
# LFM Case

# Generate waveform object With waveform() to be handed to utility.analysis
m_waveform = waveform()
m_waveform.Fs = Fs
m_waveform.T = T
m_waveform.symbolRate = 1/T



# Create analysis object
m_analysis = analysis('IF_Estimation_LFM', estimators=estimators, lossFcn='MAE')

m_analysis.axis.displayName = '$E_b/N_0$ [dB]'
m_analysis.axis.displayVector = np.linspace(EbN0End, EbN0Start, np.intc(abs(EbN0End-EbN0Start)/4))
m_analysis.axis.name = '$S/N$ [dB]'
m_analysis.axis.vector = comm.EbN0toSNRdB(m_analysis.axis.displayVector, 2, Fs, 1/T)
m_analysis.analyze(iterations=nIterations, parameter='omega_t', signalType='LFM', m_waveform=m_waveform, debug=debug)

# Write to binary file
filename = 'jobLFM'
destination = path + filename + str(m_analysis.iterations) + '.pkl'
# Save job to binary file
with open(destination,'wb') as f:
    pickle.dump(m_analysis, f)