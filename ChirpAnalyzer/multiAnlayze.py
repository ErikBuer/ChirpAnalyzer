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
from matplotlib import cm
colorMap = cm.coolwarm
import matplotlib

import pickle
import os

Fs = np.intc(802e3) # Receiver sample rate. #! Must be the same as the signals
T = np.float(6e-3)  # Pulse duration.       #! Must be the same as the signals

# Configure estimators
estimators = []
estimators.append(estimator('Derivative', estimate.instFreq, Fs=Fs, method='derivative'))
estimators.append(estimator('Barnes Two-Point FIR', estimate.instFreq, Fs=Fs, method='BarnesTwo'))
estimators.append(estimator('Hilbert Spectrum MLE', estimate.instFreq, Fs=Fs, method='maxDHHT'))
estimators.append(estimator('Piecewise Polynomial MLE', estimate.instFreq, Fs=Fs, method='polyMle', windowSize=36, order=2))
estimators.append(estimator('WVD MLE', estimate.instFreq, Fs=Fs, method='maxWVD'))

# Generate logarithmic spread of Eb/N0 values.
EbN0Start = 40
EbN0End = 0

# Create analysis object
m_analysis = analysis('IF Estimation', estimators=estimators, lossFcn='MAE')

# Generate Eb/N0 range for statistics gathering.
EbN0Start = 40
EbN0End = 0

m_analysis.axis.displayName = 'E_b/N_0'
m_analysis.axis.displayVector = np.linspace(EbN0End, EbN0Start, 41)
m_analysis.axis.name = 'SNR'
m_analysis.axis.vector = comm.EbN0toSNRdB(m_analysis.axis.displayVector, 2, Fs, 1/T)
m_analysis.analyze(iterations=100, parameter='omega_t')