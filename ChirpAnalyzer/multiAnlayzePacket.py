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
plt.style.use('masterThesis')
from matplotlib import cm
colorMap = cm.coolwarm
import matplotlib as mpl


import pickle
#import os

Fs = np.intc(802e3) # Receiver sample rate. #! Must be the same as the signals
T = np.float(6e-3)  # Pulse duration.       #! Must be the same as the signals
nIterations = 100

def packetExtractor( sig_t, Fs, T, **kwargs ):
    """
    Wrapper for extracting the single output which is used
    """
    [packet, symbolAlphabet] = estimate.inspectPackage(sig_t, Fs, T, **kwargs)
    return packet


# Configure estimators
estimators = []
estimators.append(estimator('Packet Extractor', packetExtractor, Fs=Fs, T=T, threshold=util.db2pow(-10.5)))

# Create analysis object
m_analysis = analysis('Packet_Extraction', estimators=estimators, lossFcn='MAE')

# Generate Eb/N0 range for statistics gathering.
EbN0Start = 28
EbN0End = 24

EbN0Vec = np.linspace(EbN0End, EbN0Start, 41)
SnrVec = comm.EbN0toSNRdB(EbN0Vec, 2, Fs, 1/T)

m_analysis.axis.displayName = 'S/N [dB]'
m_analysis.axis.displayVector = SnrVec
m_analysis.axis.name = 'S/N [dB]'
m_analysis.axis.vector = SnrVec
m_analysis.analyze(iterations=nIterations, parameter='packet', packetSize=10, debug=False)

# Write to binary file
path = '../jobs/'
filename = 'packetJob'
destination = path + filename + str(m_analysis.iterations) + '.pkl'
# Save job to binary file
with open(destination,'wb') as f:
    pickle.dump(m_analysis, f)


m_analysis.plotResults(pgf=True)

imagePath = '../figures/'
fileName = m_analysis.name +'_'+ str(nIterations) + '_iterations' # str(m_analysis.iterations)
plt.savefig(imagePath + fileName + '.png', bbox_inches='tight')
plt.savefig(imagePath + fileName + '.pgf', bbox_inches='tight')

plt.show()