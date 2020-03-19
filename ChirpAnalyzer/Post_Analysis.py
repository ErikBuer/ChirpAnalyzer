from scipy.signal import chirp, sweep_poly, spectrogram, welch
from scipy.special import factorial
import scipy.signal as signal
import numpy as np
import random

import rftool.radar as radar
import rftool.utility as util

import matplotlib.pyplot as plt
from matplotlib import cm
colorMap = cm.coolwarm
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

import joblib   # Parallelizations
import pickle
from waveform import *
import os

file_count = 200
# Load from binary file
imagePath = "../figures/estimation_50khz_bw/"
filename = 'CRLB'
destination = imagePath + filename + '.pkl'

# Load CRLB
with open(destination,'rb') as f:
        CRLBVector = pickle.load(f)

CRLB = CRLBVector[1]
        
filename = 'estimates'
destination = imagePath + filename + str(file_count) + '.pkl'
# Load estimation Results
with open(destination,'rb') as f:
        estimateVector = pickle.load(f)

snrVector = estimateVector[0]
EbN0Vector = estimateVector[1]
nIterations = estimateVector[2]
fCenterEstimate = estimateVector[3]
fCenterEstimate2 = estimateVector[4]
R_symbEstimate = estimateVector[5]

# Calculate Mean Absolute Normalized Error
fCenterMse = np.mean(np.power(fCenterEstimate, 2), 0)
fCenter2Mse = np.mean(np.power(fCenterEstimate2, 2), 0)

# Calculate Mean Absolute Normalized Error
R_symbEstimateVector = np.mean(R_symbEstimate, 0)

################################################################################
plt.figure()
plt.semilogy(EbN0Vector, fCenterMse, label='Cyclic Etimator', marker="+")
plt.semilogy(EbN0Vector, fCenter2Mse, label='DFT ML Estimator', marker=".")
plt.semilogy(EbN0Vector, CRLB, '--', label='CRLB')

plt.grid()
#plt.title("Center Frequency Estimation Error")
plt.xlabel('$E_b/N_0$ [dB]')
plt.ylabel('MSE')
plt.legend()
plt.tight_layout()

fileName = 'Center_Frequency_Estimation_Error' + str(nIterations)
plt.savefig(imagePath + fileName + '.png', bbox_inches='tight')
plt.savefig(imagePath + fileName + '.pgf', bbox_inches='tight')

################################################################################
maxValue = np.max(fCenterMse)
minValue = np.min(fCenterMse)
nBins = 100
histMatrix = np.empty((nBins, len(EbN0Vector)))

for i, row in enumerate(fCenterEstimate.T):
    histMatrix[:,i], bin_edges  = np.histogram(row, nBins, (minValue,maxValue))

plt.figure()
plt.pcolormesh(EbN0Vector, bin_edges, histMatrix, cmap = colorMap)
plt.colorbar()
#plt.boxplot(fCenterEstimate[:, 1::2], positions=EbN0Vector[1::2], showfliers=False)
#plt.xticks(rotation=90)
#plt.grid()
#plt.title("Center Frequency Estimation Error")
plt.xlabel('$E_b/N_0$ [dB]')
plt.ylabel('Absolute Error [Hz]')
plt.tight_layout()

fileName = 'Center_Frequency_Estimation_Error_Box' + str(nIterations)
plt.savefig(imagePath + fileName + '.png', bbox_inches='tight')
plt.savefig(imagePath + fileName + '.pgf', bbox_inches='tight')

################################################################################

plt.figure()
plt.semilogy(EbN0Vector, R_symbEstimateVector)
plt.grid()
#plt.title("Symbol Rate Estimation Error")
plt.xlabel('$E_b/N_0$ [dB]')
plt.ylabel('Mean Absolute Normalized Error')
plt.tight_layout()

fileName = 'Symbol_Rate_Estimation_Error' + str(nIterations)
plt.savefig(imagePath + fileName + '.png', bbox_inches='tight')
plt.savefig(imagePath + fileName + '.pgf', bbox_inches='tight')

#plt.show()