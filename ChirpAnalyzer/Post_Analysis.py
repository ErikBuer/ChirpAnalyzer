from scipy.signal import chirp, sweep_poly, spectrogram, welch
from scipy.special import factorial
import scipy.signal as signal
import numpy as np
import random

import rftool.radar as radar
import rftool.utility as util

import matplotlib.pyplot as plt
from matplotlib import cm
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


# Load from binary file
fileString = str(input())

with open(fileString,'rb') as f:
        estimateVector = pickle.load(f)

snrVector = estimateVector[0]
EbN0Vector = estimateVector[1]
fCenterEstimate = estimateVector[2]
fCenterEstimate2 = estimateVector[3]
R_symbEstimate = estimateVector[4]

# Calculate Mean Absolute Normalized Error
fCenterEstimateVector = np.mean(fCenterEstimate, 0)
fCenterEstimateVector2 = np.mean(fCenterEstimate2, 0)

# Calculate Mean Absolute Normalized Error
R_symbEstimateVector = np.mean(R_symbEstimate, 0)

plt.figure()
plt.semilogy(EbN0Vector, fCenterEstimateVector, label='Cyclic Etimator', marker="+")
plt.semilogy(EbN0Vector, fCenterEstimateVector2, label='DFT ML Estimator', marker=".")
plt.grid()
#plt.title("Center Frequency Estimation Error")
plt.xlabel('$E_b/N_0$ [dB]')
plt.ylabel('Mean Absolute Normalized Error')
plt.legend()
plt.tight_layout()

fileName = 'Center_Frequency_Estimation_Error' + str(nIterations)
plt.savefig(imagePath + fileName + '.png', bbox_inches='tight')
plt.savefig(imagePath + fileName + '.pgf', bbox_inches='tight')

plt.figure()
plt.semilogy(EbN0Vector, R_symbEstimateVector)
plt.grid()
#plt.title("Symbol Rate Estimation Error")
plt.xlabel('$E_b/N_0$ [dB]')
plt.ylabel('MSE')
plt.tight_layout()

fileName = 'Symbol_Rate_Estimation_Error' + str(nIterations)
plt.savefig(imagePath + fileName + '.png', bbox_inches='tight')
plt.savefig(imagePath + fileName + '.pgf', bbox_inches='tight')

plt.show()