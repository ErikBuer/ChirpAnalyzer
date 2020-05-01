from scipy.signal import chirp, sweep_poly, spectrogram, welch
from scipy.special import factorial
import scipy.signal as signal
import numpy as np
import random

import rftool.radar as radar
import rftool.utility as util
import rftool.estimation as estimate
from utility import *

import matplotlib.pyplot as plt
plt.style.use('masterThesis')
from matplotlib import cm
import matplotlib as mpl

pgf = True
if pgf==True:
    mpl.use("pgf")
    mpl.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

import joblib   # Parallelizations
import pickle
from utility import *
import os

file_count = 100
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
R_symbEstimate2 = estimateVector[6]

# Calculate Mean Absolute Normalized Error
fCenterMse = np.mean(np.power(fCenterEstimate, 2), 0)
fCenterMae= np.mean(fCenterEstimate, 0)
fCenter2Mse = np.mean(np.power(fCenterEstimate2, 2), 0)

# Calculate Mean Absolute Normalized Error
R_symbEstimateVector = np.mean(R_symbEstimate, 0)
R_symbEstimateVector2 = np.mean(R_symbEstimate2, 0)
################################################################################
plt.figure()
plt.semilogy(EbN0Vector, fCenterMse, label='Cyclic Etimator') #, marker="+")
plt.semilogy(EbN0Vector, fCenter2Mse, label='DFT ML Estimator') #, marker=".")
plt.semilogy(EbN0Vector, CRLB, '--', label='CRLB')

plt.grid()
#plt.title("Center Frequency Estimation Error")
plt.xlabel('$E_b/N_0$ [dB]')
plt.ylabel('MSE')
plt.legend()
plt.tight_layout()

if pgf==True:
    fileName = 'Center_Frequency_Estimation_Error' + str(nIterations)
    plt.savefig(imagePath + fileName + '.png', bbox_inches='tight')
    plt.savefig(imagePath + fileName + '.pgf', bbox_inches='tight')

################################################################################
maxValue = np.max(fCenterEstimate)
minValue = np.min(fCenterEstimate)
deciRdange = (maxValue-minValue)/100
nBins = 100
histMatrix = np.empty((nBins, len(EbN0Vector)))

for i, row in enumerate(np.transpose(fCenterEstimate)):
    histMatrix[:,i], bin_edges  = np.histogram(row, nBins, (minValue-deciRdange,maxValue+deciRdange))

plt.figure()
plt.pcolormesh(EbN0Vector, bin_edges, histMatrix)
plt.colorbar()
#plt.boxplot(fCenterEstimate[:, 1::2], positions=EbN0Vector[1::2], showfliers=False)
#plt.xticks(rotation=90)
#plt.grid()
#plt.title("Center Frequency Estimation Error")
plt.xlabel('$E_b/N_0$ [dB]')
plt.ylabel('Absolute Error [Hz]')
plt.tight_layout()

if pgf==True:
    fileName = 'Center_Frequency_Estimation_Error_Distribution' + str(nIterations)
    plt.savefig(imagePath + fileName + '.png', bbox_inches='tight')
    plt.savefig(imagePath + fileName + '.pgf', bbox_inches='tight')

################################################################################

plt.figure()
plt.semilogy(EbN0Vector, R_symbEstimateVector, label='$2$ dB BW', marker="+")
plt.semilogy(EbN0Vector, R_symbEstimateVector2, label='Full BW', marker=".")
plt.grid()
#plt.title("Symbol Rate Estimation Error")
plt.xlabel('$E_b/N_0$ [dB]')
plt.ylabel('Mean Absolute Error [Hz]')
plt.legend()
plt.tight_layout()

if pgf==True:
    fileName = 'Symbol_Rate_Estimation_Error' + str(nIterations)
    plt.savefig(imagePath + fileName + '.png', bbox_inches='tight')
    plt.savefig(imagePath + fileName + '.pgf', bbox_inches='tight')


################################################################################
maxValue = np.max(R_symbEstimate)
minValue = np.min(R_symbEstimate)
deciRdange = (maxValue-minValue)/100
nBins = 100
histMatrix = np.empty((nBins, len(EbN0Vector)))

for i, row in enumerate(np.transpose(R_symbEstimate)):
    histMatrix[:,i], bin_edges  = np.histogram(row, nBins, (minValue-deciRdange,maxValue+deciRdange))

plt.figure()
plt.pcolormesh(EbN0Vector, bin_edges, histMatrix)
plt.colorbar()
#plt.boxplot(R_symbEstimate[:, 1::2], positions=EbN0Vector[1::2], showfliers=False)
#plt.xticks(rotation=90)
#plt.grid()
#plt.title("")
plt.xlabel('$E_b/N_0$ [dB]')
plt.ylabel('Absolute Error [Hz]')
plt.tight_layout()

if pgf==True:
    fileName = 'Stmbol_Rate_Estimation_Error_Distribution' + str(nIterations)
    plt.savefig(imagePath + fileName + '.png', bbox_inches='tight')
    plt.savefig(imagePath + fileName + '.pgf', bbox_inches='tight')

################################################################################
maxValue = np.max(R_symbEstimate2)
minValue = np.min(R_symbEstimate2)
deciRdange = (maxValue-minValue)/100
nBins = 100
histMatrix = np.empty((nBins, len(EbN0Vector)))

for i, row in enumerate(np.transpose(R_symbEstimate2)):
    histMatrix[:,i], bin_edges  = np.histogram(row, nBins, (minValue-deciRdange,maxValue+deciRdange))

plt.figure()
plt.pcolormesh(EbN0Vector, bin_edges, histMatrix)
plt.colorbar()
#plt.boxplot(R_symbEstimate[:, 1::2], positions=EbN0Vector[1::2], showfliers=False)
#plt.xticks(rotation=90)
#plt.grid()
#plt.title("")
plt.xlabel('$E_b/N_0$ [dB]')
plt.ylabel('Absolute Error [Hz]')
plt.tight_layout()

if pgf==True:
    fileName = 'Stmbol_Rate_Estimation_Full_BW_Error_Distribution' + str(nIterations)
    plt.savefig(imagePath + fileName + '.png', bbox_inches='tight')
    plt.savefig(imagePath + fileName + '.pgf', bbox_inches='tight')