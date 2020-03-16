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

def EbN0toSNRdB(EbN0, M, Fs, Fsymb):
    """
    Calculte the necessary SNR in order to obtain a target Eb/N0
    EbN0 is the intended SNR (scalar or vector)
    sig_t is the time domain signal
    M is the order of the modultion
    Fs is the sample rate of the signal
    Fsymb is the symbol rate of the signal (pulse rate)
    """
    return util.pow2db(np.multiply(util.db2pow(EbN0), Fsymb*np.log2(M)/Fs))


# find number of files in folder
directory = '../../waveforms/'
path, dirs, files = os.walk(directory).__next__()

file_count = len(files)
nIterations = file_count

Fs = np.intc(802e3) # Receiver sample rate. #! Must be the same as the signals
T=np.float(6e-3)  # Pulse duration. #! Must be the same as the signals

# Generate logarithmic spread of Eb/N0 values.
EbN0Start = -0.1
EbN0End = -20
#EbN0Vector = -np.subtract(-EbN0End, np.logspace(np.log10(-EbN0Start), np.log10(-EbN0End), num=31, endpoint=True, base=10.0))
EbN0Vector = np.linspace(EbN0End, EbN0Start, 41)

#EbN0Vector = np.linspace(0, -30, 31)
snrVector = EbN0toSNRdB(EbN0Vector, 2, Fs, 1/T)

fCenterEstimate = np.zeros((nIterations, len(snrVector)), dtype=np.float64)
fCenterEstimate2 = np.zeros((nIterations, len(snrVector)), dtype=np.float64)
R_symbEstimate = np.zeros((nIterations, len(snrVector)))

print("np.shape(fCenterEstimate)", np.shape(fCenterEstimate))

for i in range(0, nIterations):
    print("Iteration", i+1, "of", nIterations)
    
    # Load from binary file
    filename = str(i)
    fileString = path + filename + ".pkl"

    with open(fileString,'rb') as f:
        sigObj = pickle.load(f)

    NLFM = radar.chirp(sigObj.Fs)
    NLFM.targetOmega_t = sigObj.omega_t
    NLFM.points = len(sigObj.omega_t)
    NLFM.c = sigObj.polynomial

    package = np.random.randint(0, 1, 32)
    modSig = NLFM.modulate( package )
    targetR_symb = 1/NLFM.T


    def estimator(modSig, SNR, sigObj):
        package = util.wgnSnr( modSig, SNR)

        SCD, f, alpha = radar.FAM(package, Fs = sigObj.Fs, plot=False, method='conj', scale='linear')
        fCenter, R_symb = radar.cyclicEstimator( SCD, f, alpha )
        fCenter2 =radar.carierFrequencyEstimator( package, sigObj.Fs, method='mle' )

        fCenterEstimate = np.abs(sigObj.fCenter-fCenter)/sigObj.fCenter
        fCenterEstimate2 = np.abs(sigObj.fCenter-fCenter2)/sigObj.fCenter
        R_symbEstimate = np.abs(targetR_symb-R_symb)
        return fCenterEstimate, fCenterEstimate2, R_symbEstimate

    # = joblib.Parallel(n_jobs=8, verbose=20)(joblib.delayed(estimator)(modSig, SNR, sigObj) for SNR in snrVector)
    estimates = joblib.Parallel(n_jobs=8, verbose=0)(joblib.delayed(estimator)(modSig, SNR, sigObj) for SNR in snrVector)
    estimateMat = np.asarray(estimates)
    fCenterEstimate[i,:] = estimateMat[:, 0]
    fCenterEstimate2[i,:] = estimateMat[:, 1]
    R_symbEstimate[i,:] = estimateMat[:, 2]


# Calculate Root Mean Square Normalized Error
fCenterEstimateVector = np.mean(fCenterEstimate, 0)
fCenterEstimateVector2 = np.mean(fCenterEstimate2, 0)

# Calculate MSE
R_symbEstimateVector = np.mean(np.power(R_symbEstimate, 2), 0)

imagePath = "../figures/estimation/"

plt.figure()
plt.semilogy(EbN0Vector, fCenterEstimateVector, label='Cyclic Etimator', marker="+")
plt.semilogy(EbN0Vector, fCenterEstimateVector2, label='DFT ML Estimator', marker=".")
plt.grid()
plt.title("Mean Absolute Normalized Error")
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
plt.title("Symbol Rate Estimation Error")
plt.xlabel('$E_b/N_0$ [dB]')
plt.ylabel('MSE')
plt.tight_layout()

fileName = 'Symbol_Rate_Estimation_Error' + str(nIterations)
plt.savefig(imagePath + fileName + '.png', bbox_inches='tight')
plt.savefig(imagePath + fileName + '.pgf', bbox_inches='tight')

plt.show()