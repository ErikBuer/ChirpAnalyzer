from scipy.signal import chirp, sweep_poly, spectrogram, welch
from scipy.special import factorial
import scipy.signal as signal
import numpy as np
import random

import matplotlib.pyplot as plt
from matplotlib import cm

import rftool.radar as radar
import rftool.utility as util
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
    return util.pow2db(np.multiply(EbN0, Fsymb*np.log2(M)/Fs))


# find number of files in folder
path = "../waveforms/"
path, dirs, files = next(os.walk(path))
file_count = len(files)
print(file_count)
nIterations = file_count

Fs = np.intc(802e3) # Receiver sample rate. #! Must be the same as the signals
T=np.float(6e-3)  # Pulse duration. #! Must be the same as the signals
EbN0Vector = np.linspace(0, 40, 21)
snrVector = EbN0toSNRdB(EbN0Vector, 2, Fs, 1/T)
fCenterEstimate = np.zeros((nIterations, len(snrVector)), dtype=np.float64)
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


    for j, SNR in enumerate(snrVector):
        package = util.wgnSnr( modSig, SNR)

        SCD, f, alpha = radar.FAM(package, Fs = sigObj.Fs, plot=False, method='conj', scale='linear')
        fCenter, R_symb = radar.cyclicEstimator( SCD, f, alpha )

        fCenterEstimate[i,j] = np.abs(sigObj.fCenter-fCenter)
        R_symbEstimate[i,j] = np.abs(targetR_symb-R_symb)



# Calculate MSE
fCenterEstimateVector = np.mean(np.power(fCenterEstimate, 2), 0)
R_symbEstimateVector = np.mean(np.power(R_symbEstimate, 2), 0)


plt.figure()
plt.plot(snrVector, fCenterEstimateVector)
plt.title("Center Frequency Estimation Error")
plt.xlabel('SNR')
plt.ylabel('MSE')
plt.tight_layout()

plt.figure()
plt.plot(snrVector, R_symbEstimateVector)
plt.title("Symbol Rate Estimation Error")
plt.xlabel('SNR')
plt.ylabel('MSE')
plt.tight_layout()

plt.show()