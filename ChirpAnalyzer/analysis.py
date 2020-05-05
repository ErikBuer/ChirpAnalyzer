from scipy.signal import chirp, sweep_poly, spectrogram, welch
from scipy.special import factorial
import scipy.signal as signal
import numpy as np
import random

import rftool.radar as radar
import rftool.communications as comm
import rftool.utility as util
import rftool.estimation as estimate
from utility import *

import matplotlib.pyplot as plt
plt.style.use('masterThesis')
from matplotlib import cm
import matplotlib

pgf = False
if pgf == True:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })


import joblib   # Parallelizations
import pickle
import os

# Pathes for file storages
imagePath = "../figures/estimation_50khz_bw/"

# find number of files in folder
directory = '../waveforms/'
path, dirs, files = os.walk(directory).__next__()

file_count = 100#len(files)
nIterations = file_count

Fs = np.intc(802e3) # Receiver sample rate. #! Must be the same as the signals
T=np.float(6e-3)  # Pulse duration. #! Must be the same as the signals

# Generate logarithmic spread of Eb/N0 values.
EbN0Start = 40
EbN0End = 0
#EbN0Vector = -np.subtract(-EbN0End, np.logspace(np.log10(-EbN0Start), np.log10(-EbN0End), num=31, endpoint=True, base=10.0))
EbN0Vector = np.linspace(EbN0End, EbN0Start, 41)

#EbN0Vector = np.linspace(0, -30, 31)
snrVector = comm.EbN0toSNRdB(EbN0Vector, 2, Fs, 1/T)

fCenterEstimate = np.zeros((nIterations, len(snrVector)), dtype=np.float64)
fCenterEstimate2 = np.zeros((nIterations, len(snrVector)), dtype=np.float64)
R_symbEstimate = np.zeros((nIterations, len(snrVector)), dtype=np.float64)
R_symbEstimate2 = np.zeros((nIterations, len(snrVector)), dtype=np.float64)

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

    packetSize=32
    package = np.random.randint(0, 2, packetSize)
    modSig = NLFM.modulate( package )
    targetR_symb = 1/NLFM.T

    # Calculate CRLB for the chirp in the SNR range
    if(i == 0):
        # Generate pulse
        #p_n = NLFM.genFromPoly()
        p_n = NLFM.genNumerical()
        # Number of pulses
        K = len(package)
        # Calculate discrete pulse times l_k
        l_k = np.linspace(0, K-1, K)*len(p_n)
        # Calculate Noise power N
        N =  util.db2pow(util.powerdB(p_n) - snrVector)
        CRLBOmega = estimate.pulseCarrierCRLB(p_n, K, l_k, N)
        CRLBHertz = np.power(np.sqrt(CRLBOmega)*sigObj.Fs/(2*np.pi), 2)

        CRLBVector = [snrVector, CRLBHertz]

        Destination = imagePath+'CRLB'+'.pkl'
        # Save sigObj to binary file
        with open(Destination,'wb') as f:
            pickle.dump(CRLBVector, f)


    def estimator(modSig, SNR, sigObj):
        package = util.wgnSnr( modSig, SNR)

        SCD, f, alpha = estimate.FAM(package, Fs = sigObj.Fs, plot=False, method='conj', scale='linear')
        fCenter, R_symb = estimate.cyclicEstimator( SCD, f, alpha, bandLimited=True )
        trash, R_symb2 = estimate.cyclicEstimator( SCD, f, alpha )
        fCenter2 = estimate.carierFrequencyEstimator( package, sigObj.Fs, method='mle' , nfft=459 )

        fCenterEstimate = np.abs(sigObj.fCenter-fCenter)
        fCenterEstimate2 = np.abs(sigObj.fCenter-fCenter2)
        R_symbEstimate = np.abs(targetR_symb-R_symb)
        R_symbEstimate2 = np.abs(targetR_symb-R_symb2)

        return fCenterEstimate, fCenterEstimate2, R_symbEstimate, R_symbEstimate2

    estimates = joblib.Parallel(n_jobs=1, verbose=0)(joblib.delayed(estimator)(modSig, SNR, sigObj) for SNR in snrVector) # Six jobs is optimal.
    estimateMat = np.asarray(estimates)
    fCenterEstimate[i,:] = estimateMat[:, 0]
    fCenterEstimate2[i,:] = estimateMat[:, 1]
    R_symbEstimate[i,:] = estimateMat[:, 2]
    R_symbEstimate2[i,:] = estimateMat[:, 3]

estimateVector = [snrVector, EbN0Vector, nIterations, fCenterEstimate, fCenterEstimate2, R_symbEstimate, R_symbEstimate2]

# Write to binary file
filename = 'estimates'
destination = imagePath + filename + str(file_count) + '.pkl'

# Save sigObj to binary file
with open(destination,'wb') as f:
    pickle.dump(estimateVector, f)

# For post processing (generation of figures, see Post_Analysis.py)