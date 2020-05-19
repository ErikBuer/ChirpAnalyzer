from scipy.signal import chirp, sweep_poly, spectrogram, welch
from scipy.special import factorial
import scipy.signal as signal
import numpy as np
import random
import matplotlib.pyplot as plt


import rftool.radar as radar
import rftool.utility as util
import rftool.estimation as estimate
from utility import *  # waveform object

import os
import joblib   # Parallelizations
import pickle


Fs=np.intc(802e3) # receiver sample rate

debug = False

def generator(Fs, i):
    print("Signal iteration", i)

    T=np.float(6e-3)  # Pulse duration
    # Time domain window for NLFM generation
    NLFM = radar.chirp(Fs)
    NLFM.fftLen = 2048

    sigObj = waveform()
    sigObj.Fs = Fs
    sigObj.T = T
    sigObj.symbolRate = 1/T

    # Synthesize the target autocorrelation function
    #window_t = signal.chebwin(np.intc(2048), 60)
    window_t = signal.hamming(np.intc(2048))
    #window_t = signal.gaussian(np.intc(2048), 360)
    #window_t = signal.gaussian(np.intc(2048), 400)

    """
    # Random BW
    fStart = random.uniform(10e3,100e3)
    fStop = fStart+random.uniform(10e3,100e3)
    fCenter = fStop-(fStop-fStart)/2

    path = '../../waveforms/'
    """

    # Fixed BW
    fStart = random.uniform(10e3,100e3)
    fStop = fStart+50e3
    fCenter = fStop-(fStop-fStart)/2

    sigObj.fCenter = fCenter
    sigObj.fStart = fStart
    sigObj.fStop = fStop

    path = '../waveforms/'
   
    sigObj.polynomial = NLFM.getCoefficients( window_t, targetBw=fStop-fStart, centerFreq=fCenter, T=T)
    sigObj.omega_t = NLFM.targetOmega_t

    # Center frequency is defined as the estimated center frequency in infinite SNR
    sig_t = NLFM.genNumerical()
    #sigObj.fCenter = estimate.carierFrequencyEstimator(sig_t, Fs, method='mle', nfft=len(sig_t))

    # Write to binary file
    filename = str(i)
    destination = path + filename + '.pkl'

    # Save sigObj to binary file
    with open(destination,'wb') as f:
        pickle.dump(sigObj, f)

rStart = 500
rStop = 1001

if debug == False:
    joblib.Parallel(n_jobs=4, verbose=0)(joblib.delayed(generator)(Fs, i) for i in range(rStart, rStop))
else:
    joblib.Parallel(n_jobs=1, verbose=10)(joblib.delayed(generator)(Fs, i) for i in range(rStart, rStop)) # Optimally four jobs