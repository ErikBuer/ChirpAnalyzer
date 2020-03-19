from scipy.signal import chirp, sweep_poly, spectrogram, welch
from scipy.special import factorial
import scipy.signal as signal
import numpy as np
import random
import matplotlib.pyplot as plt


import rftool.radar as radar
import rftool.utility as util
from waveform import *  # waveform object

import joblib   # Parallelizations
import pickle


Fs=np.intc(802e3) # receiver sample rate

def generator(Fs, i):
    print("Signal iteration", i)

    T=np.float(6e-3)  # Pulse duration
    # Time domain window for NLFM generation
    NLFM = radar.chirp(Fs)
    NLFM.fftLen = 2048

    sigObj = waveform()
    sigObj.Fs = Fs


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

    path = '../../waveforms_50khz_bw/'
   


    sigObj.polynomial = NLFM.getCoefficients( window_t, targetBw=fStop-fStart, centerFreq=fCenter, T=T)
    sigObj.omega_t = NLFM.targetOmega_t


    # Write to binary file

    filename = str(i)
    destination = path + filename + '.pkl'

    # Save sigObj to binary file
    with open(destination,'wb') as f:
        pickle.dump(sigObj, f)

joblib.Parallel(n_jobs=4, verbose=0)(joblib.delayed(generator)(Fs, i) for i in range(10000, 20000))
    