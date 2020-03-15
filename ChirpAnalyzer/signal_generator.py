from scipy.signal import chirp, sweep_poly, spectrogram, welch
from scipy.special import factorial
from waveform import *  # waveform object
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
import rftool.radar as radar
import rftool.utility as util
import pickle
import random

Fs=np.intc(802e3) # receiver sample rate

for i in range(0, 1000):
    print("Iteration", i)

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

    fStart = random.uniform(10e3,100e3)
    fStop = fStart+random.uniform(10e3,100e3)
    fCenter = fStop-(fStop-fStart)/2

    sigObj.fCenter = fCenter
    sigObj.fStart = fStart
    sigObj.fStop = fStop

    sigObj.polynomial = NLFM.getCoefficients( window_t, targetBw=fStop-fStart, centerFreq=fCenter, T=T)
    sigObj.omega_t = NLFM.targetOmega_t


    # Write to binary file
    path = "../waveforms/"
    filename = str(i)
    destination = path + filename + ".pkl"

    # Save sigObj to binary file
    with open(destination,'wb') as f:
        pickle.dump(sigObj, f)

    """
    snrVector = np.linspace(0, -70, 71, dtype=int)
    for SNR in snrVector:
        sigObj.SNR = SNR

        package = np.random.randint(0, 1, 32)
        modSig = NLFM.modulate( package )
        modSig = util.wgnSnr( modSig, SNR)
        sigObj.timeSeries = modSig

        # Write to binary file
        path = "../wavefrorms/"
        filename = str(i)+"_"+ str(SNR)
        destination = path + filename + ".pkl"


        # Save sigObj to binary file
        with open(destination,'wb') as f:
            pickle.dump(sigObj, f)
    """

    

"""
TODO Analyze
- List of parameters to estimate on single chirp
- Skriv i rapport hvordan HH-transform foregår.
- Vurder Om kurvene skal karakteriseres fra sentrum, eller frs siden (i tid-frekvens).

TODO Report
- Which parameters in FAM decides the estimation resolution?
"""