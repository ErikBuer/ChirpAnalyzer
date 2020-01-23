from scipy.signal import chirp, sweep_poly, spectrogram, welch
import scipy.signal as signal
import analysis as analyze
import matplotlib.pyplot as plt
import numpy as np
import rftool.radar as radar
from rftool.utility import *

Fs=np.intc(200e3)           # receiver sample rate
frameSize=np.float(128e-3)  # seconds

# NLFM chirp generation 
def indefIntegration( x_t, dt ):
    Sx_tdt = np.cumsum(x_t)*dt
    return Sx_tdt

# NLFM chirp generation 
def cascadedIntegralChirpGenerator( t_i, c, Fs ):
    """
    Generate Non.Linear Frequency Modualted (NLFM) Chirps.

    t is the length of the chirp [s]
    c is a vector of phase polynomial coefficients (arbitrary length)
    f_c is the IF frequency
    Fs is the sampling frequency [Hz]

    c_1 is the reference phase
    c_2 is the reference frequency,
    c_3 is the nominal constant chirp rate
    c is one indexed here as it is represented as a vector. In the paper, time series representation with zero-indexing is used.

    Fro symmetricsl PSD, cn = 0 for even n > 3, that is, for n = 4, 6, 8, …

    - A .W. Doerry, Generating Nonlinear FM Chirp Waveforms for Radar, Sandia National Laboratories, 2006
    """
    dt = np.divide(1,Fs)        # seconds
    t = np.linspace(-t_i/2, t_i/2-dt, np.intc(Fs*t_i))  # Time vector

    phi_t = np.full(np.intc(t_i*Fs), c[-1])

    c = np.flip(c)
    c = np.delete(c, 1) # delete c_N
    for c_n in np.nditer(c):
        phi_t = c_n + indefIntegration(phi_t, dt)

    gamma_t = np.gradient(phi_t,t)  # Instantaneous frequency
    plt.figure
    plt.plot(t, gamma_t)
    plt.xlabel('t [s]')
    plt.ylabel('f [Hz]')
    plt.title("Instantaneous Frequency")
    plt.show()

    sig = np.exp(np.multiply(1j, phi_t))
    return sig

# NLFM chirp analyze 
def chirpRate( t_i, c, Fs ):
    for i in range(3, ):
        
        

a = 10e3
b = 1e3
d = 1e3
c = np.array([10e3,10,1e6,0,a,0,b,0,d,0])
sig = cascadedIntegralChirpGenerator( frameSize, c, Fs )

f, t, Sxx = signal.spectrogram(sig, Fs)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


f, Pxx_den = welch(sig, Fs, 'flattop', 1024, scaling='density')
Pxx_den_dB = pow2db(Pxx_den)
plt.plot(f, Pxx_den_dB)
plt.grid()
plt.ylim([-120,0])
plt.title("Welch PSD Estimate")
plt.xlabel('frequency [Hz]')
plt.ylabel('dBW/Hz')
plt.show()

"""
t_i = 1 #Chirp duration
deltaFrequency = 1e3
centerFrequency = 20e3

thrdOrdRate = 10e3
linRate     = deltaFrequency/t_i

# Generate test chirp using scipy.signal.sweep_poly
p = np.poly1d([(1/t_i)*thrdOrdRate, 0, linRate, centerFrequency])
t = np.linspace(-t_i/2, t_i/2-dt, np.intc(Fs*t_i))
sig = sweep_poly(t, p)

plt.subplot(2, 1, 1)
plt.plot(t, sig)
plt.title("Sweep Poly\n with frequency ")
plt.subplot(2, 1, 2)
plt.plot(t, p(t), 'r', label='f(t)')
plt.legend()
plt.xlabel('t')
plt.tight_layout()
"""

radar.hilbert_spectrum(np.real(sig), Fs)
plt.show()
