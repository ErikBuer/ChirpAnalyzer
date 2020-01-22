from scipy.signal import chirp, sweep_poly, spectrogram, welch
import matplotlib.pyplot as plt
import numpy as np
from pyhht.visualization import plot_imfs # Hilbert-Huang TF analysis
import rftool.radar as radar

Fs=np.intc(1e6)             # receiver sample rate
frameSize=np.float(10e-3)   # seconds
dt = np.divide(1,Fs)        # seconds


# Generate test chirp using by the method of C. Lesnik




"""
# Generate test chirp using scipy.signal.sweep_poly
p = np.poly1d([0.025, -0.36, 0.25, 10000])
t = np.linspace(0, frameSize-dt, np.intc(frameSize*Fs))
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

radar.hilbert_spectrum(sig, Fs)
#plt.show()
