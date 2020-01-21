from scipy.signal import chirp, sweep_poly, spectrogram, welch, hilbert
import matplotlib.pyplot as plt
import numpy as np
from pyhht.visualization import plot_imfs # Hilbert-Huang TF analysis
from pyhht import EMD    # Hilbert-Huang TF analysis

Fs=np.intc(1e6)             # receiver sample rate
frameSize=np.float(10e-3)   # seconds
dt = np.divide(1,Fs)        # seconds

# Generate test chirp using scipy.signal.sweep_poly
p = np.poly1d([0.025, -0.36, 0.25, 10000])
t = np.linspace(0, frameSize-dt, np.intc(frameSize*Fs))
sig = sweep_poly(t, p)

print(p)

plt.subplot(2, 1, 1)
plt.plot(t, sig)
plt.title("Sweep Poly\n with frequency ")
plt.subplot(2, 1, 2)
plt.plot(t, p(t), 'r', label='f(t)')
plt.legend()
plt.xlabel('t')
plt.tight_layout()

# Hilbert-Huang
decomposer = EMD(sig)
imfs = decomposer.decompose()
#plot_imfs(sig, imfs, t)

imfAngle = np.angle(hilbert(imfs))

# Calculate instantaneous frequency
instFreq = np.divide(np.gradient(imfAngle,t,axis=1), 2*np.pi)   # Not validated, looks right

# Calculate Hilbert spectrum
# Time, frequency, magnitude

intensity = np.absolute(hilbert(imfs))
plt.figure()
for i in range(np.size(instFreq,0)):
    plt.scatter(t, instFreq[i], c=intensity[i], alpha=0.3)

plt.title("Hilbert Spectrum")
plt.xlabel('t [s]')
plt.ylabel('f [Hz]')
plt.tight_layout()

plt.show()
