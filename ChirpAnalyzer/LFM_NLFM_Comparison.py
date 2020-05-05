from scipy.signal import chirp, sweep_poly, spectrogram, welch
from scipy.special import factorial
import scipy.signal as signal
import matplotlib.pyplot as plt
plt.style.use('masterThesis')
import matplotlib as mpl

pgf=True
imagePath = '../figures/cycloDemo/'

if pgf==True:
    mpl.use("pgf")
    mpl.rcParams.update({
        "pgf.texsystem": "lualatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    
import numpy as np
import rftool.radar as radar
import rftool.utility as util
import rftool.estimation as estimate

Fs=np.intc(800e3)   # receiver sample rate
T=np.float(4e-3)    # Pulse duration
points = np.intc(Fs*T)
t = np.linspace(0, T, points)

targetBw=100e3      # Pulse BW
centerFreq=100e3    # Pulse center frequency

# Time domain window for NLFM generation
NLFM = radar.chirp(Fs)
NLFM.fftLen = 2048


# Synthesize the target autocorrelation function
#indow_t = signal.chebwin(np.intc(2048), 60)
window_t = signal.hamming(np.intc(2048))
#window_t = signal.gaussian(np.intc(2048), 360)
#window_t = signal.gaussian(np.intc(2048), 400)
NLFM.getCoefficients( window_t, targetBw=targetBw, centerFreq=centerFreq, T=T)

# Time domain window for LFM generation
LFMsig = signal.chirp(t, centerFreq-(targetBw/2), t[-1], centerFreq+(targetBw/2), method='linear')
"""plt.figure()
plt.plot(t, LFMsig)"""

NLFMsig = NLFM.genFromPoly()
signals = np.stack((LFMsig, NLFMsig), axis=-1)

"""radar.ACF(signals, label=['LFM', 'NLFM'])
plt.savefig(imagePath+'ACF_compare'+'_pgf_'+str(pgf)+'.png', bbox_inches='tight')
plt.savefig(imagePath+'ACF_compare'+'_pgf_'+str(pgf)+'.pgf', bbox_inches='tight')"""


bitStream = np.random.randint(0, 2, 32)
modSig = NLFM.modulate(bitStream)
#modSig = util.wgnSnr( modSig, -40 )

SCD, f, alpha = estimate.FAM(modSig, Fs = Fs, plot=False, method='conj', scale='linear')
estimate.cyclicEstimator( SCD, f, alpha, bandLimited=True )


SCDplt = np.abs(SCD)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# Plot for positive frequencies
im = ax.pcolormesh(alpha, f[np.intc(len(f)/2):-1], SCDplt[np.intc(len(f)/2):-1,:], edgecolors='none')
ax.ticklabel_format(useMathText=True, scilimits=(0,3))
#plt.title("Spectral Correlation Density")
ax.set_xlabel("alpha [Hz]")
ax.set_ylabel("f [Hz]")

fig.colorbar(im)
plt.tight_layout()

if pgf==True:
    plt.savefig(imagePath+'SCD_NLFM_32'+'_pgf_'+str(pgf)+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'SCD_NLFM_32'+'_pgf_'+str(pgf)+'.pgf', bbox_inches='tight')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(alpha, np.abs(SCD[144,:]))
ax.ticklabel_format(useMathText=True, scilimits=(0,3))
ax.set_xlabel("alpha [Hz]")
ax.set_ylabel("Correlation")
if pgf==True:
    plt.savefig(imagePath+'S_fc'+'_pgf_'+str(pgf)+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'S_fc'+'_pgf_'+str(pgf)+'.pgf', bbox_inches='tight')

plt.show()