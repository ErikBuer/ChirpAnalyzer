from scipy.signal import chirp, sweep_poly, spectrogram, welch
from scipy.special import factorial
import scipy.signal as signal
import matplotlib.pyplot as plt
plt.style.use('masterThesis')
import matplotlib as mpl

pgf=False
imagePath = '../figures/'

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

N=10
def fun(omega,N):
    return np.divide(np.sin(N*omega/2),np.sin(omega/2))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

k = np.linspace(-5,5,500)
omega = np.multiply(2*np.pi/N,k)
y = np.abs(fun(omega,N))
ax.plot(k,y)
    
ax.ticklabel_format(useMathText=True, scilimits=(0,3))
ax.set_xlabel("DFT Bins")
ax.set_ylabel("Magnitude")
plt.tight_layout()
if pgf==True:
    plt.savefig(imagePath+'dftResponse'+'.png', bbox_inches='tight')
    plt.savefig(imagePath+'dftResponse'+'.pgf', bbox_inches='tight')

plt.show()