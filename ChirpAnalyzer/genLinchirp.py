import scipy.signal as signal
import scipy.optimize as optimize
import scipy.integrate as integrate
import scipy.special as special
import scipy.ndimage as ndimage
import numpy as np
import numpy.polynomial.polynomial as poly

from mpl_toolkits.mplot3d import axes3d     # 3D plot
import matplotlib.pyplot as plt
plt.style.use('masterThesis')
import matplotlib as mpl

mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "lualatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

import rftool.LFM as LFM
import rftool.utility as util
import rftool.estimation as estimate

imagePath = '../figures/linFM/'

FM = LFM.chirp(30e3,5e-3, 1, 5e3, 4, direction='up')
FM.plotSymbols()

fileName = 'symbolIf'
plt.savefig(imagePath + fileName + '.png', bbox_inches='tight')
plt.savefig(imagePath + fileName + '.pgf', bbox_inches='tight')

FM.plotAutocorr()
fileName = 'symbolRxx'
plt.savefig(imagePath + fileName + '.png', bbox_inches='tight')
plt.savefig(imagePath + fileName + '.pgf', bbox_inches='tight')

FM.plotDotProd()
fileName = 'symbolDotProd'
plt.savefig(imagePath + fileName + '.png', bbox_inches='tight')
plt.savefig(imagePath + fileName + '.pgf', bbox_inches='tight')

# LoRa-like synchronization sequence
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

FM = LFM.chirp(300e3, (1/440), 1, 126e3, 8, direction='both')
dt = FM.dt

preamble = np.tile(FM.getSymbolIF(0), 8)
t = np.linspace(0, len(preamble)*dt, len(preamble))
ax.plot(t, preamble, label='Preamble')

frameSync = np.tile(FM.getSymbolIF(2), 2)
frameSync = np.append(preamble[-1], frameSync)
t = t[-1]+np.linspace(0, len(frameSync)*dt, len(frameSync))
ax.plot(t, frameSync, label='Frame Sync')

freqSync = np.tile(FM.getSymbolIF(4), 2)
quaterLen = np.intc(len(freqSync)/8)
freqSync = np.append(freqSync, freqSync[0:quaterLen-1])
freqSync = np.append(frameSync[-1], freqSync)
t = t[-1]+np.linspace(0, len(freqSync)*dt, len(freqSync))
ax.plot(t, freqSync, label='Frequency Sync')

ax.ticklabel_format(useMathText=True, scilimits=(0,3))
nSymb = np.ceil(t[-1]/FM.T)
xGrid = np.linspace(0, (nSymb-1)*FM.T, nSymb)
ax.set_xticks(xGrid)
#ax.tick_params(axis='x', which='major', pad=4)

ax.set_ylabel('f [Hz]')
ax.set_xlabel('t [s]')
plt.legend(loc='center left')
plt.grid(axis='x')
plt.tight_layout()

fileName = 'LoRaSyncWord'
plt.savefig(imagePath + fileName + '.png', bbox_inches='tight')
plt.savefig(imagePath + fileName + '.pgf', bbox_inches='tight')

plt.show()