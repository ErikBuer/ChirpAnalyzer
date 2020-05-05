from scipy.signal import chirp, sweep_poly, spectrogram, welch
from scipy.special import factorial
import scipy.signal as signal
import numpy as np
import random

import rftool.radar as radar
import rftool.utility as util
import rftool.estimation as estimate
import rftool.communications as comm
from utility import *

import matplotlib.pyplot as plt
import matplotlib as mpl

import pickle
#import os

Debug = False

Fs = np.intc(802e3) # Receiver sample rate. #! Must be the same as the signals
T = np.float(6e-3)  # Pulse duration.       #! Must be the same as the signals
nIterations = 1000
packetSize = 32

# Load alpha window function a-priori
path = '../jobs/'
filename = 'SCD_GMM'
destination = path + filename + '.pkl'
with open(destination,'rb') as f:
    alphaWindow = pickle.load(f)


# Wrapper for estimation function 
def symbolRateEstimator(sig, Fs, **kwargs):
    SCD, f, alpha = estimate.FAM(sig, Fs = Fs, plot=False, method='conj', scale='linear', **kwargs)
    fCenter, R_symb = estimate.cyclicEstimator( SCD, f, alpha, bandLimited=True , **kwargs)
    return R_symb

# Configure estimators
estimators = []
estimators.append(estimator('Cyclic MLE Method', symbolRateEstimator, Fs=Fs))
estimators.append(estimator('Cyclic MLE A-Priori $f_c$', symbolRateEstimator, Fs=Fs, alphaWindow=alphaWindow))
estimators.append(estimator('Cyclic MLE A-Priori $f_c$, $\Omega$', symbolRateEstimator, Fs=Fs, alphaWindow=alphaWindow, fWindow='triangle', fWindowWidthHertz=50e3))

# Create analysis object
m_analysis = analysis('Center_Frequency_Estimation', estimators=estimators, lossFcn='MAE')

# Generate Eb/N0 range for statistics gathering.
EbN0Start = 40
EbN0End = 10

m_analysis.axis.displayName = '$E_b/N_0$ [dB]'
m_analysis.axis.displayVector = np.linspace(EbN0End, EbN0Start, EbN0Start-EbN0End)
m_analysis.axis.name = 'S/N [dB]'
m_analysis.axis.vector = comm.EbN0toSNRdB(m_analysis.axis.displayVector, 2, Fs, 1/T)
m_analysis.analyze(iterations=nIterations, parameter='symbolRate', packetSize=packetSize, debug=Debug)

"""# Write to binary file
path = '../jobs/'
filename = 'Job'
destination = path + filename + str(m_analysis.iterations) + '.pkl'
# Save job to binary file
with open(destination,'wb') as f:
    pickle.dump(m_analysis, f)


# Read from binary file
path = '../jobs/'
filename = 'symbolRateJob'
destination = path + filename + str(iterations) + '.pkl'
with open(destination,'rb') as f:
    m_analysis = pickle.load(f)"""

# Plot results
import matplotlib.pyplot as plt
plt.style.use('masterThesis')
import matplotlib

iterations = nIterations
fig, ax = m_analysis.plotResults(pgf=not Debug, scale='semilogy', plotYlabel='MAE [Hz]')
#ax.legend(loc='lower right')
#fig.set_figheight(2.5)
plt.tight_layout()
imagePath = '../figures/centerFreqEst/'
if Debug == False:
    fileName = m_analysis.name +'_'+ str(iterations) + '_iterations' # str(m_analysis.iterations)
    plt.savefig(imagePath + fileName + '.png', bbox_inches='tight')
    plt.savefig(imagePath + fileName + '.pgf', bbox_inches='tight')

plt.show()