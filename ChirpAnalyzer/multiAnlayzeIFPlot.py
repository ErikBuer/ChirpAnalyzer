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
plt.style.use('masterThesis')
from matplotlib import cm
colorMap = cm.coolwarm
import matplotlib

import pickle
import os

iterations = 1

# Write to binary file
path = '../jobs/'
filename = 'job'
destination = path + filename + str(iterations) + '.pkl'
# Save job to binary file
with open(destination,'rb') as f:
    m_analysis = pickle.load(f)

fig, ax = m_analysis.plotResults(pgf=True, scale = 'semilogy')
ax.legend(loc='upper right')
imagePath = '../figures/IFestimation/'
fileName = m_analysis.name +'_'+ str(iterations) + '_iterations' # str(m_analysis.iterations)
plt.savefig(imagePath + fileName + '.png', bbox_inches='tight')
plt.savefig(imagePath + fileName + '.pgf', bbox_inches='tight')

plt.show()