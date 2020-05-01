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
import multiAnlayzeCenterFreq

import matplotlib.pyplot as plt
plt.style.use('masterThesis')
import matplotlib

import pickle
#import os

# The number of iterations in the estimation. Must be the same as the pickle file name.
iterations = 10
# Read from binary file
path = '../jobs/'
filename = 'centerFrequencyJob'
destination = path + filename + str(iterations) + '.pkl'
# Save job to binary file
with open(destination,'rb') as f:
    m_analysis = pickle.load(f)

m_analysis.plotResults(pgf=False)

imagePath = '../figures/'
fileName = m_analysis.name +'_'+ str(iterations) + '_iterations' # str(m_analysis.iterations)
plt.savefig(imagePath + fileName + '.png', bbox_inches='tight')
plt.savefig(imagePath + fileName + '.pgf', bbox_inches='tight')

plt.show()