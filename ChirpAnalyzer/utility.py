import numpy as np
import rftool.utility as util
import rftool.radar as radar
import rftool.estimation as estimate
import rftool.LFM as LFM
import joblib   # Parallelizations
import random
import timeit
import pickle

import matplotlib.pyplot as plt
import matplotlib as mpl

class waveform:
    Fs = 0
    fCenter = 0
    fStart = 0
    fStop = 0
    polynomial = np.array([0])
    omega_t = np.array([0])
    T = 0
    symbolRate = 0

    def returnValue(self, parameter):
        """
        Returns the true parameter of a waveform.
        """
        if parameter == 'Fs':
            value = self.Fs
        elif parameter == 'fCenter':
            value = self.fCenter
        elif parameter == 'fStart':
            value = self.fStart
        elif parameter == 'fStop':
            value = self.fStop
        elif parameter == 'polynomial':
            value = self.polynomial
        elif parameter == 'omega_t':
            value = self.omega_t
        elif parameter == 'packet':
            value = self.packet
        elif parameter == 'symbolRate':
            value = self.symbolRate
        else:
            value = None
        return value

class axis:
    displayName = 'displayName'
    displayVector = []
    name = 'SNR'
    vector = []

class estimator:
    errorMat = []
    iterations = 0
    meanTime = None
    def __init__( self, name, function, **kwargs):
        """
        Class for managing estimators in the analysis.

        name is the name of the method it will be passed on as labal in an eventual plot.
        function is the function which performes the estimation
        **kwargs ar keyword arguments to be passed to the estimatorm, in addition to the signal.
        """
        self.name = name
        self.function = function
        self.kwargs = kwargs

class analysis:
    path = '../waveforms/'
    def __init__( self, name, lossFcn, estimators):
        """
        Class for managing estimation analysis result.

        name is the name of the parameter which is estimated
        estimators are objects of the class estimator.
        lossFcn is the loss function 'MAE', 'MSE'
        """
        self.name = name
        self.estimators = estimators
        self.lossFcn = lossFcn
        self.axis = axis()
        

    def analyze(self, iterations, parameter, signalType='NLFM', **kwargs):
        """
        Iterate through the estimators and estimate.
        iterations is the number of iterations per SNR scenario.
        parameter is the parameter to be estimated as a string. Must be the same as the parametername in the wavefrom object. 
        signalType decides which signal type is to be applied. 'NLFM' uses the radar.chirp object, 'LFM' uses the LFM.chirp object for signal generation.
        """
        packetSize = kwargs.get('packetSize', 1)

        debug = kwargs.get('debug', False)
        self.iterations = iterations
        m_waveform = kwargs.get('m_waveform')

        def estimate(i, SNRVector, estimator, packetSize=1, **kwargs):
            print(estimator.name, 'iteration',i)

            time = np.empty_like(SNRVector)
            AE = np.empty_like(SNRVector)

            if signalType == 'NLFM':
                # Load from binary file
                filename = str(i)
                fileString = self.path + filename + ".pkl"

                with open(fileString,'rb') as f:
                    m_waveform = pickle.load(f)

                FM = radar.chirp(m_waveform.Fs)
                FM.targetOmega_t = m_waveform.omega_t
                FM.points = len(m_waveform.omega_t)

                if 1<packetSize:
                    bitstream = np.random.randint(0, 2, packetSize)
                    m_waveform.packet=bitstream
                    sig_t = FM.modulate( bitstream )
                else:
                    sig_t = FM.genNumerical()
                    m_waveform.packet=1

            elif signalType == 'LFM':
                m_waveform = kwargs.get('m_waveform')
                # Fixed BW, Random Center Frequency
                fStart = random.uniform(10e3,100e3)
                fStop = fStart+50e3
                fCenter = fStop-(fStop-fStart)/2

                m_waveform.fCenter = fCenter
                m_waveform.fStart = fStart
                m_waveform.fStop = fStop

                FM = LFM.chirp(Fs=m_waveform.Fs, T=m_waveform.T, fStart=fStart, fStop=fStop, nChirps=8, direction='both') # Both directions and 8 symbols are configured in order to generate the sync sequence.

                if 1<packetSize:
                    addSyncSeq = kwargs.get('syncSeq', False)
                    if addSyncSeq == True:
                        synqSeqSymbols = np.array([0,0,0,0,0,0,0,0,1,1,4,4,4])
                        symbolStream = np.random.randint(0, 4, packetSize-len(synqSeqSymbols))
                        synqSeq = FM.modulate( synqSeqSymbols )
                        # Remove the last three quartes of a symbol
                        clipLen = np.intc(FM.Fs*FM.T*0.75)
                        synqSeq = synqSeq[:-clipLen]
                        # Add a random packet to the sync sequence
                        sig_t = FM.modulate( symbolStream )
                        sig_t = np.append(synqSeq, sig_t)
                    else:
                        symbolStream = np.random.randint(0, 4, packetSize)
                        m_waveform.packet=symbolStream
                        sig_t = FM.modulate( symbolStream )
                else:
                    sig_t = FM.getSymbolSig()
                    m_waveform.packet=1

            for m, SNR in enumerate(SNRVector):
                sig_noisy_t = util.wgnSnr( sig_t, SNR)
                tic = timeit.default_timer()
                # cleanSig and SNR is passed to enable CRLB calculation, fCenter for a-priori freq info for some estimators
                estimate = estimator.function(sig_noisy_t, **estimator.kwargs, SNR=SNR, cleanSig=sig_t, fCenterPriori=m_waveform.fCenter)
                toc = timeit.default_timer()
                
                # Calculate Accumulated Absolute Error
                # For packets, the inverse of the bitstream is accepted.
                if parameter=='packet':
                    AE_temp = []
                    AE_temp.append(np.sum(np.abs(np.subtract(m_waveform.returnValue(parameter), estimate))))
                    AE_temp.append(np.sum(np.abs(np.subtract(m_waveform.returnValue(parameter), abs(1-estimate)))))
                    AE[m] = np.min(AE_temp)
                    # If the error is greater than 50% then (for binary symbols) the packet is broken.
                    if len(estimate)/2<AE[m]:
                         AE[m]=len(estimate)/2
                elif 'CRLB' in estimator.name:
                    AE[m] = estimate
                else:
                    AE[m] = np.sum(np.abs(np.subtract(m_waveform.returnValue(parameter), estimate)))
                # calculate execution time
                time[m] = toc - tic
            return AE, time 

        for estimator in self.estimators:
            # CRLB calculation, only run once for each SNR
            if 'CRLB' in estimator.name:
                estimator.errorMat = estimate(i=0, SNRVector=self.axis.vector, estimator=estimator, packetSize=packetSize)
                estimator.meanTime = np.nan
                estimator.iterations = np.nan
            else:
                if debug == True:
                    result = joblib.Parallel(n_jobs=1, verbose=10)(joblib.delayed(estimate)(i, self.axis.vector, estimator, **kwargs) for i in range(0, iterations))
                else:
                    result = joblib.Parallel(n_jobs=8, verbose=0)(joblib.delayed(estimate)(i, self.axis.vector, estimator, **kwargs) for i in range(0, iterations))
                result = np.array(result)
                error = result[:,0,:]
                time = result[:,1,:]

                estimator.errorMat = error
                estimator.meanTime = np.mean(time)
                estimator.iterations = iterations
    
    
    def plotResults(self, pgf=False, **kwargs):
        """
        Function for plotting the estimated data
        kwarg: scale defines the scale of the y-axis for the plot. allowed values: linear, semilogy, semilogx, dBmag
        """
        scale = kwargs.pop('scale', 'linear')
        plotYlabel = kwargs.pop('plotYlabel', self.lossFcn)

        if pgf==True:
            mpl.use("pgf")
            mpl.rcParams.update({
                "pgf.texsystem": "pdflatex",
                'font.family': 'serif',
                'text.usetex': True,
                'pgf.rcfonts': False,
            })

        fig, ax = plt.subplots()
        for index,estimator in enumerate(self.estimators):
            if 'CRLB' in estimator.name:    # If the name contains the string 'CRLB'
                newParam = {'linestyle':'--'}
                kwargs = {**kwargs, **newParam}

            print(estimator.name, ', mean execution time:', estimator.meanTime*1000, '[ms]')
            estimator.meanError = np.mean(estimator.errorMat, axis=0)

            if scale=='linear':
                ax.plot(self.axis.displayVector, estimator.meanError, label=estimator.name, **kwargs)
                ax.ticklabel_format(useMathText=True, scilimits=(0,3))
            elif scale=='semilogy':
                ax.semilogy(self.axis.displayVector, estimator.meanError, label=estimator.name, **kwargs)
            elif scale=='semilogx':
                ax.semilogx(self.axis.displayVector, estimator.meanError, label=estimator.name, **kwargs)
            elif scale=='dBmag':
                ax.plot(self.axis.displayVector, util.mag2db(estimator.meanError), label=estimator.name, **kwargs)
                ax.ticklabel_format(useMathText=True, scilimits=(0,3))
            else:
                ax.loglog(self.axis.displayVector, estimator.meanError, label=estimator.name, **kwargs)

        ax.set_xlabel(self.axis.displayName)
        ax.set_ylabel(plotYlabel)
        ax.legend()
        # For the minor ticks, use no labels; default NullFormatter.
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        ax.grid(which='minor',axis='x', linestyle='--', )
        ax.grid()
        plt.tight_layout()
        return fig, ax