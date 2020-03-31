import numpy as np
import rftool.utility as util
import rftool.radar as radar
import joblib   # Parallelizations
import timeit
import pickle

class waveform:
    Fs = 0
    fCenter = 0
    fStart = 0
    fStop = 0
    polynomial = np.array([0])
    omega_t = np.array([0])

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
    """
    Class for managing estimation analysis result.

    name is the name of the parameter which is estimated
    lossFcn is the loss function 'MAE', 'MSE'
    """
    path = '../waveforms/'
    def __init__( self, name, estimators, lossFcn):
        self.name = name
        self.estimators = estimators
        self.lossFcn = lossFcn
        self.axis = axis()

    def analyze(self, iterations, parameter):
        """
        Iterate through the estimators and estimate.
        iterations is the number of iterations per SNR scenario.
        parameter is the parameter to be estimated as a string. Must be the same as the parametername in the wavefrom object. 
        """
        def estmate(i, SNRVector, estimator):
            time = np.empty_like(SNRVector)
            AE = np.empty_like(SNRVector)

            # Load from binary file
            filename = str(i)
            fileString = self.path + filename + ".pkl"

            with open(fileString,'rb') as f:
                m_waveform = pickle.load(f)

            FM = radar.chirp(m_waveform.Fs)
            FM.targetOmega_t = m_waveform.omega_t
            FM.points = len(m_waveform.omega_t)
            bitstream = np.random.randint(0, 2, 32)
            modSig = FM.modulate( bitstream )

            for m, SNR in SNRVector:
                package = util.wgnSnr( modSig, SNR)
                tic = timeit.default_timer()
                estimate = estimator.funtion(**self.kwargs)
                toc = timeit.default_timer()
                
                # Calculate Accumulated Absolute Error
                AE[m] = np.sum(np.abs(np.subtract(m_waveform.returnValue(parameter), estimate)))
                # calculate execution time
                time[m] = toc - tic
            return AE, time 

        for estimator in self.estimators:
            result = joblib.Parallel(n_jobs=1, verbose=0)(joblib.delayed(estmate)(i, self.axis.vector, estimator) for i in range(0, iterations))