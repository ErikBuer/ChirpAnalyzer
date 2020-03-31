def NLFM( t_i=0.01, f_c=20e3, Delta=20e3 ):
    """
    Generate test chirp usingg the synthesis formula of:
    C. Lesnik, Nonlinear Frequency Modulated Signal Design, 2009

    t_i is the chirp duration [s]
    f_c is the chirp center frequency [Hz]
    Delta is the total frequency span [Hz]?
    """

    dt = np.divide(1,Fs)        # seconds
    t = np.linspace(0, t_i-dt, np.intc(t_i*Fs)) # time vector (variable in CT representation)
    theta = np.divide(t_i*np.sqrt(np.power(Delta,2)+4), 2*Delta) - np.power(  np.divide( np.power(t_i,2)*(np.power(Delta,2)+4) , 4*np.power(Delta,2) ) - np.power( t - np.divide(t_i, 2), 2 ), 0.5 )
    A = 10 # Magnitude
    sig = A*np.exp(1j*2*np.pi*theta)
    sig = radar.upconvert( sig, f_c, Fs )
    f = ( t - np.divide(t_i, 2) )* np.power(  np.divide( np.power(t_i,2)*(np.power(Delta,2)+4) , 4*np.power(Delta,2) ) - np.power( t - np.divide(t_i, 2), 2 ), -0.5 )


"""
t_i = 1 #Chirp duration
deltaFrequency = 1e3
centerFrequency = 20e3

thrdOrdRate = 10e3
linRate     = deltaFrequency/t_i

# Generate test chirp using scipy.signal.sweep_poly
p = np.poly1d([(1/t_i)*thrdOrdRate, 0, linRate, centerFrequency])
t = np.linspace(-t_i/2, t_i/2-dt, np.intc(Fs*t_i))
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