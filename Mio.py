import numpy as np
from numba import jit, njit, prange


def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real

def band_limited_noise(min_freq, max_freq, samples=2**20, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    return fftnoise(f)


"""def f (x, R_1, R_2):
    
        
    if abs(x) > R_2:
        return 1.0

    else :
        result= (x**2 - R_1**2)/(R_2**2 - R_1**2)/ (1 + np.exp(((R_1+0.3)-x)/0.1))
        return result"""

@njit
def f (x, R_1, R_2):
    if abs(x) < R_1:
        return 0.0
        
    if abs(x) > R_2:
        return 1.0

    else :
        result= (x**2 - R_1**2)/(R_2**2 - R_1**2)
        return result


def make_correlated_noise(n_elements, gamma=0.0):
    """Make an array of correlated noise
    
    Parameters
    ----------
    n_elements : unsigned int
        number of elements
    gamma : float, optional
        correlation coefficient, by default 0.0
    
    Returns
    -------
    ndarray
        the noise array
    """
#    correl=5
 #   noise=np.empty(0)
  #  for s in range(int(n_elements/correl)):
   #     turn=np.full(correl, np.random.binomial(1, 0.5, 1))
    #    noise=np.append(noise, turn)  
    #noise=band_limited_noise(0.305, 0.31, n_elements)
    #print(np.mean(noise)) 
    #print(noise)
    #np.random.seed(1)
    #noise=np.full(n_elements, 0.5)
    #noise=np.zeros(n_elements)
    #noise=np.random.rand(n_elements)
    # noise = np.random.normal(0.0, 1.0, n_elements)
    noise = np.random.binomial(1, 0.5, n_elements)
    
    #print(len(noise))
    if gamma != 0.0:
        for i in range(1, n_elements):
            noise[i] += gamma * noise[i - 1]
    return noise




@njit
def iterate(x, px, noise, epsilon, omega_0, omega_1, omega_2, R_1, R_2, TH_MAX, barrier_radius, start):
    """internal iteration method for symplectic map
    
    Parameters
    ----------
    x : float
        x0
    px : float
        px0
    noise : ndarray
        array of noise values
    epsilon : float
        epsilon value
    alpha : float
        alpha exponential
    beta : float
        beta exponential
    x_star : float
        nek coefficient
    delta : float
        delta coefficient for fix
    omega_0 : float
        omega 0
    omega_1 : float
        omega 1
    omega_2 : float
        omega 2
    barrier_radius : float
        barrier position 
    start : unsigned int
        starting iteration value
    
    Returns
    -------
    (float, float, unsigned int)
        (x, px, iterations)
    """    
    
    
    for i in range(len(noise)):
        
        # rot_angle = np.random.rand() * np.pi * 2
        temp1 = x
        temp2 = (px + (epsilon * 
        #((2*noise[i]-1)*sq + 0.5)
         noise[i]
         * TH_MAX * f(x, R_1, R_2) * R_2 * 914099.8357243269 / (x + 10**-16) ))
        action = (temp1 * temp1 + temp2 * temp2) * 0.5
        if (np.sqrt(action*2) >= barrier_radius):
        #if (abs(x)) >= barrier_radius:
            return 0.0, 0.0, i + start 


        rot_angle = omega_0 + (omega_1 * action) + (0.5*  omega_2 * action * action)
        x = np.cos(rot_angle) * temp1 + np.sin(rot_angle) * temp2
        px = -np.sin(rot_angle) * temp1 + np.cos(rot_angle) * temp2
        action = (x * x + px * px) * 0.5
        if (np.sqrt(action*2) >= barrier_radius):
        #if (abs(x)) >= barrier_radius:
            return 0.0, 0.0, i + start 
        

    return x, px, i + start +1

@njit(parallel=True)
def symplectic_map_personal(x, px, step_values, n_iterations, epsilon, omega_0, omega_1, omega_2, R_1, R_2, TH_MAX, barrier_radius, gamma=0.0):
    """computation for personal noise symplectic map
    
    Parameters
    ----------
    x : ndarray
        x initial condition
    px : ndarray
        px initial condition
    step_values : ndarray
        iterations already performed
    n_iterations : unsigned int
        number of iterations to perform
    epsilon : float
        epsilon value
    omega_0 : float
        ipse dixit
    omega_1 : float
        ipse dixit
    omega_2 : float
        ipse dixit
    R_1 : float
        inner radius
    R_2 : float
        outer radius
    TH_MAX : fload
        theta max value
    barrier_radius : float
        barrier radius
    f : func
        function defined in the simplectic map
    gamma : float, optional
        correlation coefficient, by default 0.0
    
    Returns
    -------
    (ndarray, ndarray, ndarray)
        x, px, step_values
    """    
    for i in prange(len(x)):
        personal_noise = make_correlated_noise(n_iterations, gamma)
        x[i], px[i], step_values[i] = iterate(x[i], px[i], personal_noise, epsilon, omega_0, omega_1, omega_2, R_1, R_2, TH_MAX, barrier_radius, step_values[i])
    return x, px, step_values 



@njit(parallel=True)
def symplectic_map_common(x, px, step_values, noise_array, epsilon, omega_0, omega_1, omega_2, R_1, R_2, TH_MAX, barrier_radius, gamma=0.0):
    """computation for personal noise symplectic map

    Parameters
    ----------
    x : ndarray
        x initial condition
    px : ndarray
        px initial condition
    step_values : ndarray
        iterations already performed
    noise_array : ndarray
        noise array for the whole group
    epsilon : float
        epsilon value
    omega_0 : float
        ipse dixit
    omega_1 : float
        ipse dixit
    omega_2 : float
        ipse dixit
    R_1 : float
        inner radius
    R_2 : float
        outer radius
    TH_MAX : fload
        theta max value
    barrier_radius : float
        barrier radius
    gamma : float, optional
        correlation coefficient, by default 0.0

    Returns
    -------
    (ndarray, ndarray, ndarray)
        x, px, step_values
    """
    for i in prange(len(x)):
        x[i], px[i], step_values[i] = iterate(x[i], px[i], noise_array, epsilon, omega_0, omega_1, omega_2, R_1, R_2, TH_MAX, barrier_radius,  step_values[i])
    return x, px, step_values 