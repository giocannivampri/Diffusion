import numpy as np
from numba import jit, njit, prange


@njit
def f (x, R_1, R_2):
    if abs(x) < R_1:
        return 0.0
        
    if abs(x) > R_2:
        return 1.0

    else :
        result= (x**2 - R_1**2)/(R_2**2 - R_1**2)
        return result

@njit
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
    np.random.seed(1)
    noise = np.random.normal(0.0, 1.0, n_elements)
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
        
        
        action = (x * x + px * px) * 0.5
        rot_angle = omega_0 + (omega_1 * action) + ( omega_2 * action * action)
        # rot_angle = np.random.rand() * np.pi * 2
        
        if (np.sqrt(action*2) >= barrier_radius):
            return 0.0, 0.0, i + start 
        
        temp1 = x
        temp2 = (px + (epsilon * noise[i] * TH_MAX * f(x, R_1, R_2) * R_2 * 914038.5712158077/ x ))
        x = np.cos(rot_angle) * temp1 + np.sin(rot_angle) * temp2
        px = -np.sin(rot_angle) * temp1 + np.cos(rot_angle) * temp2
        

    return x, px, i + start 

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