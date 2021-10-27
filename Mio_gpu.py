import numpy as np
from numba import njit, cuda
from numba.cuda import random
import math
from numba import float64, int32

@cuda.jit
def f (x, R_1, R_2):
    if abs(x) < R_1:
        return 0.0
        
    if abs(x) > R_2:
        return 1.0

    else :
        result= (x**2 - R_1**2)/(R_2**2 - R_1**2)
        return result



@cuda.jit
def symplectic_map_common(x, px, step_values, noise_array, epsilon, omega_0, omega_1, omega_2, R_1, R_2, TH_MAX, barrier_radius):
    i = cuda.threadIdx.x
    j = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    action = cuda.shared.array(shape=(512), dtype=float64)
    rot_angle = cuda.shared.array(shape=(512), dtype=float64)
    temp1 = cuda.shared.array(shape=(512), dtype=float64)
    temp2 = cuda.shared.array(shape=(512), dtype=float64)
    l_x = cuda.shared.array(shape=(512), dtype=float64)
    l_px = cuda.shared.array(shape=(512), dtype=float64)
    l_step = cuda.shared.array(shape=(512), dtype=int32)
    
    if j < x.shape[0]:
        l_x[i] = x[j]
        l_px[i] = px[j]
        l_step[i] = step_values[j]
        for k in range(noise_array.shape[0]):
            action[i] = (l_x[i] * l_x[i] + l_px[i] * l_px[i]) * 0.5
            rot_angle[i] = omega_0 + (omega_1 + action[i]) + (0.5* omega_2 * action[i] * action[i])

            if (l_x[i] == 0.0 and px[i] == 0.0) or np.sqrt(action[i]*2) >= barrier_radius:
                l_x[i] = 0.0
                l_px[i] = 0.0
                break

            temp1[i] = l_x[i]
            temp2[i] = (
                l_px[i] + (epsilon * noise_array[k] * TH_MAX * f(l_x[i], R_1, R_2) * R_2 * 914038.5712158077/ l_x[i] )
                
            )
            l_x[i] = math.cos(rot_angle[i]) * temp1[i] + \
                math.sin(rot_angle[i]) * temp2[i]
            l_px[i] = -math.sin(rot_angle[i]) * temp1[i] + \
                math.cos(rot_angle[i]) * temp2[i]

            l_step[i] += 1
        x[j] = l_x[i]
        px[j] = l_px[i]
        step_values[j] = l_step[i]


@cuda.jit
def symplectic_map_personal(x, px, step_values, n_iterations, epsilon, omega_0, omega_1, omega_2,  R_1, R_2, TH_MAX, barrier_radius, rng_states, gamma):
    i = cuda.threadIdx.x
    j = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    action = cuda.shared.array(shape=(512), dtype=float64)
    rot_angle = cuda.shared.array(shape=(512), dtype=float64)
    temp1 = cuda.shared.array(shape=(512), dtype=float64)
    temp2 = cuda.shared.array(shape=(512), dtype=float64)
    noise = cuda.shared.array(shape=(512), dtype=float64)
    l_x = cuda.shared.array(shape=(512), dtype=float64)
    l_px = cuda.shared.array(shape=(512), dtype=float64)
    l_step = cuda.shared.array(shape=(512), dtype=int32)

    noise[i] = random.xoroshiro128p_normal_float64(rng_states, j)

    if j < x.shape[0]:
        l_x[i] = x[j]
        l_px[i] = px[j]
        l_step[i] = step_values[j]
        for k in range(n_iterations):
            action[i] = (l_x[i] * l_x[i] + l_px[i] * l_px[i]) * 0.5
            rot_angle[i] = omega_0 + (omega_1 + action[i]) + \
                (0.5* omega_2 * action[i] * action[i])

            if (l_x[i] == 0.0 and l_px[i] == 0.0) or np.sqrt(action[i]*2) >= barrier_radius:
                l_x[i] = 0.0
                l_px[i] = 0.0
                break
            
            temp1[i] = l_x[i]
            temp2[i] = (
                l_px[i] + (epsilon * noise[i] * TH_MAX * f(l_x[i], R_1, R_2) * R_2 * 914038.5712158077/ l_x[i] )
            
            )
            l_x[i] = math.cos(rot_angle[i]) * temp1[i] + \
                math.sin(rot_angle[i]) * temp2[i]
            l_px[i] = -math.sin(rot_angle[i]) * temp1[i] + \
                math.cos(rot_angle[i]) * temp2[i]

            l_step[i] += 1

            noise[i] = random.xoroshiro128p_normal_float64(rng_states, j) + gamma * noise[i]
        x[j] = l_x[i]
        px[j] = l_px[i]
        step_values[j] = l_step[i]
