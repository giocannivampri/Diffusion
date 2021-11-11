from numpy.core.arrayprint import _guarded_repr_or_str
from numpy.core.function_base import linspace
from _initmio_ import make_correlated_noise, f, symplectic_map as sm
import numpy as np
import matplotlib.pyplot as plt
import crank_nicolson_numba.generic as cn
import scipy.integrate as integrate
np.random.seed(4)


    

def get_th(x, p):
    """Find theta given x and p"""
    th=[]
    for i in range(len(x)):
        th_1=np.arcsin(x[i]/np.sqrt((x[i] * x[i])+(p[i] * p[i])))
        th_2=np.arccos(p[i]/np.sqrt((x[i] * x[i])+(p[i] * p[i])))
        if np.sin(th_1) > 0 and np.cos(th_2) > 0 :
            th.append(th_1)
        if np.sin(th_1) > 0 and np.cos(th_2) < 0 :
            th.append(th_2)
        if np.sin(th_1) < 0 and np.cos(th_2) > 0 :
            th_4=(th_1 + (np.pi*2))
            th.append(th_4)
        if np.sin(th_1) < 0 and np.cos(th_2) < 0 :
         th_3=(np.pi - th_1)
         th.append(th_3)
         
    Th=np.array(th)
    return Th


def D_calculatore(I, delta=0.0001):
    """Estimates D value by using definitions given for stochastic map.
    
    Parameters
    ----------
    I : float
        sampling point
    epsilon : float
        noise coefficient
    
    Returns
    -------
    float
        diffusion value
    """
    #if abs(r_1/(delta + np.sqrt(2*I)))>1.0:
        
     #   return 0.0
    int_result = integrate.quad(
        (lambda th:
           1/epsilon_cn**2 *(np.sqrt(2*I)*np.cos(th) * f(np.sqrt(2*I)*np.sin(th), r_1, r_2) / (delta + np.sqrt(2*I)*np.sin(th)))**2),
        0.0,
        (np.pi / 2), epsabs=1e-15)
    # Check if int_result is valid, otherwise return 0.0
    #print(int_result[0], int_result[1],(int_result[1] / int_result[0] if int_result[0] != 0.0 else 0.0))

    return (int_result[0] * 0.5 * th_MAX**2 * r_2**2 * (beta/ emittance)/ (np.pi/2)
            if np.absolute(int_result[1] / int_result[0] if int_result[0] != 0.0 else 1.0) < 0.05 else 0.0)


#setting parameters
#scala=100
#epsilon = 1.0
epsilon_cn=0.1
r_1=3.59
r_2=7.18
radius=6.3245
th_MAX=  0.3*10**-6
emittance_star=2.5e-6
beta=280
gamma=(7000/0.938272088)
beta_rel=np.sqrt(1-(1/(gamma**2)))
emittance=emittance_star/(gamma*beta_rel)
#print(np.sqrt(beta/emittance))
#th_MAX=0.3*10**-6 * 0.0022 / (np.sqrt(beta*emittance)*r_2)
#omega_0x=0.31 * 2 * np.pi
omega_0x=1.1
#omega_1x= -1.73e5 * 2*np.pi * 2*emittance_star/(beta_rel*gamma)
omega_1x=0.91
omega_2x=0.1
#omega_2x= -1.87e12 * 2*np.pi * (2*emittance_star/(beta_rel*gamma))**2
print(omega_0x, omega_1x, omega_2x)
iterations=10**5 
n_particle=10**8
mean_x=0.0
sigma_x=1.0
mean_p=0.0
sigma_p=1.0
Max_I=radius**2 * 0.5
n_step=3000
Min_I=r_1**2 * 0.5
#Min_I=6.125

#creating initial distributions
#u=np.random.normal(1.0, 0.2, n_particle)
#th0=np.random.uniform(0.0, np.pi*2, n_particle)
x0=np.random.normal(mean_x, sigma_x, n_particle)
p0=np.random.normal(mean_p, sigma_p, n_particle)
I0=(x0**2 + p0**2)*0.5
p0=(p0)[I0>Min_I]
x0=(x0)[I0>Min_I]
#ro=cn.normed_normal_linspace(0.0, 4.5, 1.0, 0.2, 3000)
#j=cn.action(x0, p0)
#x0=cn.x_from_I_th(u, th0)
#p0=cn.p_from_I_th(u, th0)
azione=np.linspace(Min_I, Max_I, n_step, endpoint=False)
#x=np.array(np.sqrt(azione*2))
diffusione=np.zeros(n_step)



#calculating Diffusion Function
for i in range(len(diffusione)):
    diffusione[i]= D_calculatore(azione[i])
    #print(diffusione[i])




#greatly exaggerated way to find action distribution
def Fj(j, delta=0.0001):
    t= (cn.normed_normal_distribution(np.sqrt(j*2), mean_x, sigma_x)/(delta**2+np.sqrt(j*2)) + cn.normed_normal_distribution(-np.sqrt(j*2), mean_x, sigma_x)/(delta**2+np.sqrt(j*2 )))
    return t

def Fq(q, delta=0.0001):
    t= (cn.normed_normal_distribution(np.sqrt(q*2), mean_p, sigma_p)/(delta**2+np.sqrt(q*2)) + cn.normed_normal_distribution(-np.sqrt(q*2), mean_p, sigma_p)/(delta**2+np.sqrt(q*2 )))
    return t

def Distribution_I(I):
    """metodo alternativo e eccessivo per trovare la distribuzione di I"""
    
    int_result = integrate.quad(
        (lambda j:
           Fj(j)*Fq(I-j)) ,
        0.0, I
        )
    # Check if int_result is valid, otherwise return 0.0
    #print(int_result[0], int_result[1],(int_result[1] / int_result[0] if int_result[0] != 0.0 else 0.0))

    return (int_result[0]
            if np.absolute(int_result[1] / int_result[0] if int_result[0] != 0.0 else 1.0) < 0.05 else 0.0)



#Finding action distribution
def Idist():
    result= integrate.quad(lambda x:  np.exp(-x), Min_I, Max_I)
    return result[0]

normaliz=Idist()
ro=np.zeros(len(azione))
for i in range(len(azione)):
    ro[i]= np.exp(-azione[i])/normaliz



#initializing CN
dt=0.002
tempo_giro=0.0000909
motore=cn.cn_generic(Min_I, Max_I, ro, dt, D_calculatore)
times, current = motore.current(int(iterations*epsilon_cn**2 / dt), 1)
#motore.iterate(iterations)
a, stat=motore.get_data_with_x()



#initializing discteete map
mappa = sm.genera_istanza(omega_0x, omega_1x, omega_2x, 1.0, r_1, r_2, th_MAX, radius, x0, p0)
#print(make_correlated_noise(iterations))
mappa.compute_comon_noise(make_correlated_noise(iterations))
#mappa.compute_personale_noise(iterations)
y=mappa.get_filtered_action()
x, p, t = mappa.get_filtered_data()
th0=get_th(x0, p0)
th=get_th(x, p)
tempi_mappa, corrente_mappa = mappa.current_binning(3000)
#print(len(t))
#g=get_th(x0, p0)



print(len(x0)-len(x))
loss_cn=motore.get_particle_loss()*len(x0)
print(loss_cn)
#per=(len(x0)-len(x))/n_particle
#print(per)



#plotting
fig, ax=plt.subplots()
fig, bx=plt.subplots()
fig, cx=plt.subplots()
fig, dx=plt.subplots()
fig, ex=plt.subplots()
fig, sx=plt.subplots()



#theta distributions
ex.set_xlabel("Theta")
ex.hist(th0, 100, density=False)
ex.hist(th, 100, None, False, None, False, None, 'step')

#currents from CN and map
dx.set_xlabel("times")
dx.set_ylabel("Current")
dx.plot(tempi_mappa, corrente_mappa)
dx.plot(times/epsilon_cn**2, current*len(x0)*epsilon_cn**2)

#Diffusion as a function of I
cx.set_xlabel("I")
cx.set_ylabel("D")
cx.plot(azione, diffusione)

#action distributions from CN and map
norm=motore.get_sum()
stat=stat/norm
Io=r_1**2 /2.0
b=np.array([Io, Io, Io, Io])
stati=np.array([0.0, 0.3, 0.7, 1.0])
ax.set_xlabel("I")
ax.set_ylabel("\u03C1")
ax.set_yscale('log')
ax.hist(y, 30, None, True)
ax.plot(a, stat)
#ax.plot(azione, ro)
#ax.plot(b, stati)

#x distribution
bx.set_xlabel("x")
bx.hist(x0, 40, None, True)
bx.hist(x, 80, None, True, None, False, None, 'step')


#phase space
xf=(x)[(x**2 + p**2)*0.5 > 8.0]
pf=(p)[(x**2 + p**2)*0.5 > 8.0]
sx.set_xlabel("x")
sx.set_ylabel("P")
sx.plot(xf, pf, "r.")

plt.show()