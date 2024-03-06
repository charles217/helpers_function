import modulesForCalibration as mfc

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import scipy.integrate as integrate
import pandas as pd

from scipy.optimize import fmin, fmin_bfgs, minimize
from scipy.stats import norm

import cmath
import math

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
%matplotlib inline

from datetime import datetime
from tqdm import tqdm
from matplotlib import cm


# 1

'''
kappa  = params[0];
theta  = params[1];
sigma  = params[2];
rho    = params[3];
v0     = params[4];
'''
S0 = 4100
r = 0.0485
mu = r
params = [0.02, 1.5, 0.05, 0.18, 0.5, 0.04]

N = 10*252 # 10 years * 252 trading day per year
T = 10
dt = T/N

def heston_simulated_prices(params, N, T, S0, r, mu, plot = False):
    kappa  = params[0]
    theta  = params[1]
    sigma  = params[2]
    rho    = params[3]
    v0     = params[4]
    
    # Define discretization parameters
    dt = T/N        # time increment
    M = 1           # number of simulations
    print('T:',T,'  N:', N, '  dt:', dt)
    print(v0)

    # Generate random numbers
    Z1 = norm.rvs(size=(N, M))
    Z2 = rho*Z1 + np.sqrt(1-rho**2)*norm.rvs(size=(N, M))

    # Define arrays to store stock price and volatility paths
    S = np.zeros((N+1, M))
    v = np.zeros((N+1, M))

    # Set initial values
    S[0,:] = S0
    v[0,:] = v0 #theta
    print(v0)
    
    # Calculate paths
    for i in range(N):
        v[i+1,:] = np.maximum(0, v[i,:] + kappa*(theta-v[i,:])*dt + sigma*np.sqrt(v[i,:])*np.sqrt(dt)*Z1[i,:])
        #print(v[i+1,:])
        S[i+1,:] = S[i,:] * np.exp((mu - 0.5*v[i,:])*dt + np.sqrt(v[i,:])*np.sqrt(dt)*Z2[i,:])
        #print(S[i+1,:])
    
    # Plot results
    if plot == True:
        plt.plot(S)
        plt.title('Simulated Heston Model Stock Price Path')
        plt.xlabel('Time Steps')
        plt.ylabel('Stock Price')
        plt.show()
    
        plt.plot(v)
        plt.title('Simulated Heston Model volatility Path')
        plt.xlabel('Time Steps')
        plt.ylabel('Volatility')
        plt.show()
        
    # Reshaping the outputs
    y = np.log(S)
    S = S.T
    S = S[0]
    v = v.T
    v = v[0]
    y = y.T
    y = y[0]
    
    return S, v, y
    

#prices, v_, y = heston_simulated_prices(params, N, T, S0, r, mu, plot = True)
S0 = 4100
r = 0.0485
mu = r
params = [mu, 1.5, 0.05, 0.18, 0.5, 0.04]
#.        mu, kappa, theta, lbda, rho , v0
N = 500
T = 10

params_sim = np.zeros(5)
params_sim[0] = params[1]# kappa  
params_sim[1] = params[2]#theta  
params_sim[2] = params[3]#sigma  
params_sim[3] = params[4]#rho   
params_sim[4] = params[5]#v0  
print(params_sim)
prices, v_, y = heston_simulated_prices(params_sim, N, T, S0, r, mu, plot = True)

#2 
def Extended_Kalman_Filter(params):
    
    global y_EKF, v_EKF
    
    mu     = params[0]
    kappa  = params[1]
    theta  = params[2]
    lbda   = params[3]
    rho    = params[4]
    v0     = params[5]
    
    dt = T/N 
        
    P = np.matrix([[0.01, 0],[0, 0.01]])
    I = np.identity(2)
    F = np.matrix([[1, -1/2*dt],[0, 1-kappa*dt]])
    U = np.matrix([[np.sqrt(v0*dt), 0],[0, lbda*np.sqrt(v0*dt)]])
    Q = np.matrix([[1, rho],[rho, 1]])
    H = np.matrix([1,0])
    
    x_update = np.matrix([np.log(S0), v0]).T
    
    y_EKF = np.zeros(N)
    v_EKF = np.zeros(N)
    
    y_EKF[0] = np.log(S0)
    v_EKF[0] = v0
    
    func_obj = 0
    for i in range(1, N):
        
        v_pred = np.matrix([0,0], dtype=np.float64).T
        v_pred[0,0] = x_update[0,0] + (mu-1/2*x_update[1,0])*dt
        v_pred[1,0] = x_update[1,0] + kappa*(theta-x_update[1,0])*dt
        
        P_pred = F*P*F.T + U*Q*U.T
        
        A = H*P_pred*H.T
        
        A = A[0,0]
        
        err = y[i] - v_pred[0,0]
        
        func_obj += np.log(abs(A)) + err**2/A
        
        # Measurement

        K = P_pred*H.T/A

        x_update = v_pred + K*err
        
        # check if volatility not negative
        x_update[1,0] = max(1e-5, x_update[1,0]) 
        
        vk = x_update[1,0]
        
        U = np.matrix([[np.sqrt(vk*dt), 0],[0, lbda*np.sqrt(vk*dt)]])
        
        P = (I-K*H)*P_pred
        
        y_EKF[i] = x_update[0,0]
        v_EKF[i] = x_update[1,0]
        
    return func_obj/N

S0 = 4100
r = 0.0485
mu = r
params = [mu, 1.5, 0.05, 0.18, 0.5, 0.04]
#.        mu, kappa, theta, lbda, rho , v0
N = 10*252 # 10 years * 252 trading day per year
T = 10

params_sim = np.zeros(5)
params_sim[0] = params[1]# kappa  
params_sim[1] = params[2]#theta  
params_sim[2] = params[3]#sigma  
params_sim[3] = params[4]#rho   
params_sim[4] = params[5]#v0  
print(params_sim)
prices, v_, y = heston_simulated_prices(params_sim, N, T, S0, r, mu, plot = True)

dt = T/N
print(dt)

# true : [0.0485, 1.5, 0.05, 0.18, 0.5, 0.04]
#params_0 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.05]
#params_0 = [0.02, 1.5, 0.05, 0.18, 0.5, 0.1]
params_0 = [0.03, 1.3, 0.07, 0.3, 0.6, 0.06]

def opt_param_research():
    
    def callback(x):
        print("Current parameter vector:", x)

    
    #constraint1 = {'type': 'ineq', 'fun': lambda x: x}#[0] - 1}
    #constraint2 = {'type': 'ineq', 'fun': lambda x: 2 - x[0]}
    #constraints = [constraint1]#, constraint2]
    #bounds = [(0.000001, 2)]*6
    #bounds[1] = (0,2)
    #bounds[4] = (-1,1)
    xopt = minimize(Extended_Kalman_Filter, params_0, callback=callback, method='Nelder-Mead')
    #result_2 = minimize(obj_function_ext_KF_m, params_0, bounds=bounds, callback=callback)# method='Nelder-Mead',
    #xopt, fopt, _, _, _ = fmin(Extended_Kalman_Filter, params_0, maxiter=100, callback=callback, disp=True, retall=False, full_output=True)

    #result = fmin(Extended_Kalman_Filter, params_0, callback=callback)
    print(80*'=')
    print('Optimal parameter set:')
    print(xopt)
    print(80*'=')
    
    return xopt


result_EKF = opt_param_research()



plt.figure(figsize=(12,8))
years = np.arange(y.shape[-1]) * dt
plt.plot(years[1:], y[1:], label = 'Real simulated log Price', linewidth=0.8)
plt.plot(years[1:], y_EKF, label = 'EKF log Price', linewidth=0.8)
plt.plot()
plt.title('Evolution of the log Price')
plt.ylabel('Log(St)')
plt.xlabel('Years')
plt.legend()
plt.show()

plt.figure(figsize=(12,8))
plt.plot(years[1:], v_[1:], label = 'Real simulated volatility', linewidth=0.8)
plt.plot(years[1:], v_EKF, label = 'EKF volatility', linewidth=0.8)
plt.plot()
plt.title('Evolution of the volatility')
plt.ylabel('v')
plt.xlabel('Years')
plt.legend()
plt.show()

def proposal_distribution(N, v_prev, dy, params):
    mu, kappa, theta, lbda, rho = params
    # Calculate mean and standard deviation of the proposal distribution
    m = v_prev + kappa * (theta - v_prev) * dt + lbda * rho * (dy - (mu - 1/2 * v_prev) * dt)
    s = lbda * np.sqrt(v_prev * (1 - rho**2) * dt)
    # Sample N particles from the proposal distribution
    return norm.rvs(m, s, N)

def likelihood(y, x, v_prev, y_prev, params):
    mu, kappa, theta, lbda, rho = params
    # Calculate the mean and standard deviation of the measurement distribution
    m = y_prev + (mu - 1/2 * x) * dt
    s = np.sqrt(v_prev * dt)
    # Calculate the likelihood
    return norm.pdf(y, m, s)

def transition_proba(x, v_prev, params):
    mu, kappa, theta, lbda, rho = params
    # Calculate the mean and standard deviation of the transition distribution
    m = 1 / (1 + 1/2 * lbda * rho * dt) * (v_prev + kappa * (theta - v_prev) * dt + 1/2 * lbda * rho * v_prev * dt)
    s = 1 / (1 + 1/2 * lbda * rho * dt) * lbda * np.sqrt(v_prev * dt)
    # Calculate the transition probability
    return norm.pdf(x, m, s)

def propo(x, v_prev, dy, params):
    mu, kappa, theta, lbda, rho = params
    # Calculate the mean and standard deviation of the proposal distribution    
    m = v_prev + kappa*(theta-v_prev)*dt + lbda*rho*(dy - (mu-1/2*v_prev)*dt)
    s = lbda*np.sqrt(v_prev*(1-rho**2)*dt)
    return norm.pdf(x, m, s)

def parameter_states_init(N, bounds):
    current_params = np.zeros((len(bounds), N))
    b0, b1, b2, b3, b4 = bounds
    # Initialize each parameter state using uniform random values within the bounds
    current_params[0] = np.random.rand(N) * (b0[1] - b0[0]) + b0[0]
    current_params[1] = np.random.rand(N) * (b1[1] - b1[0]) + b1[0]
    current_params[2] = np.random.rand(N) * (b2[1] - b2[0]) + b2[0]
    current_params[3] = np.random.rand(N) * (b3[1] - b3[0]) + b3[0]
    current_params[4] = np.random.rand(N) * (b4[1] - b4[0]) + b4[0]

    return current_params

def resample_state(particles, w):
    N = len(particles)
    c_sum = np.cumsum(w)
    c_sum[-1] = 1.
    # Select new particles by randomly sampling from the cumulative sum
    idx = np.searchsorted(c_sum, np.random.rand(N))
    particles[:] = particles[idx]
    # Assign equal w to the new particles
    new_w = np.ones(len(w)) / len(w)
    
    return particles, new_w

def resample(v_pred, w, current_params):
        current_params[0], _ = resample_state(current_params[0], w)
        current_params[1], _ = resample_state(current_params[1], w)
        current_params[2], _ = resample_state(current_params[2], w)
        current_params[3], _ = resample_state(current_params[3], w)
        current_params[4], _ = resample_state(current_params[4], w)
        v_pred, w = resample_state(v_pred, w)
        return v_pred, w, current_params

def prediction_density(y, y_prev, x, mu):
    m = y_prev + (mu-1/2*x)*dt
    s = np.sqrt(x*dt)
    return norm.pdf(y, m, s)

def prediction_density_v(v, v_prev, dy, lbda,rho, theta, kappa):
    # Transition
    m = 1/(1+1/2*lbda*rho*dt) * (v_prev + kappa*(theta-v_prev)*dt + 1/2*lbda*rho*v_prev*dt)
    #print('m',m)
    s = (1/(1+1/2*lbda*rho*dt) * lbda * np.sqrt(v_prev*dt))
    return norm.pdf(v, m, s)


def predict(v_pred, particles, y_prev, mu):
    # Generate predicted observations based on the predicted states
    y_hat = y_prev + (mu - 1 / 2 * v_pred) * dt + np.sqrt(particles * dt) * norm.rvs()
    # Calculate the prediction density for each predicted observation
    prediction_densities = [prediction_density(y_hat[k], y_prev, v_pred, mu) for k in range(len(y_hat))]
    # Calculate the average prediction density for each predicted observation
    pdf_y_hat = np.array([np.mean(density) for density in prediction_densities])
    # Normalize the prediction densities
    pdf_y_hat = pdf_y_hat / np.sum(pdf_y_hat)
    # Calculate the weighted sum of the predicted observations
    return np.sum(pdf_y_hat * y_hat)

def predict_v(v_pred, particles, v_prev, mu, lbda,rho, theta, kappa, w, params, dy):
    v_hat = v_prev + (theta-1/2*particles)*dt + lbda*rho*(((mu-1/2*particles)*dt)-(mu-1/2*particles)*dt) + lbda*np.sqrt((1-rho**2)*particles*dt)*norm.rvs() + lbda*rho*np.sqrt(particles*dt)*norm.rvs()
    pdf_v_hat = np.array([np.mean(prediction_density_v(v_hat[k], particles[k], dy,lbda,rho, theta,kappa)) for k in range(len(v_hat))])
    pdf_v_hat = pdf_v_hat/sum(pdf_v_hat)
    return np.sum(pdf_v_hat * v_hat)

def inv_froeb(w):
        return 1. / np.sum(np.square(w))

def particle_filter(params, N):
    global y_PF, v_PF, v_PF_bis
    
    # Unpack the model parameters
    mu, kappa, theta, lbda, rho, v0 = params
    
    print(params[:-1])
    
    # Initialize the current parameter states
    current_params = parameter_states_init(N, params[:-1])
    
    # Initialize the arrays to store the estimated states
    y_PF = np.zeros(N)
    v_PF = np.zeros(N)
    v_PF_bis = np.zeros(N)
    
    y_PF[0] = y[0]
    v_PF[0] = v0
    v_PF_bis[0] = v0
    
    # Initialize the weights
    w = np.array([1 / N] * N)
    
    # Initialize the particles
    particles = norm.rvs(v0, 0.02, N)
    particles = np.maximum(1e-4, particles)
    
    # Initialize the array to store the parameter steps
    params_steps = np.zeros((len(params) - 1, len(y)))
    params_steps.transpose()[0] = np.mean(current_params, axis=1)
    print(N)
    
    for i in range(1, N):
        dy = y[i] - y[i - 1]
        
        # Particle prediction step
        v_pred = proposal_distribution(N, particles, dy, current_params)
        v_pred = np.maximum(1e-3, v_pred)
        
        # Likelihood calculation
        Li = likelihood(y[i], v_pred, particles, y[i - 1], current_params)
        I = propo(v_pred, particles, dy, current_params)
        T = transition_proba(v_pred, particles, current_params)
        
        # Update the weights
        w = w * (Li * T / I)
        w = w / np.sum(w)
        
        # Resampling step
        if inv_froeb(w) < 0.7 * N:
            print('Resampling')
            v_pred, w, current_params = resample(v_pred, w, current_params)
        
        # State estimation step
        y_hat = predict(v_pred, particles, y[i - 1], np.mean(current_params[0]))
        y_PF[i] = y_hat
        v_PF_bis[i] = predict_v(v_pred, particles, v_PF[i - 1], np.mean(current_params[0]),
                                np.mean(current_params[3]), np.mean(current_params[4]),
                                np.mean(current_params[2]), np.mean(current_params[1]), w, current_params, dy)
        v_PF[i] = np.sum(v_pred * w)
        particles = v_pred
        params_steps.transpose()[i] = np.sum(np.multiply(current_params, w[np.newaxis, :]), axis=1)
        
        print("Iteration {} completed".format(i))
        
    return (v_PF, v_PF_bis, params_steps, y_PF)

S0 = 4100
r = 0.0485
mu = r
params = [mu, 1.5, 0.05, 0.18, 0.5, 0.04]
#.        mu, kappa, theta, lbda, rho , v0
N = 500
T = 10

params_sim = np.zeros(5)
params_sim[0] = params[1]# kappa  
params_sim[1] = params[2]#theta  
params_sim[2] = params[3]#sigma  
params_sim[3] = params[4]#rho   
params_sim[4] = params[5]#v0  
print(params_sim)
prices, v_, y = heston_simulated_prices(params_sim, N, T, S0, r, mu, plot = True)

dt = T/N
print(dt)


mu = (0.01, 0.05)
kappa = (0.5, 3)
theta = (0.02, 0.2)
lbda = (0.01, 0.91)
rho = (-0.5, 1)
v0 = params[-1]

params_0 = [mu, kappa, theta, lbda, rho, v0]

v, v_bis, param_steps, obs = particle_filter(params_0, N)

plt.figure(figsize=(12,8))
years = np.arange(obs.shape[-1]) * (T/N)
plt.plot(years, y[:-1], label = 'Real simulated log Price', linewidth=0.8)
plt.plot(years, obs, label = 'Particle Filter log Price', linewidth=0.8)
plt.plot()
plt.title('Evolution of the log Price')
plt.ylabel('Log(St)')
plt.xlabel('Years')
plt.legend()
plt.show()

plt.figure(figsize=(12,8))
years = np.arange(v.shape[-1]) * (T/N)
plt.plot(years, v_[1:], label = 'Real simulated volatility', linewidth=0.8)
plt.plot(years[1:], v[1:], label = 'Particle Filter volatility', linewidth=0.8)
#plt.plot(years[1:], v_bis[1:], label = 'Particle Filter volatility', linewidth=0.8)
plt.plot()
plt.title('Evolution of the volatility')
plt.ylabel('v')
plt.xlabel('Years')
plt.legend()
plt.show()