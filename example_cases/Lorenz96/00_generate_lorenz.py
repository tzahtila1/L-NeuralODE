import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

plt.close('all')

def lorenz96(t, x, F):
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return dxdt

# Parameters
N           = 4           # Number of variables (can be changed)
Nt          = 80
t_span      = (0, 20) # Time interval
t_eval      = np.linspace(*t_span, Nt)

# Initial condition: small perturbation

eps     = 1e-3
fig,ax  = plt.subplots(N,1, figsize=(10, 6))
 
F_set   = np.arange(1,3 + eps,0.02)
Nf      = len(F_set)

A   = np.zeros((Nt, N, Nf))
itr = 0 
for F in F_set:
    x0 = F * np.ones(N)
    x0[0] += 0.1
    
    # Solve the system
    sol = solve_ivp(lorenz96, t_span, x0, args=(F,), t_eval=t_eval, method='RK45')
    
    # Plot results for a few components
   
    for i in range(N):
        ax[i].plot(sol.t, sol.y[i], label=f'x[{i}]')
        ax[i].set_ylabel('x' + str(i))
    plt.xlabel('Time')
    
    A[:,:,itr]  = sol['y'].transpose()
    
    itr += 1

plt.savefig('./00_lorenz_trajectories.png',dpi = 200)
np.savez('A_training.npz', A = A, t = sol['t'], F_set = F_set)