import torch
import numpy as np
import matplotlib.pyplot as plt

def minmax_scale(data):
    min_val = np.min(data, axis=0, keepdims=True)  # Compute minimum along the columns
    max_val = np.max(data, axis=0, keepdims=True)  # Compute maximum along the columns

    # Perform min-max scaling
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data, min_val, max_val

# Function to unscale the data back to its original range
def unscale_data(scaled_data, min_val, max_val):
    # Revert the scaled data back to the original scale
    unscaled_data = scaled_data * (max_val - min_val) + min_val
    return unscaled_data


def determine_curriculum_times(trajectories, folds = 40):
    
    Nt          = trajectories.shape[0]
    t           = torch.linspace(0, 1.0, Nt)
    

    t_samples   = np.array(np.linspace(1,folds, folds), int) / folds * len(t)
    tsi         = 0; loss        = 1.0; upper_ind   = 0 

    
    return tsi, t_samples, loss, upper_ind, t

def scale_inputs_trajectories(data_cfg, param_cfg, xi, trajectories):
    
    if xi.ndim == 1:
        xi = xi.reshape(1, -1)
        param_cfg.xi = xi
    xi_scaled   = np.zeros_like(xi)
    
    min_scales  = []
    max_scales  = []
    
    fig, ax = plt.subplots(1,5)
        
    itr = 0 
     
    
    for i in range(len(xi)):
            ax[i].plot(xi[itr,:], 'bo')
            
            scaled_data, min_val, max_val = minmax_scale(xi[itr,:])        
            orig    = unscale_data(scaled_data, min_val, max_val)            
            ax[i].plot(orig, 'rx')
            
    
            # min_scales.append(min_val)
            # max_scales.append(max_val)
    
            xi_scaled[itr,:] = scaled_data
            
            itr += 1
    
    
    
    trajectories_scaled = np.zeros_like(trajectories)
    num_latent_parms    = trajectories_scaled.shape[1]
    
    itr = 0 
    for itr in range(num_latent_parms):
        scaled_data, min_val, max_val = minmax_scale(trajectories[:,itr,:])          
        min_scales.append(min_val)
        max_scales.append(max_val)
        trajectories_scaled[:,itr,:] = scaled_data
        
        itr += 1
    
    data_cfg.trajectories_scaled = trajectories_scaled
    param_cfg.xi_scaled          = xi_scaled
    param_cfg.min_scales         = min_scales
    param_cfg.max_scales         = max_scales
    
    return 
