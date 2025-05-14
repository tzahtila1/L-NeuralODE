import torch
import numpy as np

def determine_curriculum_times(x_trajectories, folds = 40):
    
    Nt          = x_trajectories.shape[0]
    t           = torch.linspace(0, 1.0, Nt)
    

    t_samples   = np.array(np.linspace(1,folds, folds), int) / folds * len(t)
    tsi         = 0; loss        = 1.0; upper_ind   = 0 

    
    return tsi, t_samples, loss, upper_ind, t