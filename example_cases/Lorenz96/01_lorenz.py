import numpy as np
import sys

#%%%
package_path = "/Users/tonyzahtila/Library/CloudStorage/GoogleDrive-tzahtila@stanford.edu/My Drive/Code/PSAAP_GENERATIVE/L-NeuralODE"
#%%
sys.path.insert(0, package_path)
from src import init, preprocess, train

#%% Case set-up

#% Load in the training data
dl  = np.load('./A_training.npz')
A   = dl['A']
t   = dl['t']
xi  = dl['F_set']

#% configure the input
param_cfg = init.ParamConfig(
    xi = xi
    )

data_cfg  = init.DataConfig(
    trajectories  = A,
    time          = t,
    Nruns         = A.shape[2],
    Nvars         = A.shape[1],
    rescale       = 1
)

train_cfg  = init.TrainConfig(
    max_iters     = 200_000,
    network_width = 800,
    curric_tol    = 0.0001,
    learning_rate = 1e-3
)


config = init.NodeConfig(param_cfg = param_cfg, data_cfg=data_cfg, train_cfg=train_cfg)


#%% Pre-process the data
preprocess.scale_inputs_trajectories(data_cfg, param_cfg, config.param_cfg.xi, config.data_cfg.trajectories)

#%% Train the model
train.train(config, 
            NXiFeatures     = param_cfg.xi.shape[0],
            network_width   = train_cfg.network_width, 
            Nvars           = data_cfg.Nvars,
            trajectories    = data_cfg.trajectories_scaled, 
            max_iters       = train_cfg.max_iters, 
            curric_tol      = train_cfg.curric_tol,
            xi_scaled       = param_cfg.xi_scaled)


#%% Next steps