import numpy as np
import sys
import pickle

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
    max_iters            = 2_000,
    network_width        = 800,
    curric_tol           = 0.001,
    learning_rate        = 1e-3,
    test_train_split     = 0.8,
    batch_size           = 20
)


config = init.NodeConfig(param_cfg = param_cfg, data_cfg=data_cfg, train_cfg=train_cfg)


#%% Pre-process the data

#scale data
preprocess.scale_inputs_trajectories(data_cfg, param_cfg, config.param_cfg.xi, config.data_cfg.trajectories)

#test train split
preprocess.test_train_split(data_cfg, train_cfg)


#%% Train the model
train.train(config, 
            NXiFeatures     = param_cfg.xi.shape[0],
            network_width   = train_cfg.network_width, 
            Nvars           = data_cfg.Nvars,
            trajectories    = data_cfg.trajectories_scaled, 
            max_iters       = train_cfg.max_iters, 
            curric_tol      = train_cfg.curric_tol,
            lr              = train_cfg.learning_rate,
            xi_scaled       = param_cfg.xi_scaled)

#%% Next steps
# Save the object
with open('output/param_cfg.pkl', 'wb') as f:
    pickle.dump(param_cfg, f)
    
with open('output/train_cfg.pkl', 'wb') as f:
    pickle.dump(train_cfg, f)

with open('output/data_cfg.pkl', 'wb') as f:    # This will be deprecated
    pickle.dump(data_cfg, f)