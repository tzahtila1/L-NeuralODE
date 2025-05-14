import numpy as np
import sys

package_path = "/Users/tonyzahtila/Library/CloudStorage/GoogleDrive-tzahtila@stanford.edu/My Drive/Code/PSAAP_GENERATIVE/L-NeuralODE"
sys.path.insert(0, package_path)
from src import init 

#%% Case set-up

#% Load in the training data
dl  = np.load('./A_training.npz')
A   = dl['A']
t   = dl['t']

#% configure the input
data_cfg = init.DataConfig(
    trajectories = A,
    time         = t,
    Nruns        = A.shape[0],
    Nvars        = A.shape[2]
)

train_cfg = init.TrainConfig(
    max_iters=20e4,
    network_width=400
)



config = init.NodeConfig(data_cfg=data_cfg, train_cfg=train_cfg)