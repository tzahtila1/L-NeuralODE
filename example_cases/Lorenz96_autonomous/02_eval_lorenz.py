import numpy as np
import sys
import torch
import pickle
import matplotlib.pyplot as plt

plt.close('all')
#%%%
package_path = "/Users/tonyzahtila/Library/CloudStorage/GoogleDrive-tzahtila@stanford.edu/My Drive/Code/PSAAP_GENERATIVE/L-NeuralODE"
#%%
sys.path.insert(0, package_path)
from src import models

#%% retreive the data
with open('output/param_cfg.pkl', 'rb') as f:
    param_cfg = pickle.load(f,)

with open('output/train_cfg.pkl', 'rb') as f:
    train_cfg = pickle.load(f,)

with open('output/data_cfg.pkl', 'rb') as f:
    data_cfg  = pickle.load(f,)
    
NXiFeatures, Nvars,network_width = param_cfg.xi.shape[0], data_cfg.Nvars, train_cfg.network_width

#%% Load in the Neural ODE and inputs
node        = models.NeuralODE(func=models.ODEFunc(NXiFeatures, Nvars,network_width))

node.load_state_dict(torch.load('output/neural_ode_model.pth'))

print(node.func.net[0].weight.data[:5, :5])

# Load PyTorch tensors
loaded_torch_data       = torch.load('output/torch_data.pt')
loaded_train_trajects   = loaded_torch_data['train_trajects']
loaded_val_trajects     = loaded_torch_data['val_trajects']

# Load NumPy arrays
loaded_numpy_data       = np.load('output/numpy_data.npz')
loaded_train_xi         = loaded_numpy_data['train_xi']
loaded_val_xi           = loaded_numpy_data['val_xi']



fig, ax = plt.subplots(2,2)
#%% Run the prediction for training
batch_y0  = loaded_train_trajects[0,:,:].t()
batch_y0 = batch_y0.unsqueeze(1)
input_xi   = loaded_train_xi[:,:]

t         = np.linspace(0,1,80)
batch_t     = torch.from_numpy(t)

print(input_xi.shape, batch_y0.shape,t.shape)

pred_y = node(y0=batch_y0, t=batch_t, solver=models.rk4, m = input_xi)


pred_y = pred_y.detach().numpy()

for i in range(5):
    ax[0,0].plot(t, pred_y[:,i,0,0], 'r-')
    
    ax[0,0].plot(t, loaded_train_trajects[:,0,i], 'ks', markerfacecolor = 'None', ms = 3)
    
#%Reshape pred_y to match loaded_train_trajects
# From (80, 40, 1, 4) to (80, 4, 40)
e_py = np.squeeze(pred_y, axis=2).transpose(0, 2, 1)

# Ensure both tensors are float
e_gt = loaded_train_trajects.numpy()

e_train = np.abs((e_py - e_gt))

error_vector = np.mean(e_train, axis=(0, 1))

ax[1,0].plot(input_xi[0,:],error_vector,'ro')

ax[0,0].set_title('Training')
#%% Run the prediction for validation
batch_y0  = loaded_val_trajects[0,:,:].t()
batch_y0 = batch_y0.unsqueeze(1)
input_xi   = loaded_val_xi[:,:]

t         = np.linspace(0,1,80)
batch_t     = torch.from_numpy(t)

print(input_xi.shape, batch_y0.shape,t.shape)

pred_y = node(y0=batch_y0, t=batch_t, solver=models.rk4, m = input_xi)


pred_y = pred_y.detach().numpy()

for i in range(5):
    ax[0,1].plot(t, pred_y[:,i,0,0],'r--')
    
    ax[0,1].plot(t, loaded_val_trajects[:,0,i], 'bo', markerfacecolor = 'None', ms = 3)

ax[0,1].set_title('Validation')

e_py = np.squeeze(pred_y, axis=2).transpose(0, 2, 1)

# Ensure both tensors are float
e_gt = loaded_val_trajects.numpy()

e_val = np.abs((e_py - e_gt))

error_vector = np.mean(e_val, axis=(0, 1))

ax[1,0].plot(input_xi[0,:],error_vector,'bo')


ax[1,0].set_xlabel('F')

ax[1,0].set_ylabel('MSE')

plt.savefig('./Result.png', dpi = 400)