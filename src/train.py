# visualization
import torch
import torch.nn as nn
import torch.optim as optim
from . import preprocess
from . import visualization
from . import models
import numpy as np
import matplotlib.pyplot as plt
import sys
import os 

#%% Perform Neural ODE training
def train(config,NXiFeatures,network_width, Nvars,trajectories, max_iters, curric_tol, lr, xi_scaled):
      
    #% Initialize Model and Optimizer
    node        = models.NeuralODE(func=models.ODEFunc(NXiFeatures, Nvars,network_width)) # Move Neural ODE to GPU
    optimizer = optim.Adam(node.parameters(), lr=lr)


    #%% Training setup
    train_i         = config.train_cfg.train_inds
    val_i           = config.train_cfg.val_inds
    
    train_trajects  = trajectories[:,:,train_i]
    val_trajects    = trajectories[:,:,val_i]
    
    train_xi        = xi_scaled[:,train_i]
    val_xi          = xi_scaled[:,val_i]
    
    train_trajects  = torch.from_numpy(train_trajects)
    val_trajects    = torch.from_numpy(val_trajects)
    
    #% Curriculum Learning Setup
    tsi, t_samples, loss, upper_ind, t = preprocess.determine_curriculum_times(train_trajects)
    
    batch_size     = config.train_cfg.batch_size
    
    train_viz_ind   = np.random.choice(config.train_cfg.Ntrain, size=5, replace=False)
    val_viz_ind     = np.random.choice(config.train_cfg.Nval, size=5, replace=False)
    
    
    #%% Save 
    # Save PyTorch tensors
    torch.save({
        'train_trajects': train_trajects,
        'val_trajects': val_trajects
    }, 'output/torch_data.pt')
    
    # Save NumPy arrays
    np.savez('output/numpy_data.npz', 
             train_xi=train_xi, 
             val_xi=val_xi)

    #%% Training Loop
    loss_func       = []
    img_itr         = 0


    for iter in range(max_iters + 1):

        if loss < curric_tol or iter == 0:
            if upper_ind != -1:
                print('Loss is:', loss, 'More training... Now at:')
                upper_ind   = int(t_samples[tsi])
                
                tsi += 1 
                if upper_ind == len(t):
                    upper_ind = -1
                    tsi       = -1
                    

        optimizer.zero_grad()

        batch_ind      = np.random.choice(config.train_cfg.Ntrain, size=batch_size, replace=False)

        
        input_xi = train_xi[:,batch_ind]
        
        # Generate arguments
      
        batch_y0 = train_trajects[0,:, batch_ind].t()
        batch_y0 = batch_y0.unsqueeze(1)
        

         
        batch_t = t[:]
        
        batch_y  = train_trajects[:,:, batch_ind]

        batch_y  = batch_y.permute(0, 2, 1)
        
        
        #%% Implement curriculum
        batch_t = batch_t[:upper_ind]
        batch_y = batch_y[:upper_ind,:,:]
        
        #%%

        pred_y = node(y0=batch_y0, t=batch_t, solver=models.rk4, m = input_xi)

        pred_y = pred_y.squeeze(-1).squeeze(2)

        loss = torch.mean(torch.abs(pred_y - batch_y)**2) #+ lambda_p*torch.mean(torch.abs(pred_y_curve - batch_y_curve)**2)
        # print(torch.abs(pred_y - batch_y)**2)
        loss.backward()
        optimizer.step()
         

        loss_func.append(loss.detach().numpy())

        if iter % 100 == 1:
            with torch.no_grad():
                plt.close('all')
                
                #%% Training 
                # Select parameteters
                input_xi = train_xi[:,train_viz_ind]
                
                
                # Generate arguments
                batch_y0 = train_trajects[0,:, train_viz_ind].t()
                batch_y0 = batch_y0.unsqueeze(1)
                 
                batch_t = t[:]
                
                batch_y  = train_trajects[:,:, train_viz_ind]

                batch_y  = batch_y.permute(0, 2, 1)

                # Implement curriculum
                batch_t = batch_t[:upper_ind]
                batch_y_train = batch_y[:upper_ind,:,:]
                   
                pred_y = node(y0=batch_y0, t=batch_t, solver=models.rk4, m = input_xi)
                
                pred_y_train = pred_y.squeeze(-1).squeeze(2)
                
                #%% Validation 
                # Select parameteters
                input_xi = val_xi[:,val_viz_ind]
                
                
                # Generate arguments
                batch_y0 = val_trajects[0,:, val_viz_ind].t()
                batch_y0 = batch_y0.unsqueeze(1)
                 
                batch_t  = t[:]
                
                batch_y  = val_trajects[:,:, val_viz_ind]

                batch_y  = batch_y.permute(0, 2, 1)

                # Implement curriculum
                batch_t = batch_t[:upper_ind]
                batch_y_val = batch_y[:upper_ind,:,:]
                   
                pred_y = node(y0=batch_y0, t=batch_t, solver=models.rk4, m = input_xi)
                
                pred_y_val = pred_y.squeeze(-1).squeeze(2)
                
                #%% Viz
                visualization.viz_training(batch_t, batch_y_train, pred_y_train,batch_y_val, pred_y_val, t, img_itr)
                
                
                
                #%% Report 
                loss = torch.mean(torch.abs(pred_y_train - batch_y_train)**2)
                print('Iter {:04d} | Total Loss {:.6f}'.format(iter, loss.item()))
                sys.stdout.flush()  # Force output to appear immediately 
                

                # plt.close('all')
                img_itr += 1
                
                    
        if iter % 2000 == 1:
                 #%% Saving
                 # training history for validation
                 output_dir = 'output'
                 loss_dir = os.path.join(output_dir, 'Loss')
                
                 # Create directories if they don't exist
                 os.makedirs(loss_dir, exist_ok=True)

                 # Create the save string
                 save_string = f'LossFunc_niter_{max_iters:02d}_width_{network_width:02d}.npz'
                
                 # Full path for saving
                 save_path = os.path.join(loss_dir, save_string)
                
                 # Save the file
                 np.savez(save_path, loss_func = loss_func)
         
                 # model
                 torch.save(node.state_dict(), 'output/neural_ode_model.pth')
         
                 # save compression de-compression
                 np.savez('output/min_max_scales.npz', min_scales = config.param_cfg.min_scales, max_scales = config.param_cfg.max_scales)

    #%% Saving
    # training history for validation
    # Create the save string
    save_string = f'LossFunc_niter_{max_iters:02d}_width_{network_width:02d}.npz'
    # Full path for saving
    save_path = os.path.join(loss_dir, save_string)
    np.savez(save_path, loss_func = loss_func)
            
    # model
    torch.save(node.state_dict(), 'output/neural_ode_model.pth')
        
    # save compression de-compression
    np.savez('output/min_max_scales.npz', min_scales = config.param_cfg.min_scales, max_scales = config.param_cfg.max_scales)