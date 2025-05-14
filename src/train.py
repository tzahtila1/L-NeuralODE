# ========================== Imports ==========================
import torch
import torch.nn as nn
import torch.optim as optim
import preprocess
import numpy as np
import matplotlib.pyplot as plt

#%% Perform Neural ODE training
def train(NXiFeatures,network_width, Nvars,x_trajectories, max_iter, curric_tol,xi_scaled):
    def rk4(func, t, dt, y, m):
        _one_sixth = 1/6
        half_dt = dt * 0.5
        
        k1 = func(t,m, y)
        k2 = func(t + half_dt, m, y)
        k3 = func(t + half_dt, m, y)
        k4 = func(t + dt, m, y)
    
        return (k1 + 2 * (k2 + k3) + k4) * dt * _one_sixth
    
    class NeuralODE(nn.Module):
        def __init__(self, func):
            super().__init__()
            self.func = func
    
        def forward(self, y0, t, solver, m):
            solution = torch.empty(
                len(t), *y0.shape, dtype=y0.dtype, device=y0.device)
            solution[0] = y0
    
            j = 1
            for t0, t1 in zip(t[:-1], t[1:]):
                dy = solver(self.func, t0, t1 - t0, y0, m)
                y1 = y0 + dy
                solution[j] = y1
                j += 1
                y0 = y1
                
            return solution
        
    # define dynamic function
    class ODEFunc(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(NXiFeatures + 1 + Nvars, network_width),
                                      nn.Tanh(),
                                      nn.Linear(network_width, network_width),
                                      nn.Tanh(),      
                                      nn.Linear(network_width, Nvars))
        def forward(self, t, xi, y):
            
            # This needs to be cleaned up
            
            t       = t.unsqueeze(0)#.unsqueeze(1)#.unsqueeze(2)
            t       = t.repeat(y.shape[0], 1)
            xi       = xi.transpose()
            xi       = torch.from_numpy(xi)
            y       = y.squeeze(2)
            y       = y.squeeze(1)
            txiy      = torch.cat((t, xi,y), dim=1).float()       
            txiy      = txiy.unsqueeze(1).squeeze(-1)
            # print('T shape:', t.shape, 'm.shape', m.shape, 'y.shape', y.shape)
            output  = self.net(txiy)
    
            return output
        
    #% Initialize Model and Optimizer
    node = NeuralODE(func=ODEFunc()) # Move Neural ODE to GPU
    optimizer = optim.Adam(node.parameters(), lr=1e-4)

    #% Curriculum Learning Setup
    tsi, t_samples, loss, upper_ind, t = preprocess.determine_curriculum_times(x_trajectories)
    #%% Training Loop
    loss_func       = []
    img_itr         = 0
    # Nruns           = latent_params['params'].shape[1]
    NumTraining     = 95  # Leave the complement for validation



    for iter in range(max_iter + 1):

        if loss < curric_tol or iter == 0:
            if upper_ind != -1:
                print('Loss is:', loss, 'More training... Now at:')
                upper_ind   = int(t_samples[tsi])
                
                tsi += 1 
                if upper_ind == len(t):
                    upper_ind = -1
                    tsi       = -1
                    

        optimizer.zero_grad()

        batch_ind      = np.array(np.round(np.random.rand(NumTraining)*(NumTraining-1))[:], int)
        
        input_xi = xi_scaled[:,batch_ind]
        
        # Generate arguments
      
        batch_y0 = x_trajectories[0,:, batch_ind].t()
        batch_y0 = batch_y0.unsqueeze(1)
        

         
        batch_t = t[:]
        
        batch_y  = x_trajectories[:,:, batch_ind]

        batch_y  = batch_y.permute(0, 2, 1)
        
        
        #%% Implement curriculum
        batch_t = batch_t[:upper_ind]
        batch_y = batch_y[:upper_ind,:,:]
        
        #%%

        pred_y = node(y0=batch_y0, t=batch_t, solver=rk4, m = input_xi)

        pred_y = pred_y.squeeze(-1).squeeze(2)

        loss = torch.mean(torch.abs(pred_y - batch_y)**2) #+ lambda_p*torch.mean(torch.abs(pred_y_curve - batch_y_curve)**2)
        # print(torch.abs(pred_y - batch_y)**2)
        loss.backward()
        optimizer.step()
         

        loss_func.append(loss.detach().numpy())

        if iter % 500 == 1:
            with torch.no_grad():
                plt.close('all')
                
                
                
                # Select parameteters
                ind     = [0,1,2,3,4]#,-3,-2,-1]
            
                input_xi = xi_scaled[:,ind]
                
                
                # Generate arguments
                batch_y0 = x_trajectories[0,:, ind].t()
                batch_y0 = batch_y0.unsqueeze(1)
                 
                batch_t = t[:]
                
                batch_y  = x_trajectories[:,:, ind]

                batch_y  = batch_y.permute(0, 2, 1)

                #%% Implement curriculum
                batch_t = batch_t[:upper_ind]
                batch_y = batch_y[:upper_ind,:,:]
                
                
                fig, ax = plt.subplots(2,2)
                
                pred_y = node(y0=batch_y0, t=batch_t, solver=rk4, m = input_xi)

                plt.subplots_adjust(wspace = 0.3)
                
                pred_y = pred_y.squeeze(-1).squeeze(2)
                   
                #%% Full data
                
                FUNCS.viz_training(ax, batch_t, batch_y, pred_y, t, trajectories_scaled, img_itr)
                
                loss = torch.mean(torch.abs(pred_y - batch_y)**2)
                print('Iter {:04d} | Total Loss {:.6f}'.format(iter, loss.item()))
                sys.stdout.flush()  # Force output to appear immediately 
                

                # plt.close('all')
                img_itr += 1
                
                    
        if iter % 20000 == 1:
                 #%% Saving
                 # training history for validation
                 save_string = 'LossFunc_niter_' + '{:2d}'.format(niters) + '_width_' + '{:2d}'.format(network_width) + '.npz'
                 np.savez(save_string, loss_func = loss_func)
         
                 # model
                 torch.save(node.state_dict(), 'neural_ode_model.pth')
         
                 # save compression de-compression
                 np.savez('min_max_scales.npz', min_scales = min_scales[5:], max_scales = max_scales[5:])

        #%% Saving
        # training history for validation
        save_string = 'LossFunc_niter_' + '{:2d}'.format(niters) + '_width_' + '{:2d}'.format(network_width) + '.npz'
        np.savez(save_string, loss_func = loss_func)
                
        # model
        torch.save(node.state_dict(), 'neural_ode_model.pth')

        # save compression de-compression
        np.savez('min_max_scales.npz', min_scales = min_scales[5:], max_scales = max_scales[5:])

    