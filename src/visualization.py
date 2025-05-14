import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os 
import torch
import scipy.signal as signal

def use_tex():
    
    # if False:
    #     plt.rcParams["font.family"] = "serif"
    #     plt.rcParams["font.serif"] = "Times New Roman"
    #     plt.rcParams["text.usetex"] = True
    #     os.environ['PATH'] = '/Library/Frameworks/Python.framework/Versions/2.7/bin:/Users/tzahtila/Software/anaconda3/bin:/Users/tzahtila/Software/anaconda3/condabin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/System/Cryptexes/App/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Library/TeX/texbin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/local/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/appleinternal/bin'
    
        
    return

def viz_training(ax, batch_t, batch_y, pred_y, t, trajectories_scaled, img_itr):
    #Style
    try:
        use_tex()
    except:
        pass
    
    # Visualize full t
    itr = 0 
    for i in range(2):
        for j in range(2):
            ax[i,j].plot(t, trajectories_scaled[:,itr,:].detach().cpu().numpy(), '-', color = 'grey', alpha = 0.1)
            
            itr+=1
    #%% V1
    
    ax[0,0].plot(batch_t.numpy(), batch_y.numpy()[:, 0, 0], 'ko', ms = 2, label = 'Training')
    ax[0,0].plot(batch_t.numpy(), pred_y.numpy()[:, 0, 0], 'r-', linewidth = 1)
    
    ax[0,0].plot(batch_t.numpy(), batch_y.numpy()[:, 1, 0], 'ko', markerfacecolor = 'None', ms = 2, linewidth = 3)
    ax[0,0].plot(batch_t.numpy(), pred_y.numpy()[:, 1, 0], 'r-', linewidth = 1)

    if False:
        ax[0,0].plot(batch_t.numpy(), batch_y.numpy()[:, -1, 0], 'bo', markerfacecolor = 'None', ms = 2, label = 'Validation')
        ax[0,0].plot(batch_t.numpy(), pred_y.numpy()[:, -1, 0], 'r-', linewidth = 1)
    
        ax[0,0].plot(batch_t.numpy(), batch_y.numpy()[:, -2, 0], 'bo', markerfacecolor = 'None', ms = 2)
        ax[0,0].plot(batch_t.numpy(), pred_y.numpy()[:, -2, 0], 'r-', linewidth = 1)
    
        ax[0,0].plot(batch_t.numpy(), batch_y.numpy()[:, -3, 0], 'bo', markerfacecolor = 'None', ms = 2)
        ax[0,0].plot(batch_t.numpy(), pred_y.numpy()[:, -3, 0], 'r-', linewidth = 1)

    ax[0,0].set_xlabel(r'$t$')
    ax[0,0].set_ylabel(r'$v_1$')

    ax[0,0].legend()
    
    #%% V2
    
    ax[0,1].plot(batch_t.numpy(), batch_y.numpy()[:, 0, 1], 'ko', markerfacecolor = 'None', ms = 2, linewidth = 3)
    ax[0,1].plot(batch_t.numpy(), pred_y.numpy()[:, 0, 1], 'r-', linewidth = 1)
    
    ax[0,1].plot(batch_t.numpy(), batch_y.numpy()[:, 1, 1], 'ko', markerfacecolor = 'None', ms = 2, linewidth = 3)
    ax[0,1].plot(batch_t.numpy(), pred_y.numpy()[:, 1, 1], 'r-', linewidth = 1)

    if False:
        ax[0,1].plot(batch_t.numpy(), batch_y.numpy()[:, -1, 1], 'bo', markerfacecolor = 'None', ms = 2)
        ax[0,1].plot(batch_t.numpy(), pred_y.numpy()[:, -1, 1], 'r-', linewidth = 1)
    
        ax[0,1].plot(batch_t.numpy(), batch_y.numpy()[:, -2, 1], 'bo', markerfacecolor = 'None', ms = 2)
        ax[0,1].plot(batch_t.numpy(), pred_y.numpy()[:, -2, 1], 'r-', linewidth = 1)
    
        ax[0,1].plot(batch_t.numpy(), batch_y.numpy()[:, -3, 1], 'bo', markerfacecolor = 'None', ms = 2)
        ax[0,1].plot(batch_t.numpy(), pred_y.numpy()[:, -3, 1], 'r-', linewidth = 1)

    ax[0,1].set_xlabel(r'$t$')
    ax[0,1].set_ylabel(r'$v_2$')

    #%% V3
    
    ax[1,0].plot(batch_t.numpy(), batch_y.numpy()[:, 0, 2], 'ko', markerfacecolor = 'None', ms = 2, linewidth = 3)
    ax[1,0].plot(batch_t.numpy(), pred_y.numpy()[:, 0, 2], 'r-', linewidth = 1)
    
    ax[1,0].plot(batch_t.numpy(), batch_y.numpy()[:, 1, 2], 'ko', markerfacecolor = 'None', ms = 2, linewidth = 3)
    ax[1,0].plot(batch_t.numpy(), pred_y.numpy()[:, 1, 2], 'r-', linewidth = 1)

    if False:
        ax[1,0].plot(batch_t.numpy(), batch_y.numpy()[:, -1, 2], 'bo', markerfacecolor = 'None', ms = 2)
        ax[1,0].plot(batch_t.numpy(), pred_y.numpy()[:, -1, 2], 'r-', linewidth = 1)
    
        ax[1,0].plot(batch_t.numpy(), batch_y.numpy()[:, -2, 2], 'bo', markerfacecolor = 'None', ms = 2)
        ax[1,0].plot(batch_t.numpy(), pred_y.numpy()[:, -2, 2], 'r-', linewidth = 1)
    
        ax[1,0].plot(batch_t.numpy(), batch_y.numpy()[:, -3, 2], 'bo', markerfacecolor = 'None', ms = 2)
        ax[1,0].plot(batch_t.numpy(), pred_y.numpy()[:, -3, 2], 'r-', linewidth = 1)

    ax[1,0].set_xlabel(r'$t$')
    ax[1,0].set_ylabel(r'$v_3$')

    #%% V4
    
    ax[1,1].plot(batch_t.numpy(), batch_y.numpy()[:, 0, 3], 'ko', markerfacecolor = 'None', ms = 2, linewidth = 3)
    ax[1,1].plot(batch_t.numpy(), pred_y.numpy()[:, 0, 3], 'r-', linewidth = 1)
    
    ax[1,1].plot(batch_t.numpy(), batch_y.numpy()[:, 1, 3], 'ko', markerfacecolor = 'None', ms = 2, linewidth = 3)
    ax[1,1].plot(batch_t.numpy(), pred_y.numpy()[:, 1, 3], 'r-', linewidth = 1)

    if False:
        ax[1,1].plot(batch_t.numpy(), batch_y.numpy()[:, -1, 3], 'bo', markerfacecolor = 'None', ms = 2)
        ax[1,1].plot(batch_t.numpy(), pred_y.numpy()[:, -1, 3], 'r-', linewidth = 1)
    
        ax[1,1].plot(batch_t.numpy(), batch_y.numpy()[:, -2, 3], 'bo', markerfacecolor = 'None', ms = 2)
        ax[1,1].plot(batch_t.numpy(), pred_y.numpy()[:, -2, 3], 'r-', linewidth = 1)
    
        ax[1,1].plot(batch_t.numpy(), batch_y.numpy()[:, -3, 3], 'bo', markerfacecolor = 'None', ms = 2)
        ax[1,1].plot(batch_t.numpy(), pred_y.numpy()[:, -3, 3], 'r-', linewidth = 1)

    ax[1,1].set_xlabel(r'$t$')
    ax[1,1].set_ylabel(r'$v_4$')
    

    for i in range(2):
        for j in range(2):
            ax[i,j].set_xlim([0.0,1.0])
            ax[i,j].set_ylim([0.0, 1.0])
    
    save_dir = './training_images'
    os.makedirs(save_dir, exist_ok=True)

    # plt.text(s = 'mass =' + str(m), x = 5.0, y = 0.8)
    plt.savefig('./training_images/' + '{:03d}'.format(img_itr), dpi=400)
    np.savez('./training_images/data_' + '{:03d}'.format(img_itr), batch_t = batch_t, batch_y = batch_y, pred_y = pred_y, trajectories_scaled = trajectories_scaled, t = t)
    

    return