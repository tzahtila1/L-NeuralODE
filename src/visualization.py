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

def viz_training(batch_t, batch_y_train, pred_y_train,batch_y_val, pred_y_val, t, img_itr):
    #Style
    try:
        use_tex()
    except:
        pass
    
    fig, ax = plt.subplots(2,2)
    plt.subplots_adjust(wspace = 0.3)
    
    t = batch_t.numpy()
    y_train = batch_y_train.numpy()
    py_train = pred_y_train.numpy()
    y_val = batch_y_val.numpy()
    py_val = pred_y_val.numpy()

    for i in range(4):
        row, col = divmod(i, 2)
        
        # Training data
        ax[row, col].plot(t, y_train[:, 0, i], 'ks', markerfacecolor='None', ms=3, label='Training' if i == 0 else None)
        ax[row, col].plot(t, py_train[:, 0, i], 'r-', linewidth=1)
        
        ax[row, col].plot(t, y_train[:, 1, i], 'ks', markerfacecolor='None', ms=3, linewidth=1)
        ax[row, col].plot(t, py_train[:, 1, i], 'r-', linewidth=1)
        
        # Validation data
        ax[row, col].plot(t, y_val[:, 0, i], 'bo', markerfacecolor='None', ms=3, label='Validation' if i == 0 else None)
        ax[row, col].plot(t, py_val[:, 0, i], 'r--', linewidth=1)
        
        ax[row, col].plot(t, y_val[:, 1, i], 'bo', markerfacecolor='None', ms=3, linewidth=1)
        ax[row, col].plot(t, py_val[:, 1, i], 'r--', linewidth=1)

        ax[row, col].set_xlabel(r'$t$')
        ax[row, col].set_ylabel(f'$v_{i+1}$')

    ax[0, 0].legend()

    
    for i in range(2):
        for j in range(2):
            ax[i,j].set_xlim([0.0,1.0])
            ax[i,j].set_ylim([0.0, 1.0])
    
    save_dir = './output/training_images'
    os.makedirs(save_dir, exist_ok=True)

    # plt.text(s = 'mass =' + str(m), x = 5.0, y = 0.8)
    plt.savefig('./output/training_images/' + '{:03d}'.format(img_itr), dpi=400)
    # np.savez('./training_images/data_' + '{:03d}'.format(img_itr))
    
    plt.close('all')

    return