# visualization
import torch
import torch.nn as nn


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
    def __init__(self, NXiFeatures, Nvars, network_width):
        super().__init__()
        self.NXiFeatures = NXiFeatures
        self.Nvars       = Nvars
        self.net         = nn.Sequential(nn.Linear(NXiFeatures + 1 + Nvars, network_width),
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
        txiy      = torch.cat((t,xi,y), dim=1).float()       
        txiy      = txiy.unsqueeze(1).squeeze(-1)
        # print('T shape:', t.shape, 'm.shape', m.shape, 'y.shape', y.shape)
        output  = self.net(txiy)

        return output