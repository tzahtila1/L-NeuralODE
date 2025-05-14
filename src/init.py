import numpy as np

class ParamConfig:
    def __init__(
        self,
        xi: np.ndarray = None
       
    ):
        self.xi = xi


class DataConfig:
    def __init__(
        self,
        trajectories: np.ndarray = None,
        time: np.ndarray = None,
        Nruns: int = None,
        Nvars: int = None,
        rescale: bool=True
    ):
        self.trajectories = trajectories
        self.time = time
        self.Nruns = Nruns
        self.Nvars = Nvars
        self.rescale = rescale


class TrainConfig:
    def __init__(
        self, 
        max_iters: int = 10,
        network_width: int = 400,
        curric_tol: float=1e-3
    ):
        self.max_iters = max_iters
        self.network_width = network_width
        self.curric_tol = curric_tol


class NodeConfig:
    def __init__(self, param_cfg: ParamConfig = None, data_cfg: DataConfig = None, train_cfg: TrainConfig = None):
        self.data_cfg  = data_cfg
        self.train_cfg = train_cfg
        self.param_cfg = param_cfg