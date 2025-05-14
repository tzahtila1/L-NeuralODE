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



    max_iters            = 200_000,
    network_width        = 800,
    learning_rate        = 1e-3,
    test_train_split     = 0.8
    
class TrainConfig:
    def __init__(
        self, 
        max_iters: int = 10,
        network_width: int = 400,
        curric_tol: float=1e-3,
        learning_rate: float=1e-3,
        test_train_split: float=0.8,
        batch_size: int=20
    ):
        self.max_iters = max_iters
        self.network_width = network_width
        self.curric_tol = curric_tol
        self.learning_rate = learning_rate
        self.test_train_split = test_train_split
        self.batch_size = batch_size


class NodeConfig:
    def __init__(self, param_cfg: ParamConfig = None, data_cfg: DataConfig = None, train_cfg: TrainConfig = None):
        self.data_cfg  = data_cfg
        self.train_cfg = train_cfg
        self.param_cfg = param_cfg