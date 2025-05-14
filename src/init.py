import numpy as np

class DataConfig:
    def __init__(
        self,
        trajectories: np.ndarray = None,
        time: np.ndarray = None,
        Nruns: int = None,
        Nvars: int = None
    ):
        self.trajectories = trajectories
        self.time = time
        self.Nruns = Nruns
        self.Nvars = Nvars

    def __repr__(self):
        return f"DataConfig(Nruns={self.Nruns}, Nvars={self.Nvars}, " \
               f"trajectories_shape={None if self.trajectories is None else self.trajectories.shape})"


class TrainConfig:
    def __init__(self, node_id: str):
        self.node_id = node_id

    def __repr__(self):
        return f"TrainConfig(node_id={self.node_id})"


class NodeConfig:
    def __init__(self, data_cfg: DataConfig = None, train_cfg: TrainConfig = None):
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg

    def __repr__(self):
        return f"NodeConfig(data={self.data}, train={self.train})"