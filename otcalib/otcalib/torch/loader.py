import numpy as np
import torch as T

from . import torch_utils as utils

DATALOADER_args = {
    # "pin_memory": False,
    "num_workers": 4,
    # "prefetch_factor": 4,
    # "persistent_workers": False,
}

class DictDataset(T.utils.data.Dataset):
    def __init__(self, data:dict, keys=None):
        self.data = data
        self.keys = list(data.keys())

    def __getitem__(self, index):
        return {i:self.data[i][index] for i in self.keys}
    
    def __len__(self):
        return len(self.data[self.keys[0]])



class Iterator:
    def __init__(self) -> None:
        self.noncvx_dim = None
        self.cvx_dim = None
        self.dataset = None
        self.device = None

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        if hasattr(self, '_iterator') and self._iterator is None:
            self._iterator = iter(self.dataset)
        else:
            self._reset()
        return self._iterator

    def _reset(self) -> None:
        self._iterator = iter(self.dataset)

    def __next__(self):
        try:
            batch = next(self._iterator)
        except (StopIteration, AttributeError):
            self._reset()
            batch = next(self._iterator)

        
        # setting batch as flow
        batch = batch.float()
        
        # check if label in columns (+1)
        if self.data.shape[1] != self.noncvx_dim + self.cvx_dim + 1:
            raise ValueError("Data dimensions and cvx/noncvx dimensions do not match - check either network input parameters or data parameters")
        
        # getting label
        mask = batch[:, -1:].bool()

        # making grad tracable
        batch.requires_grad = True

        # move to device
        batch = batch.to(self.device)

        batch = [
            batch[:, : self.noncvx_dim],
            batch[
                :,
                self.noncvx_dim : self.noncvx_dim + self.cvx_dim,
            ],
            mask.bool().to(self.device),
        ]

        return batch
    
class Dataset(Iterator):
    "class that initialize the tfrecords pipeline for torch"

    def __init__(
        self,
        data,
        cvx_dim,
        noncvx_dim=0,
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        dataloader_kwargs = kwargs.get("dataloader_kwargs")
        
        if dataloader_kwargs is None:
            dataloader_kwargs = DATALOADER_args.copy()
            dataloader_kwargs.update({"shuffle": False, "drop_last": True, 'batch_size': 512})
        
        self.data = data
        self.cvx_dim = cvx_dim
        self.noncvx_dim = noncvx_dim
        self.device = device

        # 
        if not isinstance(self.data, T.Tensor):
            self.data = T.tensor(self.data).float()

        # adding label column if missing
        if self.data.shape[1] == self.noncvx_dim + self.cvx_dim:
            self.data = T.concat(
                [self.data, T.ones((len(self.data), 1))],1
                )

        self.dataset = T.utils.data.DataLoader(
            self.data.cpu().detach(),
            **dataloader_kwargs,
        )


class BaseDistribution(Iterator):
    # add __next__ to this or utils.Dataset as init
    def __init__(
        self, distribution, dims: int, batch_size: int = 1024, device: str = "cuda"
    ) -> None:
        if (not hasattr(distribution, "log_prob")) and (
            not hasattr(distribution, "sample")
        ):
            raise TypeError("distribution requires log_prob & sample as attributs")
        super().__init__()
        self.distribution = distribution
        self.device = device
        self.dims = dims
        self.sample_loader = None
        self.batch_size = batch_size
        self.mu = 0
        self.sigma = 1

    def sample(self, conds_dist, transport_dist=None) -> None:
        self.conds_dist = conds_dist
        self.transport_dist = transport_dist
        if self.transport_dist is None:
            if isinstance(self.distribution, Dirichlet) or isinstance(
                self.distribution, T.distributions.MultivariateNormal
            ):
                self.dist = self.distribution.sample((len(conds_dist)))
            elif isinstance(
                self.distribution,
                T.distributions.relaxed_bernoulli.LogitRelaxedBernoulli,
            ) or isinstance(self.distribution, T.distributions.Normal):
                self.dist = self.distribution.sample([len(conds_dist)])
            else:
                self.dist = self.distribution.sample((len(conds_dist), self.dims))
        else:
            idx = T.randperm(len(self.transport_dist))
            self.dist = self.transport_dist[idx].detach()

        self.data = T.tensor(
            np.c_[conds_dist.detach().cpu(), self.dist.cpu()], requires_grad=True
        ).float()
        self.noncvx_dim = conds_dist.shape[1]
        self.cvx_dim = self.data.shape[1] - self.noncvx_dim
        self.dataset = T.utils.data.DataLoader(
            self.data.detach(),
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=True,
            num_workers=4,
        )

    def log_prob(self, values):
        return self.distribution.log_prob(values)

    def _reset(self) -> None:
        self.sample(self.conds_dist, self.transport_dist)
        self._iterator = iter(self.dataset)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    base = BaseDistribution(
        "uniform", device="cpu", batch_size=1024, logit=True, dims=3
    )
    base.generate_ot_sample(T.tensor(np.random.randint(3, size=100_000)).view(-1, 1))

    data = base.data.detach().numpy()
    data[:, 1:] = utils.probsfromlogits(data[:, 1:])

    style = {"histtype": "step", "bins": 11, "range": [-5, 5]}
    for i in range(3):
        mask = data[:, 0] == i
        plt.figure()
        for j in range(3):
            plt.hist(utils.logit(data[:, 1 + j][mask]), **style)

    size = 100_000
    columns = 5
    first = np.random.uniform(size=(size, 1))
    second = np.random.uniform(0, 1, size=(size, columns))

    data = np.c_[second[second.sum(1) < 1], 1 - second[second.sum(1) < 1].sum(1)]
    # data = np.r_[data, 1-data/np.sum(1-data, 1)[:,None]]
    plt.hist(data)
