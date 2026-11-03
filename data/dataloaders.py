from dataclasses import dataclass
import torch

def collate_fn(data_batch):

    x0, xT, sample_id = zip(*data_batch)

    x0 = torch.stack(x0)
    xT = torch.stack(xT)

    t = torch.rand((xT.shape[0], 1))

    return Batch(x0=x0, xT=xT, t=t, sample_id=sample_id)


@dataclass
class Batch:
    x0: torch.Tensor
    xT: torch.Tensor
    t: torch.Tensor
    sample_id: list

    def to(self, device):
        self.x0 = self.x0.to(device)
        self.xT = self.xT.to(device)
        self.t = self.t.to(device)
        return self
