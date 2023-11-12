import torch.nn as nn

import copy


class EMA(nn.Module):
    def __init__(self, model, beta=0.99, step_start=2000):
        super(EMA, self).__init__()
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.beta = beta
        self.step_start = step_start

    def update_model(self, model, step=None):
        if step is not None and step < self.step_start:
            return None
        for ema, orig in zip(self.model.parameters(), model.parameters()):
            ema.data = ema.data * self.beta + orig.data * (1 - self.beta)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
