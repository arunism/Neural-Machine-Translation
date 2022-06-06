import torch.nn as nn

class BaseEncoder(nn.Module):
    def __init__(self, config, input_size) -> None:
        super(BaseEncoder, self).__init__()
    
    def forward(self, x):
        raise NotImplementedError


class BaseDecoder(nn.Module):
    def __init__(self, config, output_size) -> None:
        super(BaseDecoder, self).__init__()
    
    def forward(self, x, hidden_state, cell_state):
        raise NotImplementedError