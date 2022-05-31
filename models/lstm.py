import torch.nn as nn
from models.base import BaseEncoder, BaseDecoder

class Encoder(BaseEncoder):
    def __init__(self, config) -> None:
        super(BaseEncoder, self).__init__(config)
    
    def forward(self, x):
        pass


class Decoder(BaseDecoder):
    def __init__(self, config) -> None:
        super(BaseDecoder, self).__init__(config)
    
    def forward(self, x):
        pass


class LstmModel(nn.Module):
    def __init__(self) -> None:
        super(LstmModel, self).__init__()
        
    def forward(self):
        pass