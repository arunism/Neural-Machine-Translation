import torch.nn as nn

class BaseEncoder(nn.Module):
    def __init__(self, config) -> None:
        super(BaseEncoder, self).__init__()


class BaseDecoder(nn.Module):
    def __init__(self, config) -> None:
        super(BaseDecoder, self).__init__()