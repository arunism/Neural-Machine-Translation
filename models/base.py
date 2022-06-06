import torch.nn as nn

class BaseEncoder(nn.Module):
    def __init__(self, config, input_size) -> None:
        super(BaseEncoder, self).__init__()
        self.config = config
        self.input_size = input_size
        self.embed_size = self.config.EMBED_SIZE
        self.layers_count = self.config.LAYERS_COUNT
        self.hidden_size = self.config.HIDDEN_SIZE
        self.dropout = self.config.DROPOUT
        self.embed_layer = nn.Embedding(self.input_size, self.embed_size)
    
    def forward(self, x):
        raise NotImplementedError


class BaseDecoder(nn.Module):
    def __init__(self, config, output_size) -> None:
        super(BaseDecoder, self).__init__()
        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        self.embed_size = self.config.EMBED_SIZE
        self.layers_count = self.config.LAYERS_COUNT
        self.hidden_size = self.config.HIDDEN_SIZE
        self.dropout = self.config.DROPOUT
        self.embed_layer = nn.Embedding(self.input_size, self.embed_size)
    
    def forward(self, x, hidden_state, cell_state):
        raise NotImplementedError