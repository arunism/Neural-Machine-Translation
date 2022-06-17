import torch
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
        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
    
    def forward(self):
        raise NotImplementedError


class BaseDecoder(nn.Module):
    def __init__(self, config, input_size, output_size) -> None:
        super(BaseDecoder, self).__init__()
        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        self.embed_size = self.config.EMBED_SIZE
        self.layers_count = self.config.LAYERS_COUNT
        self.hidden_size = self.config.HIDDEN_SIZE
        self.dropout = self.config.DROPOUT
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
    
    def forward(self):
        raise NotImplementedError


class BaseModel(nn.Module):
    def __init__(self, config, input_size, output_size) -> None:
        super(BaseModel, self).__init__()
        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        self.encoder_optimizer_name = self.config.ENCODER_OPTIMIZER
        self.decoder_optimizer_name = self.config.DECODER_OPTIMIZER
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self):
        raise NotImplementedError
    
    def get_optimizer(self, optimizer, model):
        if optimizer == 'adam':
            return torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        elif optimizer == 'adadelta':
            return torch.optim.Adadelta(model.parameters(), lr=self.config.LEARNING_RATE)
        elif optimizer == 'adagrad':
            return torch.optim.Adagrad(model.parameters(), lr=self.config.LEARNING_RATE)
        elif optimizer == 'rmsprop':
            return torch.optim.RMSprop(model.parameters(), lr=self.config.LEARNING_RATE)
        elif optimizer == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=self.config.LEARNING_RATE)
        return None