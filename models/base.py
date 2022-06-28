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
        self.dropout = nn.Dropout(self.config.DROPOUT)
        self.embedding = nn.Embedding(self.input_size, self.embed_size)
    
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
        self.dropout = nn.Dropout(self.config.DROPOUT)
        self.embedding = nn.Embedding(self.input_size, self.embed_size)
    
    def forward(self):
        raise NotImplementedError


class BaseModel(nn.Module):
    def __init__(self, config, en_input_size, de_input_size, output_size) -> None:
        super(BaseModel, self).__init__()
        self.loss = 0
        self.config = config
        self.en_input_size = en_input_size
        self.de_input_size = de_input_size
        self.output_size = output_size
        self.batch_size = self.config.BATCH_SIZE
        self.encoder_optimizer_name = self.config.ENCODER_OPTIMIZER
        self.decoder_optimizer_name = self.config.DECODER_OPTIMIZER
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self):
        raise NotImplementedError
    
    def get_optimizer(self, optimizer, model):
        if optimizer.lower() == 'adam':
            return torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        elif optimizer.lower() == 'adadelta':
            return torch.optim.Adadelta(model.parameters(), lr=self.config.LEARNING_RATE)
        elif optimizer.lower() == 'adagrad':
            return torch.optim.Adagrad(model.parameters(), lr=self.config.LEARNING_RATE)
        elif optimizer.lower() == 'rmsprop':
            return torch.optim.RMSprop(model.parameters(), lr=self.config.LEARNING_RATE)
        elif optimizer.lower() == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=self.config.LEARNING_RATE)
        return None