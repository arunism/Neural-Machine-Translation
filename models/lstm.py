from torch import dropout
import torch.nn as nn
from models.base import BaseEncoder, BaseDecoder

class Encoder(BaseEncoder):
    def __init__(self, config, input_size) -> None:
        super(Encoder, self).__init__(config, input_size)
        self.config = config
        self.input_size = input_size
        self.embed_size = self.config.EMBED_SIZE
        self.layers_count = self.config.LAYERS_COUNT
        self.hidden_size = self.config.HIDDEN_SIZE
        self.dropout = self.config.DROPOUT

        self.embed_layer = nn.Embedding(self.input_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.layers_count, dropout=self.dropout)
        print(self.lstm)
    
    def forward(self, x):
        embedding = self.dropout(self.embed_layer(x))
        output, (hidden_state, cell_state) = self.lstm(embedding)
        return hidden_state, cell_state


class Decoder(BaseDecoder):
    def __init__(self, config, input_size, output_size) -> None:
        super(Decoder, self).__init__(config, output_size)
        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        self.embed_size = self.config.EMBED_SIZE
        self.layers_count = self.config.LAYERS_COUNT
        self.hidden_size = self.config.HIDDEN_SIZE
        self.dropout = self.config.DROPOUT

        self.embed_layer = nn.Embedding(self.input_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.layers_count, dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, x, hidden_state, cell_state):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embed_layer(x))
        output, (hidden_state, cell_state) = self.lstm(embedding, (hidden_state, cell_state))
        predictions = self.fc(output)
        predictions = predictions.squeeze(0)
        return predictions, hidden_state, cell_state


class LstmModel(nn.Module):
    def __init__(self, config, input_size, output_size) -> None:
        super(LstmModel, self).__init__()
        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        self.encoder = Encoder(self.config, self.input_size)
        self.decoder = Decoder(self.config, self.input_size, self.output_size)
        
    def forward(self):
        pass