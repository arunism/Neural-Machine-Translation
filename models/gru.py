import torch.nn as nn
from models.base import BaseEncoder, BaseDecoder

class GruEncoder(BaseEncoder):
    def __init__(self, config, input_size) -> None:
        super(GruEncoder, self).__init__(config, input_size)
        self.gru = nn.GRU(self.embed_size, self.hidden_size, self.layers_count, dropout=self.config.DROPOUT)
    
    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.gru(embedding)
        return hidden, cell


class GruDecoder(BaseDecoder):
    def __init__(self, config, input_size, output_size) -> None:
        super(GruDecoder, self).__init__(config, input_size, output_size)
        self.gru = nn.GRU(self.embed_size, self.hidden_size, self.layers_count, dropout=self.config.DROPOUT)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.gru(embedding, (hidden, cell))
        prediction = self.fc(output)
        prediction = prediction.squeeze(0)
        return prediction, hidden, cell