import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseEncoder, BaseDecoder, BaseModel

class GruEncoder(BaseEncoder):
    def __init__(self, config, input_size) -> None:
        super(GruEncoder, self).__init__(config, input_size)
        self.gru = nn.GRU(self.embed_size, self.hidden_size, self.layers_count, dropout=self.config.DROPOUT)
    
    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.gru(embedding)
        return hidden, cell