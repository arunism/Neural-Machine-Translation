import random
from turtle import forward
from unicodedata import bidirectional
import torch
import torch.nn as nn
from models.base import BaseEncoder, BaseDecoder, BaseModel

class LstmAttenEncoder(BaseEncoder):
    def __init__(self, config, input_size) -> None:
        super(LstmAttenEncoder, self).__init__(config, input_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.layers_count, 
                            dropout=self.config.DROPOUT, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size*2, self.hidden_size)
    
    def forward(self, input):
        embedding = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.lstm(embedding)
        hidden = self.fc(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc(torch.cat((cell[0:1], cell[1:2]), dim=2))
        return output, hidden, cell


class Attention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = self.config.HIDDEN_SIZE
        self.energy = nn.Linear(self.hidden_size*3, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, output, hidden, cell):
        src_len = output.size(0)
        hidden = hidden.repeat(src_len, 1, 1)
        energy = self.relu(self.energy(torch.cat((hidden, output), dim=2)))
        attention = self.softmax(energy)
        return attention.permute(1, 2, 0)


class LstmAttenDecoder(BaseDecoder):
    def __init__(self, config, input_size, output_size) -> None:
        super(LstmAttenDecoder, self).__init__(config, input_size, output_size)
        self.attention = Attention(self.config)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.layers_count, dropout=self.config.DROPOUT)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, input, output, hidden, cell):
        input = input.unsqueeze(0)
        embedding = self.dropout(self.embedding(input))
        attention = self.attention(output, hidden, cell)
        output = output.permute(1, 0, 2)
        context_vector = torch.bmm(attention, output).permute(1, 0, 2)
        input = torch.cat((context_vector, embedding), dim=2)
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        prediction = self.fc(output)
        prediction = prediction.squeeze(0)
        return prediction, hidden, cell