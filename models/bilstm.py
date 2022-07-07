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


class LstmAttenModel(BaseModel):
    def __init__(self, config, en_input_size, de_input_size, output_size) -> None:
        super(LstmAttenModel, self).__init__(config, en_input_size, de_input_size, output_size)
        self.encoder = LstmAttenEncoder(self.config, self.en_input_size)
        self.decoder = LstmAttenDecoder(self.config, self.de_input_size, self.output_size)
        self.encoder_optimizer = self.get_optimizer(self.encoder_optimizer_name, self.encoder)
        self.decoder_optimizer = self.get_optimizer(self.decoder_optimizer_name, self.decoder)

    def forward(self, src_tensor, target_tensor, tf=0.5):
        target_len = target_tensor.size(0)
        batch_len = src_tensor.size(1)
        encoder_output, encoder_hidden, encoder_cell = self.encoder(src_tensor)
        outputs = torch.zeros(target_len, batch_len, self.output_size).to(self.device)
        x = target_tensor[0]
        for i in range(target_len):
            output, hidden, cell = self.decoder(x, encoder_output, encoder_hidden, encoder_cell)
            outputs[i] = output
            prediction = output.argmax(1)
            x = target_tensor[i] if random.random() < tf else prediction
        return outputs