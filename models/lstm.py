import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseEncoder, BaseDecoder, BaseModel

class LstmEncoder(BaseEncoder):
    def __init__(self, config, input_size) -> None:
        super(LstmEncoder, self).__init__(config, input_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.layers_count, dropout=self.dropout)
    
    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        output, (hidden_state, cell_state) = self.lstm(embedding)
        return hidden_state, cell_state


class LstmDecoder(BaseDecoder):
    def __init__(self, config, input_size, output_size) -> None:
        super(LstmDecoder, self).__init__(config, input_size, output_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.layers_count, dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x, hidden):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        output = F.relu(embedding)
        output, hidden = self.lstm(output, hidden)
        prediction = self.softmax(self.fc(output[0]))
        return prediction, hidden


class LstmModel(BaseModel):
    def __init__(self, config, input_size, output_size) -> None:
        super(LstmModel, self).__init__(config, input_size, output_size)
        self.encoder = LstmEncoder(self.config, self.input_size)
        self.decoder = LstmDecoder(self.config, self.input_size, self.output_size)
        self.encoder_optimizer = self.get_optimizer(self.encoder_optimizer_name, self.encoder)
        self.decoder_optimizer = self.get_optimizer(self.decoder_optimizer_name, self.decoder)

    def forward(self, src, target, tf=0.5):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        src_length = src.shape[0]
        target_length = target.shape[0]
        print(src_length)
        print(target_length)